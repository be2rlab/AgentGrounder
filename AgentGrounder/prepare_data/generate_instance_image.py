import argparse
import os
import sys
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image
import open3d as o3d
from scipy.spatial import ConvexHull

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from inference.utils import load_bboxes
from utils.config_loader import load_configuration

renderer = None

def get_parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default=Path('PCGrounder/configs/scanrefer.yaml'), help="Path to config")

    args = parser.parse_args()
    return args


def load_mesh(ply_path: str) -> o3d.geometry.TriangleMesh:
    """Load a triangle mesh from a PLY file and compute normals."""
    mesh = o3d.io.read_triangle_mesh(ply_path)
    mesh.compute_vertex_normals()
    return mesh


def read_align_matrix(meta_file_path: str) -> np.ndarray:
    """
    Read the axis alignment matrix from a ScanNet meta (.txt) file.
    Returns a 4x4 numpy array (identity if not found).
    """
    axis_align_matrix = None
    with open(meta_file_path, 'r') as f:
        for line in f:
            if 'axisAlignment' in line:
                axis_align_matrix = [float(x) for x in line.rstrip().strip('axisAlignment = ').split()]
                break

    if axis_align_matrix is not None:
        return np.array(axis_align_matrix).reshape((4, 4))
    return np.eye(4)


# ---------------------------------------------------------------------------
#  Camera-position helpers
# ---------------------------------------------------------------------------

def is_point_in_bbox(point, bbox):
    """
    Check if a 3D point is inside an axis-aligned bounding box.

    Args:
        point: np.array [x, y, z] – the 3D point to test.
        bbox: [cx, cy, cz, w, l, h] – center and dimensions of the box.

    Returns:
        True if the point is inside the bbox.
    """
    cx, cy, cz, w, l, h = bbox
    return (
        abs(point[0] - cx) <= w / 2
        and abs(point[1] - cy) <= l / 2
        and abs(point[2] - cz) <= h / 2
    )


def _point_in_polygon_2d(px, py, polygon):
    """
    Ray-casting algorithm for point-in-polygon test in 2D.

    Casts a horizontal ray from (px, py) to +∞ and counts how many
    edges of the polygon it crosses.  An odd count means the point
    is inside.

    Args:
        px, py: Coordinates of the test point.
        polygon: np.ndarray (N, 2) – ordered polygon vertices.

    Returns:
        True if (px, py) is strictly inside the polygon.
    """
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and \
           (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _shrink_polygon(vertices_2d, margin):
    """
    Shrink a convex polygon inward by *margin* metres.

    Each edge is moved inward (in the direction of the inward-pointing
    normal) by *margin*, and the new polygon is formed by intersecting
    the shifted half-planes.  Falls back to centroid-based scaling if
    the geometric offset produces a degenerate result.

    Args:
        vertices_2d: np.ndarray (N, 2) – convex polygon vertices in CCW order.
        margin: Distance (metres) to shrink inward.

    Returns:
        np.ndarray (N, 2) – shrunk polygon vertices.
    """
    n = len(vertices_2d)
    if n < 3 or margin <= 0:
        return vertices_2d.copy()

    centroid = vertices_2d.mean(axis=0)

    # --- Attempt: offset each edge inward and intersect consecutive edges ---
    #  Edge i goes from vertices_2d[i] to vertices_2d[(i+1) % n].
    #  Inward normal for a CCW polygon: rotate edge direction 90° clockwise.
    new_pts = []
    success = True
    for i in range(n):
        p1 = vertices_2d[i]
        p2 = vertices_2d[(i + 1) % n]
        edge = p2 - p1
        length = np.linalg.norm(edge)
        if length < 1e-12:
            success = False
            break
        # Inward normal (90° CW rotation of CCW edge)
        inward_normal = np.array([edge[1], -edge[0]]) / length
        # Verify inward direction points towards centroid
        if np.dot(inward_normal, centroid - p1) < 0:
            inward_normal = -inward_normal
        new_pts.append((p1 + margin * inward_normal, p2 + margin * inward_normal))

    if success:
        # Intersect consecutive offset edges to get new vertices
        shrunk = []
        for i in range(n):
            # Line i: new_pts[i][0] -> new_pts[i][1]
            # Line i+1: new_pts[(i+1)%n][0] -> new_pts[(i+1)%n][1]
            a1, a2 = new_pts[i]
            b1, b2 = new_pts[(i + 1) % n]
            da = a2 - a1
            db = b2 - b1
            denom = da[0] * db[1] - da[1] * db[0]
            if abs(denom) < 1e-12:
                success = False
                break
            t = ((b1[0] - a1[0]) * db[1] - (b1[1] - a1[1]) * db[0]) / denom
            shrunk.append(a1 + t * da)

    if success and len(shrunk) == n:
        shrunk = np.array(shrunk)
        # Sanity: all new vertices should be closer to centroid than originals
        orig_dists = np.linalg.norm(vertices_2d - centroid, axis=1)
        new_dists = np.linalg.norm(shrunk - centroid, axis=1)
        if np.all(new_dists < orig_dists + 1e-6):
            return shrunk

    # Fallback: scale uniformly toward centroid
    max_dist = np.max(np.linalg.norm(vertices_2d - centroid, axis=1))
    if max_dist < 1e-6:
        return vertices_2d.copy()
    scale = max(0.0, 1.0 - margin / max_dist)
    return centroid + scale * (vertices_2d - centroid)


def compute_room_polygon(vertices, z_margin=0.1, xy_margin=0.15):
    """
    Build a 2D convex-hull boundary polygon on the XY plane from the
    mesh vertices, together with Z-axis bounds.

    The polygon is shrunk inward by *xy_margin* so that cameras placed
    right against a wall (or just outside it due to floating-point
    imprecision) are correctly rejected.

    Args:
        vertices: np.ndarray (N, 3) – all mesh vertices.
        z_margin: Extra margin (metres) added above and below the
                  vertex Z range so cameras just outside the floor /
                  ceiling are still accepted.
        xy_margin: Distance (metres) to shrink the hull inward.
                   Set to 0 to disable.

    Returns:
        dict with:
            'hull_vertices_2d': np.ndarray (K, 2) – ordered (CCW)
                convex-hull vertices on XY, shrunk inward.
            'z_min': float – lowest Z minus z_margin.
            'z_max': float – highest Z plus z_margin.
    """
    xy = vertices[:, :2]  # project onto XY plane
    hull = ConvexHull(xy)

    # Ordered hull vertices (CCW for 2-D hulls)
    hull_pts = xy[hull.vertices]

    # Shrink inward so cameras don't sit right at the wall
    if xy_margin > 0:
        hull_pts = _shrink_polygon(hull_pts, xy_margin)

    z_min = float(vertices[:, 2].min()) - z_margin
    z_max = float(vertices[:, 2].max()) + z_margin

    return {
        "hull_vertices_2d": hull_pts,
        "z_min": z_min,
        "z_max": z_max,
    }


def is_point_in_room(point, room_bounds):
    """
    Check if a 3D point is within the room boundaries.

    Supports two formats for *room_bounds*:
      1. **Polygon-based** (preferred): dict with 'hull_vertices_2d',
         'z_min', 'z_max'.  XY is tested with a ray-casting
         point-in-polygon algorithm; Z is range-checked.
      2. **Legacy AABB**: dict with 'min' and 'max' np.arrays.

    Args:
        point: np.array [x, y, z].
        room_bounds: dict – see above.

    Returns:
        True if the point is inside the room.
    """
    if "hull_vertices_2d" in room_bounds:
        in_z = room_bounds["z_min"] <= point[2] <= room_bounds["z_max"]
        if not in_z:
            return False
        return _point_in_polygon_2d(
            point[0], point[1], room_bounds["hull_vertices_2d"],
        )
    else:
        # Fallback: simple AABB check
        return bool(
            np.all(point >= room_bounds["min"])
            and np.all(point <= room_bounds["max"])
        )


def extract_object_face_data(mesh, bbox):
    """
    Extract face normals and areas for triangles whose centroids lie
    inside the object's bounding box.

    Args:
        mesh: Open3D TriangleMesh (with vertex normals computed).
        bbox: [cx, cy, cz, w, l, h] – object bounding box.

    Returns:
        face_normals: np.ndarray (K, 3) – unit normals of matching faces.
        face_areas:   np.ndarray (K,)   – area of each matching face.
        Returns (None, None) if no faces are found inside the bbox.
    """
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    if len(triangles) == 0:
        return None, None

    # Compute centroids of all faces
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    centroids = (v0 + v1 + v2) / 3.0

    # Filter faces whose centroid lies inside the bbox
    cx, cy, cz, w, l, h = bbox
    half = np.array([w / 2, l / 2, h / 2])
    center_bbox = np.array([cx, cy, cz])
    mask = np.all(np.abs(centroids - center_bbox) <= half, axis=1)

    if not np.any(mask):
        return None, None

    v0 = v0[mask]
    v1 = v1[mask]
    v2 = v2[mask]

    # Compute face normals via cross product
    edge1 = v1 - v0
    edge2 = v2 - v0
    cross = np.cross(edge1, edge2)
    areas = np.linalg.norm(cross, axis=1) * 0.5

    # Normalise to unit normals (avoid division by zero)
    norms = np.linalg.norm(cross, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    face_normals = cross / norms

    return face_normals, areas


def compute_surface_visibility(eye, center, face_normals, face_areas):
    """
    Compute the fraction of the object's surface area that is visible
    (front-facing) from the given camera position.

    A face is considered visible when its normal has a positive dot
    product with the view direction (camera → object centre).

    Args:
        eye: np.array [x, y, z] – camera position.
        center: np.array [x, y, z] – object centre.
        face_normals: np.ndarray (K, 3) – unit face normals.
        face_areas: np.ndarray (K,) – face areas.

    Returns:
        float in [0, 1] – area-weighted visibility ratio.
    """
    view_dir = center - eye
    norm = np.linalg.norm(view_dir)
    if norm < 1e-12:
        return 0.0
    view_dir = view_dir / norm

    # dot > 0 means the face normal points towards the camera
    dots = face_normals @ view_dir
    visible_area = np.sum(face_areas[dots > 0])
    total_area = np.sum(face_areas)

    if total_area < 1e-12:
        return 0.0
    return float(visible_area / total_area)


def compute_candidate_eye_positions(center, dist, num_candidates=100, seed=None):
    """
    Randomly sample *num_candidates* camera positions on the upper
    hemisphere of a sphere of radius *dist* centred at *center*.

    A large pool of candidates is generated so that the best ones can
    be selected later.

    Args:
        center: np.array [x, y, z] – look-at target.
        dist: Radius of the sampling sphere.
        num_candidates: Number of candidate viewpoints to generate.
        seed: Optional random seed for reproducibility.

    Returns:
        List of np.array eye positions.
    """
    rng = np.random.default_rng(seed)

    eye_positions = []
    for _ in range(num_candidates):
        # Uniform azimuth in [0, 2π)
        azimuth = rng.uniform(0, 2 * np.pi)
        # Polar angle from upper hemisphere only: θ ∈ [0, π/2)
        # Use cosine-weighted sampling for uniform distribution on the sphere
        cos_polar = rng.uniform(0, 1)          # cos(θ) ∈ (0, 1]
        polar = np.arccos(cos_polar)           # θ ∈ [0, π/2)

        offset = np.array([
            dist * np.sin(polar) * np.cos(azimuth),
            dist * np.sin(polar) * np.sin(azimuth),
            dist * np.cos(polar),               # always positive → above center
        ])
        eye_positions.append(center + offset)

    return eye_positions


def filter_eye_positions(eye_positions, other_bboxes, room_bounds):
    """
    Remove camera positions that are invalid:
      1. Outside the room boundary polygon (XY) or Z range.
      2. Inside another object's bounding box.

    The room-boundary check is performed first because it is the
    cheapest bulk filter and rejects the most candidates.

    Args:
        eye_positions: List of np.array [x, y, z].
        other_bboxes: List of [cx, cy, cz, w, l, h] for other objects.
        room_bounds: dict with 'hull_vertices_2d' / 'z_min' / 'z_max',
                     or legacy dict with 'min' / 'max', or None.

    Returns:
        Filtered list of valid eye positions.
    """
    valid = []
    for eye in eye_positions:
        # 1. Room boundary check (polygon + Z range)
        if room_bounds is not None and not is_point_in_room(eye, room_bounds):
            continue
        # 2. Collision with other objects
        if any(is_point_in_bbox(eye, ob) for ob in other_bboxes):
            continue
        valid.append(eye)
    return valid


def score_eye_positions(
    eye_positions, center, dist, other_bboxes,
    face_normals=None, face_areas=None,
):
    """
    Score each eye position based on how well it can see the target object.

    Scoring criteria (each normalised to [0, 1], then combined):
      1. **Surface visibility** (weight 0.35):
         Area-weighted fraction of the object's faces that are front-facing
         to the camera.  Computed from mesh face normals inside the bbox.
         Falls back to a neutral 0.5 if face data is unavailable.
      2. **Elevation quality** (weight 0.25):
         Prefer moderate elevation angles (≈30-50°) that show both the top
         and the side of the object.  A Gaussian centred at 40° is used.
      3. **Clearance from other objects** (weight 0.20):
         Prefer cameras that are far from other objects so the view is
         less likely to be occluded.  Uses the minimum distance to any
         other bbox centre, normalised by *dist*.
      4. **Horizon visibility** (weight 0.20):
         Prefer cameras that are not too close to the ground plane
         (z ≈ center_z) nor directly above (z ≈ center_z + dist).
         A gentle penalty on extreme polar angles.

    Args:
        eye_positions: List of np.array [x, y, z] – valid candidates.
        center: np.array [x, y, z] – object centre.
        dist: Sphere radius (used for normalisation).
        other_bboxes: List of [cx, cy, cz, w, l, h] bboxes.
        face_normals: np.ndarray (K, 3) – unit normals of object faces,
                      or None if unavailable.
        face_areas: np.ndarray (K,) – area of each object face,
                    or None if unavailable.

    Returns:
        np.array of scores, same length as *eye_positions*.
    """
    has_face_data = face_normals is not None and face_areas is not None
    scores = np.zeros(len(eye_positions))

    for i, eye in enumerate(eye_positions):
        direction = eye - center
        elevation_rad = np.arcsin(np.clip(direction[2] / dist, -1.0, 1.0))
        elevation_deg = np.degrees(elevation_rad)

        # 1. Surface visibility
        if has_face_data:
            visibility_score = compute_surface_visibility(
                eye, center, face_normals, face_areas,
            )
        else:
            visibility_score = 0.5  # neutral fallback

        # 2. Elevation quality – Gaussian centred at 40°, σ = 25°
        elev_score = np.exp(-0.5 * ((elevation_deg - 40.0) / 25.0) ** 2)

        # 3. Clearance from other objects
        if other_bboxes:
            min_dist_to_other = min(
                np.linalg.norm(eye - np.array(ob[:3])) for ob in other_bboxes
            )
            clearance_score = np.clip(min_dist_to_other / (dist * 2), 0.0, 1.0)
        else:
            clearance_score = 1.0

        # 4. Horizon visibility – penalise very low (<10°) or very high (>80°)
        if elevation_deg < 10:
            horizon_score = elevation_deg / 10.0
        elif elevation_deg > 80:
            horizon_score = (90.0 - elevation_deg) / 10.0
        else:
            horizon_score = 1.0

        scores[i] = (
            0.35 * visibility_score
            + 0.25 * elev_score
            + 0.20 * clearance_score
            + 0.20 * horizon_score
        )

    return scores


def select_best_eye_positions(eye_positions, scores, center, max_views=9,
                              min_score=0.15):
    """
    Select up to *max_views* viewpoints from the candidates using
    **greedy farthest-point sampling weighted by quality score**.

    Candidates whose score is below *min_score* are discarded first —
    these are typically cameras that sit inside the target object and
    see almost no front-facing surface.

    This ensures that the selected views are both high-quality AND
    well-spread around the object (angular diversity).

    Algorithm:
      0. Remove all candidates with score < min_score.
      1. Start with the candidate that has the highest score.
      2. For each remaining candidate compute:
            priority = score * min_angular_distance_to_already_selected
      3. Pick the candidate with the highest priority.
      4. Repeat until *max_views* are selected or candidates exhausted.

    Args:
        eye_positions: List of np.array [x, y, z].
        scores: np.array of per-candidate scores.
        center: np.array [x, y, z] – object centre.
        max_views: Maximum number of views to keep.
        min_score: Minimum acceptable score. Candidates below this
                   threshold are dropped (likely inside the object).

    Returns:
        List of np.array – the selected eye positions.
    """
    # 0. Drop low-quality candidates (e.g. cameras inside the object)
    keep = [i for i in range(len(eye_positions)) if scores[i] >= min_score]
    if len(keep) == 0:
        # Fallback: keep the single best candidate rather than returning nothing
        best = int(np.argmax(scores))
        return [eye_positions[best]]

    eye_positions = [eye_positions[i] for i in keep]
    scores = scores[keep]
    n = len(eye_positions)

    if n <= max_views:
        return eye_positions

    # Precompute unit direction vectors from center
    directions = np.array([eye - center for eye in eye_positions])
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    unit_dirs = directions / norms

    selected_indices = []
    remaining = set(range(n))

    # 1. Seed with the highest-scored candidate
    best_idx = int(np.argmax(scores))
    selected_indices.append(best_idx)
    remaining.discard(best_idx)

    # 2. Greedy farthest-point sampling
    while len(selected_indices) < max_views and remaining:
        best_priority = -1.0
        best_candidate = None

        for idx in remaining:
            # Angular distance to the closest already-selected view
            min_ang_dist = min(
                np.arccos(np.clip(np.dot(unit_dirs[idx], unit_dirs[s]), -1.0, 1.0))
                for s in selected_indices
            )
            priority = scores[idx] * min_ang_dist
            if priority > best_priority:
                best_priority = priority
                best_candidate = idx

        selected_indices.append(best_candidate)
        remaining.discard(best_candidate)

    return [eye_positions[i] for i in selected_indices]


# ---------------------------------------------------------------------------
#  Rendering helpers
# ---------------------------------------------------------------------------

def _get_renderer(image_size):
    """Return (and lazily create) the global offscreen renderer."""
    global renderer
    if renderer is None:
        renderer = o3d.visualization.rendering.OffscreenRenderer(image_size, image_size)
        renderer.scene.set_background([0.0, 0.0, 0.0, 1.0])
    return renderer


def render_views(mesh, center, eye_positions, image_size):
    """
    Render the mesh from each eye position looking at *center*.

    Returns:
        List of np.ndarray images (H, W, 3).
    """
    rdr = _get_renderer(image_size)

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"
    mat.base_color = [0.85, 0.85, 0.85, 1.0]

    if rdr.scene.has_geometry("scene"):
        rdr.scene.remove_geometry("scene")
    rdr.scene.add_geometry("scene", mesh, mat)

    views = []
    for eye in eye_positions:
        rdr.scene.camera.look_at(
            center.tolist(),
            eye.tolist(),
            [0.0, 0.0, 1.0],
        )
        img_o3d = rdr.render_to_image()
        img_np = np.asarray(img_o3d)
        views.append(img_np[:, :, :3] if img_np.shape[-1] == 4 else img_np)
    return views


def _optimal_grid_layout(n):
    """
    Find (rows, cols) such that rows * cols >= n, the grid is as close
    to a square as possible, and cols >= rows (landscape orientation).

    Args:
        n: Number of views to arrange.

    Returns:
        (grid_rows, grid_cols)
    """
    if n <= 0:
        return (1, 1)

    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    return (rows, cols)


def build_grid_image(views, image_size):
    """
    Arrange rendered views into a near-square grid, padding with black
    if the number of views doesn't fill the grid exactly.

    Args:
        views: List of np.ndarray images (H, W, 3).
        image_size: Resolution of each individual view.

    Returns:
        np.ndarray of the stitched grid image.
    """
    grid_rows, grid_cols = _optimal_grid_layout(len(views))

    total = grid_rows * grid_cols
    black = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    while len(views) < total:
        views.append(black)

    rows = []
    for r in range(grid_rows):
        row = np.hstack([views[r * grid_cols + c] for c in range(grid_cols)])
        rows.append(row)
    return np.vstack(rows)


# ---------------------------------------------------------------------------
#  Main image-generation entry point
# ---------------------------------------------------------------------------

def gen_img(
    bbox,
    mesh,
    save_path,
    other_bboxes=None,
    room_bounds=None,
    image_size=512,
    dist=2.0,
    max_views=9,
    num_candidates=100,
    seed=None,
):
    """
    Render multi-view images of a mesh around a bbox center and save as
    a near-square grid.  A large pool of candidate camera positions is
    generated, invalid ones are removed, and the best *max_views* are
    selected for rendering.

    Selection criteria:
      - Elevation quality  (prefer ~40° for clear object visibility)
      - Clearance from other objects  (avoid occlusion)
      - Angular diversity  (views spread evenly around the object)

    Args:
        bbox: [x, y, z, w, l, h] – center and dimensions of the object.
        mesh: Open3D TriangleMesh (already aligned).
        save_path: Output image path.
        other_bboxes: Bboxes of other objects (list of [x,y,z,w,l,h]).
        room_bounds: dict {'min': np.array, 'max': np.array} or None.
        image_size: Resolution of each rendered view.
        dist: Camera distance from the bbox center.
        max_views: Maximum number of views to keep after selection.
        num_candidates: Number of random candidates to generate.
        seed: Optional random seed for reproducibility.
    """
    if other_bboxes is None:
        other_bboxes = []

    center = np.array(bbox[:3])

    # 1. Generate a large pool of candidate camera positions
    eye_positions = compute_candidate_eye_positions(
        center, dist, num_candidates=num_candidates, seed=seed,
    )

    # 2. Filter out invalid positions (inside other objects / outside room)
    eye_positions = filter_eye_positions(eye_positions, other_bboxes, room_bounds)

    if len(eye_positions) == 0:
        print(f"[WARN] All camera views are invalid, skipping {save_path}")
        return

    # 3. Extract object face normals for surface-visibility scoring
    face_normals, face_areas = extract_object_face_data(mesh, bbox)

    # 4. Score & select the best views
    scores = score_eye_positions(
        eye_positions, center, dist, other_bboxes,
        face_normals=face_normals, face_areas=face_areas,
    )
    eye_positions = select_best_eye_positions(
        eye_positions, scores, center, max_views=max_views,
    )

    # 5. Render & stitch
    views = render_views(mesh, center, eye_positions, image_size)
    grid = build_grid_image(views, image_size)
    Image.fromarray(grid).save(save_path)


def generate_images_for_room(
    room: str,
    ply_dir: str,
    olt_dir: str,
    output_dir: str,
    image_size: int = 512,
):
    """
    Generate multi-view rendered images for every object in a room.

    Args:
        room: Scene id, e.g. "scene0131_00".
        ply_dir: Directory with *_vh_clean_2.ply mesh files.
        meta_file_dir: Directory with *.txt meta files (axis alignment).
        olt_dir: Directory with per-scene OLT JSON files (bounding boxes).
        output_dir: Root directory to save rendered images.
        image_size: Resolution of each rendered view.
    """
    # --- Load and align mesh ---
    ply_path = os.path.join(ply_dir, room, f"{room}_vh_clean_2.ply")
    if not os.path.exists(ply_path):
        # Try alternative naming: <room>.ply
        ply_path = os.path.join(ply_dir, room, f"{room}.ply")
    if not os.path.exists(ply_path):
        print(f"[WARN] PLY file not found for {room}, skipping.")
        return

    mesh = load_mesh(ply_path)

    meta_path = os.path.join(ply_dir, room, f"{room}.txt")
    if os.path.exists(meta_path):
        axis_align_matrix = read_align_matrix(meta_path)
        mesh.transform(axis_align_matrix)
    else:
        print(f"[WARN] Meta file not found for {room}, using identity alignment.")

    # --- Load bounding boxes ---
    pred_bboxes = load_bboxes(room, olt_dir)

    # Compute room bounds using a convex-hull polygon on the XY plane.
    # This is much more accurate than a simple axis-aligned bounding box
    # for irregularly shaped rooms (L-shape, corridors, etc.).
    vertices = np.asarray(mesh.vertices)
    room_bounds = compute_room_polygon(vertices, z_margin=0.3)

    # Create output folder
    room_output_dir = os.path.join(output_dir, room)
    os.makedirs(room_output_dir, exist_ok=True)

    for obj_id in list(pred_bboxes.keys()):
        bbox = pred_bboxes[obj_id]["bbox_3d"]
        x, y, z, w, l, h = bbox

        # Half-diagonal of the bbox (circumradius): accounts for all three
        # dimensions simultaneously and guarantees the object fits in view.
        # Scale by 1.5 for comfortable framing; clamp to a minimum of 0.05 m.
        half_diag = 0.5 * np.sqrt(w ** 2 + l ** 2 + h ** 2)
        suitable_dist = max(half_diag * 1.5, 0.05)

        save_path = os.path.join(room_output_dir, f"object_{obj_id}.png")
        if os.path.exists(save_path):
            print(f"File {save_path} already exists, skipping")
            continue

        # Collect bboxes of all OTHER objects to avoid placing cameras inside them
        other_bboxes = [
            pred_bboxes[oid]["bbox_3d"]
            for oid in pred_bboxes
            if oid != obj_id
        ]

        gen_img(
            bbox=bbox,
            mesh=mesh,
            save_path=save_path,
            other_bboxes=other_bboxes,
            room_bounds=room_bounds,
            image_size=image_size,
            dist=suitable_dist,
            max_views=4,
        )

        print(f"Scene {room}: Saved projection image for object {obj_id}.")


# ---------------------------------------------------------------------------
#  Parallel worker – must be a top-level callable so it can be pickled by
#  multiprocessing.  Each worker process owns its own OffscreenRenderer
#  (the module-level ``renderer`` global starts as None in every child).
# ---------------------------------------------------------------------------

def _worker_generate_room(args):
    """
    Thin wrapper used by ProcessPoolExecutor.

    Returns:
        (room, error_str_or_None)
    """
    room, ply_dir, olt_dir, output_dir, image_size = args
    try:
        generate_images_for_room(
            room=room,
            ply_dir=ply_dir,
            olt_dir=olt_dir,
            output_dir=output_dir,
            image_size=image_size,
        )
        return room, None
    except Exception as exc:  # noqa: BLE001
        return room, str(exc)


if __name__ == "__main__":
    args = get_parser_args()
    config = load_configuration(yaml_path=args.config_path)

    dir_paths = config.experiment.data

    ply_dir = str(dir_paths.ply_dir)
    olt_dir = str(dir_paths.pred_bbox_dir)
    output_dir = str(dir_paths.multi_view_projection_dir)

    scan_ids = sorted([i.split(".")[0] for i in os.listdir(olt_dir)])

    num_workers = min(args.num_workers, len(scan_ids))
    print(
        f"Processing {len(scan_ids)} scenes with {num_workers} worker(s) "
        f"(image_size={args.image_size})"
    )

    worker_args = [
        (room, ply_dir, olt_dir, output_dir, args.image_size)
        for room in scan_ids
    ]

    errors = []

    # Use spawn context to avoid issues with CUDA / OpenGL state inherited
    # from the parent process when using multiprocessing.
    mp_ctx = multiprocessing.get_context("spawn")

    with ProcessPoolExecutor(max_workers=num_workers, mp_context=mp_ctx) as executor:
        futures = {
            executor.submit(_worker_generate_room, wa): wa[0]
            for wa in worker_args
        }

        if tqdm is not None:
            it = tqdm(as_completed(futures), total=len(futures), unit="scene")
        else:
            it = as_completed(futures)

        for future in it:
            room_id = futures[future]
            try:
                _, err = future.result()
                if err:
                    print(f"[ERROR] {room_id}: {err}")
                    errors.append((room_id, err))
            except Exception as exc:  # noqa: BLE001
                print(f"[ERROR] {room_id}: {exc}")
                errors.append((room_id, str(exc)))

    if errors:
        print(f"\n{len(errors)} scene(s) failed:")
        for room_id, err in errors:
            print(f"  {room_id}: {err}")
    else:
        print("\nAll scenes rendered successfully.")

