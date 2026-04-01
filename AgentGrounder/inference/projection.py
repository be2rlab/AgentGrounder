import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    PointsRenderer,
    PointsRasterizationSettings,
    PointsRasterizer,
    AlphaCompositor,
    FoVPerspectiveCameras,
    PerspectiveCameras,
    look_at_view_transform,
)
from PIL import Image, ImageDraw, ImageFont
import os
import cv2
import json
import re
import random
from scipy.spatial import ConvexHull


def render_point_cloud_with_pytorch3d_with_objects(
    objects,
    targets,
    anchors,
    center,
    scan_pc,
    save_dir,
    image_size=680,
    use_color_image=True,
    draw_bbox=False,
    draw_id=False,
    draw_img=False,
    draw_mask=False,
    draw_contour=False,
    device="cuda",
    return_rasterizer=False
):
    """
    Render point cloud with PyTorch3D and annotate with objects, targets, and anchors.

    Args:
        objects (list): List of objects to render.
        targets (list): List of target objects.
        anchors (list): List of anchor objects.
        center (array): Center of the point cloud.
        scan_pc (array): Point cloud data.
        save_dir (str): Directory to save the rendered image.
        image_size (int): Size of the output image.
        use_color_image (bool): Whether to use a color image.
        draw_bbox (bool): Whether to draw bounding boxes.
        draw_id (bool): Whether to draw object IDs.
        draw_img (bool): Whether to draw the image.
        draw_mask (bool): Whether to draw masks.
        draw_contour (bool): Whether to draw contours.
        device (str): Device to use for rendering.

    Returns:
        str: Path to the saved image.
    """
    point_cloud = create_point_cloud(scan_pc, device)
    os.makedirs(save_dir, exist_ok=True)

    obj_bbox_3d = choose_camera_focus(anchors, targets, center)

    cameras = setup_camera(
        anchor_bbox_3d=obj_bbox_3d,
        center=center,
        image_size=image_size,
        camera_distance_factor=1,
        camera_lift=1.5,
        device=device,
        point_cloud=point_cloud,
        calibrate=False,
    )

    image_np, rasterizer = render_point_cloud(point_cloud, cameras, image_size, device)

    color_image = Image.fromarray((image_np * 255).astype(np.uint8))

    if not draw_img:
        width, height = color_image.size
        color_image = Image.new("RGB", (width, height), (255, 255, 255))

    font = ImageFont.truetype(
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf", 15, encoding="unic"
    )

    color = (128, 128, 128)  # grey color
    annotate_image(
        color_image,
        anchors,
        targets,
        cameras,
        image_size,
        font,
        scan_pc=scan_pc,
        depth_map=None,
        bbox_color=color,
        draw_bbox=draw_bbox,
        draw_mask=draw_mask,
        draw_id=draw_id,
        draw_contour=draw_contour,
    )

    f_name = f"{save_dir}/rendered.png"
    color_image.save(f_name)
    print(f"Annotated image saved at {f_name}")
    color_image.close()

    if return_rasterizer:
        return f_name, rasterizer

    return f_name


def choose_camera_focus(anchors, targets, center):
    '''
    Determine focus point for the camera using mean target and mean anchor positions.
    '''
    target_mean = (
        torch.stack(
            [torch.tensor(t["bbox_3d"][:3], dtype=torch.float32) for t in targets], dim=0
        ).mean(dim=0)
        if targets
        else None
    )
    anchor_mean = (
        torch.stack(
            [torch.tensor(a["bbox_3d"][:3], dtype=torch.float32) for a in anchors], dim=0
        ).mean(dim=0)
        if anchors
        else None
    )

    means = [m for m in (target_mean, anchor_mean) if m is not None]
    obj_bbox_3d = (
        torch.stack(means, dim=0).mean(dim=0)
        if means
        else torch.tensor(center, dtype=torch.float32)
    )

    return obj_bbox_3d


def annotate_image(
    color_image,
    anchors,
    targets,
    cameras,
    image_size,
    font,
    depth_map,
    scan_pc,
    bbox_color=(0, 255, 0),
    draw_bbox=False,
    draw_mask=False,
    draw_contour=False,
    draw_id=False,
):
    """
    Annotate the image with bounding boxes, masks, contours, and IDs.

    Args:
        color_image (Image): The image to annotate.
        anchors (list): List of anchor objects.
        targets (list): List of target objects.
        cameras (PerspectiveCameras): Camera settings.
        image_size (int): Size of the output image.
        font (ImageFont): Font for drawing text.
        depth_map (array): Depth map of the point cloud.
        scan_pc (array): Point cloud data.
        bbox_color (tuple): Color for bounding boxes.
        draw_bbox (bool): Whether to draw bounding boxes.
        draw_mask (bool): Whether to draw masks.
        draw_contour (bool): Whether to draw contours.
        draw_id (bool): Whether to draw object IDs.
    """
    draw = ImageDraw.Draw(color_image, "RGBA")

    if draw_mask:
        draw_masks(draw, targets, cameras, scan_pc, image_size)

    if draw_contour:
        draw_contours(draw, targets, cameras, scan_pc, image_size)

    if draw_bbox:
        draw_bboxes(draw, anchors + targets, cameras, image_size, bbox_color)

    if draw_id:
        draw_ids(draw, anchors + targets, cameras, image_size, font)

    return


def draw_masks(draw, targets, cameras, scan_pc, image_size):
    """
    Draw masks on the image.

    Args:
        draw (ImageDraw): ImageDraw object.
        targets (list): List of target objects.
        cameras (PerspectiveCameras): Camera settings.
        scan_pc (array): Point cloud data.
        image_size (int): Size of the output image.
    """
    for bbox in targets:
        bbox_id = bbox["bbox_id"]
        obj_label = bbox["label"]
        x, y, z, w, l, h = bbox["bbox_3d"]

        in_bbox_points = scan_pc[
            (scan_pc[:, 0] >= x - w / 2)
            & (scan_pc[:, 0] <= x + w / 2)
            & (scan_pc[:, 1] >= y - l / 2)
            & (scan_pc[:, 1] <= y + l / 2)
            & (scan_pc[:, 2] >= z - h / 2)
            & (scan_pc[:, 2] <= z + h / 2)
        ]

        projected_points = cameras.transform_points_screen(
            torch.tensor(in_bbox_points[:, :3]).cuda(),
            image_size=(image_size, image_size),
        )
        projected_points = projected_points[..., :2]

        visible_points = [(int(px), int(py)) for px, py in projected_points]

        mask_color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
            100,
        )  # Random color with transparency
        draw.polygon(visible_points, fill=mask_color)


def draw_contours(draw, targets, cameras, scan_pc, image_size):
    """
    Draw contours on the image.

    Args:
        draw (ImageDraw): ImageDraw object.
        targets (list): List of target objects.
        cameras (PerspectiveCameras): Camera settings.
        scan_pc (array): Point cloud data.
        image_size (int): Size of the output image.
    """
    for bbox in targets:
        bbox_id = bbox["bbox_id"]
        obj_label = bbox["label"]
        x, y, z, w, l, h = bbox["bbox_3d"]

        in_bbox_points = scan_pc[
            (scan_pc[:, 0] >= x - w / 2)
            & (scan_pc[:, 0] <= x + w / 2)
            & (scan_pc[:, 1] >= y - l / 2)
            & (scan_pc[:, 1] <= y + l / 2)
            & (scan_pc[:, 2] >= z - h / 2)
            & (scan_pc[:, 2] <= z + h / 2)
        ]

        projected_points = cameras.transform_points_screen(
            torch.tensor(in_bbox_points[:, :3]).cuda(),
            image_size=(image_size, image_size),
        )
        projected_points = projected_points[..., :2]

        visible_points = [(int(px), int(py)) for px, py in projected_points]

        points_array = np.array(visible_points)
        try:
            hull = ConvexHull(points_array)  # Compute the convex hull
            contour_points = points_array[hull.vertices]  # Get contour points in order
            contour_points = [(int(x), int(y)) for x, y in contour_points]

            contour_color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
                255,
            )  # Random color with transparency
            draw.line(
                contour_points + [contour_points[0]], fill=contour_color, width=3
            )  # Close the contour loop
        except:
            pass


def draw_bboxes(draw, bboxes, cameras, image_size, bbox_color):
    """
    Draw bounding boxes on the image.

    Args:
        draw (ImageDraw): ImageDraw object.
        bboxes (list): List of bounding boxes.
        cameras (PerspectiveCameras): Camera settings.
        image_size (int): Size of the output image.
        bbox_color (tuple): Color for bounding boxes.
    """
    for bbox in bboxes:
        x, y, z, w, l, h = bbox["bbox_3d"]

        # Define the eight corners of the 3D bounding box
        corners = [
            [x - w / 2, y - l / 2, z - h / 2],
            [x - w / 2, y + l / 2, z - h / 2],
            [x + w / 2, y - l / 2, z - h / 2],
            [x + w / 2, y + l / 2, z - h / 2],
            [x - w / 2, y - l / 2, z + h / 2],
            [x - w / 2, y + l / 2, z + h / 2],
            [x + w / 2, y - l / 2, z + h / 2],
            [x + w / 2, y + l / 2, z + h / 2],
        ]

        # Project the 3D corners to the 2D image plane
        corners_2d = cameras.transform_points_screen(
            torch.tensor(corners).cuda(), image_size=(image_size, image_size)
        )
        corners_2d = corners_2d[..., :2].cpu().numpy()

        # Check if each corner is within the image boundaries
        valid_corners = [
            (0 <= x < image_size and 0 <= y < image_size) for x, y in corners_2d
        ]

        # Skip drawing if all corners are out of image boundaries
        if not any(valid_corners):
            continue

        # Draw the 3D bounding box
        draw_bbox_function(draw, corners_2d, valid_corners, bbox_color)


def draw_ids(draw, bboxes, cameras, image_size, font):
    """
    Draw object IDs on the image.

    Args:
        draw (ImageDraw): ImageDraw object.
        bboxes (list): List of bounding boxes.
        cameras (PerspectiveCameras): Camera settings.
        image_size (int): Size of the output image.
        font (ImageFont): Font for drawing text.
    """
    for bbox in bboxes:
        bbox_id = bbox["bbox_id"]
        x, y, z, w, l, h = bbox["bbox_3d"]

        # Define the eight corners of the 3D bounding box
        corners = [
            [x - w / 2, y - l / 2, z - h / 2],
            [x - w / 2, y + l / 2, z - h / 2],
            [x + w / 2, y - l / 2, z - h / 2],
            [x + w / 2, y + l / 2, z - h / 2],
            [x - w / 2, y - l / 2, z + h / 2],
            [x - w / 2, y + l / 2, z + h / 2],
            [x + w / 2, y - l / 2, z + h / 2],
            [x + w / 2, y + l / 2, z + h / 2],
        ]

        # Project the 3D corners to the 2D image plane
        corners_2d = cameras.transform_points_screen(
            torch.tensor(corners).cuda(), image_size=(image_size, image_size)
        )
        corners_2d = corners_2d[..., :2].cpu().numpy()

        # Check if each corner is within the image boundaries
        valid_corners = [
            (0 <= x < image_size and 0 <= y < image_size) for x, y in corners_2d
        ]

        # Skip drawing if all corners are out of image boundaries
        if not any(valid_corners):
            continue

        # Draw the label and bbox_id
        draw_label(draw, corners_2d, bbox_id, font, image_size)


def draw_label(draw, corners_2d, bbox_id, font, image_size):
    """
    Draw label and bbox_id at the center of the top face of the bounding box.

    Args:
        draw (ImageDraw): ImageDraw object.
        corners_2d (array): 2D coordinates of the bounding box corners.
        bbox_id (int): Bounding box ID.
        font (ImageFont): Font for drawing text.
        image_size (int): Size of the output image.
    """
    # Find the center of the top face
    center_x = int(
        (corners_2d[4][0] + corners_2d[5][0] + corners_2d[6][0] + corners_2d[7][0]) / 4
    )
    center_y = int(
        (corners_2d[4][1] + corners_2d[5][1] + corners_2d[6][1] + corners_2d[7][1]) / 4
    )
    if 0 <= center_x < image_size and 0 <= center_y < image_size:
        text = f"{bbox_id}"
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        background_x0 = center_x - text_width // 2 - 2
        background_y0 = center_y - text_height // 2 - 2
        background_x1 = center_x + text_width // 2 + 2
        background_y1 = center_y + text_height // 2 + 2
        draw.rectangle(
            [background_x0, background_y0, background_x1, background_y1],
            fill=(255, 255, 255),
        )
        draw.text(
            (center_x - text_width // 2, center_y - text_height // 2),
            text,
            font=font,
            fill=(255, 0, 0),
        )

    return


def draw_bbox_function(draw, corners_2d, valid_corners, bbox_color):
    """
    Draw the 3D bounding box by connecting the projected corners.

    Args:
        draw (ImageDraw): ImageDraw object.
        corners_2d (array): 2D coordinates of the bounding box corners.
        valid_corners (list): List of booleans indicating if each corner is within the image boundaries.
        bbox_color (tuple): Color for the bounding box.
    """
    # Draw the 3D bounding box by connecting the projected corners
    for i, (start, end) in enumerate(
        [
            (0, 1),
            (1, 3),
            (3, 2),
            (2, 0),
            (4, 5),
            (5, 7),
            (7, 6),
            (6, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ]
    ):
        if valid_corners[start] and valid_corners[end]:
            draw.line(
                [tuple(corners_2d[start]), tuple(corners_2d[end])],
                fill=bbox_color,
                width=2,
            )
    return


def create_point_cloud(scan_pc, device):
    """
    Create a point cloud from scan data.

    Args:
        scan_pc (np.ndarray): The scan data containing points and colors.
        device (str): The device to use for computation.

    Returns:
        Pointclouds: The created point cloud.
    """
    points = torch.tensor(scan_pc[:, :3], dtype=torch.float32)
    colors = torch.tensor(scan_pc[:, 3:], dtype=torch.float32)
    point_cloud = Pointclouds(points=[points], features=[colors]).to(device)
    return point_cloud


def setup_camera(
    point_cloud,
    anchor_bbox_3d,
    center,
    image_size,
    camera_distance_factor=1.0,
    camera_lift=1.0,
    device="cuda",
    calibrate=True,
):
    """
    Set up the camera for rendering the point cloud.

    Args:
        point_cloud (Pointclouds): The point cloud to render.
        anchor_bbox_3d (torch.Tensor): The 3D bounding box of the anchor.
        center (np.ndarray): The center of the point cloud.
        image_size (int): The size of the output image.
        camera_distance_factor (float): The factor to adjust camera distance.
        camera_lift (float): The lift to apply to the camera.
        device (str): The device to use for computation.
        calibrate (bool): Whether to calibrate the camera.

    Returns:
        PerspectiveCameras: The set up camera.
    """
    # Compute the bounding box of the point cloud
    min_bounds = point_cloud.points_padded().min(dim=1)[0]
    max_bounds = point_cloud.points_padded().max(dim=1)[0]

    center = torch.tensor(center, dtype=torch.float32)
    center[2] += camera_lift
    camera_position = center + camera_distance_factor * (center - anchor_bbox_3d)
    R, T = look_at_view_transform(
        dist=1,
        elev=0,
        azim=0,
        at=anchor_bbox_3d.unsqueeze(0),
        eye=camera_position.unsqueeze(0),
        up=((0, 0, 1),),
    )

    focal_length = torch.tensor([[1.0, 1.0]]).to(
        point_cloud.device
    )  # Initial focal length, shape (1, 2)
    principal_point = torch.tensor([[0.0, 0.0]]).to(
        point_cloud.device
    )  # Initial principal point, shape (1, 2)

    cameras = PerspectiveCameras(
        device=device,
        R=R,
        T=T,
        focal_length=focal_length,
        principal_point=principal_point,
    )

    if calibrate:
        if isinstance(image_size, int):
            image_size_tensor = torch.tensor(
                [[image_size, image_size]]
            )  # Convert integer to 2D tensor
        assert image_size_tensor.shape[-1] == 2

        # Get the projection of the point cloud
        points_2d = cameras.transform_points_screen(
            point_cloud.points_padded(), image_size=image_size_tensor
        )
        points_2d = points_2d[..., :2]

        # Compute the bounding box of the projected points
        min_proj = points_2d.min(dim=1)[0]
        max_proj = points_2d.max(dim=1)[0]

        # Adjust focal length and principal point to ensure all points are within the image
        new_focal_length = (
            focal_length
            * (max_proj - min_proj).max()
            / image_size_tensor.to(point_cloud.device)
        )
        new_principal_point = (min_proj + max_proj) / 2

        # Update camera intrinsics
        cameras = PerspectiveCameras(
            device=device,
            R=R,
            T=T,
            focal_length=new_focal_length,
            principal_point=new_principal_point,  # Ensure principal point is 2D
        )
    return cameras


def render_point_cloud(point_cloud, cameras, image_size, device):
    """
    Render the point cloud.

    Args:
        point_cloud (Pointclouds): The point cloud to render.
        cameras (PerspectiveCameras): The camera settings.
        image_size (int): The size of the output image.
        device (str): The device to use for computation.

    Returns:
        np.ndarray: The rendered image.
        PointsRasterizer: The rasterizer used for rendering.
    """
    raster_settings = PointsRasterizationSettings(
        image_size=image_size, radius=0.01, points_per_pixel=10
    )
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    renderer = PointsRenderer(
        rasterizer=rasterizer, compositor=AlphaCompositor(background_color=255)
    )
    images = renderer(point_cloud)
    image_np = images[0, ..., :3].cpu().numpy()
    return image_np, rasterizer


def compute_depth_map(rasterizer, point_cloud):
    """
    Compute the depth map of the point cloud.

    Args:
        rasterizer (PointsRasterizer): The rasterizer used for rendering.
        point_cloud (Pointclouds): The point cloud to render.

    Returns:
        np.ndarray: The computed depth map.
    """
    fragments = rasterizer(point_cloud)
    depth_map = fragments.zbuf[0].cpu().numpy()
    depth_map = np.min(depth_map, axis=2)

    return depth_map


def get_3d_bbox_from_masks(scan_pc, rasterizer, masks, device='cuda'):
    """
    Calculates the 3D axis-aligned bounding box (AABB) of the points 
    visible within a specific 2D mask.

    Args:
        point_cloud (Pointclouds): The PyTorch3D point cloud object.
        rasterizer (PointsRasterizer): The configured rasterizer.
        masks (np.ndarray or torch.Tensor): A 2D boolean mask of shape (N, H, W).

    Returns:
        np.ndarray: [cx, cy, cz, w, l, h] or None.
    """
    point_cloud = create_point_cloud(scan_pc, device=device)

    # 1. Run the rasterizer to get the fragment data
    fragments = rasterizer(point_cloud)
    
    # 2. Extract the indices of the points rendered at each pixel
    # Shape of fragments.idx is (batch_size, H, W, points_per_pixel)
    idx = fragments.idx[0]
    
    # Ensure mask is a boolean PyTorch tensor on the same device
    if isinstance(masks, np.ndarray):
        masks = torch.from_numpy(masks).to(idx.device)
    masks = masks.bool()
        
    # 3. Apply the 2D mask to the indices to get only the object's pixels
    # This results in a tensor of shape (Num_True_Pixels, points_per_pixel)

    bboxes = []

    for i in range(masks.shape[0]):
        mask = masks[i]
        masked_idx = idx[mask.permute(1, 2, 0).repeat(1, 1, idx.shape[-1])]
        
        # 4. Filter out background pixels (PyTorch3D assigns -1 to background)
        valid_idx = masked_idx[masked_idx != -1]
        
        if len(valid_idx) == 0:
            continue # The mask contains no rendered points
            
        # 5. Get unique point indices to avoid redundant calculations
        unique_idx = torch.unique(valid_idx)
        
        # 6. Extract the actual 3D coordinates from the original point cloud
        # .points_list()[0] gets the (N, 3) tensor for the first item in the batch
        points_3d = point_cloud.points_list()[0]
        object_points = points_3d[unique_idx]
        
        # 7. Compute the 3D bounding box by finding the min and max along X, Y, and Z
        min_coords, _ = torch.min(object_points, dim=0)
        max_coords, _ = torch.max(object_points, dim=0)

        center_coords = (max_coords + min_coords) / 2
        sizes = max_coords - min_coords
        
        bbox_3d = torch.cat([center_coords, sizes])

        bboxes.append(bbox_3d.cpu().numpy())
    
    return bboxes