import numpy as np
import json
import os.path as osp
import os
from plyfile import PlyData, PlyElement

# import cupy as cp
import argparse
import sys

def read_dict(file_path):
    with open(file_path) as fin:
        return json.load(fin)


class MeshProcessor:
    def __init__(self, dataset, axis_alignment_info_file, output_dir):
        self.dataset = dataset
        self.scans_axis_alignment_matrices = read_dict(axis_alignment_info_file)
        self.output_dir = output_dir

    def get_axis_alignment_matrix(self, scan_id):
        return self.scans_axis_alignment_matrices[scan_id]

    def align_to_axes(self, vertices, colors, room):
        """
        Align the mesh to xyz axes using the alignment matrix found in Scannet.
        """
        # Get the axis alignment matrix
        alignment_matrix = self.get_axis_alignment_matrix(room)
        alignment_matrix = np.array(alignment_matrix, dtype=np.float32).reshape(4, 4)

        # Transform the vertices
        pts = np.ones((vertices.shape[0], 4), dtype=vertices.dtype)
        pts[:, 0:3] = vertices
        vertices = np.dot(pts, alignment_matrix.transpose())[:, :3]

        # Ensure no NaN values are introduced after transformation
        assert np.sum(np.isnan(vertices)) == 0
        print(f"Aligned mesh to axes of {room}")

        return vertices

    def save_mesh(self, vertices, colors, faces, file_path):
        """
        Save the mesh vertices, colors, and faces to a PLY file.

        Parameters:
        - vertices: A numpy array of shape (N, 3) containing the mesh vertices.
        - colors: A numpy array of shape (N, 3) containing the vertex colors.
        - faces: A numpy array of shape (M, 3) containing the mesh faces.
        - file_path: The path where the PLY file will be saved.
        """
        # Create a structured array for the vertices and colors
        vertex_data = np.array(
            [(v[0], v[1], v[2], c[0], c[1], c[2]) for v, c in zip(vertices, colors)],
            dtype=[
                ("x", "f4"),
                ("y", "f4"),
                ("z", "f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
            ],
        )

        # Create a PlyElement for the vertices
        vertex_element = PlyElement.describe(vertex_data, "vertex")

        elements = [vertex_element]

        if faces is not None:
            # Create a structured array for the faces
            face_data = np.array(
                [(f,) for f in faces], dtype=[("vertex_indices", "i4", (3,))]
            )
            face_element = PlyElement.describe(face_data, "face")
            elements.append(face_element)

        # Write the vertices and faces to a PLY file
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        PlyData(elements, text=True).write(file_path)
        print(f"Saved mesh to {file_path}")
        return

    def process_mesh_for_room(self, room, scan_data_dir, mesh_suffix, processor):
        """
        Process mesh for a single room (scan).

        Parameters:
        - room: The room (scan) identifier.
        - scan_data_dir: Directory containing scan data.
        - mesh_suffix: Suffix for mesh files.
        - processor: MeshProcessor object to handle alignment and saving.

        Returns:
        - None
        """
        scan_data_file = osp.join(scan_data_dir, room, room + mesh_suffix)
        print(f"Processing {scan_data_file}")

        # Load mesh data
        vertices, colors, faces = load_mesh_data(scan_data_file)

        # Align vertices to axes
        aligned_vertices = processor.align_to_axes(vertices, colors, room)

        # Save the aligned mesh to the output directory
        output_path = osp.join(self.output_dir, f"{room}.ply")
        self.save_mesh(aligned_vertices, colors, faces, output_path)
        return


def read_file_to_list(file_path):
    with open(file_path, "r") as file:
        return sorted(file.read().splitlines())


def load_mesh_data(scan_data_file):
    """
    Load mesh data (vertices and colors) from a PLY file.

    Parameters:
    - scan_data_file: Path to the PLY file.

    Returns:
    - vertices: A numpy array of shape (N, 3) containing the mesh vertices.
    - colors: A numpy array of shape (N, 3) containing the mesh vertex colors.
    - faces: A numpy array of shape (M, 3) containing the mesh faces (if available).
    """
    data = PlyData.read(scan_data_file)

    # Extract vertex data
    x = np.asarray(data.elements[0].data["x"])
    y = np.asarray(data.elements[0].data["y"])
    z = np.asarray(data.elements[0].data["z"])
    red = np.asarray(data.elements[0].data["red"])
    green = np.asarray(data.elements[0].data["green"])
    blue = np.asarray(data.elements[0].data["blue"])
    vertices = np.stack([x, y, z], axis=1)
    colors = np.stack([red, green, blue], axis=1)

    # Read faces if available
    faces = None
    if len(data.elements) > 1:
        faces = np.asarray(data.elements[1].data["vertex_indices"])

    return vertices, colors, faces


def process_all_rooms(scans, scan_data_dir, mesh_suffix, processor):
    """
    Process meshes for all rooms in the scan list.

    Parameters:
    - scans: List of scan (room) identifiers.
    - scan_data_dir: Directory containing scan data.
    - mesh_suffix: Suffix for mesh files.
    - processor: MeshProcessor object to handle alignment and saving.

    Returns:
    - None
    """
    for room in scans:
        process_mesh_for_room(room, scan_data_dir, mesh_suffix, processor)


if __name__ == "__main__":
    # Parse arguments from the command line
    parser = argparse.ArgumentParser(description="Process and align meshes to axes.")
    parser.add_argument(
        "--scannet_file",
        type=str,
        default="PCGrounder/data/scannet/scannetv2_val.txt",
        help="Path to the ScanNet scene file list.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="PCGrounder/data/scannet/global_aligned_mesh_clean_2/",
        help="Directory where the aligned meshes will be saved. Original data from https://github.com/referit3d/referit3d/blob/eccv/referit3d/data/scannet/scans_axis_alignment_matrices.json",
    )
    parser.add_argument(
        "--axis_alignment_info_file",
        type=str,
        default="PCGrounder/data/scannet/scans_axis_alignment_matrices.json",
        help="Path to the axis alignment info JSON file.",
    )
    parser.add_argument(
        "--mesh_suffix",
        type=str,
        default="_vh_clean_2.ply",
        help="Suffix of the mesh PLY file.",
    )
    parser.add_argument(
        "--scan_dir",
        type=str,
        default="/datasets/scannet/scans",
        help="Directory containing ScanNet scans.",
    )
    parser.add_argument("--dataset", type=str, default="scannet")

    args = parser.parse_args()

    scans = read_file_to_list(args.scannet_file)

    print(f"Number of scenes: {len(scans)}")

    processor = MeshProcessor(
        args.dataset,
        axis_alignment_info_file=args.axis_alignment_info_file,
        output_dir=args.output_dir,
    )

    # Process all rooms
    for room in scans:
        processor.process_mesh_for_room(
            room, args.scan_dir, args.mesh_suffix, processor
        )
