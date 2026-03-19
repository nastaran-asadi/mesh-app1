from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal

import numpy as np

MeshMethod = Literal["poisson", "ball_pivoting"]


@dataclass
class ReconstructionSettings:
    voxel_size: float = 0.03
    normal_radius: float = 0.08
    max_nn: int = 32
    remove_outliers: bool = True
    nb_neighbors: int = 20
    std_ratio: float = 2.0
    method: MeshMethod = "poisson"
    poisson_depth: int = 9
    poisson_density_quantile: float = 0.02
    bpa_radii_scale: tuple[float, float, float] = (1.0, 2.0, 4.0)
    smooth_iterations: int = 5
    simplify_triangles: int = 200_000


@dataclass
class ReconstructionResult:
    point_cloud: o3d.geometry.PointCloud
    mesh: o3d.geometry.TriangleMesh
    points_before: int
    points_after: int
    triangles: int
    watertight: bool


def _save_upload_to_disk(upload_bytes: bytes, suffix: str) -> Path:
    with NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(upload_bytes)
        return Path(handle.name)


def load_point_cloud(upload_bytes: bytes, filename: str) -> o3d.geometry.PointCloud:
    suffix = Path(filename).suffix or ".ply"
    temp_path = _save_upload_to_disk(upload_bytes, suffix)
    cloud = o3d.io.read_point_cloud(str(temp_path))
    temp_path.unlink(missing_ok=True)

    if cloud.is_empty():
        raise ValueError(
            "The imported file did not contain a readable point cloud. "
            "Use a supported format such as PLY, XYZ, XYZN, XYZRGB, PCD, LAS, or LAZ."
        )

    return cloud


def _estimate_normals(cloud: o3d.geometry.PointCloud, settings: ReconstructionSettings) -> None:
    cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=settings.normal_radius,
            max_nn=settings.max_nn,
        )
    )
    cloud.orient_normals_consistent_tangent_plane(settings.max_nn)


def preprocess_point_cloud(
    cloud: o3d.geometry.PointCloud,
    settings: ReconstructionSettings,
) -> o3d.geometry.PointCloud:
    working = cloud.voxel_down_sample(settings.voxel_size)

    if settings.remove_outliers:
        working, _ = working.remove_statistical_outlier(
            nb_neighbors=settings.nb_neighbors,
            std_ratio=settings.std_ratio,
        )

    _estimate_normals(working, settings)
    return working


def _poisson_mesh(
    cloud: o3d.geometry.PointCloud,
    settings: ReconstructionSettings,
) -> o3d.geometry.TriangleMesh:
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        cloud,
        depth=settings.poisson_depth,
    )
    density_array = np.asarray(densities)
    cutoff = np.quantile(density_array, settings.poisson_density_quantile)
    mesh.remove_vertices_by_mask(density_array < cutoff)
    return mesh


def _ball_pivoting_mesh(
    cloud: o3d.geometry.PointCloud,
    settings: ReconstructionSettings,
) -> o3d.geometry.TriangleMesh:
    distances = cloud.compute_nearest_neighbor_distance()
    avg_distance = float(np.mean(distances)) if distances else settings.voxel_size
    radii = o3d.utility.DoubleVector(
        [avg_distance * scale for scale in settings.bpa_radii_scale]
    )
    return o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(cloud, radii)


def postprocess_mesh(
    mesh: o3d.geometry.TriangleMesh,
    settings: ReconstructionSettings,
) -> o3d.geometry.TriangleMesh:
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    if settings.smooth_iterations > 0:
        mesh = mesh.filter_smooth_taubin(number_of_iterations=settings.smooth_iterations)

    if settings.simplify_triangles and len(mesh.triangles) > settings.simplify_triangles:
        mesh = mesh.simplify_quadric_decimation(settings.simplify_triangles)

    mesh.compute_vertex_normals()
    return mesh


def reconstruct_mesh(
    cloud: o3d.geometry.PointCloud,
    settings: ReconstructionSettings,
) -> ReconstructionResult:
    points_before = len(cloud.points)
    processed_cloud = preprocess_point_cloud(cloud, settings)
    points_after = len(processed_cloud.points)

    if settings.method == "poisson":
        mesh = _poisson_mesh(processed_cloud, settings)
    else:
        mesh = _ball_pivoting_mesh(processed_cloud, settings)

    mesh = postprocess_mesh(mesh, settings)

    return ReconstructionResult(
        point_cloud=processed_cloud,
        mesh=mesh,
        points_before=points_before,
        points_after=points_after,
        triangles=len(mesh.triangles),
        watertight=bool(mesh.is_watertight()),
    )


def export_mesh(mesh: o3d.geometry.TriangleMesh, file_format: str) -> bytes:
    suffix = ".obj" if file_format.lower() == "obj" else ".ply"
    with NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        temp_path = Path(handle.name)

    try:
        ok = o3d.io.write_triangle_mesh(str(temp_path), mesh, write_vertex_normals=True)
        if not ok:
            raise ValueError("Mesh export failed.")
        return temp_path.read_bytes()
    finally:
        temp_path.unlink(missing_ok=True)
