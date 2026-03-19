from __future__ import annotations

import streamlit as st

from src.mesh_pipeline import (
    ReconstructionSettings,
    export_mesh,
    load_point_cloud,
    reconstruct_mesh,
)

st.set_page_config(page_title="Point Cloud Soft Mesh App", layout="wide")

st.title("Point Cloud to Soft Mesh")
st.write(
    "Upload an interior point cloud of a school or building, then reconstruct a cleaner, "
    "smoother mesh that you can export as PLY or OBJ."
)

with st.sidebar:
    st.header("Reconstruction settings")
    method = st.selectbox(
        "Mesh method",
        options=["poisson", "ball_pivoting"],
        help="Poisson usually creates softer closed surfaces. Ball pivoting can preserve sharper details.",
    )
    voxel_size = st.slider("Voxel downsample size", 0.005, 0.20, 0.03, 0.005)
    normal_radius = st.slider("Normal estimation radius", 0.01, 0.30, 0.08, 0.01)
    max_nn = st.slider("Max neighbors for normals", 8, 128, 32, 4)
    remove_outliers = st.checkbox("Remove noisy points", value=True)
    nb_neighbors = st.slider("Outlier neighbors", 5, 100, 20, 1)
    std_ratio = st.slider("Outlier std ratio", 0.5, 5.0, 2.0, 0.1)
    poisson_depth = st.slider("Poisson depth", 6, 12, 9, 1)
    poisson_density_quantile = st.slider("Poisson density trim", 0.0, 0.2, 0.02, 0.005)
    smooth_iterations = st.slider("Taubin smoothing iterations", 0, 30, 5, 1)
    simplify_triangles = st.slider("Max triangles after simplify", 10_000, 500_000, 200_000, 10_000)

uploaded_file = st.file_uploader(
    "Upload point cloud",
    type=["ply", "pcd", "xyz", "xyzn", "xyzrgb", "las", "laz"],
)

st.info(
    "For dense indoor scans with books, tables, and chairs, start with Poisson depth 9-10, "
    "voxel size 0.02-0.05, and 3-8 smoothing iterations."
)

if uploaded_file is not None:
    settings = ReconstructionSettings(
        voxel_size=voxel_size,
        normal_radius=normal_radius,
        max_nn=max_nn,
        remove_outliers=remove_outliers,
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio,
        method=method,
        poisson_depth=poisson_depth,
        poisson_density_quantile=poisson_density_quantile,
        smooth_iterations=smooth_iterations,
        simplify_triangles=simplify_triangles,
    )

    with st.spinner("Loading point cloud and generating mesh..."):
        cloud = load_point_cloud(uploaded_file.getvalue(), uploaded_file.name)
        result = reconstruct_mesh(cloud, settings)

    left, right = st.columns(2)
    with left:
        st.subheader("Input summary")
        st.metric("Original points", f"{result.points_before:,}")
        st.metric("Processed points", f"{result.points_after:,}")
    with right:
        st.subheader("Mesh summary")
        st.metric("Triangles", f"{result.triangles:,}")
        st.metric("Watertight", "Yes" if result.watertight else "No")

    st.success(
        "Mesh reconstruction finished. If furniture looks too blobby, reduce voxel size or smoothing; "
        "if holes remain, increase Poisson depth or switch methods."
    )

    export_format = st.radio("Export format", ["ply", "obj"], horizontal=True)
    export_bytes = export_mesh(result.mesh, export_format)
    st.download_button(
        label=f"Download mesh as {export_format.upper()}",
        data=export_bytes,
        file_name=f"reconstructed_mesh.{export_format}",
        mime="application/octet-stream",
    )
else:
    st.caption("Upload a point cloud file to begin reconstruction.")
