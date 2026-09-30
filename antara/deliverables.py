"""
Stage 4 — Standard-format deliverables.

Exports the georeferenced reconstruction to the formats PS-17 requires:
  * PLY  — coloured point cloud (also what the Three.js viewer loads)
  * LAS  — GIS point cloud (laspy)
  * OBJ  — surface mesh via Open3D screened-Poisson
  * GeoTIFF — top-down orthomosaic with real WGS-84 bounds (rasterio)

This is the one part of the legacy pipelines that was already real and identical
across all four copies, so it is carried over largely verbatim — just deduplicated
and given real (rather than synthetic) point coordinates as input.

SuGaR hook: a photorealistic textured mesh via SuGaR 3DGS is the approved LATER
phase. `export_mesh` has a documented seam where that will slot in.
"""

from __future__ import annotations

import os

import numpy as np

from .config import METERS_PER_DEG_LAT, get_utm_transformer


def export_ply(points, colors, path):
    """Write an ASCII PLY with per-vertex RGB. colors in [0, 1]."""
    points = np.asarray(points, dtype=np.float64)
    if points.size == 0: points = np.empty((0, 3))
    colors = np.asarray(colors, dtype=np.float64)
    if colors.size == 0: colors = np.empty((0, 3))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, colors):
            r, g, b = (np.clip(c[:3], 0, 1) * 255).astype(int)
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {r} {g} {b}\n")
    return path


def export_las(points, colors, path):
    """Write a LAS 1.4 point cloud (point format 3, with RGB).

    Keeps the dynamic-scale fix from the legacy code: LAS stores integer-scaled
    coordinates, so scale must be derived from the actual point span or precision
    is lost / overflowed.
    """
    import laspy

    points = np.asarray(points, dtype=np.float64)
    if points.size == 0: points = np.empty((0, 3))
    colors = np.asarray(colors, dtype=np.float64)
    if colors.size == 0: colors = np.empty((0, 3))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    if len(points) == 0:
        min_pt = np.zeros(3)
        span = np.ones(3)
    else:
        min_pt = points.min(axis=0)
        span = np.maximum(points.max(axis=0) - min_pt, 1.0)

    header = laspy.LasHeader(point_format=3, version="1.4")
    header.offsets = min_pt
    header.scales = np.maximum(span / 1e7, 0.001)

    las = laspy.LasData(header)
    if len(points) > 0:
        las.x, las.y, las.z = points[:, 0], points[:, 1], points[:, 2]
        rgb = np.clip(colors * 65535, 0, 65535).astype(np.uint16)
        las.red, las.green, las.blue = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    las.write(path)
    return path


def export_coordinate_metadata(path, origin):
    """Describe the coordinate frame used by local-metric LAS/PLY outputs.

    The cloud coordinates are local ENU metres, not longitude/latitude.  A
    sidecar prevents GIS users from mistaking the local numbers for a global
    CRS until a projected-CRS exporter is added.
    """
    import json

    metadata_path = f"{path}.json"
    with open(metadata_path, "w") as f:
        json.dump({
            "coordinate_frame": "local_enu_meters",
            "origin_latitude": float(origin[0]),
            "origin_longitude": float(origin[1]),
            "origin_altitude_m": float(origin[2]),
            "horizontal_reference": "WGS-84 tangent-plane approximation",
            "vertical_reference": "input telemetry altitude_m; datum not inferred",
            "axis_order": {"x": "east_m", "y": "north_m", "z": "up_m"},
        }, f, indent=2)
    return metadata_path


def export_mesh(ply_path, obj_path, poisson_depth=9, sugar_mesh_path=None):
    """Surface mesh via Open3D screened Poisson or a pre-generated SuGaR mesh.

    `sugar_mesh_path` is the approved future path for a textured OBJ export.
    When present, it is copied to `obj_path` and used in place of the Poisson
    fallback. This keeps the pipeline robust even before external 3DGS/SuGaR
    dependencies are installed.
    """
    if sugar_mesh_path:
        sugar_mesh_path = os.path.abspath(sugar_mesh_path)
        if os.path.isdir(sugar_mesh_path):
            candidates = [
                os.path.join(sugar_mesh_path, name)
                for name in sorted(os.listdir(sugar_mesh_path))
                if name.lower().endswith(".obj")
            ]
            if not candidates:
                raise FileNotFoundError(f"No OBJ file found in SuGaR output dir: {sugar_mesh_path}")
            sugar_mesh_path = candidates[0]
        if not os.path.exists(sugar_mesh_path):
            raise FileNotFoundError(f"SuGaR mesh not found: {sugar_mesh_path}")
        os.makedirs(os.path.dirname(obj_path) or ".", exist_ok=True)
        import shutil
        shutil.copy2(sugar_mesh_path, obj_path)
        mtl_path = os.path.splitext(sugar_mesh_path)[0] + ".mtl"
        if os.path.exists(mtl_path):
            target_mtl = os.path.splitext(obj_path)[0] + ".mtl"
            shutil.copy2(mtl_path, target_mtl)
            # Fix the mtllib pointer inside the OBJ file to match the new MTL filename
            old_mtl_name = os.path.basename(mtl_path)
            new_mtl_name = os.path.basename(target_mtl)
            with open(obj_path, "r", encoding="utf-8", errors="replace") as f:
                obj_content = f.read()
            obj_content = obj_content.replace(f"mtllib {old_mtl_name}", f"mtllib {new_mtl_name}")
            with open(obj_path, "w", encoding="utf-8") as f:
                f.write(obj_content)
        return obj_path

    import open3d as o3d

    pcd = o3d.io.read_point_cloud(ply_path)
    if len(pcd.points) == 0:
        raise ValueError(f"Point cloud at {ply_path} is empty; cannot mesh.")
    pcd.estimate_normals()
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=poisson_depth
    )
    
    # Trim the low-density extrapolated edges (the "sky bowl" artifact)
    import numpy as np
    densities = np.asarray(densities)
    # Remove the bottom 5% of vertices by density
    density_threshold = np.quantile(densities, 0.05)
    vertices_to_remove = densities < density_threshold
    mesh.remove_vertices_by_mask(vertices_to_remove.tolist())
    
    o3d.io.write_triangle_mesh(obj_path, mesh)
    return obj_path


def export_geotiff(points, colors, origin, path, size=1024, background=110):
    """Top-down orthomosaic as a georeferenced GeoTIFF (EPSG:4326).

    `origin` is (lat, lon, alt) — the same tangent-plane origin used for GPS
    conversion, so the raster bounds map back to real-world coordinates.
    """
    import rasterio
    from rasterio.transform import from_bounds

    points = np.asarray(points, dtype=np.float64)
    if points.size == 0: points = np.empty((0, 3))
    colors = np.asarray(colors, dtype=np.float64)
    if colors.size == 0: colors = np.empty((0, 3))
    
    if len(points) == 0:
        raise ValueError("Cannot export GeoTIFF from an empty point cloud.")

    origin_lat, origin_lon, _ = origin

    img = np.full((size, size, 3), background, dtype=np.uint8)
    min_x, max_x = points[:, 0].min(), points[:, 0].max()
    min_y, max_y = points[:, 1].min(), points[:, 1].max()
    
    # Pad degenerate bounds to prevent rasterio division-by-zero
    if max_x == min_x:
        min_x -= 0.5
        max_x += 0.5
    if max_y == min_y:
        min_y -= 0.5
        max_y += 0.5
        
    dx = (max_x - min_x)
    dy = (max_y - min_y)

    for pt, col in zip(points, colors):
        gx = max(0, min(size - 1, int(((pt[0] - min_x) / dx) * (size - 1))))
        gy = max(0, min(size - 1, (size - 1) - int(((pt[1] - min_y) / dy) * (size - 1))))
        rgb = (np.clip(col[:3], 0, 1) * 255).astype(np.uint8)
        img[gy, gx] = rgb

    to_utm, from_utm = get_utm_transformer(origin_lat, origin_lon)
    if from_utm is not None:
        origin_e, origin_n = to_utm.transform(origin_lon, origin_lat)
        min_lon, min_lat = from_utm.transform(origin_e + min_x, origin_n + min_y)
        max_lon, max_lat = from_utm.transform(origin_e + max_x, origin_n + max_y)
    else:
        m_per_lon = METERS_PER_DEG_LAT * np.cos(np.radians(origin_lat))
        min_lon = origin_lon + min_x / m_per_lon
        max_lon = origin_lon + max_x / m_per_lon
        min_lat = origin_lat + min_y / METERS_PER_DEG_LAT
        max_lat = origin_lat + max_y / METERS_PER_DEG_LAT

    transform = from_bounds(min_lon, min_lat, max_lon, max_lat, size, size)
    with rasterio.open(
        path, "w", driver="GTiff", height=size, width=size, count=3,
        dtype=img.dtype, crs="EPSG:4326", transform=transform,
    ) as dst:
        for i in range(3):
            dst.write(img[:, :, i], i + 1)
    return path


def export_transforms(frame_names, cam_centers, rotation, colmap_dir,
                      intrinsics=None):
    """Write transforms.json (NeRF/SuGaR convention) for the later 3DGS phase."""
    import json

    os.makedirs(colmap_dir, exist_ok=True)
    frames = []
    for name, center in zip(frame_names, cam_centers):
        mat = np.eye(4)
        mat[:3, :3] = rotation
        mat[:3, 3] = center
        frames.append({
            "file_path": f"../frames/{name}",
            "transform_matrix": mat.tolist(),
        })
    data = {"camera_angle_x": 1.047, "frames": frames}
    if intrinsics:
        data.update(intrinsics)
    path = os.path.join(colmap_dir, "transforms.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def export_glb(obj_path, glb_path):
    """Convert OBJ mesh to GLB (binary glTF) format for web/AR viewers."""
    import trimesh

    if not os.path.isfile(obj_path):
        raise FileNotFoundError(f"OBJ file not found for GLB conversion: {obj_path}")

    os.makedirs(os.path.dirname(glb_path) or ".", exist_ok=True)
    mesh = trimesh.load(obj_path, force="mesh")
    mesh.export(glb_path, file_type="glb")
    print(f"[Deliverables] GLB exported: {glb_path}")
    return glb_path


def export_fbx(obj_path, fbx_path):
    """Convert OBJ mesh to FBX format for professional 3D software (Blender, Unity, Unreal).

    Uses the Open Asset Import Library (assimp) backend via pyassimp.
    """
    if not os.path.isfile(obj_path):
        raise FileNotFoundError(f"OBJ file not found for FBX conversion: {obj_path}")

    os.makedirs(os.path.dirname(fbx_path) or ".", exist_ok=True)

    # Ensure conda Library/bin is on PATH on Windows so assimp.dll can be loaded
    import sys
    conda_bin = os.path.join(sys.prefix, "Library", "bin")
    if os.path.isdir(conda_bin) and conda_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = conda_bin + os.pathsep + os.environ.get("PATH", "")

    try:
        import pyassimp
        with pyassimp.load(obj_path) as scene:
            pyassimp.export(scene, fbx_path, "fbx")

        if os.path.isfile(fbx_path) and os.path.getsize(fbx_path) > 0:
            print(f"[Deliverables] FBX exported: {fbx_path} ({os.path.getsize(fbx_path)} bytes)")
            return fbx_path
        else:
            raise RuntimeError(f"FBX export failed to create non-empty file at {fbx_path}")
    except Exception as e:
        print(f"[Deliverables] FBX export skipped ({e}). "
              "Ensure assimp is installed: conda install -c conda-forge assimp")
        return None