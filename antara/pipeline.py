"""
Orchestrator — wires the five stages into one run() used by BOTH entrypoints.

Previously the CLI (`run_pipeline_cloud.py` / `antara_master.py`) and the Flask
server (`antara_server.py`) each carried their own inline copy of the pipeline,
which had drifted apart (different thresholds, different bugs, one faked stage in
each). This is the single shared core; `run_pipeline.py` and `antara_server.py`
are thin shells over `run()`.

Stage flow
----------
  1. preprocess   video + GPS -> sharp keyframes, dynamic-object masks, synced GPS
  2. reconstruct  frames -> REAL SLAM3R world cloud + per-keyframe camera centers
  3. georeference cloud + cameras -> aligned to GPS (RANSAC-Umeyama + TPS)
  4. deliverables PLY / LAS / OBJ / GeoTIFF / transforms.json
  5. evaluate     honest alignment-residual + hold-out ATE report

Output layout matches the existing Three.js viewer (output/index.html):
  output/colmap_format/georeferenced_cloud.ply
  output/colmap_format/transforms.json
  output/antara_model.las
  output/antara_model.obj
  output/antara_orthomosaic.tif
  output/accuracy_report.json

Hard-fail policy propagates from reconstruct.py: any failure of the real path
raises; no fabricated geometry is ever written.
"""

from __future__ import annotations

import os
import json
import uuid
import sys
from dataclasses import replace

import numpy as np

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

try:
    from .preprocess import convert_srt_to_json, auto_detect_gps_for_video
except ImportError:
    convert_srt_to_json = None
    auto_detect_gps_for_video = None

from .config import PipelineConfig
from . import preprocess as _pre
from . import reconstruct as _rec
from . import georeference as _geo
from . import deliverables as _dl
from . import evaluate as _eval
from . import renderers
from . import colmap_sfm as _colmap


def _noop(pct, detail):
    pass


def run(video_path, gps_path=None, cfg: PipelineConfig | None = None, progress=None):
    """Run the full ANTARA pipeline. Returns a dict of output paths + the report.

    progress(pct:int, detail:str) is called throughout so the Flask server can
    stream status; the CLI passes a printer. Defaults to a no-op.
    """
    cfg = cfg or PipelineConfig()
    progress = progress or _noop

    # Auto-detect or convert SRT telemetry if applicable
    if gps_path and gps_path.lower().endswith((".srt", ".sub")):
        if convert_srt_to_json:
            progress(1, f"Converting SRT telemetry '{gps_path}' to GPS JSON...")
            gps_path = convert_srt_to_json(gps_path)
            cfg = replace(cfg, gps_verified=True)
    elif not gps_path:
        if auto_detect_gps_for_video:
            detected_gps = auto_detect_gps_for_video(video_path)
            if detected_gps:
                progress(1, f"Auto-detected GPS telemetry file: {detected_gps}")
                gps_path = detected_gps
                cfg = replace(cfg, gps_verified=True)

    # Every run gets a private staging directory.  Final viewer-compatible
    # deliverables remain under output_dir, but frames, masks, telemetry, and
    # SLAM3R predictions can no longer leak between runs.
    run_id = cfg.run_id or uuid.uuid4().hex[:12]
    staging_dir = cfg.staging_dir or os.path.join(cfg.output_dir, ".runs", run_id)
    cfg = replace(
        cfg,
        run_id=run_id,
        staging_dir=staging_dir,
        slam3r_test_name=f"antara_{run_id}",
    )

    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.working_dir(), exist_ok=True)
    colmap_dir = cfg.colmap_dir()
    os.makedirs(colmap_dir, exist_ok=True)

    # --- Stage 1: preprocess -------------------------------------------------
    progress(2, "Stage 1/5 — extracting keyframes and masking dynamic objects...")
    pre = _pre.preprocess(video_path, gps_path, cfg, progress=progress)
    frame_names = pre["frame_names"]
    gps_records = pre["gps_records"]

    # --- Stage 2: REAL SLAM3R reconstruction ---------------------------------
    progress(27, "Stage 2/5 — SLAM3R reconstruction from video pixels...")
    rec = _rec.reconstruct(
        pre["frames_dir"], len(frame_names), cfg,
        masks=pre["masks"], frame_names=frame_names, progress=progress,
    )
    cam_centers = rec["cam_centers"]
    world_points = rec["world_points"]
    world_colors = rec["world_colors"]

    # --- Stage 2.5: COLMAP SfM camera pose estimation ----------------------
    progress(60, "Stage 2.5/5 — COLMAP SfM for real camera poses...")
    colmap_model_dir = None
    colmap_cams = None
    try:
        colmap_root, colmap_cams = _colmap.run_colmap_sfm(
            pre["frames_dir"], cfg, progress=progress,
            gps_records=gps_records, frame_names=frame_names
        )
        colmap_model_dir = colmap_root
    except Exception as exc:
        progress(61, f"COLMAP SfM unavailable ({exc}); continuing with proxy camera centers.")
        colmap_model_dir = None
        colmap_cams = None

    # --- Stage 3: georeference ----------------------------------------------
    progress(66, "Stage 3/5 — georeferencing to GPS (RANSAC-Umeyama + TPS)...")
    if colmap_cams is not None and len(colmap_cams) >= len(frame_names):
        active_cams = colmap_cams[:len(frame_names)]
        progress(67, "Georeferencing with true optical camera centers from COLMAP SfM.")
    else:
        active_cams = cam_centers
        progress(67, "Georeferencing with proxy camera centers from pointmap centroids.")

    gps_coords, origin = _geo.gps_to_local_metric(gps_records)
    k = min(len(active_cams), len(gps_coords))
    if k < len(active_cams) or k < len(gps_coords):
        active_cams = active_cams[:k]
        gps_coords = gps_coords[:k]
        frame_names = frame_names[:k]
    geo = _geo.georeference(active_cams, world_points, gps_coords, cfg, gps_records=gps_records[:k])
    corrected_points = geo["corrected_points"]
    corrected_cams = geo["corrected_cams"]

    progress(78, f"Aligned. scale={geo['scale']:.4f}, "
                 f"{len(geo['inliers'])}/{k} keyframe inliers"
                 f"{', gravity-corrected' if geo.get('gravity_aligned') else ''}.")

    # Write the cloud before optional 3DGS training; the Gaussian trainer uses
    # this exported point cloud as its initialization input.
    ply_path = os.path.join(colmap_dir, "georeferenced_cloud.ply")
    _dl.export_ply(corrected_points, world_colors, ply_path)

    # --- Stage 3.5 / 3.6: optional 3DGS + SuGaR photorealistic mesh ---
    gaussian_splat_path = None
    sugar_mesh_path = None
    if cfg.enable_gaussian_splatting or cfg.enable_sugar:
        try:
            gs_dir = os.path.join(cfg.output_dir, "gaussian_splat", run_id)
            gaussian_splat_path = renderers.train_gaussian_splat(
                ply_path=ply_path,
                model_dir=gs_dir,
                cfg=cfg,
                colmap_model_dir=colmap_model_dir,
                progress=progress,
            )
            if cfg.enable_sugar:
                sugar_mesh_path = renderers.extract_sugar_mesh(
                    gaussian_splat_path=gaussian_splat_path,
                    output_dir=os.path.join(cfg.output_dir, "sugar_mesh", run_id),
                    cfg=cfg,
                    progress=progress,
                )
        except Exception as exc:  # Keep the baseline point-cloud export functional.
            progress(82, f"3DGS / SuGaR stage skipped ({exc}). Falling back to baseline mesh.")
            gaussian_splat_path = None
            sugar_mesh_path = None

    # --- Stage 4: deliverables ----------------------------------------------
    progress(80, "Stage 4/5 — exporting PLY / LAS / OBJ / GeoTIFF...")
    las_path = os.path.join(cfg.output_dir, "antara_model.las")
    obj_path = os.path.join(cfg.output_dir, "antara_model.obj")
    tif_path = os.path.join(cfg.output_dir, "antara_orthomosaic.tif")

    progress(84, "PLY written. Exporting LAS...")
    _dl.export_las(corrected_points, world_colors, las_path)
    coordinate_metadata_path = _dl.export_coordinate_metadata(las_path, origin)
    progress(87, "LAS written. Exporting GeoTIFF orthomosaic...")
    try:
        _dl.export_geotiff(corrected_points, world_colors, origin, tif_path)
    except Exception as e:
        progress(88, f"GeoTIFF export skipped ({e}). Continuing pipeline...")
        tif_path = None
    progress(90, "GeoTIFF done. Meshing (Poisson or SuGaR fallback)...")
    try:
        _dl.export_mesh(ply_path, obj_path, sugar_mesh_path=sugar_mesh_path)
    except Exception as e:  # meshing is the most fragile export; don't lose the cloud
        progress(91, f"Mesh export skipped ({e}). Point cloud deliverables intact.")
        obj_path = None

    # --- GLB and FBX export (NTRO PS-17 compliance) ---
    glb_path = None
    fbx_path = None
    if obj_path and os.path.isfile(obj_path):
        glb_path = os.path.join(cfg.output_dir, "antara_model.glb")
        fbx_path = os.path.join(cfg.output_dir, "antara_model.fbx")
        try:
            _dl.export_glb(obj_path, glb_path)
        except Exception as e:
            progress(91, f"GLB export skipped ({e}).")
            glb_path = None
        try:
            _dl.export_fbx(obj_path, fbx_path)
        except Exception as e:
            progress(91, f"FBX export skipped ({e}).")
            fbx_path = None

    _dl.export_transforms(frame_names, corrected_cams, geo["rotation"], colmap_dir)
    progress(92, "transforms.json written.")

    # --- Stage 5: honest evaluation -----------------------------------------
    progress(94, "Stage 5/5 — honest accuracy evaluation (hold-out ATE)...")
    report_path = os.path.join(cfg.output_dir, "accuracy_report.json")
    report = _eval.evaluate(
        active_cams, world_points, gps_records[:k], cfg,
        output_report=report_path, gps_source=pre["gps_source"],
    )

    synthetic_gps = pre["gps_source"] == "synthetic"
    gps_verified = pre["gps_source"] == "verified"
    manifest = {
        "run_id": run_id,
        "video_path": os.path.abspath(video_path),
        "gps_source": pre["gps_source"],
        "gps_sync_mode": pre["gps_sync_mode"],
        "gps_verified": gps_verified,
        "staging_dir": os.path.abspath(cfg.working_dir()),
        "telemetry_path": os.path.abspath(pre["telemetry_path"]),
        "transforms_path": os.path.abspath(os.path.join(colmap_dir, "transforms.json")),
        "accuracy_report_path": os.path.abspath(report_path),
        "coordinate_metadata_path": os.path.abspath(coordinate_metadata_path),
        "gaussian_splat_path": os.path.abspath(gaussian_splat_path) if gaussian_splat_path else None,
        "sugar_mesh_path": os.path.abspath(sugar_mesh_path) if sugar_mesh_path else None,
        "glb_path": os.path.abspath(glb_path) if glb_path else None,
        "fbx_path": os.path.abspath(fbx_path) if fbx_path else None,
        "frame_count": len(frame_names),
        "mask_shape": list(pre["masks"].shape),
        "mask_fraction": float(np.mean(pre["masks"] > 0)),
        "reconstructed_points": int(len(corrected_points)),
        "mask_filter": rec.get("mask_filter", {}),
    }
    manifest_path = os.path.join(cfg.output_dir, "run_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    progress(100, "Pipeline complete.")

    return {
        "outputs": {
            "ply": ply_path,
            "las": las_path,
            "obj": obj_path,
            "glb": glb_path,
            "fbx": fbx_path,
            "geotiff": tif_path,
            "transforms": os.path.join(colmap_dir, "transforms.json"),
            "accuracy_report": report_path,
            "synced_telemetry": pre["telemetry_path"],
            "run_manifest": manifest_path,
            "coordinate_metadata": coordinate_metadata_path,
            "slam3r_ply": rec["ply_path"],
            "gaussian_splat": gaussian_splat_path,
            "sugar_mesh": sugar_mesh_path,
        },
        "num_points": int(len(corrected_points)),
        "num_keyframes": int(k),
        "scale": float(geo["scale"]),
        "origin": origin,
        "synthetic_gps": synthetic_gps,
        "gps_source": pre["gps_source"],
        "gps_sync_mode": pre["gps_sync_mode"],
        "gps_verified": gps_verified,
        "mask_filter": rec.get("mask_filter", {}),
        "accuracy": report,
    }
