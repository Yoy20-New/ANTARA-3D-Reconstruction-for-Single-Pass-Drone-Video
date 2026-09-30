"""
Stage 2 — REAL SLAM3R reconstruction (the part that used to be faked).

The legacy pipelines all fabricated this stage: a hardcoded camera line
(`idx * 0.75`), `np.random` depths, and a procedural point grid. No SLAM3R code
ran. This module replaces that with a genuine subprocess invocation of the
official SLAM3R `recon.py`, so the point cloud is derived from the actual video
pixels.

What SLAM3R gives us
--------------------
SLAM3R (DUSt3R lineage) regresses dense *pointmaps*, not explicit camera poses.
Its Local-to-World (L2W) model registers each keyframe's local pointmap into a
shared world frame. We ask recon.py to write:

  * `<scene_id>_recon.ply`         — the fused, confidence-filtered world cloud
                                      (this is the reconstruction deliverable).
  * `preds/registered_pcds.npy`    — every keyframe's registered world pointmap,
                                      concatenated (K·H·W, 3).
  * `preds/registered_confs.npy`   — matching per-pixel L2W confidence (K, H, W).
  * `preds/metadata.json`          — scene_id, kf_stride, etc.

Camera-center proxy
-------------------
SLAM3R exposes no per-frame camera pose. Georeferencing needs one position per
keyframe to align against GPS, so we recover it as the **confidence-weighted
centroid of each keyframe's registered world pointmap**. This is a documented
proxy, not a silent hack — the centroid of a frame's back-projected points sits
near the scene the camera was looking at, which tracks camera motion well enough
for a global similarity + TPS solve. If SLAM3R ever exposes real poses, swap them
in here.

Hard-fail policy
----------------
No synthetic fallback. If CUDA is unavailable, the SLAM3R repo/weights are
missing, recon.py errors, OOMs, or produces no cloud, we raise and stop. We never
emit fabricated geometry.
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import sys

import numpy as np

from .config import PipelineConfig, SLAM3R_DIR


class ReconstructionError(RuntimeError):
    """Raised when the real SLAM3R path cannot run. Never falls back to fake data."""


# ---------------------------------------------------------------------------
# Preconditions (fail early, fail clearly)
# ---------------------------------------------------------------------------
def _check_cuda():
    try:
        import torch
    except ImportError as e:
        raise ReconstructionError(
            "PyTorch is not installed in this environment. Create the 'antara' "
            "conda env per SETUP_FROM_SCRATCH.md. (No synthetic fallback by policy.)"
        ) from e
    if not torch.cuda.is_available():
        raise ReconstructionError(
            "CUDA is not available. SLAM3R reconstruction requires an NVIDIA GPU. "
            "Verify with `python -c \"import torch; print(torch.cuda.is_available())\"`. "
            "Fix the driver/toolkit (see SETUP_FROM_SCRATCH.md). There is no synthetic fallback."
        )


def _check_slam3r():
    recon_py = os.path.join(SLAM3R_DIR, "recon.py")
    if not os.path.isdir(SLAM3R_DIR) or not os.path.exists(recon_py):
        raise ReconstructionError(
            f"SLAM3R repo not found at {SLAM3R_DIR}. Clone it there "
            "(git clone https://github.com/PKU-VCL-3DV/SLAM3R.git) and "
            "the Windows-safe dependency setup in SETUP_FROM_SCRATCH.md."
        )
    return recon_py


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------
def _load_fused_cloud(result_dir, frames_dir=None):
    """Load `<scene_id>_recon.ply` -> (points (P,3), colors (P,3) in [0,1])."""
    plys = [p for p in glob.glob(os.path.join(result_dir, "*_recon.ply"))]
    # This SLAM3R fork uses the absolute img_dir as scene_id, which causes
    # os.path.join(save_dir, scene_id + "_recon.ply") to escape save_dir.
    if frames_dir:
        external_ply = os.path.abspath(frames_dir) + "_recon.ply"
        if os.path.exists(external_ply):
            plys.append(external_ply)
    if not plys:
        raise ReconstructionError(
            f"SLAM3R produced no *_recon.ply in {result_dir}. The run likely failed "
            "silently or was killed (OOM). Check the SLAM3R log above."
        )
    ply_path = max(plys, key=os.path.getmtime)

    import trimesh

    cloud = trimesh.load(ply_path, process=False)
    pts = np.asarray(cloud.vertices, dtype=np.float64)
    if pts.shape[0] == 0:
        raise ReconstructionError(f"SLAM3R cloud {ply_path} is empty.")

    colors = None
    vc = getattr(getattr(cloud, "visual", None), "vertex_colors", None)
    if vc is not None and len(vc) == len(pts):
        colors = np.asarray(vc, dtype=np.float64)[:, :3]
    else:
        c = getattr(cloud, "colors", None)
        if c is not None and len(c) == len(pts):
            colors = np.asarray(c, dtype=np.float64)[:, :3]
            
    if colors is not None:
        if colors.max() > 1.0:
            colors /= 255.0
    else:
        # No vertex colours in the PLY — fall back to mid-grey so exports stay valid.
        colors = np.full((len(pts), 3), 0.5, dtype=np.float64)
    return pts, colors, ply_path


def _camera_centers_from_preds(result_dir, num_keyframes, cfg: PipelineConfig):
    """Recover per-keyframe camera centers as confidence-weighted pointmap centroids.

    Reads `preds/registered_pcds.npy` (K·H·W, 3) and `preds/registered_confs.npy`
    (K, H, W). Returns (K, 3). This is the documented pose proxy — SLAM3R exposes
    no explicit camera poses.
    """
    preds_dir = os.path.join(result_dir, "preds")
    pcds_path = os.path.join(preds_dir, "registered_pcds.npy")
    confs_path = os.path.join(preds_dir, "registered_confs.npy")
    if not os.path.exists(pcds_path):
        raise ReconstructionError(
            f"Per-frame predictions missing at {pcds_path}. reconstruct.py invokes "
            "recon.py with --save_preds; if this file is absent the SLAM3R run did "
            "not finish. Check the log above."
        )

    reg = np.load(pcds_path)  # (K, H, W, 3) or (K*H*W, 3)
    confs = np.load(confs_path) if os.path.exists(confs_path) else None

    if reg.ndim == 4 and reg.shape[0] == num_keyframes and reg.shape[-1] == 3:
        reg = reg.reshape(num_keyframes, -1, 3)
    elif reg.ndim == 3 and reg.shape[-1] == 3:
        if reg.shape[0] != num_keyframes:
            raise ReconstructionError(
                f"registered_pcds has {reg.shape[0]} frames, expected {num_keyframes}. "
                "SLAM3R keyframe count differs from the extracted frames — the pipeline "
                "cannot align cameras to GPS."
            )
        reg = reg.reshape(num_keyframes, -1, 3)
    elif reg.ndim == 2 and reg.shape[-1] == 3:
        # Flattened array: (K*H*W, 3)
        total = reg.shape[0]
        if total % num_keyframes != 0:
            raise ReconstructionError(
                f"registered_pcds has {total} points (ndim=2), not divisible by "
                f"{num_keyframes} keyframes. SLAM3R keyframe count differs "
                "from the extracted frames."
            )
        reg = reg.reshape(num_keyframes, -1, 3)
    else:
        raise ReconstructionError(
            f"Unexpected registered_pcds shape {reg.shape}; expected "
            f"({num_keyframes}, H, W, 3) or (K*H*W, 3)."
        )

    if confs is not None:
        if confs.ndim >= 1 and confs.shape[0] == num_keyframes:
            confs = confs.reshape(num_keyframes, -1)
        else:
            raise ReconstructionError(
                f"Unexpected registered_confs shape {confs.shape}; expected "
                f"the first dimension to contain {num_keyframes} keyframes."
            )

    centers = np.empty((num_keyframes, 3), dtype=np.float64)
    for i in range(num_keyframes):
        pts = reg[i]
        if confs is not None:
            w = confs[i]
            keep = w > cfg.conf_thres_l2w
            if keep.sum() < 8:  # too few confident points; use all non-zero
                keep = np.abs(pts).sum(axis=1) > 1e-9
            if keep.sum() == 0:
                centers[i] = pts.mean(axis=0)
                continue
            centers[i] = pts[keep].mean(axis=0)
        else:
            nz = np.abs(pts).sum(axis=1) > 1e-9
            centers[i] = pts[nz].mean(axis=0) if nz.any() else pts.mean(axis=0)

    if not np.all(np.isfinite(centers)):
        raise ReconstructionError(
            "Recovered camera centers contain non-finite values — SLAM3R output is "
            "corrupt. Aborting rather than georeferencing garbage."
        )
    return centers


def _masked_registered_cloud(result_dir, frames_dir, frame_names, masks, cfg):
    """Build a colored world cloud from registered predictions and masks.

    SLAM3R's fused PLY is confidence-filtered, but it does not know ANTARA's
    dynamic-object masks.  The registered per-frame predictions are the correct
    place to apply those masks because each pixel still has a known frame/pixel
    correspondence.
    """
    preds_dir = os.path.join(result_dir, "preds")
    pcds_path = os.path.join(preds_dir, "registered_pcds.npy")
    confs_path = os.path.join(preds_dir, "registered_confs.npy")
    imgs_path = os.path.join(preds_dir, "input_imgs.npy")
    if not os.path.exists(pcds_path) or not os.path.exists(confs_path):
        raise ReconstructionError(
            "SLAM3R per-frame pointmaps/confidences are required for mask "
            "filtering, but registered_pcds.npy or registered_confs.npy is missing."
        )

    reg = np.load(pcds_path)
    confs = np.load(confs_path)
    
    if confs.ndim != 3 or confs.shape[0] != len(frame_names):
        raise ReconstructionError(
            f"Registered confidence shape {confs.shape} does not match "
            f"{len(frame_names)} extracted frames."
        )
        
    K, H, W = confs.shape
    
    if reg.ndim == 2 and reg.shape == (K * H * W, 3):
        reg = reg.reshape(K, H, W, 3)
    elif reg.ndim == 3 and reg.shape == (K, H * W, 3):
        reg = reg.reshape(K, H, W, 3)
        
    if reg.ndim != 4 or reg.shape[0] != len(frame_names) or reg.shape[-1] != 3:
        raise ReconstructionError(
            f"Registered pointmap shape {reg.shape} does not match "
            f"{len(frame_names)} extracted frames."
        )

    height, width = reg.shape[1:3]
    imgs = np.load(imgs_path) if os.path.exists(imgs_path) else None
    if imgs is not None and (imgs.ndim != 4 or imgs.shape[0] != len(frame_names)):
        imgs = None

    import cv2

    all_points = []
    all_colors = []
    masked_pixels = 0
    total_pixels = len(frame_names) * height * width

    for i, name in enumerate(frame_names):
        pts = np.asarray(reg[i], dtype=np.float64)
        conf = np.asarray(confs[i], dtype=np.float64)
        mask = np.asarray(masks[i], dtype=np.uint8)
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

        dynamic = mask > 0
        masked_pixels += int(dynamic.sum())
        valid = (
            ~dynamic
            & (conf > cfg.conf_thres_l2w)
            & np.isfinite(pts).all(axis=2)
            & (np.abs(pts).sum(axis=2) > 1e-9)
        )
        if not valid.any():
            continue

        if imgs is not None:
            colors = np.asarray(imgs[i])
            if colors.shape[:2] != (height, width):
                colors = cv2.resize(colors, (width, height), interpolation=cv2.INTER_LINEAR)
            colors = colors[..., :3].astype(np.float64)
            if colors.size and colors.max() > 1.5:
                colors /= 255.0
        else:
            image_path = os.path.join(frames_dir, name)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                colors = np.full((height, width, 3), 0.5, dtype=np.float64)
            else:
                image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
                colors = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0

        all_points.append(pts[valid])
        all_colors.append(np.clip(colors[valid], 0.0, 1.0))

    if not all_points:
        raise ReconstructionError(
            "Dynamic masks and SLAM3R confidence filtering removed every point. "
            "Inspect mask ratios and confidence thresholds."
        )

    points = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0)
    if len(points) > cfg.num_points_save:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Estimate initial voxel size from bounding box
        bbox = pcd.get_axis_aligned_bounding_box()
        bbox_extent = np.maximum(bbox.get_max_bound() - bbox.get_min_bound(), 1e-4)
        volume = np.prod(bbox_extent)
        # Target voxel size to get approximately num_points_save points
        voxel_size = max((volume / cfg.num_points_save) ** (1.0 / 3.0), 1e-4)
        
        # Iteratively refine voxel size to hit target point count
        for _ in range(5):
            pcd_down = pcd.voxel_down_sample(voxel_size)
            n_down = len(pcd_down.points)
            if n_down == 0:
                voxel_size /= 2.0
                continue
            if abs(n_down - cfg.num_points_save) / cfg.num_points_save < 0.1:
                break  # Within 10% of target
            ratio = (n_down / cfg.num_points_save) ** (1.0 / 3.0)
            voxel_size *= max(ratio, 0.1)
        
        points = np.asarray(pcd_down.points, dtype=np.float32)
        colors = np.asarray(pcd_down.colors, dtype=np.float64)
        print(f"[Reconstruct] Voxel downsampled: {len(points):,} points "
              f"(voxel_size={voxel_size:.4f})")

    return points, colors, {
        "total_pixels": int(total_pixels),
        "masked_pixels": int(masked_pixels),
        "masked_fraction": float(masked_pixels / total_pixels) if total_pixels else 0.0,
        "valid_points_after_filter": int(len(points)),
    }


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
def reconstruct(frames_dir, num_keyframes, cfg: PipelineConfig,
                masks=None, frame_names=None, progress=None):
    """Run real SLAM3R on the extracted keyframes.

    Returns dict:
        world_points  (P, 3) float64  — fused world cloud
        world_colors  (P, 3) float64  — RGB in [0, 1]
        cam_centers   (K, 3) float64  — per-keyframe camera-center proxy
        ply_path      str             — path to the fused SLAM3R PLY
    Raises ReconstructionError on any failure (no synthetic fallback).
    """
    def report(pct, detail):
        if progress:
            progress(pct, detail)

    _check_cuda()
    recon_py = _check_slam3r()

    frames_dir = os.path.abspath(frames_dir)
    frame_names = frame_names or sorted(
        name for name in os.listdir(frames_dir) if name.lower().endswith(".jpg")
    )
    if masks is not None and len(masks) != len(frame_names):
        raise ReconstructionError(
            f"Mask count {len(masks)} does not match frame count {len(frame_names)}."
        )
    result_dir = os.path.join(SLAM3R_DIR, "results", cfg.slam3r_test_name)
    # Clear stale output so we never mistake a previous run's cloud for this one.
    if os.path.isdir(result_dir):
        for f in glob.glob(os.path.join(result_dir, "*_recon.ply")):
            try:
                os.remove(f)
            except OSError:
                pass
    external_ply = os.path.abspath(frames_dir) + "_recon.ply"
    if os.path.exists(external_ply):
        try:
            os.remove(external_ply)
        except OSError as e:
            raise ReconstructionError(
                f"Cannot remove stale SLAM3R output {external_ply}: {e}"
            ) from e

    report(28, "Launching real SLAM3R reconstruction (this is the heavy stage)...")

    cmd = [
        sys.executable, "recon.py",
        "--img_dir", frames_dir,
        "--test_name", cfg.slam3r_test_name,
        "--save_dir", "results",
        "--keyframe_stride", str(cfg.keyframe_stride),
        "--win_r", str(cfg.win_r),
        "--initial_winsize", str(cfg.initial_winsize),
        "--conf_thres_i2p", str(cfg.conf_thres_i2p),
        "--conf_thres_l2w", str(cfg.conf_thres_l2w),
        "--num_points_save", str(cfg.num_points_save),
        "--save_preds",       # writes preds/registered_pcds.npy + confs
        "--save_all_views",   # writes per-frame frame_{i}.ply (useful for debug)
    ]
    if cfg.enable_tensorrt:
        cmd.append("--tensorrt")

    # Stream SLAM3R's stdout so tqdm progress is visible; keep the tail for errors.
    proc = subprocess.Popen(
        cmd, cwd=SLAM3R_DIR,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    tail = []
    for line in proc.stdout:
        line = line.rstrip("\n")
        tail.append(line)
        if len(tail) > 60:
            tail.pop(0)
        low = line.lower()
        if "registering" in low or "i2p" in low:
            report(45, "SLAM3R registering keyframes into world frame...")
        elif "resampling" in low or "save_to" in low:
            report(58, "SLAM3R fusing and saving world cloud...")
    proc.wait()

    if proc.returncode != 0:
        blob = "\n".join(tail)
        oom = "out of memory" in blob.lower() or "cuda error" in blob.lower()
        hint = (
            "\n\nThis looks like a GPU OOM on the 6GB card. Lower num_points_save or "
            "raise keyframe_stride in antara/config.py, or feed a shorter clip."
            if oom else
            "\n\nSee the SLAM3R output above for the underlying error."
        )
        raise ReconstructionError(
            f"SLAM3R recon.py exited with code {proc.returncode}.{hint}\n"
            f"--- last SLAM3R output ---\n{blob}"
        )

    report(60, "SLAM3R finished. Parsing world cloud and recovering camera centers...")

    points, colors, ply_path = _load_fused_cloud(result_dir, frames_dir)
    cam_centers = _camera_centers_from_preds(result_dir, num_keyframes, cfg)

    mask_filter = {}
    if masks is not None:
        points, colors, mask_filter = _masked_registered_cloud(
            result_dir, frames_dir, frame_names, masks, cfg
        )
        report(64, f"Applied dynamic masks: {len(points):,} valid world points remain.")

    report(65, f"Reconstruction complete: {len(points):,} points, "
               f"{len(cam_centers)} camera centers.")
    return {
        "world_points": points,
        "world_colors": colors,
        "cam_centers": cam_centers,
        "ply_path": ply_path,
        "mask_filter": mask_filter,
    }
