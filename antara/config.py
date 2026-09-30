"""
Single source of truth for all pipeline constants.

Previously these values were duplicated (and had drifted) across the four
pipeline scripts: blur threshold was 35 / 40 / 50, the SLAM drift factor was
0.92 / 0.95, motion-diff thresholds were 28 / 30 / 35. Consolidating here so
CLI and server behave identically.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Filesystem layout
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLAM3R_DIR = os.path.join(REPO_ROOT, "SLAM3R")
SAM2_DIR = os.path.join(REPO_ROOT, "sam2")
GAUSSIAN_SPLATTING_DIR = os.path.join(REPO_ROOT, "gaussian-splatting")
SUGAR_DIR = os.path.join(REPO_ROOT, "SuGaR")
# The project uses the locally installed SAM 2.1 Tiny checkpoint.  A
# checkpoint must always be paired with its matching YAML configuration.
SAM2_CHECKPOINT = os.path.join(REPO_ROOT, "checkpoints", "sam2.1_hiera_tiny.pt")
SAM2_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"


# ---------------------------------------------------------------------------
# Geographic constants (WGS-84 local tangent-plane approximation)
# ---------------------------------------------------------------------------
METERS_PER_DEG_LAT = 111_320.0


def get_utm_transformer(lat, lon):
    """Get a pyproj transformer from WGS-84 to the correct UTM zone for given lat/lon.
    
    Returns (to_utm, from_utm) transformer pair, or (None, None) if pyproj is unavailable.
    """
    try:
        from pyproj import Transformer
        # Determine UTM zone
        zone_number = (int((lon + 180) // 6) % 60) + 1
        hemisphere = 'north' if lat >= 0 else 'south'
        epsg_code = 32600 + zone_number if hemisphere == 'north' else 32700 + zone_number
        
        to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg_code}", always_xy=True)
        from_utm = Transformer.from_crs(f"EPSG:{epsg_code}", "EPSG:4326", always_xy=True)
        print(f"[Config] Using UTM zone {zone_number}{hemisphere[0].upper()} (EPSG:{epsg_code})")
        return to_utm, from_utm
    except ImportError:
        print("[Config] pyproj not installed, falling back to flat-earth approximation")
        return None, None


@dataclass
class PipelineConfig:
    """Runtime configuration threaded through every stage."""

    # --- Stage 1: preprocessing ---
    target_fps: int = 5                 # Reduced to 5 to extract fewer frames for faster processing
    blur_threshold: float = 40.0        # Variance-of-Laplacian; below => dropped
    use_sam: bool = True                # False => motion-diff debug path (--no-sam)
    sam2_ckpt: str = SAM2_CHECKPOINT
    motion_diff_threshold: int = 30     # only used when use_sam is False
    motion_diff_dilate_iters: int = 2
    # SAM2 mask cleanup performed in ANTARA with OpenCV. This is the portable
    # equivalent of SAM2's optional connected-components CUDA extension.
    sam_mask_cleanup_area: int = 400
    gps_max_interpolation_gap_ms: float = 2000.0
    gps_max_interpolation_gap_frames: int = 60
    # "auto" accepts telemetry containing exactly one synchronization field.
    # Use an explicit mode when a legacy file contains both frame_idx and
    # timestamp_ms but only one of them is authoritative.
    gps_sync_mode: str = "auto"

    # --- Stage 2: SLAM3R reconstruction ---
    keyframe_stride: int = 4         # Process every 4th frame (much faster for long videos)
    win_r: int = 3                      # I2P input window radius
    initial_winsize: int = 5
    conf_thres_i2p: float = 1.5
    conf_thres_l2w: float = 10.0
    # num_points_save is the total points in the fused cloud (SLAM3R default: 2M).
    # For 5-6GB VRAM (RTX 3050), use 200K-500K for balance between density and memory.
    # 200K produces dense reconstruction while avoiding OOM during point cloud processing.
    num_points_save: int = 500_000
    slam3r_test_name: str = "antara_run"
    enable_tensorrt: bool = False

    # --- Stage 3: georeferencing ---
    ransac_max_iter: int = 300
    ransac_inlier_thresh: float = 1.5   # metres
    tps_smoothing: float = 1.0

    # --- Stage 3.5 / 3.6: photorealistic mesh path ---
    enable_gaussian_splatting: bool = True
    enable_sugar: bool = True
    gs_iterations: int = 7000

    # --- Stage 5: evaluation ---
    holdout_fraction: float = 0.2       # GPS keyframes withheld from alignment
    target_accuracy_m: float = 1.0      # NTRO PS-17 criterion

    # --- I/O ---
    output_dir: str = "output"
    frames_dirname: str = "frames"
    colmap_dirname: str = "colmap_format"
    # A unique staging directory is assigned by the orchestrator for each run.
    # Keeping this separate from output_dir prevents old frames/predictions from
    # being accidentally consumed by a later reconstruction.
    staging_dir: str | None = None
    run_id: str | None = None
    # Synthetic GPS is demo-only and must be explicitly enabled by the caller.
    allow_synthetic_gps: bool = False
    # Supplied telemetry is unverified by default. This flag is an operator
    # assertion that the file came from the actual flight.
    gps_verified: bool = False

    def working_dir(self) -> str:
        """Return the isolated per-run staging directory, if configured."""
        return self.staging_dir or self.output_dir

    def frames_dir(self) -> str:
        return os.path.join(self.working_dir(), self.frames_dirname)

    def colmap_dir(self) -> str:
        return os.path.join(self.output_dir, self.colmap_dirname)
