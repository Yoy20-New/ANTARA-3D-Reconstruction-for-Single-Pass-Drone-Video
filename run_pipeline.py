#!/usr/bin/env python3
"""
Project ANTARA — CLI entrypoint (thin shell over antara.pipeline).

Replaces the four overlapping legacy drivers (01_preprocess.py,
02_track_and_scale.py, run_pipeline_cloud.py, antara_master.py), all of which
faked the SLAM3R stage. This calls the single shared core.

Usage
-----
  python run_pipeline.py --video drone.mp4 --gps telemetry.json
  python run_pipeline.py --video drone.mp4 --gps legacy.json --gps-sync-mode frame_idx
  python run_pipeline.py --video drone.mp4 --no-sam      # motion-diff debug mask
  python run_pipeline.py --video drone.mp4 --allow-synthetic-gps  # demo only
  python run_pipeline.py --evaluate-only                 # re-score existing output

Hard-fail policy: if CUDA / SLAM3R weights / SAM 2 are unavailable, the run
raises and stops. No synthetic geometry is ever produced.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from antara.config import PipelineConfig
from antara.pipeline import run as run_pipeline

def _print_progress(pct, detail):
    bar = "#" * (pct // 4) + "-" * (25 - pct // 4)
    sys.stdout.write(f"\r[{bar}] {pct:3d}%  {detail[:60]:<60}")
    sys.stdout.flush()
    if pct >= 100:
        sys.stdout.write("\n")


def build_parser():
    ap = argparse.ArgumentParser(description="Project ANTARA reconstruction pipeline.")
    ap.add_argument("--video", help="Path to the drone video (mp4).")
    ap.add_argument("--gps", help="Path to GPS telemetry JSON or SRT (auto-detected from videos/ or gps/ if omitted).")
    ap.add_argument("--output", default="output", help="Output directory.")
    ap.add_argument("--no-sam", action="store_true",
                    help="Motion-diff debug masking instead of SAM 2.")
    ap.add_argument("--allow-synthetic-gps", action="store_true",
                    help="Demo mode only: permit generated GPS and mark accuracy unverified.")
    ap.add_argument("--verified-gps", action="store_true",
                    help="Explicitly assert that supplied telemetry is real flight GPS.")
    ap.add_argument("--gps-sync-mode", choices=("auto", "frame_idx", "timestamp_ms"),
                    default="auto",
                    help="Telemetry synchronization field. Required explicitly when "
                         "records contain both frame_idx and timestamp_ms.")
    ap.add_argument("--evaluate-only", action="store_true",
                    help="Re-score an existing run's output; no reconstruction.")
    ap.add_argument("--gaussian-splatting", "--gs", action="store_true",
                    dest="gaussian_splatting",
                    help="Enable the optional 3D Gaussian Splatting stage.")
    ap.add_argument("--sugar", action="store_true",
                    help="Enable SuGaR textured mesh extraction after Gaussian splatting.")
    ap.add_argument("--gs-iterations", type=int, default=7000,
                    help="Iteration count for Gaussian splatting training (default: 7000).")
    return ap


def _evaluate_only(cfg: PipelineConfig):
    """Re-run the honest accuracy report against already-exported output.

    Reads the georeferenced cloud + transforms.json + synced telemetry from a
    previous run. This does NOT re-reconstruct; it re-scores what's on disk.
    """
    import numpy as np
    from antara.evaluate import evaluate
    from antara.deliverables import export_transforms  # noqa: F401 (doc ref)

    telemetry = os.path.join(cfg.output_dir, "synced_telemetry.json")
    manifest_path = os.path.join(cfg.output_dir, "run_manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        telemetry = manifest.get("telemetry_path", telemetry)
        gps_source = manifest.get("gps_source", "unknown")
    else:
        gps_source = "unknown"
    transforms = os.path.join(cfg.colmap_dir(), "transforms.json")
    if not os.path.exists(telemetry) or not os.path.exists(transforms):
        print("No prior run found (need output/synced_telemetry.json and "
              "output/colmap_format/transforms.json). Run a full pipeline first.")
        return 1

    with open(telemetry) as f:
        gps_records = json.load(f)
    with open(transforms) as f:
        tf = json.load(f)

    cam_centers = np.array([m["transform_matrix"] for m in tf["frames"]],
                           dtype=np.float64)[:, :3, 3]
    # world_points unused by the scorer beyond being transformable; a light proxy.
    evaluate(
        cam_centers, cam_centers.copy(), gps_records, cfg,
        output_report=os.path.join(cfg.output_dir, "accuracy_report.json"),
        gps_source=gps_source,
    )
    return 0


def main():
    ap = build_parser()
    args = ap.parse_args()

    cfg = PipelineConfig()
    cfg.output_dir = args.output
    cfg.use_sam = not args.no_sam
    cfg.allow_synthetic_gps = args.allow_synthetic_gps
    cfg.gps_verified = args.verified_gps
    cfg.gps_sync_mode = args.gps_sync_mode
    cfg.enable_gaussian_splatting = bool(args.gaussian_splatting)
    cfg.enable_sugar = bool(args.sugar)
    cfg.gs_iterations = args.gs_iterations

    if cfg.enable_sugar and not cfg.enable_gaussian_splatting:
        cfg.enable_gaussian_splatting = True

    if args.evaluate_only:
        return _evaluate_only(cfg)

    if not args.video:
        ap.error("--video is required (or use --evaluate-only).")
    if not os.path.exists(args.video):
        ap.error(f"Video not found: {args.video}")

    result = run_pipeline(args.video, args.gps, cfg=cfg, progress=_print_progress)

    print("\n" + "=" * 60)
    print(" RECONSTRUCTION COMPLETE")
    print("=" * 60)
    print(f"  Points:     {result['num_points']:,}")
    print(f"  Keyframes:  {result['num_keyframes']}")
    print(f"  Scale:      {result['scale']:.4f}")
    if result["synthetic_gps"]:
        print("  [!] Synthetic GPS was used — accuracy numbers are NOT substantiated.")
    elif not result["gps_verified"]:
        print("  [!] GPS was supplied but not marked verified — accuracy claims are disabled.")
    print("  Outputs:")
    for name, path in result["outputs"].items():
        if path:
            print(f"    - {name:16s} {path}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
