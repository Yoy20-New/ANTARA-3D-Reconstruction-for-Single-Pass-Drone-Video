from __future__ import annotations
import os
import subprocess
import sys
from typing import Callable

from .config import GAUSSIAN_SPLATTING_DIR, SUGAR_DIR, PipelineConfig

def _progress(progress: Callable | None, pct: int, msg: str):
    if progress is not None:
        progress(pct, msg)

def _run_streaming(cmd, cwd, progress, label="Model"):
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in process.stdout:
        detail = line.strip()
        if detail:
            _progress(progress, 86, f"{label}: {detail[-180:]}")
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)

def train_gaussian_splat(ply_path: str, model_dir: str, cfg: PipelineConfig, colmap_model_dir: str | None = None, progress=None) -> str:
    """Run the optional Gaussian Splatting training stage as a real subprocess."""
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"Input point cloud not found: {ply_path}")

    if not os.path.isdir(GAUSSIAN_SPLATTING_DIR):
        raise FileNotFoundError(
            "Gaussian Splatting repo not found. Clone it into the project root as "
            f"{GAUSSIAN_SPLATTING_DIR} and install its CUDA extension."
        )

    train_script = os.path.join(GAUSSIAN_SPLATTING_DIR, "train.py")
    if not os.path.exists(train_script):
        raise FileNotFoundError(f"Missing train.py in {GAUSSIAN_SPLATTING_DIR}")

    os.makedirs(model_dir, exist_ok=True)
    _progress(progress, 78, "Stage 3.5 — launching Gaussian Splatting training...")

    if not colmap_model_dir:
        raise RuntimeError("Gaussian Splatting requires a successful COLMAP model with camera poses.")
    source_path = colmap_model_dir
    python_exe = sys.executable
    if not python_exe:
        raise RuntimeError("No Python executable found for Gaussian Splatting subprocess.")

    cmd = [
        python_exe,
        train_script,
        "--source_path", source_path,
        "--model_path", model_dir,
        "--iterations", str(cfg.gs_iterations),
    ]
    _run_streaming(cmd, GAUSSIAN_SPLATTING_DIR, progress, "3DGS")

    output_candidates = []
    for root, _, files in os.walk(model_dir):
        if "point_cloud.ply" in files:
            output_candidates.append(os.path.join(root, "point_cloud.ply"))
    if not output_candidates:
        raise FileNotFoundError(
            f"Gaussian Splatting completed without a point_cloud.ply under {model_dir}"
        )
    output_ply = sorted(output_candidates)[-1]

    _progress(progress, 90, f"Gaussian Splatting output ready: {output_ply}")
    return output_ply

def extract_sugar_mesh(gaussian_splat_path: str, output_dir: str, cfg: PipelineConfig, progress=None) -> str:
    """Run the optional SuGaR mesh extraction stage as a real subprocess."""
    if not gaussian_splat_path or not os.path.exists(gaussian_splat_path):
        raise FileNotFoundError(f"Gaussian Splat path not found: {gaussian_splat_path}")

    if not os.path.isdir(SUGAR_DIR):
        raise FileNotFoundError(
            "SuGaR repo not found. Clone it into the project root as "
            f"{SUGAR_DIR} before enabling --sugar."
        )

    train_script = os.path.join(SUGAR_DIR, "train.py")
    if not os.path.exists(train_script):
        raise FileNotFoundError(f"Missing train.py in {SUGAR_DIR}")

    os.makedirs(output_dir, exist_ok=True)
    _progress(progress, 82, "Stage 3.6 — launching SuGaR mesh extraction...")

    python_exe = sys.executable
    if not python_exe:
        raise RuntimeError("No Python executable found for SuGaR subprocess.")

    model_root = os.path.dirname(os.path.dirname(os.path.dirname(gaussian_splat_path)))
    colmap_model_dir = os.path.join(cfg.output_dir, "colmap_sfm", "dataset")

    cmd = [
        python_exe,
        train_script,
        "--scene_path", colmap_model_dir,
        "--checkpoint_path", model_root,
        "--output_dir", output_dir,
        "--mesh_output_dir", output_dir,
    ]
    _run_streaming(cmd, SUGAR_DIR, progress, "SuGaR")

    candidates = []
    for root, _, files in os.walk(output_dir):
        for filename in files:
            if filename.lower().endswith(".obj"):
                candidates.append(os.path.join(root, filename))

    if not candidates:
        raise FileNotFoundError(
            f"No OBJ mesh was generated in {output_dir}. Install SuGaR and run its mesh export step."
        )

    selected = sorted(candidates)[0]
    _progress(progress, 90, f"SuGaR mesh ready: {selected}")
    return selected
