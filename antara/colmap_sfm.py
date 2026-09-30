from __future__ import annotations

import os
import shutil
import subprocess
from typing import Callable, Tuple

import numpy as np


def _progress(progress: Callable | None, pct: int, msg: str):
    if progress is not None:
        progress(pct, msg)


def _require_colmap():
    colmap = shutil.which("colmap")
    if colmap:
        return colmap
    
    # Check the user's explicit COLMAP directory
    explicit_path = r"C:\Users\elite\COLMAP\COLMAP.bat"
    if os.path.exists(explicit_path):
        return explicit_path

    raise RuntimeError(
        "COLMAP is not installed or not on PATH. Please add your COLMAP folder to your system PATH."
    )


def run_colmap_sfm(frames_dir: str, cfg, progress=None, gps_records=None, frame_names=None) -> Tuple[str, np.ndarray | None]:
    """Run COLMAP feature extraction + matching + mapping on the extracted keyframes.

    This solves the real camera-pose problem described in the implementation plan:
    the external COLMAP pipeline estimates valid camera intrinsics/extrinsics from
    the keyframe images. The output model is written into the project output dir,
    where the 3DGS stage can consume it and the viewer can still use the legacy
    point cloud as the baseline reconstruction.
    """
    if not os.path.isdir(frames_dir):
        raise FileNotFoundError(f"Keyframe directory not found: {frames_dir}")

    colmap_bin = _require_colmap()
    sfm_root = os.path.join(cfg.output_dir, "colmap_sfm")
    os.makedirs(sfm_root, exist_ok=True)
    database_path = os.path.join(sfm_root, "database.db")

    if os.path.exists(database_path):
        os.remove(database_path)
        print(f"[COLMAP] Removed stale database: {database_path}")

    def run_cmd(cmd):
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        except subprocess.CalledProcessError as e:
            cmd_str = ' '.join(str(c) for c in cmd)
            raise RuntimeError(f"COLMAP command failed: {cmd_str}\nOutput:\n{e.output}")

    _progress(progress, 30, "Stage 2.5 — COLMAP feature extraction...")
    feature_cmd = [
        colmap_bin,
        "feature_extractor",
        "--database_path", database_path,
        "--image_path", frames_dir,
        "--ImageReader.camera_model", "OPENCV",
        "--FeatureExtraction.use_gpu", "1",
    ]
    run_cmd(feature_cmd)

    _progress(progress, 35, "Stage 2.5 — COLMAP feature matching...")
    match_cmd = [
        colmap_bin,
        "sequential_matcher",
        "--database_path", database_path,
        "--SequentialMatching.overlap", "15",
    ]
    run_cmd(match_cmd)

    _progress(progress, 40, "Stage 2.5 — COLMAP mapper...")
    mapper_output = os.path.join(sfm_root, "model")
    os.makedirs(mapper_output, exist_ok=True)
    mapper_cmd = [
        colmap_bin,
        "mapper",
        "--database_path", database_path,
        "--image_path", frames_dir,
        "--output_path", mapper_output,
    ]
    run_cmd(mapper_cmd)

    model_dir = os.path.join(mapper_output, "0")
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"COLMAP mapper did not produce a model at {model_dir}")

    geo_reg_path = os.path.join(sfm_root, "geo_registration.txt")
    if gps_records and frame_names:
        with open(geo_reg_path, "w") as f:
            for name, rec in zip(frame_names, gps_records):
                gps = rec.get("gps", rec)
                f.write(f"{name} {gps['longitude']} {gps['latitude']} {gps['altitude_m']}\n")

    if gps_records and os.path.isfile(geo_reg_path):
        align_cmd = [
            colmap_bin,
            "model_aligner",
            "--input_path", model_dir,
            "--output_path", model_dir,
            "--ref_images_path", geo_reg_path,
            "--ref_is_gps", "1",
            "--alignment_type", "enu",
            "--alignment_max_error", "3.0",
        ]
        # align_cmd is allowed to fail (e.g. if GPS is completely disjoint)
        subprocess.run(align_cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    # Match the directory contract expected by the reference 3DGS trainer:
    # <source>/images and <source>/sparse/0/{cameras,images,points3D}.bin.
    source_dir = os.path.join(sfm_root, "dataset")
    images_dir = os.path.join(source_dir, "images")
    sparse_dir = os.path.join(source_dir, "sparse", "0")
    if not os.path.exists(images_dir):
        shutil.copytree(frames_dir, images_dir)
    os.makedirs(sparse_dir, exist_ok=True)
    model_converter_cmd = [
        colmap_bin,
        "model_converter",
        "--input_path", model_dir,
        "--output_path", sparse_dir,
        "--output_type", "BIN",
    ]
    run_cmd(model_converter_cmd)
    
    # Also output TXT format so we can easily parse the camera poses for pipeline integration
    txt_dir = os.path.join(source_dir, "sparse_txt")
    os.makedirs(txt_dir, exist_ok=True)
    model_converter_txt_cmd = [
        colmap_bin,
        "model_converter",
        "--input_path", model_dir,
        "--output_path", txt_dir,
        "--output_type", "TXT",
    ]
    run_cmd(model_converter_txt_cmd)

    colmap_cams = None
    if frame_names:
        colmap_cams = parse_colmap_camera_centers(txt_dir, frame_names)
        if colmap_cams is not None:
            print(f"[COLMAP] Extracted true optical camera centers for {len(colmap_cams)} keyframes.")

    _progress(progress, 45, f"COLMAP SfM complete: {source_dir}")
    return source_dir, colmap_cams


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]
    ])


def parse_colmap_camera_centers(sparse_txt_dir: str, frame_names: list[str]) -> np.ndarray | None:
    """Parse COLMAP sparse model images.txt to extract optical camera centers C = -R.T @ T."""
    images_txt = os.path.join(sparse_txt_dir, "images.txt")
    if not os.path.isfile(images_txt) or not frame_names:
        return None

    poses: dict[str, np.ndarray] = {}
    with open(images_txt, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue

        parts = line.split()
        if len(parts) >= 10:
            name = " ".join(parts[9:])
            try:
                # Validate it's an image line (has integer IDs)
                _ = int(parts[0])  # IMAGE_ID
                _ = int(parts[8])  # CAMERA_ID
                qvec = np.array([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
                tvec = np.array([float(parts[5]), float(parts[6]), float(parts[7])])
                R = qvec2rotmat(qvec)
                center = -R.T @ tvec
                poses[name] = center
            except (ValueError, IndexError):
                pass

        # COLMAP images.txt has strictly two lines per image
        i += 2

    if not poses:
        return None

    cams: list[np.ndarray | None] = []
    missing_count = 0
    for name in frame_names:
        base_name = os.path.basename(name)
        if base_name in poses:
            cams.append(poses[base_name])
        elif name in poses:
            cams.append(poses[name])
        else:
            missing_count += 1
            cams.append(None)

    valid_indices = [i for i, c in enumerate(cams) if c is not None]
    if len(valid_indices) < 4 or missing_count > len(frame_names) * 0.5:
        return None

    valid_coords = np.array([cams[i] for i in valid_indices])
    result = np.empty((len(frame_names), 3), dtype=np.float64)
    for dim in range(3):
        result[:, dim] = np.interp(range(len(frame_names)), valid_indices, valid_coords[:, dim])
    return result
