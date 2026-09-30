# Project ANTARA: Windows 11 Prototype Implementation Guide
*Reconstruction architecture for SIH Problem Statement 17 (NTRO)*

> **This document was rewritten to match the real pipeline.** The earlier version
> described an obsolete workflow — three separate conda envs
> (`antara_sam2` / `antara_slam` / `antara_sugar`), four standalone step scripts
> (`01_preprocess.py`, `02_track_and_scale.py`, …), and an in-scope SuGaR meshing
> stage. That has all been replaced by a **single `antara` conda env** and one
> shared `antara/` package driven by `run_pipeline.py` (CLI) or `antara_server.py`
> (web). **For setup and run instructions, follow [setup_windows.md](setup_windows.md)
> — it is the canonical guide.** This file keeps only the architecture overview
> and the licensing-compliance reference.

---

## Architecture Overview

```
[Drone Video + GPS Telemetry]
        │
        ▼ (Stage 1: Preprocessing — antara/preprocess.py)
[Frame Extractor (Variance-of-Laplacian) + SAM 2 Metadata Masking] ──► masks.npz
        │   raw pixels never altered; masks stored as separate metadata
        ▼ (Stage 2: SLAM3R Reconstruction — antara/reconstruct.py)
[Raw Frames ──► real SLAM3R recon.py ──► fused world pointcloud]
        │   per-keyframe camera center = centroid of registered pointmap (proxy)
        │   HARD FAIL if CUDA/weights missing — no synthetic fallback
        ▼ (Stage 3: Georeferencing — antara/georeference.py)
[Post-inference SAM 2 filter + Global RANSAC-Umeyama + TPS] ──► transforms.json + .ply
        │
        ▼ (Stage 4: Deliverables — antara/deliverables.py)
[PLY + LAS + OBJ (Open3D Poisson) + GeoTIFF (rasterio)]
        │
        ▼ (Stage 5: Honest Evaluation — antara/evaluate.py)
[Alignment residual (diagnostic) + Hold-out ATE (real signal)] ──► accuracy_report.json
        │
        ▼ (Viewer)
[Three.js output/index.html]
```

**Not in this pipeline (approved *later* phases):**
- **SuGaR** surface-aligned Gaussian textured meshing — the seam is documented in
  `antara/deliverables.py:export_mesh`; today's mesh is Open3D screened Poisson.
- **Potree** WebGL conversion — the deliverable viewer is the Three.js `output/index.html`.
- **Symmetry facade completion** — was never implemented; removed from the claim.

---

## Setup and Execution

See **[setup_windows.md](setup_windows.md)**. In short:

```bash
conda env create -f environment.yml
conda activate antara
# clone SLAM3R + facebookresearch/sam2, fetch the SAM 2 tiny checkpoint (see doc)
python run_pipeline.py --video test_drone_flight.mp4 --gps test_gps_telemetry.json
```

The pipeline hard-fails (rather than emitting fabricated geometry) if CUDA, the
SLAM3R weights, or the SAM 2 checkpoint are missing. Use `--no-sam` for the
motion-diff debug mask, `--evaluate-only` to re-score an existing run.

---

## Defense Licensing Compliance Reference

| Package | License | NTRO Hackathon Status |
| :--- | :--- | :--- |
| **SAM 2** (`facebookresearch/sam2`) | Apache 2.0 | Approved: Unrestricted, edge-deployable, zero telemetry. |
| **SLAM3R** | CC BY-NC-SA 4.0 | Approved for research prototype. |
| **Open3D / laspy / rasterio** | MIT / BSD | Approved: Fully permissive. |
| **SuGaR / 3DGS** | Inria Non-commercial | Approved for research prototype (later phase, not yet integrated). |
| **Meta VGGT** | FAIR Noncommercial + AUP | **Disqualified:** Prohibits military/surveillance applications. |
| **Ultralytics YOLO** | AGPL-3.0 | **Disqualified:** Copyleft viral licensing — the exact restriction this table rules out. |

> Note: official Meta SAM 2 lives at `github.com/facebookresearch/sam2` (the older
> `segment-anything-2` URL redirects there). The Ultralytics SAM wrapper is
> deliberately **not** used — it is AGPL-3.0, which the compliance table disqualifies.
