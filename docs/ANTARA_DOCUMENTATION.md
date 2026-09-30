# PROJECT ANTARA — Comprehensive Documentation

**SIH Problem Statement 17 (NTRO) — Automated Edge 3D Reconstruction Console**
**Team HNM-OG**
Last updated: 2026-09-19

---

## Table of Contents

1. [Architecture of the Project](#1-architecture-of-the-project)
2. [Flow of the Project](#2-flow-of-the-project)
3. [Missing & Unclear Elements](#3-missing--unclear-elements)
4. [Existing Problems (ANTARA Criteria)](#4-existing-problems-antara-criteria)
5. [Prototype History](#5-prototype-history)

---

## 1. Architecture of the Project

### 1.1 High-Level System Overview

ANTARA is a **local, on-device drone-video-to-3D-model reconstruction system**. It takes a drone-captured video as input and produces a georeferenced, coloured 3D point cloud, surface mesh, and orthomosaic — with dynamic objects (vehicles, people) automatically removed.

```
┌────────────────────────────────────────────────────────────────────┐
│                      PROJECT ANTARA                                │
│                                                                    │
│  ┌──────────┐   ┌──────────┐   ┌───────────┐   ┌──────────────┐  │
│  │  Drone   │──▶│  Stage 1 │──▶│  Stage 2  │──▶│  Stage 2.5   │  │
│  │  Video   │   │ Preproc. │   │  SLAM3R   │   │ COLMAP SfM   │  │
│  │ + GPS    │   │ + Masking│   │ Reconst.  │   │ (Optional)   │  │
│  └──────────┘   └──────────┘   └───────────┘   └──────────────┘  │
│                                      │                │           │
│                                      ▼                ▼           │
│  ┌──────────────┐   ┌──────────────────────────────────────────┐  │
│  │   Stage 5    │◀──│  Stage 3: GPS Georeferencing             │  │
│  │  Evaluation  │   │  (RANSAC-Umeyama + Gravity + TPS)        │  │
│  └──────────────┘   └──────────────────────────────────────────┘  │
│         ▲                              │                          │
│         │           ┌──────────────────┼──────────────────┐       │
│         │           ▼                  ▼                  ▼       │
│         │    ┌────────────┐   ┌──────────────┐   ┌────────────┐  │
│         │    │ Stage 3.5  │   │  Stage 3.6   │   │  Stage 4   │  │
│         │    │   3DGS     │──▶│   SuGaR      │   │Deliverables│  │
│         │    │ (Optional) │   │ (Optional)   │   │PLY/LAS/OBJ │  │
│         │    └────────────┘   └──────────────┘   │GeoTIFF/JSON│  │
│         │                                        └────────────┘  │
│         └────────────────────────────────────────────┘            │
└────────────────────────────────────────────────────────────────────┘
```

### 1.2 Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Dynamic Object Detection** | YOLOE (Ultralytics) | Open-vocabulary bounding-box detection of cars, people, trucks, etc. |
| **Instance Segmentation** | SAM 2 (Meta) | Pixel-perfect mask generation from YOLOE box prompts |
| **3D Reconstruction** | SLAM3R (PKU-VCL) | DUSt3R-lineage dense pointmap regression from video frames |
| **Camera Pose Estimation** | COLMAP SfM | Structure-from-Motion for real camera intrinsics/extrinsics |
| **Photorealistic Meshing** | 3D Gaussian Splatting + SuGaR | Optional textured mesh via Gaussian radiance fields |
| **Baseline Meshing** | Open3D (Poisson) | Untextured surface mesh fallback |
| **Georeferencing** | Custom (NumPy / SciPy) | RANSAC-Umeyama similarity + TPS residual correction |
| **Output Formats** | PLY, LAS (laspy), OBJ, GeoTIFF (rasterio) | Standard GIS and 3D formats |
| **Viewer** | Three.js (browser) | Interactive 3D point cloud viewer |
| **Web Portal** | Flask | Local upload server for video + GPS telemetry |
| **GPU Runtime** | PyTorch 2.5.1 + CUDA 11.8 | GPU inference for SAM 2 and SLAM3R |
| **Platform** | Windows + Conda (`antara` env) | Target deployment environment |

### 1.3 Directory Structure

```
E:\SIH/
├── antara/                     # Core pipeline package
│   ├── __init__.py
│   ├── config.py               # Single source of truth for all constants
│   ├── pipeline.py             # Orchestrator — wires all stages
│   ├── preprocess.py           # Stage 1: keyframes, blur filter, SAM 2 masks, GPS sync
│   ├── reconstruct.py          # Stage 2: real SLAM3R subprocess invocation
│   ├── colmap_sfm.py           # Stage 2.5: COLMAP camera pose estimation
│   ├── georeference.py         # Stage 3: Umeyama + gravity + TPS alignment
│   ├── gaussians.py            # Stage 3.5: 3D Gaussian Splatting training
│   ├── sugar.py                # Stage 3.6: SuGaR textured mesh extraction
│   ├── deliverables.py         # Stage 4: PLY / LAS / OBJ / GeoTIFF / transforms.json
│   └── evaluate.py             # Stage 5: honest alignment-residual + hold-out ATE
│
├── run_pipeline.py             # CLI entrypoint (thin shell over antara.pipeline)
├── antara_server.py            # Flask web upload portal (thin shell over antara.pipeline)
├── environment.yml             # Conda environment specification
│
├── SLAM3R/                     # External: PKU-VCL SLAM3R repository
├── sam2/                       # External: Meta SAM 2 repository
├── gaussian-splatting/         # External: Inria 3DGS reference implementation
├── SuGaR/                      # External: SuGaR mesh extraction
├── checkpoints/                # SAM 2 model checkpoint (sam2.1_hiera_tiny.pt)
│
├── videos/                     # User video input files
├── output/                     # Generated outputs
│   ├── index.html              # Three.js interactive viewer
│   ├── colmap_format/          # georeferenced_cloud.ply + transforms.json
│   ├── antara_model.las        # LAS point cloud
│   ├── antara_model.obj        # Surface mesh (Poisson or SuGaR)
│   ├── antara_orthomosaic.tif  # Georeferenced top-down image
│   ├── accuracy_report.json    # Honest accuracy evaluation
│   └── run_manifest.json       # Full run metadata and provenance
│
├── tests/                      # Automated tests
├── PROJECT_HISTORY.md          # Historical project state document
├── SETUP_FROM_SCRATCH.md       # Clean installation guide
└── ANTARA_DOCUMENTATION.md     # This file
```

### 1.4 Module Dependency Graph

```
run_pipeline.py ──┐
                  ├──▶ antara.pipeline (orchestrator)
antara_server.py ─┘         │
                            ├──▶ antara.config        (constants & PipelineConfig dataclass)
                            ├──▶ antara.preprocess     (YOLOE + SAM 2 + keyframes + GPS)
                            ├──▶ antara.reconstruct    (SLAM3R subprocess)
                            ├──▶ antara.colmap_sfm     (COLMAP subprocess)
                            ├──▶ antara.georeference   (Umeyama + Gravity + TPS)
                            ├──▶ antara.gaussians      (3DGS subprocess)
                            ├──▶ antara.sugar           (SuGaR subprocess)
                            ├──▶ antara.deliverables    (PLY / LAS / OBJ / GeoTIFF export)
                            └──▶ antara.evaluate        (hold-out ATE scoring)
```

### 1.5 Entrypoints

| Entrypoint | Type | Description |
|-----------|------|-------------|
| `run_pipeline.py` | CLI | `argparse`-based command line. Supports `--video`, `--gps`, `--allow-synthetic-gps`, `--gaussian-splatting`, `--sugar`, `--evaluate-only` |
| `antara_server.py` | Web (Flask) | Local upload portal at `http://127.0.0.1:8000`. Accepts video + GPS via multipart POST. Streams progress via `/api/status` polling |

Both entrypoints are **thin shells** over the same `antara.pipeline.run()` function. This eliminates the legacy problem where CLI and server had drifted apart with different thresholds and different bugs.

---

## 2. Flow of the Project

### 2.1 Complete Pipeline Flow

```
┌─────────────────────────────────────────────────────────┐
│               INPUT: Drone Video (.mp4)                 │
│              + Optional GPS Telemetry (.json)           │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
╔═════════════════════════════════════════════════════════╗
║  STAGE 1: PREPROCESSING (2%–25%)                       ║
║                                                         ║
║  1. Extract keyframes at target FPS (default: 3 fps)    ║
║  2. Blur filter (Laplacian variance < 40 → dropped)     ║
║  3. YOLOE detects dynamic objects (cars, people, etc.)  ║
║  4. SAM 2 segments pixel-perfect masks from YOLOE boxes ║
║  5. Masks saved as separate metadata (masks.npz)        ║
║     ⚠ Raw image pixels are NEVER altered                ║
║  6. GPS telemetry loaded, validated, and interpolated    ║
║     to each extracted keyframe                          ║
║  7. SAM 2 + YOLOE freed from GPU memory                ║
╚═════════════════════════╤═══════════════════════════════╝
                          │
     Output: frames/, masks.npz, synced_telemetry.json
                          │
                          ▼
╔═════════════════════════════════════════════════════════╗
║  STAGE 2: SLAM3R RECONSTRUCTION (27%–65%)              ║
║                                                         ║
║  1. Verify CUDA + SLAM3R repo + weights                 ║
║  2. Invoke SLAM3R recon.py as subprocess                ║
║  3. SLAM3R regresses dense pointmaps (not camera poses) ║
║  4. L2W model registers each keyframe into world frame  ║
║  5. Fused confidence-filtered world cloud exported      ║
║  6. Camera-center PROXY: confidence-weighted centroid    ║
║     of each frame's registered pointmap                 ║
║  7. Dynamic-object masks applied POST-INFERENCE         ║
║     (points in masked regions removed from cloud)       ║
║  ⚠ Hard-fail policy: no synthetic fallback              ║
╚═════════════════════════╤═══════════════════════════════╝
                          │
     Output: world_points, world_colors, cam_centers,
             <scene>_recon.ply
                          │
                          ▼
╔═════════════════════════════════════════════════════════╗
║  STAGE 2.5: COLMAP SfM (Optional) (60%–61%)           ║
║                                                         ║
║  1. COLMAP feature extraction (SIFT)                    ║
║  2. Exhaustive feature matching                         ║
║  3. Incremental mapper → sparse model                   ║
║  4. Model converted to BIN format                       ║
║  5. Dataset directory assembled for 3DGS trainer        ║
║  ⚠ Graceful fallback if COLMAP unavailable              ║
╚═════════════════════════╤═══════════════════════════════╝
                          │
                          ▼
╔═════════════════════════════════════════════════════════╗
║  STAGE 3: GPS GEOREFERENCING (66%–78%)                 ║
║                                                         ║
║  1. GPS lat/lon/alt → local ENU metric coordinates      ║
║  2. RANSAC-Umeyama: robust 7-DOF similarity transform   ║
║     (scale + rotation + translation) on RANSAC inliers  ║
║  3. Gravity alignment: PCA-based tilt correction        ║
║     (for near-collinear drone paths at constant alt)    ║
║  4. TPS (Thin-Plate-Spline) residual correction         ║
║     applied ONLY on RANSAC inliers (not GPS outliers)   ║
║  5. Point cloud deformation bounded to physical GPS     ║
║     residual magnitude (prevents TPS extrapolation      ║
║     from flinging distant points to infinity)           ║
╚═════════════════════════╤═══════════════════════════════╝
                          │
                          ▼
╔═════════════════════════════════════════════════════════╗
║  STAGE 3.5: 3D Gaussian Splatting (Optional) (~78%)    ║
║                                                         ║
║  1. Requires successful COLMAP model                    ║
║  2. Runs gaussian-splatting train.py as subprocess      ║
║  3. Input: COLMAP poses + SLAM3R cloud + keyframes      ║
║  4. Output: Gaussian scene .ply                         ║
║  5. Configurable iterations (7K fast / 30K quality)     ║
╚═════════════════════════╤═══════════════════════════════╝
                          │
                          ▼
╔═════════════════════════════════════════════════════════╗
║  STAGE 3.6: SuGaR Mesh Extraction (Optional) (~82%)   ║
║                                                         ║
║  1. Requires 3DGS trained checkpoint                    ║
║  2. Runs SuGaR train.py as subprocess                   ║
║  3. Output: textured OBJ + MTL + texture atlas          ║
║  4. Fallback: Open3D Poisson mesh if SuGaR fails        ║
╚═════════════════════════╤═══════════════════════════════╝
                          │
                          ▼
╔═════════════════════════════════════════════════════════╗
║  STAGE 4: DELIVERABLES EXPORT (80%–92%)                ║
║                                                         ║
║  Exports produced:                                      ║
║  • PLY   — coloured point cloud (Three.js viewer)       ║
║  • LAS   — GIS point cloud (1.4, point format 3, RGB)   ║
║  • OBJ   — surface mesh (Poisson or SuGaR)             ║
║  • GeoTIFF — top-down orthomosaic (EPSG:4326)           ║
║  • transforms.json — NeRF/SuGaR camera convention       ║
║  • coordinate_metadata.json — CRS sidecar               ║
║  • run_manifest.json — full provenance metadata          ║
╚═════════════════════════╤═══════════════════════════════╝
                          │
                          ▼
╔═════════════════════════════════════════════════════════╗
║  STAGE 5: HONEST ACCURACY EVALUATION (94%–100%)        ║
║                                                         ║
║  Two clearly separated metrics:                         ║
║                                                         ║
║  1. ALIGNMENT RESIDUAL (fit diagnostic only)            ║
║     Error on keyframes USED to fit the transform.       ║
║     Expected near zero. NOT accuracy.                   ║
║                                                         ║
║  2. HOLD-OUT ATE (the real accuracy signal)             ║
║     A fraction (default 20%) of GPS keyframes are       ║
║     withheld from the fit; error measured on those.     ║
║     THIS is what speaks to real accuracy.               ║
║                                                         ║
║  Accuracy claims DISABLED unless GPS is explicitly      ║
║  marked as verified real flight telemetry.              ║
║  Target: ≤ 1.0 m RMSE (NTRO PS-17 criterion)          ║
╚═════════════════════════════════════════════════════════╝
```

### 2.2 Data Flow Diagram

```
Drone Video ──┬──▶ Keyframes (JPEG)
              │         │
              │         ├──▶ YOLOE boxes ──▶ SAM 2 masks ──▶ masks.npz
              │         │
              │         └──▶ SLAM3R ──▶ registered_pcds.npy
              │                  │             + registered_confs.npy
              │                  │
              │                  ├──▶ Fused world cloud (*_recon.ply)
              │                  │
              │                  └──▶ Camera-center proxy (K,3)
              │
GPS JSON ─────┴──▶ Synced telemetry ──▶ Local ENU coordinates
                         │                       │
                         ▼                       ▼
                   Georeferencing ──▶ Corrected cloud + cameras
                         │
                         ├──▶ PLY, LAS, OBJ, GeoTIFF
                         ├──▶ transforms.json
                         ├──▶ accuracy_report.json
                         └──▶ run_manifest.json
```

### 2.3 GPU Memory Lifecycle

The pipeline is designed for a **6 GB VRAM** budget (RTX 3050):

| Phase | VRAM Usage | Strategy |
|-------|-----------|----------|
| Stage 1 (SAM 2 + YOLOE) | ~2–3 GB | Freed via `torch.cuda.empty_cache()` before Stage 2 |
| Stage 2 (SLAM3R) | ~4–5 GB | Runs as subprocess; memory released on exit |
| Stage 2.5 (COLMAP) | ~1 GB | CPU-heavy; GPU only for SIFT |
| Stage 3.5 (3DGS training) | ~4–5 GB | Runs after SLAM3R, not simultaneously |
| Stage 3.6 (SuGaR) | ~3–4 GB | Runs after 3DGS, not simultaneously |

---

## 3. Missing & Unclear Elements

> These are elements related to the **project's topic and domain** (drone-based 3D mapping for NTRO PS-17), not code bugs.

### 3.1 Unverified / Missing Domain Elements

| # | Element | Status | Detail |
|---|---------|--------|--------|
| 1 | **Real GPS telemetry** | ❌ Missing | No actual drone flight GPS has been used. All georeferenced outputs so far use **synthetic straight-line GPS** (base: 28.6139°N, 77.2090°E — New Delhi). Accuracy claims are meaningless without real flight telemetry. |
| 2 | **NTRO PS-17 compliance criteria** | ⚠️ Unclear | The exact acceptance criteria from NTRO PS-17 beyond "sub-metre accuracy" are not fully enumerated in the project. What specific deliverable formats, metadata schemas, or reporting standards does NTRO expect? |
| 3 | **"Single-pass" definition** | ⚠️ Unclear | The project title says "Single-Pass Drone Video 3D Mapping" — but what constitutes a "single pass"? Is it one continuous flight? One orbit? A linear fly-over? This affects how georeferencing handles trajectory geometry. |
| 4 | **Edge deployment target** | ⚠️ Unclear | PS-17 mentions "Automated **Edge** 3D Reconstruction Console." What is the edge device? An embedded GPU (Jetson)? A ruggedized laptop with a discrete GPU? The current pipeline requires an RTX 3050+ (6 GB VRAM) which is a desktop/laptop class GPU, not a typical edge device. |
| 5 | **Camera intrinsics** | ❌ Missing | The pipeline currently uses a **hardcoded** `camera_angle_x: 1.047` (≈60°) in `transforms.json`. Real camera intrinsics (focal length, principal point, distortion) from the actual drone camera are unknown and not calibrated. |
| 6 | **Vertical datum for GPS altitude** | ⚠️ Unclear | The GPS `altitude_m` field has no specified datum. Is it WGS-84 ellipsoidal height? EGM96 geoid height? MSL? The `coordinate_metadata.json` says "datum not inferred." This matters for real-world accuracy claims. |
| 7 | **Ground Control Points (GCPs)** | ❌ Not implemented | Professional photogrammetry uses GCPs for sub-centimetre accuracy validation. ANTARA has no mechanism to ingest GCPs or use them as independent accuracy checkpoints. |
| 8 | **Flight speed and altitude assumptions** | ⚠️ Unverified | The keyframe extraction rate (`target_fps: 3`) and stride (`keyframe_stride: 6`) assume a certain flight speed and altitude. These have not been validated against real drone flight profiles. |
| 9 | **Symmetry-based facade completion** | ❌ Not implemented | Documented as a future feature. For urban reconstruction of buildings, occluded sides could be inferred from architectural symmetry. No algorithm or approach has been selected. |
| 10 | **Multi-sensor fusion** | ❌ Not explored | Real drones often carry LiDAR, IMU, and barometric sensors. The pipeline only uses monocular video + GPS. Fusion with LiDAR or IMU could dramatically improve reconstruction and georeferencing. |
| 11 | **Regulatory and operational context** | ⚠️ Unclear | What are the DGCA (Directorate General of Civil Aviation) or Ministry of Defence regulations governing drone surveys in the deployment context? Does the output need to comply with Survey of India standards? |
| 12 | **Scale of target area** | ⚠️ Unclear | What is the expected area coverage? A single building? A city block? A 1 km² region? This determines whether the current `num_points_save: 200,000` point budget is adequate or severely limiting. |

### 3.2 Unclear Technical Terms

| Term | Used Where | What's Unclear |
|------|-----------|----------------|
| **PS-17 sub-metre criterion** | `config.py` (`target_accuracy_m: 1.0`) | Is this RMSE? Mean error? p95? The codebase uses RMSE but it's unclear if NTRO specifies the statistical measure. |
| **"Edge" in "Edge 3D Reconstruction Console"** | Problem Statement | Does "edge" mean edge computing (near-device), edge of network, or edge of the battlefield? Affects hardware constraints. |
| **"Automated"** | Problem Statement | Fully autonomous (no operator input) or semi-automated (operator triggers, machine processes)? Current design requires manual video upload and GPS file pairing. |
| **"Console"** | Problem Statement | A desktop GUI application? A web dashboard? A terminal-based tool? Currently implemented as CLI + basic Flask web server. |
| **DUSt3R lineage** | `reconstruct.py` | SLAM3R is described as having "DUSt3R lineage" but the specific architectural differences and capability differences versus DUSt3R v1/v2 for this use case are not documented. |

---

## 4. Existing Problems (ANTARA Criteria)

> Problems evaluated against the ANTARA project's own documented criteria and design principles from `PROJECT_HISTORY.md`, `config.py`, `evaluate.py`, and the implementation plan.

### 4.1 Accuracy & Georeferencing Problems

| # | Problem | Severity | Detail |
|---|---------|----------|--------|
| **A1** | **No real GPS = no accuracy claims** | 🔴 Critical | Every run so far used synthetic GPS. The evaluation module correctly refuses to substantiate accuracy claims, but this means the project has **zero validated accuracy measurements**. |
| **A2** | **Camera-center proxy, not real poses** | 🟡 Major | SLAM3R provides pointmaps, not camera extrinsics. The "camera center" is a confidence-weighted centroid — a proxy that sits *near the scene the camera was looking at*, not at the camera's optical center. For GPS alignment this works tolerably; for 3DGS training it is **insufficient** (3DGS needs full 6-DOF R\|t). |
| **A3** | **Hardcoded camera_angle_x** | 🟡 Major | `transforms.json` uses a fixed `camera_angle_x: 1.047` (~60° FOV). Real drone cameras vary (DJI Mini: ~83°, DJI Mavic: ~77°, etc.). Wrong intrinsics produce distorted 3DGS and SuGaR meshes. |
| **A4** | **Circular accuracy in legacy evaluation** | 🟢 Fixed | The old code measured RMSE between TPS-snapped cameras and the GPS they were snapped to (≈0 by construction). The new `evaluate.py` correctly separates alignment residual from hold-out ATE. |
| **A5** | **Hold-out ATE trivially passes with synthetic GPS** | 🟡 Misleading | Synthetic GPS is a straight line; even the hold-out set is trivially interpolable. A "PASSED" verdict with synthetic GPS does not mean anything. The code correctly prints a caveat but the report structure could still mislead a reviewer. |

### 4.2 Reconstruction Quality Problems

| # | Problem | Severity | Detail |
|---|---------|----------|--------|
| **R1** | **200K point budget may be sparse** | 🟡 Major | `num_points_save: 200,000` is a compromise for 6 GB VRAM. For a building or city block, this yields ~0.2 points/m² at typical drone altitudes — far below the ~50–100 points/m² that LiDAR or dense MVS typically achieves. |
| **R2** | **No texture on baseline mesh** | 🟡 Major | The Open3D Poisson mesh (`antara_model.obj`) is **untextured**. It's a grey surface with vertex colors at best. SuGaR integration exists in code but is optional and untested end-to-end with real data. |
| **R3** | **No multi-view consistency check** | 🟡 Moderate | The pipeline does not verify that SLAM3R's registered pointmaps are globally consistent. Drift or registration failures in SLAM3R produce a corrupted cloud that propagates through all downstream stages. |
| **R4** | **Orthomosaic is point-projection, not mosaic** | 🟡 Moderate | `export_geotiff()` projects 3D points onto a 1024×1024 grid. This is not a true orthorectified mosaic (which stitches actual image pixels). The result has large gaps between projected points and no visual detail. |

### 4.3 Pipeline Robustness Problems

| # | Problem | Severity | Detail |
|---|---------|----------|--------|
| **P1** | **Single active run under output/** | 🟡 Major | All runs overwrite the same `output/` directory. There is no multi-run history, per-run browser routing, or run comparison. The `.runs/<run_id>/` staging directory isolates intermediate files but final deliverables clobber each other. |
| **P2** | **No run resume / checkpoint** | 🟡 Moderate | If SLAM3R takes 30+ minutes and Stage 4 fails (e.g., Open3D mesh OOM), the entire pipeline must be re-run from scratch. There is no way to restart from Stage 4 with existing SLAM3R output. |
| **P3** | **No input validation for video codec** | 🟢 Minor | `cv2.VideoCapture` silently fails on some codecs. The pipeline checks `cap.isOpened()` but does not validate codec, resolution, or duration before committing to a long SLAM3R run. |
| **P4** | **Flask server has no authentication** | 🟡 Major | The web portal defaults to localhost but can be bound to `0.0.0.0` with `--host`. There is no auth, CSRF protection, or rate limiting. Anyone on the network can trigger GPU jobs. |
| **P5** | **COLMAP failure is non-fatal, silently disables 3DGS** | 🟢 Minor | If COLMAP SfM fails (e.g., insufficient features), the pipeline continues with proxy camera centers. But 3DGS training then hard-fails because it requires real COLMAP poses. The error path could be clearer. |

### 4.4 Honesty & Provenance Problems

| # | Problem | Severity | Detail |
|---|---------|----------|--------|
| **H1** | **Synthetic GPS outputs could be misinterpreted** | 🟡 Major | Demo-mode outputs (LAS, GeoTIFF) contain real-looking WGS-84 coordinates derived from synthetic GPS. A downstream user who receives just the LAS file without reading `run_manifest.json` could mistake these for genuine survey data. |
| **H2** | **Viewer does not display provenance** | 🟡 Moderate | The Three.js viewer (`output/index.html`) does not show whether the model was generated with synthetic or real GPS, or whether accuracy is verified. `PROJECT_HISTORY.md` recommends labelling (e.g., "Demo Reconstruction — Synthetic GPS") but this is not implemented in the viewer. |
| **H3** | **No watermark or metadata in exported files** | 🟢 Minor | PLY, OBJ, and GeoTIFF files have no embedded provenance (synthetic vs. real GPS, pipeline version, run ID). The separate `run_manifest.json` and `coordinate_metadata.json` sidecars exist but can be separated from the deliverable files. |

### 4.5 Evaluation Criteria Compliance (NTRO PS-17)

| NTRO Criterion (Inferred) | Status | Gap |
|---------------------------|--------|-----|
| Sub-metre spatial accuracy | ❌ Unverified | No real GPS data used; synthetic GPS trivially passes |
| Automated processing | ⚠️ Partial | Requires manual video upload + GPS pairing; no autonomous flight integration |
| Edge deployment | ❌ Not met | Requires desktop GPU (RTX 3050 minimum, 6 GB VRAM) |
| Real-time or near-real-time | ❌ Not met | Full pipeline takes 30–60 minutes per scene |
| Standard output formats | ✅ Met | PLY, LAS, OBJ, GeoTIFF all implemented |
| Dynamic object removal | ✅ Met | YOLOE + SAM 2 pipeline works on real video |

---

## 5. Prototype History

### Prototype 0: Legacy Multi-Script Era (Pre-cleanup)

**Architecture:** Four separate, overlapping pipeline scripts:
- `01_preprocess.py`
- `02_track_and_scale.py`
- `run_pipeline_cloud.py`
- `antara_master.py`

**Key characteristics:**
- Each script carried its own inline copy of the pipeline
- **SLAM3R stage was faked** in all four scripts:
  - Camera positions: `idx * 0.75` (hardcoded linear trajectory)
  - Depths: `np.random` (random noise, not reconstruction)
  - Point cloud: procedural point grid (fabricated geometry)
- Configuration values had drifted across copies:
  - Blur threshold: 35 / 40 / 50 (three different values)
  - SLAM drift factor: 0.92 / 0.95 (two different values)
  - Motion-diff thresholds: 28 / 30 / 35 (three different values)
- **Umeyama scale bug**: `scale = np.trace(np.diag(S)) / np.var(src, axis=0).sum()` — incorrect formula that does not implement Umeyama (1991)
- Server bound to `0.0.0.0` with no authentication
- Circular accuracy evaluation (measuring TPS fit-to-self)

**Problems carried forward:** Everything was broken. No real reconstruction ever ran.

---

### Prototype 1: Unified Pipeline + Real SLAM3R

**Changes from Prototype 0:**
1. **Consolidated four scripts into one `antara/` package** with a single `pipeline.py` orchestrator
2. Both `run_pipeline.py` (CLI) and `antara_server.py` (Flask) became thin shells over the same `antara.pipeline.run()`
3. **Real SLAM3R reconstruction** via subprocess invocation of `recon.py` — no more fabricated geometry
4. **Umeyama scale formula corrected** to proper Umeyama (1991) closed-form: `scale = trace(D @ S_diag) / variance_src`
5. **Configuration centralised** in `config.py` as a `PipelineConfig` dataclass — one source of truth
6. **Hard-fail policy introduced**: if CUDA, SLAM3R weights, or SAM 2 are unavailable, the run raises and stops. No synthetic geometry fallback.
7. **SAM 2 dynamic masking** with OpenCV connected-components cleanup (avoids Windows `sam2._C` CUDA extension dependency)
8. **Separate mask metadata**: raw image pixels are never altered. Masks stored in `masks.npz` and applied to reconstructed 3D points post-inference
9. **Camera-center proxy documented**: confidence-weighted centroid of registered pointmaps used as camera position proxy (SLAM3R doesn't expose camera poses)
10. **GPS validation**: input telemetry checked for completeness, numeric validity, and coverage before reconstruction starts
11. **RANSAC-Umeyama**: robust similarity with inlier selection instead of least-squares-on-everything
12. **TPS residual correction**: thin-plate-spline applied only on RANSAC inliers (not GPS outliers)
13. **Server defaults to localhost** with warning for network binding
14. **Per-run staging** via `.runs/<run_id>/` directories to isolate intermediate files

**Verified:** SLAM3R completed a real reconstruction producing a 200,000-point cloud. SAM 2 model loading, one-frame inference, and short-video preprocessing succeeded.

---

### Prototype 2: Honest Evaluation + GPS Provenance

**Changes from Prototype 1:**
1. **Honest accuracy evaluation** (`evaluate.py`):
   - Separated alignment residual (fit diagnostic) from hold-out ATE (real accuracy signal)
   - 20% of GPS keyframes withheld from the transform fit
   - Accuracy claims disabled unless GPS is explicitly `--verified-gps`
2. **GPS provenance tracking**: every output carries `gps_source` field — `"synthetic"`, `"provided_unverified"`, or `"verified"`
3. **Honesty caveat** printed at runtime and embedded in report: "Synthetic or unverified test telemetry does not substantiate a sub-metre claim"
4. **`run_manifest.json`**: comprehensive run metadata including video path, GPS source, sync mode, staging directory, artifact paths, mask statistics, and point counts
5. **`coordinate_metadata.json`**: sidecar for LAS/PLY explaining the local ENU metric coordinate frame and its WGS-84 origin
6. **GPS synchronization modes**: explicit `--gps-sync-mode` for `frame_idx` vs `timestamp_ms` with auto-detection and rejection of ambiguous records

---

### Prototype 3: YOLOE + SAM 2 Box-Prompted Masking

**Changes from Prototype 2:**
1. **Replaced SAM 2 grid-based masking with YOLOE + SAM 2 box-prompted approach**:
   - Old: `SAM2AutomaticMaskGenerator` (144 grid points per frame, slow, class-unaware) + motion-diff overlap to identify dynamic objects
   - New: YOLOE open-vocabulary detector → bounding boxes for dynamic object classes → SAM 2 `ImagePredictor` with box prompts → pixel-perfect masks
2. **Faster and more accurate**: Only N detections instead of 144 grid points; class-aware (no false positives from shadows or camera shake)
3. **Configurable dynamic classes**: `["car", "person", "truck", "bus", "motorcycle", "bicycle", "van", "dog", "bird", "boat"]`
4. **MobileClip backbone**: YOLOE uses `mobileclip_blt.ts` (600 MB) for text-prompted detection

---

### Prototype 4: Gravity Alignment + 3DGS/SuGaR Integration (Current)

**Changes from Prototype 3:**
1. **Gravity alignment correction** in georeferencing:
   - PCA-based detection of the cloud's "thin" (height) axis
   - Rodrigues rotation to align the thin axis with ENU +Z
   - Safety checks: skips correction if GPS has strong altitude variation (>15% of horizontal range) or if scene is not clearly planar (flatness ratio > 0.35)
2. **COLMAP SfM module** (`colmap_sfm.py`):
   - Runs feature extraction → exhaustive matching → incremental mapping
   - Exports standard `<source>/images` + `<source>/sparse/0/` directory layout for 3DGS
   - Graceful fallback if COLMAP not installed
3. **3D Gaussian Splatting module** (`gaussians.py`):
   - Subprocess invocation of Inria reference `train.py`
   - Requires COLMAP model (hard requirement, not optional)
   - Configurable iteration count (`--gs-iterations`, default 7000)
4. **SuGaR mesh extraction module** (`sugar.py`):
   - Subprocess invocation of SuGaR `train.py`
   - Produces textured OBJ + MTL + texture atlas
   - Falls back to Open3D Poisson mesh if SuGaR fails
5. **Pipeline wiring**: 3DGS + SuGaR stages gated by `--gaussian-splatting` / `--sugar` CLI flags
6. **TPS deformation bounding**: point-cloud TPS extrapolation capped at `max(5.0, 2× max GPS residual)` to prevent unbounded drift far from the trajectory

**Current state:** Code is wired but 3DGS/SuGaR path is **untested end-to-end with real data** (COLMAP, gaussian-splatting, and SuGaR repos present but not verified on a real flight).

---

### Prototype 5: Automated Edge Console & Pipeline UI (Current)

**Changes from Prototype 4:**
1. **Minimalist 3D Web Viewer (`output/index.html`)**:
   - Completely redesigned from a cluttered dashboard to a sleek, monochrome, clamp-responsive UI.
   - Removed broken ambient/directional light sliders (PointsMaterial ignores lights).
   - Fixed ENU-to-ThreeJS coordinate alignment for OBJ meshes (`alignObjMesh()`).
   - Added Drag-and-Drop file loading for local `.ply` and `.obj` files directly into the browser.
2. **Live Geographic Readout (HUD)**:
   - Added a `THREE.Raycaster` HUD that tracks the cursor and reverse-calculates exact real-world ENU coordinates (East, North, Elevation). This visually proves the "measurement and analysis" requirements.
3. **Decoupled FastAPI Backend (`backend/`)**:
   - Replaced the old Flask server with a modern FastAPI architecture (`server.py`) and a dedicated job state machine (`job_manager.py`).
   - Implemented streaming XHR file uploads safely handling up to 10GB 4K videos.
   - Added a `/viewer` 302 redirect route to correctly serve the static 3D HTML and assets.
4. **Advanced Operator Console (`upload_page.html`)**:
   - Built a sleek upload interface with a live progress bar tracking all 5 stages.
   - **Exposed Pipeline Parameters:** Added an "Advanced Processing Settings" panel allowing the user to bypass hardcoded settings and tweak:
     - *Masking Engine:* Disable SAM 2 for fast motion-diff.
     - *Rendering:* Disable Gaussian Splatting (saves 10+ mins).
     - *Frame Extraction:* 2, 3, 5, or 10 FPS targets.
     - *SLAM Stride:* Jump interval configuration (2, 4, 6).
     - *Point Density:* 200k, 500k, 1M vertex limits.
5. **Job Cancellation Architecture**:
   - Added a `/api/cancel` route and a `JobCancelledError` exception in `job_manager.py`.
   - Allows users to gracefully abort a running background thread instantly via a red "Cancel Processing" UI button, resetting the system cleanly.
6. **Poisson Mesh Cleanup**:
   - Fixed the "sky bowl" extrapolation artifact in `deliverables.py` by capturing `densities` and filtering out the bottom 5% lowest-density vertices using `mesh.remove_vertices_by_mask()`.

---

### Summary Timeline

```
Prototype 0  →  Prototype 1  →  Prototype 2  →  Prototype 3  →  Prototype 4    →  Prototype 5
  (Legacy)       (Real SLAM)     (Honest)       (YOLOE)        (Gravity+3DGS)     (Edge Console)
                                                                                        ▲
    ✗ Fake          ✓ Real           ✓ Honest        ✓ Better       ✓ Configured       CURRENT
    geometry        recon            evaluation      masking        3DGS/SuGaR         ✓ Advanced UI
    ✗ Buggy         ✓ Fixed          ✓ Provenance    ✓ Faster       ✓ Gravity          ✓ Pipeline control
    Umeyama         Umeyama          tracking        class-aware    alignment          ✓ Decoupled backend
    ✗ Drifted       ✓ Unified        ✓ GPS source                                      ✓ Job cancel
    configs         config           labelling                                         ✓ Live GPS readout
```

---

> **Document maintained by:** Project ANTARA development team
> **Next review due:** Before SIH final submission
