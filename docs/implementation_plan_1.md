# 3D Gaussian Splatting + SuGaR Integration — ANTARA Pipeline

## Background

The ANTARA pipeline currently produces a **dense colored point cloud** (200K points via SLAM3R) and an **untextured Poisson mesh** (via Open3D). The goal of this phase is to replace the Poisson mesh step with a **photorealistic textured mesh** via the 3DGS → SuGaR pathway.

The existing codebase already has a documented seam for this in [`antara/deliverables.py#L98`](file:///e:/SIH/antara/deliverables.py#L95-L113) and exports `transforms.json` specifically to enable this later phase.

---

## Critical Architecture Problem to Solve First

> [!IMPORTANT]
> **SLAM3R does not output real camera pose matrices.** The current pipeline uses a **proxy** (confidence-weighted centroid of each frame's back-projected pointmap) as the camera "position". This gives a position-only proxy, not a full 6-DOF pose (rotation matrix + translation vector).
>
> 3DGS training requires proper **camera extrinsics (R|t)** per frame, not just positions.
>
> **Solution**: Run **COLMAP SfM** on the already-extracted keyframes to get real camera intrinsics and extrinsics. Use SLAM3R's dense point cloud as the 3DGS **initialization** (coloured SfM point cloud substitute). This is the standard photogrammetry → 3DGS pipeline used by the gaussian-splatting reference implementation.

---

## Pipeline After Integration

```
[Drone Video]
    │
    ▼
Stage 1: YOLOE + SAM 2 (Dynamic Masking)                      [EXISTING]
    │
    ▼
Stage 2: SLAM3R (Dense World Point Cloud)                      [EXISTING]
    │
    ├──► Stage 2.5: COLMAP SfM on keyframes                   [NEW]
    │       Output: cameras.bin / images.bin / points3D.bin
    │       SLAM3R cloud is merged as initialization
    │
    ▼
Stage 3: GPS Georeferencing (Umeyama + TPS)                    [EXISTING]
    │
    ▼
Stage 3.5: 3D Gaussian Splatting Training                     [NEW]
    │   Input: COLMAP poses + SLAM3R cloud + keyframe images
    │   Output: output/gaussian_splat/ (.ply Gaussian scene)
    │
    ▼
Stage 3.6: SuGaR Mesh Extraction                              [NEW]
    │   Input: Trained 3DGS scene + training images
    │   Output: output/sugar_mesh/ (.obj + .mtl + texture atlas)
    │
    ▼
Stage 4: Deliverables (PLY, LAS, GeoTIFF, 3DGS, SuGaR Mesh)   [MODIFIED]
    │
    ▼
Stage 5: Honest Accuracy Evaluation                            [EXISTING]
```

---

## Open Questions

> [!IMPORTANT]
> **Q1 — Storage**: gaussian-splatting CUDA extension build takes ~2 GB. The full 3DGS checkpoint for a typical drone scene is ~200–500 MB. SuGaR adds another ~100–300 MB. Do you have ~1.5 GB free?

> [!IMPORTANT]
> **Q2 — Training Time**: On an RTX 3050 6GB, 3DGS training at 30K iterations (default) takes ~30–45 minutes per scene. SuGaR post-processing takes another ~10–20 minutes. Is this acceptable for the pipeline, or should we add `--iter 7000` fast mode?

> [!WARNING]
> **Q3 — COLMAP availability**: COLMAP needs to be installed (`conda install -c conda-forge colmap`). Do you have it, or does it need to be installed first?

---

## Proposed Changes

---

### Stage 2.5 — COLMAP SfM Camera Pose Estimation

#### [NEW] [`antara/colmap_sfm.py`](file:///e:/SIH/antara/colmap_sfm.py)

New module that:
1. Calls COLMAP `feature_extractor` → `exhaustive_matcher` → `mapper` on the masked keyframes
2. Optionally seeds COLMAP's sparse model with SLAM3R camera centers to guide bundle adjustment
3. Reads back `cameras.txt` / `images.txt` / `points3D.txt` (text format)
4. Exports a standard `colmap_sparse/` directory consumable directly by the gaussian-splatting trainer
5. Falls back gracefully: if COLMAP is not installed, skips to 3DGS with position-only proxy and logs a warning

---

### Stage 3.5 — 3D Gaussian Splatting Training

#### [NEW] [`antara/gaussians.py`](file:///e:/SIH/antara/gaussians.py)

New module that:
1. Clones / checks for the `gaussian-splatting` repo in `REPO_ROOT/gaussian-splatting/`
2. Builds CUDA diff-rasterization extension on first run (one-time ~5 min build)
3. Runs `train.py` as a subprocess with:
   - `--source_path` pointing to the COLMAP sparse model + keyframe images
   - `--model_path` → `output/gaussian_splat/<run_id>/`
   - `--iterations 7000` (fast mode for 6GB GPU) or `30000` (quality)
   - Streams progress back to pipeline's `progress()` callback
4. Returns path to the trained `.ply` Gaussian scene file
5. The 3DGS `.ply` (Gaussian format) is kept separate from the SLAM3R point cloud `.ply`

---

### Stage 3.6 — SuGaR Textured Mesh Extraction

#### [NEW] [`antara/sugar.py`](file:///e:/SIH/antara/sugar.py)

New module that:
1. Clones / checks for `SuGaR` repo in `REPO_ROOT/SuGaR/`
2. Runs SuGaR's `train.py` as subprocess with:
   - `--scene_path` → COLMAP sparse directory
   - `--checkpoint_path` → trained 3DGS checkpoint
   - `--output_dir` → `output/sugar_mesh/<run_id>/`
   - `--mesh_output_dir` → `output/`
3. Renames the final textured mesh to `output/antara_textured_mesh.obj` + `.mtl` + texture images
4. Returns paths; if SuGaR fails, existing Poisson mesh is kept as fallback

---

### Modified Existing Files

#### [MODIFY] [`antara/config.py`](file:///e:/SIH/antara/config.py)

Add new config fields:
```python
# gaussian-splatting repo dir
GS_DIR = os.path.join(REPO_ROOT, "gaussian-splatting")
SUGAR_DIR = os.path.join(REPO_ROOT, "SuGaR")

# In PipelineConfig:
enable_gaussian_splatting: bool = False  # opt-in flag
gs_iterations: int = 7000               # 7K = fast, 30K = quality
enable_sugar: bool = False               # opt-in; requires enable_gaussian_splatting
```

#### [MODIFY] [`antara/pipeline.py`](file:///e:/SIH/antara/pipeline.py)

Add Stage 2.5 (COLMAP SfM) and Stages 3.5–3.6 (3DGS + SuGaR) after georeferencing, guarded by `cfg.enable_gaussian_splatting`. The existing Poisson path continues to run as baseline. The manifest and deliverables dict gains `gaussian_splat` and `sugar_mesh` keys.

#### [MODIFY] [`antara/deliverables.py`](file:///e:/SIH/antara/deliverables.py)

Fill in the documented SuGaR seam at `export_mesh()` — branch on a `sugar_ply_path` argument; if provided, copy the SuGaR output instead of running Open3D Poisson.

#### [MODIFY] [`run_pipeline.py`](file:///e:/SIH/run_pipeline.py)

Add CLI flags:
- `--gaussian-splatting` / `--gs`
- `--sugar`
- `--gs-iterations N` (default 7000)

#### [MODIFY] [`output/index.html`](file:///e:/SIH/output/index.html)

Add a `Splat` tab or button in the bottom bar that loads the SuGaR textured OBJ mesh using Three.js `OBJLoader` + `MTLLoader`, toggleable alongside the existing point cloud view.

---

## Installation Plan (one-time setup)

```powershell
# 1. COLMAP (for real camera poses)
conda install -c conda-forge colmap

# 2. gaussian-splatting (Inria reference implementation)
cd E:\SIH
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
cd gaussian-splatting
pip install plyfile tqdm simple-knn
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn

# 3. SuGaR
cd E:\SIH
git clone https://github.com/Anttwo/SuGaR
cd SuGaR
pip install -r requirements.txt
```

---

## Verification Plan

### Automated Tests
- `python run_pipeline.py --video videos/drone_footage.mp4 --allow-synthetic-gps --gaussian-splatting --gs-iterations 1000` — smoke test with 1K iterations (2 min, proves the wiring works)
- Check `output/gaussian_splat/` exists with a `.ply` file > 1 MB
- Check `output/antara_textured_mesh.obj` exists

### Manual Verification
- Open `output/antara_textured_mesh.obj` in MeshLab or Blender — should show a photorealistic textured 3D model of the scene
- Compare against the untextured Poisson `antara_model.obj`
- Inspect the 3D web console — switch between point cloud and textured mesh view

---

## VRAM Budget (RTX 3050 6GB)

| Stage | VRAM | Notes |
|---|---|---|
| COLMAP feature extraction | ~1 GB | CPU-heavy, GPU only for SIFT |
| 3DGS training (7K iter) | ~4–5 GB | Tight but within 6GB budget |
| SuGaR mesh extraction | ~3–4 GB | Runs after 3DGS, not simultaneously |
| Viewer (Three.js OBJ) | ~200 MB | CPU/WebGL, separate from training |
