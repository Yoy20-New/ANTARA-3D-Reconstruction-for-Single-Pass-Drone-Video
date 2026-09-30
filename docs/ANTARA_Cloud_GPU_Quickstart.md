# Project ANTARA: Cloud GPU Quickstart Guide
*Zero-Friction Prototype Deployment on Cloud GPUs (Lightning AI / RunPod)*

This guide allows Team HNM-OG to spin up a **16GB – 24GB Cloud GPU**, run the full end-to-end Project ANTARA reconstruction pipeline, and host the live **Potree 3D Web Viewer** in under 15 minutes.

---

## 1. Launching Your Cloud Instance

### Recommended: Lightning AI Studio (Free Tier)
1. Go to **[lightning.ai](https://lightning.ai/)** and log in.
2. Click **"New Studio"**.
3. In the top-right hardware selector, switch from **CPU** to **T4 GPU (16 GB)** or **A10G**.
4. Open the terminal at the bottom of the screen.

*(Alternative: On **RunPod**, choose a `PyTorch 2.1 / CUDA 12.1` template on an RTX 3090/4090 and open the Web Terminal).*

---

## 2. One-Click Automated Setup

Upload your project files (`cloud_setup.sh`, `run_pipeline.py`, the `antara/` package, and your drone video) into the cloud workspace, or clone your repository.

In the cloud terminal, run the setup script:

```bash
chmod +x cloud_setup.sh
./cloud_setup.sh
```

### What this script automatically installs:
* PyTorch with native CUDA 12.1 acceleration.
* Meta's official **SAM 2** (Apache 2.0) and downloads the `sam2.1_hiera_tiny.pt` edge-checkpoint.
* **SuGaR** (Surface-Aligned Gaussian Splatting) with all compiled CUDA submodules (`diff-gaussian-rasterization`, `simple-knn`).
* Geospatial libraries: `rasterio`, `laspy`, `open3d`, `trimesh`.
* Linux binary for **PotreeConverter 2.1**.

---

## 3. Running the End-to-End Pipeline

Run the master pipeline with a single command:

```bash
python run_pipeline.py \
    --video test_drone_flight.mp4 \
    --gps test_gps_telemetry.json
```

### What Happens Automatically:
1. **Stage 1 (Preprocessing):** Extracts keyframes at 5 FPS, drops motion-blurred frames using Laplacian scoring, and computes SAM 2 dynamic object masks (stored as metadata; raw pixels are never altered).
2. **Stage 2 (SLAM3R Reconstruction):** Invokes the real SLAM3R `recon.py` to regress dense world pointmaps from the video pixels, and recovers a per-keyframe camera-center proxy from each frame's registered pointmap. Hard-fails if CUDA/weights are unavailable — no synthetic fallback.
3. **Stage 3 (Georeferencing):** Removes points inside SAM 2 dynamic masks, computes a global flight-wide RANSAC-Umeyama transform, and smooths residual drift with Thin-Plate Spline (TPS) deformation.
4. **Stage 4 (Deliverables):** Exports PLY, LAS point cloud, OBJ mesh (Open3D screened Poisson), and an orthomosaic top-down GeoTIFF (`antara_orthomosaic.tif`) mapped to real-world GPS coordinates.
5. **Stage 5 (Honest Evaluation):** Reports the alignment residual (fit diagnostic) and hold-out ATE (the real accuracy signal) to `accuracy_report.json`.

> **SuGaR textured meshing** (surface-aligned 3D Gaussians) is an approved *later* phase, not part of this pipeline. The seam is documented in `antara/deliverables.py`. **Potree** conversion is likewise not wired in; the deliverable viewer is the Three.js `output/index.html`.

---

## 4. Viewing the 3D Model Live

### In Lightning AI:
1. Click the **"Ports"** icon on the right-hand sidebar.
2. Enter Port **`8000`** and click open.
3. Lightning AI will generate a secure, public HTTPS link that you can open on any phone, laptop, or projector during your hackathon presentation!

### In RunPod:
1. In your Pod dashboard, click **"Connect"** $\rightarrow$ **"HTTP Port 8000"**.
2. It will open your live, interactive 3D model in your browser.

---

## Output Deliverables Directory (`output/`)

| File | Format | Deliverable Met |
| :--- | :--- | :--- |
| `antara_model.obj` | 3D Mesh | Textured 3D Mesh for CAD / 3D software |
| `antara_model.las` | LAS | Standard GIS Point Cloud format |
| `antara_orthomosaic.tif` | GeoTIFF | 2D Georeferenced Orthomosaic for QGIS / ArcGIS |
| `web_viewer/` | WebGL / Potree | Interactive browser viewer for live presentation |
