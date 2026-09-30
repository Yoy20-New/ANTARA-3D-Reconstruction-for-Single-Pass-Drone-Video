# ANTARA: 3D Reconstruction for Single-Pass Drone Video 🚁

ANTARA is a high-speed, edge-deployable AI pipeline that converts single-pass monocular drone videos into metric-accurate 3D point clouds, solid meshes (OBJ), and Georeferenced Orthomosaics (GeoTIFF) **without the need for Ground Control Points (GCPs)**. 

Built for the **Smart India Hackathon 2025**, this pipeline targets sub-meter accuracy and processes 10-minute 4K video feeds in under 15 minutes using true multiprocessing and Vision Transformers.

## ✨ Key Features
- **No GCPs Required:** Relies on GPS telemetry (Lat/Lon/Alt) from DJI drones and robust Vision Transformers.
- **Adaptive Keyframing:** Parses GPS to calculate 3D Haversine distance, dropping redundant "hover" frames to save 80% compute time.
- **Dynamic Object Masking:** Automatically removes moving vehicles and pedestrians using YOLOE + Meta SAM 2 to prevent 3D ghosting.
- **Transformer Reconstruction:** Bypasses traditional COLMAP SfM mapping using **SLAM3R / DUSt3R** for instantaneous dense point matching.
- **Metric Georeferencing:** Rigid 7-DOF Procrustes alignment coupled with Thin Plate Spline (TPS) non-linear deformation to correct IMU drift.
- **Edge-Ready Web Console:** A decoupled FastAPI backend that keeps the 3D Viewer (Three.js) lag-free while CUDA/C++ processes heavy tasks.

---

## 🛠️ Installation & Setup (Windows / Linux)

Because this pipeline integrates complex 3D Gaussian Splatting and CUDA dependencies, we **highly recommend using Miniconda**.

### 1. OS-Level Prerequisites
- **FFmpeg**: Required for extracting frames from .mp4 video files.
  - *Windows:* Download from [Gyan.dev](https://www.gyan.dev/ffmpeg/builds/) and add to your system PATH.
  - *Linux:* sudo apt install ffmpeg
- **COLMAP (>= 3.8)**: Required *only* for the camera initialization step in Gaussian Splatting. 
  - Download from [GitHub Releases](https://github.com/colmap/colmap/releases) and add to PATH.

### 2. Clone the Repository
Because this project links to external AI repositories (like SAM2 and SLAM3R), you must clone recursively:
\\\ash
git clone --recursive https://github.com/Yoy20-New/ANTARA-3D-Reconstruction-for-Single-Pass-Drone-Video.git
cd ANTARA-3D-Reconstruction-for-Single-Pass-Drone-Video
\\\

### 3. Create the Conda Environment
\\\ash
conda create -n antara python=3.10 -y
conda activate antara
pip install -r backend/requirements.txt
\\\
*(Ensure you install the correct PyTorch 2.x version matching your CUDA toolkit, e.g., CUDA 11.8 or 12.1).*

### 4. Download AI Weights
We have provided a script to automatically download the heavy .pt files required for SAM2 and SLAM3R.
\\\ash
python scripts/download_weights.py
python scripts/apply_patches.py
\\\

---

## 🚀 Running the Pipeline

You can run the full pipeline in two ways:

**1. Headless CLI Mode:**
\\\ash
python run_pipeline.py --video videos/drone_footage.mp4 --gps videos/DJI_gps.SRT --output output/
\\\

**2. Interactive Web UI:**
Start the FastAPI background server:
\\\ash
python backend/server.py
\\\
Then navigate to http://localhost:8000 to upload videos and view live metrics in the 3D Geographic HUD.

---
*Built for SIH 2025. This project uses code from SLAM3R, Meta Segment Anything 2, and Inria Gaussian Splatting.*
