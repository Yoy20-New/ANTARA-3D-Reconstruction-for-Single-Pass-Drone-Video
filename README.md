# ANTARA: 3D Reconstruction for Single-Pass Drone Video 🚁

ANTARA is a high-speed, edge-deployable AI pipeline that converts single-pass monocular drone videos into metric-accurate 3D point clouds, solid meshes (OBJ), and Georeferenced Orthomosaics (GeoTIFF) **without the need for Ground Control Points (GCPs)**. 

Built for the **Smart India Hackathon 2025**, this pipeline targets sub-meter accuracy and processes 10-minute 4K video feeds in under 15 minutes using true multiprocessing and Vision Transformers.

---

## 🛠️ Step-by-Step Installation Guide (Windows / Linux)

Because this pipeline integrates complex C++ compilations (3D Gaussian Splatting) and heavy AI models, **you must follow these steps precisely**. 

### Phase 1: System-Level Software (Install these FIRST)

You cannot just use pip install for this project. Your operating system needs the correct compilers and rendering engines installed first.

#### 1. NVIDIA CUDA Toolkit (Required for 3DGS & PyTorch)
*   **Official Link:** [CUDA Toolkit 11.8 Archive](https://developer.nvidia.com/cuda-11-8-0-download-archive)
*   **What to do:** Download and install the version for your OS.
*   **⚠️ Crucial ANTARA Warning:** The official NVIDIA site will try to push CUDA 12.x on you. **Do NOT install CUDA 12.x.** The 3D Gaussian Splatting rasterizer is highly unstable on Windows with CUDA 12. You *must* install version 11.8 to compile the C++ extensions error-free.

#### 2. Visual Studio 2022 C++ Build Tools (Windows Only)
*   **Official Link:** [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
*   **What to do:** Install it and select the **"Desktop development with C++"** workload. 
*   **⚠️ Crucial ANTARA Warning:** The 
vcc CUDA compiler needs this to build the SuGaR and 3DGS meshes.

#### 3. COLMAP (Required for Camera Initialization)
*   **Official Link:** [COLMAP GitHub Releases](https://github.com/colmap/colmap/releases)
*   **What to do:** 
    *   *Windows:* Download COLMAP-x.x-windows-cuda.zip. Extract it to C:\COLMAP. You **must** add this folder to your Windows System PATH environment variable.
    *   *Linux:* Run sudo apt-get install colmap
*   **⚠️ Crucial ANTARA Warning:** Our pipeline uses SLAM3R for the point cloud, but the 3DGS engine *strictly* requires COLMAP's cameras.bin files to start training. 

#### 4. FFmpeg (Required for Video Extraction)
*   **Official Link:** [FFmpeg Download](https://ffmpeg.org/download.html) (For Windows, use [Gyan.dev](https://www.gyan.dev/ffmpeg/builds/))
*   **What to do:** Extract the ZIP and add the in/ folder to your System PATH. Verify by typing fmpeg -version in your terminal.

#### 5. Miniconda (Required Python Manager)
*   **Official Link:** [Miniconda3 Installers](https://docs.conda.io/en/latest/miniconda.html)
*   **What to do:** Install the 64-bit version. We strongly advise against using standard Python/pip, as it routinely corrupts PyTorch CUDA environments.

---

### Phase 2: Project Setup

Once the system prerequisites are installed, open your **Anaconda Prompt** (Windows) or Terminal (Linux) and run:

**1. Clone the Repository**
Because this project links to external AI repositories (SAM2, SLAM3R, 3DGS), you **must** use the --recursive flag:
\\\ash
git clone --recursive https://github.com/Yoy20-New/ANTARA-3D-Reconstruction-for-Single-Pass-Drone-Video.git
cd ANTARA-3D-Reconstruction-for-Single-Pass-Drone-Video
\\\

**2. Create the Conda Environment**
\\\ash
conda create -n antara python=3.10 -y
conda activate antara
\\\

**3. Install PyTorch 2.x (CUDA 11.8)**
*Do not install standard PyTorch.* Force the CUDA 11.8 version:
\\\ash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
\\\

**4. Install Project Requirements**
\\\ash
pip install -r backend/requirements.txt
\\\

**5. Download AI Weights & Apply Patches**
We have written automated scripts to fetch the heavy AI models (like YOLO and SAM2) and apply our custom TensorRT modifications to the submodules.
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
*Built for SIH 2025. This project utilizes code from SLAM3R, Meta Segment Anything 2, and Inria Gaussian Splatting.*
