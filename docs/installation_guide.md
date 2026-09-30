# ANTARA Prototype: Full Installation & Setup Guide

This guide provides step-by-step instructions to completely rebuild the ANTARA 3D Reconstruction prototype from scratch. It includes both Windows and Linux instructions, ensuring anyone can deploy the system quickly.

---

## 1. Prerequisites (System Hardware)
* **OS:** Windows 10/11 or Ubuntu 20.04/22.04
* **GPU:** NVIDIA GPU with at least 6GB VRAM (RTX 3060 or better recommended).
* **Base Software:** [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda installed.

---

## 2. System Binaries (FFmpeg & COLMAP)
The pipeline relies on system-level binaries for video extraction and dense 3D meshing.

### On Ubuntu/Linux:
```bash
sudo apt update
sudo apt install ffmpeg colmap -y
```

### On Windows:
1. **FFmpeg:** Download the Windows build from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) and extract it.
2. **COLMAP:** Download the Windows standalone binary from the [COLMAP GitHub Releases](https://github.com/colmap/colmap/releases).
3. **Environment Variables:** Add the `bin` folders of both FFmpeg and COLMAP to your Windows `PATH` environment variable so they can be called from the terminal.

---

## 3. Python Environment & Core AI
Open your terminal (Anaconda Prompt on Windows) and run the following commands to isolate the project dependencies.

```bash
# 1. Create and activate a fresh environment
conda create -n antara python=3.10 -y
conda activate antara

# 2. Install PyTorch with CUDA (Hardware Acceleration)
# Note: Adjust cu118/cu121 based on your installed CUDA version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## 4. Pipeline & Backend Dependencies
Install the exact libraries required for the mathematical pipeline and the FastAPI web server.

```bash
# 1. Math, Geometry, and Computer Vision
pip install numpy scipy opencv-python open3d pyproj

# 2. Web Backend (FastAPI, Server, and Upload handling)
pip install fastapi uvicorn aiofiles python-multipart
```

---

## 5. Directory Structure & Checkpoints
Ensure your project folder contains the required AI model weights and structure.
Your root folder (`ANTARA/`) must look like this:

```text
ANTARA/
├── backend/                  # FastAPI server files
│   ├── server.py
│   ├── job_manager.py
│   └── upload_page.html
├── antara/                   # Core Python pipeline
│   ├── pipeline.py
│   ├── config.py
│   └── ...
├── output/                   # Static 3D viewer & generated meshes
│   ├── index.html            # WebGL Viewer
├── checkpoints/              # AI Weights
│   └── sam2.1_hiera_tiny.pt  # Segment Anything 2 weights
└── uploads/                  # Temporary video upload storage
```

---

## 6. Starting the System
Once everything is installed and your folders are in place, you can start the Automated Edge Console.

```bash
# 1. Navigate to the project root
cd /path/to/ANTARA

# 2. Activate the environment
conda activate antara

# 3. Start the Backend Server on port 8765
python -m backend.server --port 8765
```

**Access the UI:** 
Open your web browser and go to `http://127.0.0.1:8765/`. 
You can now upload a video, configure advanced settings, and generate a 3D model.
