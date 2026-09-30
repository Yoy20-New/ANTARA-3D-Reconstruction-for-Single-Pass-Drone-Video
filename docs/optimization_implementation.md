# Implementation Plan: Pipeline Optimizations (Phase 2)

This document serves as the technical blueprint for implementing the major speed optimizations into the ANTARA codebase. These upgrades shift the prototype from a "brute-force" pipeline to an intelligent, production-ready system capable of hitting the <15 minute processing requirement.

---

## 1. YOLOv11 Dynamic Object Masking (Replacing SAM 2)
**Goal:** Reduce masking time by 90% by switching from pixel-perfect segmentation (SAM 2) to bounding-box segmentation (YOLO).
**Target File:** `antara/preprocess.py`

**Implementation Steps:**
1. **Install Ultralytics:** `pip install ultralytics`
2. **Load YOLO-nano:** In `preprocess.py`, load the lightweight model:
   ```python
   from ultralytics import YOLO
   model = YOLO('yolov11n.pt') # extremely fast
   ```
3. **Generate Mask:** Instead of feeding the image to SAM 2, run YOLO inference.
4. **Draw Bounding Boxes:** Loop through detected objects (filtering for classes like `person`, `car`, `truck`). Use OpenCV to draw solid black rectangles over those coordinates on a white mask.
5. **Feed to SLAM3R:** Pass this new binary mask into the SLAM3R pipeline. Since SLAM3R only looks for keypoints, a black box effectively hides the dynamic objects 10x faster than SAM 2's precise outlines.

---

## 2. IMU-Driven Adaptive Keyframing
**Goal:** Stop extracting redundant video frames when the drone is hovering or rotating slowly, cutting total frames processed by up to 50%.
**Target File:** `antara/preprocess.py` (specifically `extract_frames()`)

**Implementation Steps:**
1. **Reverse the Order:** Currently, the pipeline extracts video frames first, then parses GPS telemetry. We must reverse this. Parse the DJI `.srt` or `.json` telemetry *first*.
2. **Calculate Distance Deltas:** Iterate through the telemetry arrays. Calculate the 3D distance between $GPS_{t}$ and $GPS_{t-1}$.
3. **Threshold Logic:** Only mark a timestamp for extraction if the drone has moved $> 2.0$ meters or the Yaw has changed $> 5$ degrees.
4. **Selective FFmpeg:** Pass the specific list of approved timestamps to FFmpeg (using the `-ss` seek flag in a loop, or by generating a custom FFmpeg `select` filter string) instead of a blind `fps=5` filter.

---

## 3. TensorRT Compilation for AI Inference
**Goal:** Double the speed of the SLAM3R AI by compiling it specifically for NVIDIA hardware.
**Target Files:** `SLAM3R/models/` and `antara/reconstruct.py`

**Implementation Steps:**
1. **Export to ONNX:** Write a temporary script to load the PyTorch `.pth` weights for the Dust3R/SLAM3R transformer models. Export them using `torch.onnx.export()`.
2. **Compile Engine:** Use NVIDIA's command-line tool `trtexec` to compile the ONNX files into TensorRT `.engine` files (this bakes memory optimizations directly into the binary).
3. **Replace Inference Code:** In the SLAM3R inference loop, replace `model(images)` with the TensorRT Python bindings (`tensorrt` library), which requires passing CUDA memory pointers directly to the engine.

---

## 4. Microservices (Celery + Redis Worker Queue)
**Goal:** Completely decouple the FastAPI web server from the heavy GPU pipeline so the web interface never lags or crashes during processing.
**Target Files:** `backend/job_manager.py` and `backend/server.py`

**Implementation Steps:**
1. **Install Stack:** Install Redis (via Docker or WSL) and `pip install celery`.
2. **Setup Celery App:** Create `backend/celery_app.py` and configure it to point to `redis://localhost:6379`.
3. **Migrate Threading:** Open `job_manager.py`. Remove the `threading.Thread` logic. Wrap the `_run_job` function with `@celery.task`.
4. **Launch Workers:** Run the FastAPI server in one terminal, and start the Celery worker in a separate terminal: `celery -A backend.celery_app worker --loglevel=info`.
5. **State Management:** Switch the `JobState` dictionary to read/write from Redis instead of a local Python memory lock, allowing the UI to poll the database directly.
