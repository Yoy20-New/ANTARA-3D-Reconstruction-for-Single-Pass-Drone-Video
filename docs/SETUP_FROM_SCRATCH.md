# ANTARA: Clean Windows Setup From Scratch

This guide rebuilds ANTARA manually on Windows. Follow the commands in order. Do not copy old virtual environments, old output folders, or old installed packages into the new setup.

## 1. Keep the Source and Video Files

Keep these before cleaning the workspace:

```text
antara/
run_pipeline.py
antara_server.py
environment.yml
PROJECT_HISTORY.md
SETUP_FROM_SCRATCH.md
*.mp4
```

The following will be installed or recreated later:

```text
sam2/
SLAM3R/
checkpoints/
output/
uploads/
```

## 2. Install System Prerequisites

Install these manually before opening PowerShell:

1. NVIDIA driver compatible with your GPU.
2. Miniconda or Anaconda for Windows.
3. Git for Windows.
4. Reliable Internet access for the repositories and model checkpoints.

Microsoft Visual C++ Build Tools and the full CUDA Toolkit are **not** needed
for the normal ANTARA pipeline. The optional SAM 2 native extension is
deliberately disabled; ANTARA uses OpenCV for the equivalent mask cleanup.
Only install those build tools later if you explicitly decide to maintain that
native extension.

Open **Anaconda Prompt** or PowerShell after Miniconda is available.

Check the basic tools:

```powershell
git --version
conda --version
nvidia-smi
```

## 3. Create and Normalize the Python Environment

From the project root:

```powershell
cd E:\SIH
conda env create -f environment.yml
conda activate antara
python -m pip install --upgrade pip
```

`environment.yml` pins the compatible CUDA 11.8 PyTorch trio. Reinstall that
same trio now and after installing the two external repositories; it prevents
an installer from silently replacing only one PyTorch package.

```powershell
python -m pip install --force-reinstall `
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 `
  --index-url https://download.pytorch.org/whl/cu118
```

Verify Python, the complete PyTorch trio, and CUDA:

```powershell
python -c "import torch, torchvision, torchaudio; print('Torch:', torch.__version__); print('Torchvision:', torchvision.__version__); print('Torchaudio:', torchaudio.__version__); print('CUDA:', torch.cuda.is_available()); print('Torch CUDA:', torch.version.cuda)"
```

The expected versions are `2.5.1+cu118`, `0.20.1+cu118`, and
`2.5.1+cu118`; `CUDA: True` is required for SLAM3R and SAM 2 GPU inference.
Stop here if CUDA is false. Do not continue with CPU mode: ANTARA intentionally
hard-fails instead of generating a substitute reconstruction.

## 4. Install SLAM3R

If the `SLAM3R` directory is absent, clone it:

```powershell
cd E:\SIH
git clone https://github.com/PKU-VCL-3DV/SLAM3R.git
```

Install the required SLAM3R packages, excluding `pycuda`:

```powershell
cd E:\SIH\SLAM3R
python -m pip install `
  roma gradio matplotlib tqdm opencv-python scipy einops trimesh `
  tensorboard "pyglet<2" "huggingface-hub[torch]>=0.22" viser
cd E:\SIH
```

`pycuda` appears in SLAM3R's upstream requirements file but is not used by the
ANTARA pipeline. On Windows it commonly attempts a local C++ build and fails.
Do not install it for this project.

Verify the repository is present:

```powershell
Test-Path E:\SIH\SLAM3R\recon.py
```

Download and cache the two SLAM3R inference models before the first real job.
This makes a later reconstruction fail at setup time rather than halfway
through a user upload:

```powershell
cd E:\SIH\SLAM3R
python -c "from slam3r.models import Image2PointsModel, Local2WorldModel; Image2PointsModel.from_pretrained('siyan824/slam3r_i2p'); Local2WorldModel.from_pretrained('siyan824/slam3r_l2w'); print('SLAM3R inference weights cached')"
cd E:\SIH
```

The weights are stored in Hugging Face's cache for the Windows user running the
command. Keep that cache when you need later offline inference. If this command
fails, confirm Internet access, run `python -m pip install --upgrade
"huggingface-hub[torch]"`, then repeat the same command.

## 5. Install SAM 2

If the `sam2` directory is absent, clone it:

```powershell
cd E:\SIH
git clone https://github.com/facebookresearch/sam2.git
```

Install SAM 2 in editable mode, deliberately skipping its optional native CUDA
extension:

```powershell
cd E:\SIH\sam2
$env:SAM2_BUILD_CUDA = "0"
python -m pip install -e .
Remove-Item Env:SAM2_BUILD_CUDA -ErrorAction SilentlyContinue
cd E:\SIH
```

Reapply the single compatible PyTorch trio after the repository installations,
then validate the dependency set:

```powershell
python -m pip install --force-reinstall `
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 `
  --index-url https://download.pytorch.org/whl/cu118
python -m pip check
```

Do not attempt to build `sam2._C` unless you deliberately choose to maintain a
matching CUDA Toolkit, PyTorch CUDA build, and MSVC toolchain. ANTARA already
performs small-island removal and hole filling with OpenCV, so normal inference
does not lose post-processing when this extension is disabled.

## 6. Download the SAM 2 Tiny Checkpoint

Create the checkpoint directory:

```powershell
New-Item -ItemType Directory -Force E:\SIH\checkpoints
```

Download the model:

```powershell
Invoke-WebRequest `
  -Uri "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt" `
  -OutFile "E:\SIH\checkpoints\sam2.1_hiera_small.pt"
```

Verify it exists:

```powershell
Get-Item E:\SIH\checkpoints\sam2.1_hiera_tiny.pt
```

## 7. Verify the Complete Inference Stack

From `E:\SIH` with the `antara` environment active:

```powershell
python -c "import torch; import cv2; import laspy; import rasterio; print('Core dependencies OK')"
python -c "from antara.reconstruct import _check_cuda, _check_slam3r; _check_cuda(); print(_check_slam3r()); print('SLAM3R integration OK')"
python -c "from antara.preprocess import _load_sam2_generator; _load_sam2_generator(); print('SAM 2 checkpoint, configuration, and GPU inference are ready')"
```

The final command intentionally loads the SAM 2 checkpoint on the GPU. It is
the setup gate for future inference; if it succeeds, exit the command normally
and GPU memory is released with the Python process.

## 8. Run a Demo Reconstruction Without Real GPS

When real flight telemetry is unavailable, run explicit demo mode. This still uses real video pixels, SAM 2, and SLAM3R; only the GPS trajectory is synthetic.

```powershell
cd E:\SIH
conda activate antara

python run_pipeline.py `
  --video drone_footage.mp4 `
  --allow-synthetic-gps
```

Do not use `--verified-gps` in demo mode. The result is for visual testing and local coordinates only.

For a run that exercises the motion-difference debug mask instead of SAM 2:

```powershell
python run_pipeline.py `
  --video drone_footage.mp4 `
  --allow-synthetic-gps `
  --no-sam
```

`--no-sam` skips only SAM 2. It still performs the full GPU SLAM3R
reconstruction, so it is not a lightweight or CPU-only preflight.

## 9. Run With Real GPS Later

When you have real flight telemetry, use one authoritative synchronization field:

- `timestamp_ms` for time-based telemetry
- `frame_idx` for video-frame-index telemetry

The JSON file must be a non-empty array. Each record must have numeric
`latitude`, `longitude`, and `altitude_m`, either at the top level or inside a
`gps` object. For timestamp telemetry, a valid minimal structure is:

```json
[
  {"timestamp_ms": 0, "latitude": 12.971600, "longitude": 77.594600, "altitude_m": 920.4},
  {"timestamp_ms": 1000, "latitude": 12.971605, "longitude": 77.594615, "altitude_m": 920.8}
]
```

Capture records across the *entire* video, including its beginning and end.
The current default accepts at most a 2,000 ms gap between timestamp samples
or 60 frames between frame-index samples. Records with both synchronization
fields are allowed only when you explicitly choose which field is authoritative.

Example with timestamp telemetry:

```powershell
python run_pipeline.py `
  --video drone_footage.mp4 `
  --gps real_telemetry.json `
  --gps-sync-mode timestamp_ms `
  --verified-gps
```

The GPS file must cover the entire video. Only use `--verified-gps` when the
telemetry comes from the actual flight; it is an operator assertion, not a
method for making synthetic or unrelated coordinates trustworthy.

## 10. Use the CLI or Local Upload Portal for Future Inference

The CLI command in Steps 8–9 is the most direct way to reconstruct a local
video. The upload portal is the equivalent local web workflow:

```powershell
cd E:\SIH
conda activate antara
python antara_server.py --allow-synthetic-gps
```

Open `http://127.0.0.1:8000`, upload an MP4, and upload telemetry when it is
available. Omit `--allow-synthetic-gps` when the portal must reject video-only
uploads. The portal allows one reconstruction at a time and serves the current
model from `output/`; it is intentionally bound to localhost by default.

## 11. Open the Local Viewer

After the pipeline succeeds:

```powershell
cd E:\SIH
python -m http.server 8000 --directory output
```

Open this address in your browser:

```text
http://localhost:8000
```

Use `Ctrl+F5` after a new run to refresh cached browser assets.

## 12. Expected Outputs

```text
output/
  index.html
  colmap_format/georeferenced_cloud.ply
  colmap_format/transforms.json
  antara_model.las
  antara_model.las.json
  antara_model.obj
  antara_orthomosaic.tif
  accuracy_report.json
  run_manifest.json
```

For a demo run, `run_manifest.json` must show `"gps_source": "synthetic"` and the viewer must not be treated as a real-world georeferenced map.

## 13. Common Problems and Recovery

### `CUDA: False`

Install a compatible NVIDIA driver and a PyTorch CUDA build. SLAM3R and SAM 2 GPU inference will not run without CUDA.

### SAM 2 checkpoint missing

Repeat Step 6 and confirm this path exists:

```text
E:\SIH\checkpoints\sam2.1_hiera_tiny.pt
```

### SAM 2 says it cannot find `_C`, or post-processing is skipped

For this project, the native `_C` extension is intentionally disabled. Make
sure Step 5 used `SAM2_BUILD_CUDA = "0"`; do not compile the extension merely
to remove this notice. ANTARA's own OpenCV cleanup remains active. If `sam2`
itself cannot import, run these commands from `E:\SIH\sam2` with the `antara`
environment active, then repeat Steps 5 and 7:

```powershell
python -m pip uninstall -y SAM-2
$env:SAM2_BUILD_CUDA = "0"
python -m pip install -e .
Remove-Item Env:SAM2_BUILD_CUDA -ErrorAction SilentlyContinue
```

### `pycuda` fails to install or asks for Microsoft Visual C++

Do not fix or retry it. `pycuda` is an unused upstream SLAM3R dependency for
this ANTARA pipeline. Return to Step 4 and use the explicit package command
that excludes it.

### SLAM3R attempts a download or cannot find its models

Repeat the model-cache command in Step 4 while online. It downloads the
`siyan824/slam3r_i2p` and `siyan824/slam3r_l2w` models before any production
video is submitted.

### GPS coverage error

Use real telemetry that covers the full video and has no timestamp gap larger
than 2,000 ms (or frame-index gap larger than 60), or use
`--allow-synthetic-gps` for a clearly labelled demo run. If records contain
both `frame_idx` and `timestamp_ms`, add `--gps-sync-mode frame_idx` or
`--gps-sync-mode timestamp_ms`; do not rely on automatic selection.

### GPU out-of-memory error

Use a shorter video first, then reduce `num_points_save` or increase `keyframe_stride` in `antara/config.py`.

### A new run has dependency conflicts

Run `python -m pip check`. If it reports a PyTorch mismatch, repeat the exact
three-package PyTorch command in Step 5, then repeat the verification commands
in Step 7. Do not mix conda-installed PyTorch with a separately installed pip
PyTorch build in the same `antara` environment.
