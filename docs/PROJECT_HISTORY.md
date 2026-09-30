# ANTARA Project History and Current State

Last reviewed: 2026-09-12

## Purpose

ANTARA is a local drone-video reconstruction prototype. It uses SAM 2 to identify moving regions, SLAM3R to reconstruct a dense point cloud from video frames, and optional GPS telemetry to place the result in a local ENU metric frame.

## Current Working Pipeline

```text
Drone video
  -> keyframe extraction and blur filtering
  -> SAM 2 dynamic-object masks
  -> SLAM3R dense pointmap reconstruction
  -> remove dynamic points from the reconstructed cloud
  -> optional GPS alignment (RANSAC-Umeyama plus TPS)
  -> PLY, LAS, OBJ, GeoTIFF, transforms, report, and browser viewer
```

The source code for this pipeline is in `antara/`, with command-line entrypoint `run_pipeline.py` and optional local web server `antara_server.py`.

## Important Implementation Decisions

- Raw image pixels are not modified for masking. SAM 2 masks are stored separately and applied to reconstructed 3D points after SLAM3R inference.
- SLAM3R does not provide explicit camera poses. ANTARA uses confidence-weighted centroids of registered pointmaps as a camera-position proxy for GPS alignment.
- GPS is required for real georeferencing. Demo mode can use synthetic GPS only when explicitly enabled.
- Synthetic or unverified GPS must never be presented as real geographic accuracy.
- SAM 2 post-processing is implemented with OpenCV connected-components cleanup. This avoids depending on the optional `sam2._C` CUDA extension on Windows.
- The active browser viewer is `output/index.html`. The root `index.html` is older and should not be treated as the active viewer.

## Verified Status

- SLAM3R completed a real reconstruction and produced a 200,000-point cloud.
- SAM 2 model loading, one-frame inference, and short-video preprocessing succeeded.
- A complete demo run exists using synthetic GPS. Its output is useful for visualization only; accuracy claims are disabled.
- Current outputs are still organised as a single active run under `output/`. A multi-run history, per-run browser routing, and ZIP downloads have not yet been implemented.

## Known Limits

- No real GPS telemetry is currently available. Current georeferenced outputs therefore use synthetic/demo coordinates and are not valid for real-world mapping claims.
- SuGaR textured meshing is not implemented. The current OBJ comes from Open3D Poisson meshing.
- Symmetry-based facade completion is not implemented.
- The optional native SAM 2 CUDA extension is not built. The OpenCV fallback supplies the required hole and small-island cleanup.

## Files That Matter After the Cleanup

```text
antara/                 Core pipeline package
run_pipeline.py         Command-line entrypoint
antara_server.py        Optional local web upload server
environment.yml         Conda environment specification
SETUP_FROM_SCRATCH.md   Installation instructions
PROJECT_HISTORY.md      This history document
*.mp4                   User video input files
sam2/                   SAM 2 source repository after installation
SLAM3R/                 SLAM3R source repository after installation
checkpoints/            Downloaded SAM 2 model checkpoint
output/                 Generated model output; can be recreated
```

## Clean Setup Baseline

The clean-install procedure is maintained in `SETUP_FROM_SCRATCH.md`. It now
includes the complete inference readiness path: the CUDA 11.8 PyTorch 2.5.1
trio, Windows-safe SLAM3R dependencies without unused `pycuda`, pre-caching of
both SLAM3R models, SAM 2 installation with its optional native extension
disabled, SAM 2 checkpoint download, and GPU-backed verification commands.

The design is intentional: OpenCV supplies SAM 2 mask cleanup on Windows, so
the project does not require MSVC or a system CUDA Toolkit merely to run the
normal inference pipeline.

## Recommended Next Development Stage

Before adding SuGaR or facade completion, rebuild the project around run-isolated output:

```text
output/runs/<run_id>/
  manifest.json
  artifacts/
  viewer/
  download.zip
```

The viewer must label each model according to provenance, for example:

- `Demo Reconstruction — Synthetic GPS — Local coordinates only`
- `Verified Georeferenced Cloud — Flight telemetry supplied`
