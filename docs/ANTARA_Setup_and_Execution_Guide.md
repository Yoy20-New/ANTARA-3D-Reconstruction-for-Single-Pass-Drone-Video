# Project ANTARA: Setup & Execution Guide

> **Superseded.** This guide described an obsolete and partly incorrect
> architecture — three separate conda envs, a g2o pose-graph SLAM formulation fed
> by ORB/SuperPoint features and explicit camera poses, and an in-scope SuGaR
> meshing stage. **SLAM3R does none of that**: it is a DUSt3R-lineage network that
> regresses dense *pointmaps* and exposes no explicit camera poses, so there are
> no features to mask at the solver level and no `poses.txt` to feed g2o. The real
> pipeline recovers a per-keyframe camera-center *proxy* from each frame's
> registered pointmap instead.
>
> The project now uses a **single `antara` conda env** and one shared `antara/`
> package driven by `run_pipeline.py` (CLI) or `antara_server.py` (web).

## Where to go instead

- **Setup + run (canonical):** [setup_windows.md](setup_windows.md)
- **Architecture overview + licensing table:** [ANTARA_Windows11_Prototype_Guide.md](ANTARA_Windows11_Prototype_Guide.md)
- **Cloud GPU quickstart:** [ANTARA_Cloud_GPU_Quickstart.md](ANTARA_Cloud_GPU_Quickstart.md)

Quick version:

```bash
conda env create -f environment.yml
conda activate antara
# clone SLAM3R (github.com/PKU-VCL-3DV/SLAM3R) + facebookresearch/sam2, fetch the
# SAM 2 tiny checkpoint — see setup_windows.md for the exact commands
python run_pipeline.py --video test_drone_flight.mp4 --gps test_gps_telemetry.json
```

Deliverables (PLY, LAS, OBJ, GeoTIFF) land in `output/`; open the Three.js viewer
at `output/index.html`. The pipeline hard-fails rather than emitting fabricated
geometry if CUDA, the SLAM3R weights, or the SAM 2 checkpoint are missing.
