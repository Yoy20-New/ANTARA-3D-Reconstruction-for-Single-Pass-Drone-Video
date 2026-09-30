"""
Project ANTARA — Automated Edge 3D Reconstruction (SIH PS-17, NTRO).

Single shared core behind both the CLI (run_pipeline.py) and the Flask upload
server (antara_server.py). This package replaces the four near-duplicate scripts
(01_preprocess.py, 02_track_and_scale.py, run_pipeline_cloud.py, antara_master.py)
that previously drifted out of sync.

The reconstruction is REAL: Stage 2 invokes SLAM3R on the actual video frames.
There is no synthetic fallback — if the real path cannot run, the pipeline raises.
"""

__version__ = "1.0.0"
__team__ = "HNM-OG"
