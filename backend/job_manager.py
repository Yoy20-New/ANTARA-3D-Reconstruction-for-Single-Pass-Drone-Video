"""
ANTARA — Job Manager
Handles single-job-at-a-time state, background thread runner, and progress callbacks.
"""
from __future__ import annotations

import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from antara.config import PipelineConfig, REPO_ROOT
from antara.pipeline import run as run_pipeline

# ── Directories ──────────────────────────────────────────────────────────────
UPLOAD_DIR = os.path.join(REPO_ROOT, "uploads")
OUTPUT_DIR = os.path.join(REPO_ROOT, "output")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_VIDEO_BYTES    = 10 * 1024 ** 3   # 10 GB
MAX_TELEMETRY_BYTES = 50 * 1024 ** 2  # 50 MB

# ── Job State ─────────────────────────────────────────────────────────────────
@dataclass
class JobState:
    status: str = "idle"          # idle | uploading | processing | completed | error
    progress: int = 0
    stage_name: str = "Ready"
    stage_detail: str = "Waiting for upload…"
    job_id: str = ""
    metrics: Dict[str, Any] = field(default_factory=lambda: {
        "points_count": 0,
        "keyframes": 0,
        "rmse": "—",
        "elapsed": "—",
        "gps_source": "none",
        "accuracy_verified": False,
        "accuracy_passed": False,
    })
    error_message: str = ""

    def to_dict(self) -> dict:
        return {
            "status":        self.status,
            "progress":      self.progress,
            "stage_name":    self.stage_name,
            "stage_detail":  self.stage_detail,
            "job_id":        self.job_id,
            "metrics":       self.metrics,
            "error_message": self.error_message,
        }


class JobCancelledError(Exception):
    pass

# Module-level singleton
_state = JobState()
_lock  = threading.Lock()


def get_state() -> dict:
    return _state.to_dict()


def cancel_job() -> bool:
    """Request the current job to cancel."""
    with _lock:
        if _state.status in ("uploading", "processing"):
            _state.status = "canceling"
            return True
        return False


def is_busy() -> bool:
    return _state.status in ("uploading", "processing", "canceling")


def reset():
    """Reset to idle — only if not currently processing."""
    with _lock:
        if _state.status in ("processing", "canceling"):
            return False
        _state.__init__()
        return True


# ── Progress callback passed into the pipeline ────────────────────────────────
def _progress(pct: int, detail: str):
    if _state.status == "canceling":
        raise JobCancelledError("Job was cancelled by the user.")
    
    _state.progress = int(pct)
    _state.stage_detail = detail

    if pct < 10:
        _state.stage_name = "Initialising"
    elif pct < 27:
        _state.stage_name = "Stage 1 — Preprocessing & dynamic masking"
    elif pct < 66:
        _state.stage_name = "Stage 2 — SLAM3R 3D reconstruction"
    elif pct < 80:
        _state.stage_name = "Stage 3 — Georeferencing (GPS alignment)"
    elif pct < 94:
        _state.stage_name = "Stage 4 — Exporting deliverables"
    elif pct < 100:
        _state.stage_name = "Stage 5 — Accuracy evaluation"
    else:
        _state.stage_name = "Reconstruction complete"


# ── Background runner ─────────────────────────────────────────────────────────
def _run_job(video_path: str, gps_path: Optional[str], cfg: PipelineConfig):
    start = time.time()
    
    with _lock:
        if _state.status == "canceling":
            is_cancelled = True
        else:
            is_cancelled = False
            _state.status       = "processing"
            _state.progress     = 2
            _state.stage_name   = "Initialising"
            _state.stage_detail = "Starting pipeline…"
            _state.error_message = ""

    try:
        if is_cancelled:
            raise JobCancelledError("Job was cancelled by the user.")

        result = run_pipeline(video_path, gps_path, cfg=cfg, progress=_progress)
        elapsed = time.time() - start

        ho   = result.get("accuracy", {}).get("holdout_ate", {})
        rmse = f"{ho['rmse_m']:.2f} m" if "rmse_m" in ho else "n/a"

        _state.status      = "completed"
        _state.progress    = 100
        _state.stage_name  = "Reconstruction complete"
        note = " (accuracy unverified — no GPS)" if not result.get("gps_verified") else ""
        _state.stage_detail = (
            f"Done in {elapsed:.0f}s — "
            f"{result.get('num_points', 0):,} points, "
            f"{result.get('num_keyframes', 0)} keyframes.{note}"
        )
        _state.metrics = {
            "points_count":       result.get("num_points", 0),
            "keyframes":          result.get("num_keyframes", 0),
            "rmse":               rmse,
            "elapsed":            f"{elapsed:.0f}s",
            "gps_source":         result.get("gps_source", "none"),
            "accuracy_verified":  result.get("gps_verified", False),
            "accuracy_passed":    bool(ho.get("passed", False)) if result.get("gps_verified") else False,
        }

    except JobCancelledError:
        _state.status       = "error"
        _state.stage_name   = "Cancelled"
        _state.stage_detail = "Job was cancelled by the user."
        _state.error_message = "Reconstruction aborted."
        print(f"[{_state.job_id}] Cancelled by user.")

    except Exception as err:
        _state.status       = "error"
        _state.stage_name   = "Reconstruction failed"
        _state.stage_detail = str(err)
        _state.error_message = str(err)
        print(f"[ANTARA Backend] Pipeline error: {err}")


def launch_job(
    video_path: str,
    gps_path: Optional[str],
    allow_synthetic_gps: bool = False,
    use_sam: bool = True,
    enable_gs: bool = True,
    target_fps: int = 5,
    keyframe_stride: int = 4,
    num_points_save: int = 500_000
) -> str:
    """Spawn a background thread for the pipeline. Returns job_id."""
    job_id = uuid.uuid4().hex[:12]
    _state.job_id  = job_id
    _state.status  = "uploading"
    _state.metrics = JobState().metrics  # reset metrics

    cfg = PipelineConfig(output_dir=OUTPUT_DIR)
    cfg.allow_synthetic_gps = allow_synthetic_gps
    cfg.use_sam = use_sam
    cfg.enable_gaussian_splatting = enable_gs
    cfg.target_fps = target_fps
    cfg.keyframe_stride = keyframe_stride
    cfg.num_points_save = num_points_save

    t = threading.Thread(
        target=_run_job,
        args=(video_path, gps_path, cfg),
        daemon=True,
    )
    t.start()
    return job_id


def cleanup_old_uploads(max_age_hours: int = 24):
    """Remove uploads older than max_age_hours."""
    cutoff = time.time() - (max_age_hours * 3600)
    for fname in os.listdir(UPLOAD_DIR):
        fpath = os.path.join(UPLOAD_DIR, fname)
        try:
            if os.path.isfile(fpath) and os.path.getmtime(fpath) < cutoff:
                os.remove(fpath)
                print(f"[Cleanup] Removed: {fpath}")
        except OSError:
            pass
