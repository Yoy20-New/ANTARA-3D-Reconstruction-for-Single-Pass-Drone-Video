"""
ANTARA — FastAPI Backend Server
Serves the upload page, handles video + GPS uploads, runs the pipeline,
and serves the 3D viewer + output files.

Usage:
    conda activate antara
    python backend/server.py --port 8765
"""
from __future__ import annotations

import argparse
import os
import sys
import uuid

import aiofiles
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# Allow importing from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import backend.job_manager as jm

# ── Paths ─────────────────────────────────────────────────────────────────────
BACKEND_DIR  = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR   = jm.OUTPUT_DIR
UPLOAD_DIR   = jm.UPLOAD_DIR
UPLOAD_PAGE  = os.path.join(BACKEND_DIR, "upload_page.html")

# Ensure directories exist so StaticFiles doesn't crash on startup
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="ANTARA Upload Portal", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8765", "http://localhost:8765"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve everything under /output/ statically (PLY, OBJ, GeoTIFF, etc.)
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def upload_page():
    """Serve the upload UI."""
    if not os.path.exists(UPLOAD_PAGE):
        raise HTTPException(status_code=404, detail="Upload page not found.")
    return FileResponse(UPLOAD_PAGE)


@app.get("/viewer")
async def viewer():
    """Redirect to the static viewer so relative PLY/OBJ paths resolve correctly."""
    viewer_path = os.path.join(OUTPUT_DIR, "index.html")
    if not os.path.exists(viewer_path):
        raise HTTPException(status_code=404, detail="3D viewer not found. Run a reconstruction first.")
    return RedirectResponse(url="/output/index.html", status_code=302)


@app.get("/api/status")
async def get_status():
    """Return current job state — polled by the UI every 2 seconds."""
    return JSONResponse(jm.get_state())


@app.post("/api/reset")
async def reset_job():
    """Reset idle/error/completed job back to idle."""
    ok = jm.reset()
    if not ok:
        raise HTTPException(status_code=409, detail="Cannot reset while a job is processing.")
    return {"status": "idle"}


@app.post("/api/cancel")
async def cancel_job():
    """Cancel the currently running job."""
    if jm.cancel_job():
        return {"message": "Job cancellation requested."}
    return {"message": "No job is currently running that can be cancelled."}


@app.post("/api/upload")
async def upload_and_process(
    video: UploadFile = File(..., description="Drone video (.mp4 / .mov / .avi)"),
    gps:   UploadFile = File(None, description="GPS/IMU telemetry (.json or .srt) — optional"),
    demo_mode: bool   = Form(False, description="Allow GPS-free run (accuracy unverified)"),
    use_sam: bool     = Form(True, description="Use SAM 2 for dynamic masking"),
    enable_gs: bool   = Form(True, description="Enable Gaussian Splatting"),
    target_fps: int   = Form(5, description="Frames to extract per second of video"),
    keyframe_stride: int = Form(4, description="SLAM3R keyframe stride"),
    point_density: int   = Form(500000, description="Number of points to save"),
):
    """
    Accept a drone video + optional GPS/IMU file, save them to disk,
    and launch the ANTARA reconstruction pipeline in a background thread.
    """
    # ── Guard: only one job at a time ──
    if jm.is_busy():
        raise HTTPException(
            status_code=409,
            detail="A reconstruction job is already running. Please wait for it to finish."
        )

    # ── Validate video extension ──
    if video.filename == "":
        raise HTTPException(status_code=400, detail="No video file provided.")
    ext = os.path.splitext(video.filename)[1].lower()
    if ext not in (".mp4", ".mov", ".avi", ".mkv"):
        raise HTTPException(status_code=400, detail=f"Unsupported video format: {ext}. Use .mp4, .mov, .avi, or .mkv.")

    # ── Require GPS unless demo mode ──
    has_gps = gps is not None and gps.filename not in ("", None)
    if not has_gps and not demo_mode:
        raise HTTPException(
            status_code=400,
            detail="GPS/IMU telemetry is required for a verified reconstruction. Enable 'Demo mode' to skip it (accuracy will be unverified)."
        )

    # ── Clean up stale uploads ──
    jm.cleanup_old_uploads(max_age_hours=24)

    # ── Stream video to disk ──
    job_id = None  # will be set by launch_job, but we need path now
    tmp_id = uuid.uuid4().hex[:12]
    video_path = os.path.join(UPLOAD_DIR, f"{tmp_id}_drone{ext}")

    written = 0
    try:
        async with aiofiles.open(video_path, "wb") as out:
            while chunk := await video.read(8 * 1024 * 1024):   # 8 MB chunks
                written += len(chunk)
                if written > jm.MAX_VIDEO_BYTES:
                    raise ValueError("VIDEO_TOO_LARGE")
                await out.write(chunk)
    except ValueError:
        if os.path.exists(video_path):
            os.remove(video_path)
        raise HTTPException(status_code=413, detail="Video exceeds 10 GB limit.")
    except Exception as e:
        if os.path.exists(video_path):
            os.remove(video_path)
        raise HTTPException(status_code=500, detail=f"Video upload failed: {str(e)}")

    # ── Stream GPS to disk if provided ──
    gps_path = None
    if has_gps:
        gps_ext = os.path.splitext(gps.filename)[1].lower()
        gps_path = os.path.join(UPLOAD_DIR, f"{tmp_id}_telemetry{gps_ext}")
        gps_written = 0
        try:
            async with aiofiles.open(gps_path, "wb") as out:
                while chunk := await gps.read(1 * 1024 * 1024):   # 1 MB chunks
                    gps_written += len(chunk)
                    if gps_written > jm.MAX_TELEMETRY_BYTES:
                        raise ValueError("GPS_TOO_LARGE")
                    await out.write(chunk)
        except ValueError:
            if os.path.exists(gps_path):
                os.remove(gps_path)
            if os.path.exists(video_path):
                os.remove(video_path)
            raise HTTPException(status_code=413, detail="GPS file exceeds 50 MB limit.")
        except Exception as e:
            if os.path.exists(gps_path):
                os.remove(gps_path)
            if os.path.exists(video_path):
                os.remove(video_path)
            raise HTTPException(status_code=500, detail=f"GPS upload failed: {str(e)}")

    # ── Launch pipeline ──
    job_id = jm.launch_job(
        video_path=video_path,
        gps_path=gps_path,
        allow_synthetic_gps=demo_mode,
        use_sam=use_sam,
        enable_gs=enable_gs,
        target_fps=target_fps,
        keyframe_stride=keyframe_stride,
        num_points_save=point_density
    )
    return {
        "message": "Upload successful. 3D reconstruction has started.",
        "job_id":  job_id,
        "status":  "processing",
    }


# ── Entrypoint ────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="ANTARA web upload server.")
    ap.add_argument("--host", default="127.0.0.1",
                    help="Bind address. Default 127.0.0.1 (localhost). Use 0.0.0.0 to expose on network (no auth).")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    print("\n" + "=" * 62)
    print(f"  PROJECT ANTARA — Upload Portal")
    print(f"  http://{args.host}:{args.port}/")
    if args.host != "127.0.0.1":
        print("  [!] WARNING: exposed on network with NO authentication.")
    print("=" * 62 + "\n")

    uvicorn.run(
        "backend.server:app",
        host=args.host,
        port=args.port,
        reload=False,
        log_level="info",
        # Allow up to 10.1 GB request body (10 GB video + headers overhead)
        limit_concurrency=4,
    )


if __name__ == "__main__":
    main()
