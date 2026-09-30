# ANTARA — Backend Upload Server: Implementation Plan

## Goal
Build a **FastAPI backend** at `e:/SIH/backend/` that lets end-users upload a drone video (+ optional GPS/IMU telemetry), runs the full ANTARA reconstruction pipeline in the background, and redirects them to the live 3D viewer when done.

---

## User Review Required

> [!IMPORTANT]
> The existing `antara_server.py` at the root already contains a Flask-based version of this idea. The plan below **replaces it** with a cleaner FastAPI version living in `backend/`. The old Flask file will be left untouched but will no longer be the primary entry point.

> [!WARNING]
> Video uploads up to 10 GB require streaming upload handling — the full file must be written to disk before the pipeline starts. On an average machine this may take 1-5 minutes for large files even before reconstruction begins.

---

## Open Questions

> [!IMPORTANT]
> **GPS/IMU optional?** The existing pipeline can run in `--allow-synthetic-gps` demo mode (no GPS). Should the upload UI clearly mark GPS as optional and warn the user that accuracy will be unverified? *(Recommended: Yes)*

---

## Proposed File Structure

```
e:/SIH/
└── backend/
    ├── server.py          ← FastAPI app (main entry point)
    ├── job_manager.py     ← Job state, queue, background runner
    ├── upload_page.html   ← Self-contained upload UI (served at /)
    ├── requirements.txt   ← fastapi, uvicorn, aiofiles, python-multipart
    └── start.bat          ← One-click launcher for Windows
```

---

## Proposed Changes

### `backend/requirements.txt` [NEW]
List of Python packages needed beyond what's already in the `antara` conda env:
- `fastapi` ✅ already installed
- `uvicorn` ✅ already installed
- `aiofiles` ❌ needs `pip install aiofiles`
- `python-multipart` (for form/file uploads) — check needed

---

### `backend/job_manager.py` [NEW]
Single source of truth for the job queue and state machine.

**JobState dataclass:**
```
status: idle | uploading | processing | completed | error
progress: 0–100
stage_name: str
stage_detail: str
job_id: str
metrics: { points, keyframes, rmse, elapsed, gps_source }
error_message: str
```

**Key design decisions:**
- Only **1 job at a time** (GPU is a shared resource). Any upload attempt while a job is running returns HTTP 409.
- Job state persists in memory (no DB needed for local use).
- The pipeline runs in a **background thread** (not async) because `antara.pipeline.run()` is synchronous and CPU/GPU bound.
- A `progress_callback(pct, detail)` function is passed to `run_pipeline()` — same pattern as the existing Flask server.

---

### `backend/server.py` [NEW]
**API Endpoints:**

| Method | Path | Description |
|---|---|---|
| `GET` `/` | — | Serve `upload_page.html` |
| `POST` `/api/upload` | `multipart/form-data` | Accept `video` (required, max 10 GB) + `gps` (optional, max 50 MB). Returns `job_id`. |
| `GET` `/api/status` | — | Returns current `JobState` as JSON. Polled by the UI every 2s. |
| `GET` `/api/reset` | — | Resets job to `idle` (only if not currently processing). |
| `GET` `/viewer` | — | Serves `output/index.html` (the 3D viewer). |
| `GET` `/output/{filename}` | — | Serves any file from `output/` (PLY, OBJ, etc.). |

**Upload endpoint logic:**
1. Check job not already running → 409 if busy.
2. Stream video file to `uploads/{job_id}_drone.mp4`.
3. If GPS file present, save to `uploads/{job_id}_telemetry.json`.
4. Validate video file is readable (check header magic bytes for mp4/mov/avi).
5. Spawn background thread → `job_manager.run_job(video_path, gps_path)`.
6. Return `{"job_id": ..., "status": "processing"}` immediately.

**CORS:** Enabled for localhost only.

**Max upload size:** 10 GB configured via FastAPI's `app.state` and uvicorn's `--limit-request-body`.

---

### `backend/upload_page.html` [NEW]
A clean, self-contained HTML page matching the minimalist style of the 3D viewer.

**Sections:**
1. **Top bar** — "ANTARA" brand badge (same as viewer).
2. **Upload zone** — drag-and-drop area for video file. Shows filename + size after selection.
3. **GPS/IMU section** — Optional file picker for `.json`/`.srt` telemetry. Toggle button to mark as "no GPS (demo mode)".
4. **Submit button** — Disabled until a video is selected.
5. **Progress panel** (hidden until upload starts):
   - Upload progress bar (using `XMLHttpRequest.upload.onprogress`).
   - Live status log showing the current pipeline stage (polled via `/api/status` every 2s).
   - Stage breakdown: `Stage 1: Preprocessing → Stage 2: SLAM3R → Stage 3: Georeferencing → Stage 4: Deliverables → Stage 5: Accuracy`.
   - Final metrics card: Points, Keyframes, RMSE, Elapsed time.
6. **"View in 3D" button** — appears when job completes, links to `/viewer`.

---

## Verification Plan

### Automated Tests
```bash
# Install missing dep
conda run -n antara pip install aiofiles python-multipart

# Start the server
conda run -n antara python backend/server.py

# Test status endpoint
curl http://127.0.0.1:8765/api/status

# Test upload (small test video)
curl -X POST http://127.0.0.1:8765/api/upload \
     -F "video=@videos/test_clip.mp4"

# Check job progress
curl http://127.0.0.1:8765/api/status
```

### Manual Verification
1. Open `http://127.0.0.1:8765/` in browser → Upload page loads.
2. Drag a drone `.mp4` onto the zone → filename appears.
3. Click **Start Reconstruction** → progress bar fills, stage names update.
4. On completion → "View in 3D" button appears → 3D viewer opens with the new model.

---

## Start Command (after implementation)
```bat
cd e:\SIH
conda activate antara
python backend/server.py --port 8765
```
Then open: `http://127.0.0.1:8765/`
