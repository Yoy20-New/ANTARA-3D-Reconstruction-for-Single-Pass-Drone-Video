# ANTARA Bug Fix Guide — Step-by-Step Manual Commands
**Date**: September 27, 2026
**Total Fixes**: 10 issues across 7 files
**Estimated Total Time**: ~2 hours

---

## How to Use This Guide

Each fix follows this format:
1. **Open the file** in your editor
2. **Find the exact lines** (line numbers given)
3. **Replace the BEFORE code** with the **AFTER code**
4. **Save the file**
5. **Run the verification command** (if provided)

> **IMPORTANT**: Follow the fixes in **P0 → P1 → P2** order. P0 fixes are critical and must be done first.

---

## P0 — Fix 1: COLMAP Database Corruption (CRITICAL)

**File**: `e:\SIH\antara\colmap_sfm.py`
**Lines**: 37-39
**Problem**: Running the pipeline twice on different videos reuses the same COLMAP database, silently corrupting the 3D reconstruction.

### Step 1: Open the file
`e:\SIH\antara\colmap_sfm.py`

### Step 2: Find this code (lines 37-39)

**BEFORE:**
`python
    sfm_root = os.path.join(cfg.output_dir, "colmap_sfm")
    os.makedirs(sfm_root, exist_ok=True)
    database_path = os.path.join(sfm_root, "database.db")
`

### Step 3: Replace with this

**AFTER:**
`python
    sfm_root = os.path.join(cfg.output_dir, "colmap_sfm")
    os.makedirs(sfm_root, exist_ok=True)
    database_path = os.path.join(sfm_root, "database.db")

    # --- FIX: Clean stale COLMAP database from previous runs ---
    if os.path.exists(database_path):
        os.remove(database_path)
        print(f"[COLMAP] Removed stale database: {database_path}")
`

### Step 4: Save the file

### Verification
No terminal command needed. The fix is passive — it will auto-clean on the next pipeline run. You can verify by running the pipeline twice on different videos and checking that the output is correct each time.

---

## P0 — Fix 2: GPS Hemisphere Swap Bug (CRITICAL)

**File**: `e:\SIH\videos\srt_to_json.py`
**Lines**: 104-117
**Problem**: For Indian locations (both Lat and Lon under 90 degrees), the parser silently swaps coordinates, geolocating the model in Finland instead of India.

### Step 1: Open the file
`e:\SIH\videos\srt_to_json.py`

### Step 2: Find this code (lines 104-117)

**BEFORE:**
`python
def _extract_gps_parens(text: str):
    """Try the GPS(a, b, c) format. Auto-detect (lon,lat,alt) vs (lat,lon,alt)."""
    m = _GPS_PARENS.search(text)
    if not m:
        return None
    a, b, c = float(m.group(1)), float(m.group(2)), float(m.group(3))
    # DJI typically uses GPS(lon, lat, alt), but some firmware does (lat, lon, alt).
    # If |a| > 90 it must be longitude; if |b| > 90 it must be longitude.
    if _looks_like_longitude(a) and not _looks_like_longitude(b):
        # a = lon, b = lat
        return b, a, c
    else:
        # a = lat, b = lon  (or both <= 90 -- assume lat-first)
        return a, b, c
`

### Step 3: Replace with this

**AFTER:**
`python
def _extract_gps_parens(text: str):
    """Try the GPS(a, b, c) format. Auto-detect (lon,lat,alt) vs (lat,lon,alt)."""
    m = _GPS_PARENS.search(text)
    if not m:
        return None
    a, b, c = float(m.group(1)), float(m.group(2)), float(m.group(3))
    # DJI documented standard is GPS(lon, lat, alt).
    # If |a| > 90, it is definitely longitude -> return (b=lat, a=lon, c=alt).
    # If |b| > 90, it is definitely longitude -> return (a=lat, b=lon, c=alt).
    # If BOTH are <= 90 (e.g., India: Lat 28, Lon 77), default to DJI standard
    # which is (lon, lat, alt) -> return (b=lat, a=lon, c=alt).
    if _looks_like_longitude(b) and not _looks_like_longitude(a):
        # b = lon, a = lat  (non-standard firmware)
        return a, b, c
    else:
        # DJI standard: a = lon, b = lat (covers both confirmed and ambiguous cases)
        return b, a, c
`

### Step 4: Save the file

### Verification
`powershell
cd e:\SIH
python -c "from videos.srt_to_json import _extract_gps_parens; print(_extract_gps_parens('GPS(77.209, 28.614, 50)'))"
`
**Expected output**: `(28.614, 77.209, 50.0)` — lat=28, lon=77 (India, not Finland)

---

## P0 — Fix 3: Add GLB and FBX Export (NTRO Compliance)

**File**: `e:\SIH\antara\deliverables.py`
**Lines**: Add at the end of file (after line 202)
**Problem**: NTRO PS-17 requires GLB and FBX formats, but they are completely missing.

### Step 1: Install the required library
`powershell
cd e:\SIH
pip install trimesh
`

### Step 2: Open the file
`e:\SIH\antara\deliverables.py`

### Step 3: Add the following code at the END of the file (after line 202)

**ADD THIS AT THE BOTTOM:**
`python


def export_glb(obj_path, glb_path):
    """Convert OBJ mesh to GLB (binary glTF) format for web/AR viewers."""
    import trimesh

    if not os.path.isfile(obj_path):
        raise FileNotFoundError(f"OBJ file not found for GLB conversion: {obj_path}")

    os.makedirs(os.path.dirname(glb_path) or ".", exist_ok=True)
    mesh = trimesh.load(obj_path, force="mesh")
    mesh.export(glb_path, file_type="glb")
    print(f"[Deliverables] GLB exported: {glb_path}")
    return glb_path


def export_fbx(obj_path, fbx_path):
    """Convert OBJ mesh to FBX format for professional 3D software (Blender, Unity, Unreal).

    NOTE: trimesh FBX support requires the assimp backend.
    Install with:  pip install pyassimp
    If pyassimp is not available, this function will print a warning and skip.
    """
    os.makedirs(os.path.dirname(fbx_path) or ".", exist_ok=True)
    try:
        import trimesh
        mesh = trimesh.load(obj_path, force="mesh")
        mesh.export(fbx_path, file_type="fbx")
        print(f"[Deliverables] FBX exported: {fbx_path}")
        return fbx_path
    except Exception as e:
        print(f"[Deliverables] FBX export skipped ({e}). "
              "Install pyassimp for FBX support: pip install pyassimp")
        return None
`

### Step 4: Save the file

### Step 5: Now wire GLB/FBX into the pipeline

**File**: `e:\SIH\antara\pipeline.py`
**Find this code** (around line 188-194):

**BEFORE:**
`python
    try:
        _dl.export_mesh(ply_path, obj_path, sugar_mesh_path=sugar_mesh_path)
    except Exception as e:  # meshing is the most fragile export; don't lose the cloud
        progress(91, f"Mesh export skipped ({e}). Point cloud deliverables intact.")
        obj_path = None
    _dl.export_transforms(frame_names, corrected_cams, geo["rotation"], colmap_dir)
    progress(92, "transforms.json written.")
`

**AFTER:**
`python
    try:
        _dl.export_mesh(ply_path, obj_path, sugar_mesh_path=sugar_mesh_path)
    except Exception as e:  # meshing is the most fragile export; don't lose the cloud
        progress(91, f"Mesh export skipped ({e}). Point cloud deliverables intact.")
        obj_path = None

    # --- GLB and FBX export (NTRO PS-17 compliance) ---
    if obj_path and os.path.isfile(obj_path):
        glb_path = os.path.join(cfg.output_dir, "antara_model.glb")
        fbx_path = os.path.join(cfg.output_dir, "antara_model.fbx")
        try:
            _dl.export_glb(obj_path, glb_path)
        except Exception as e:
            progress(91, f"GLB export skipped ({e}).")
        try:
            _dl.export_fbx(obj_path, fbx_path)
        except Exception as e:
            progress(91, f"FBX export skipped ({e}).")

    _dl.export_transforms(frame_names, corrected_cams, geo["rotation"], colmap_dir)
    progress(92, "transforms.json written.")
`

### Step 6: Save the file

### Verification
`powershell
cd e:\SIH
python -c "from antara.deliverables import export_glb, export_fbx; print('GLB/FBX functions imported OK')"
`
**Expected output**: `GLB/FBX functions imported OK`

---

## P1 — Fix 4: OBJ Material Link Broken After Rename

**File**: `e:\SIH\antara\deliverables.py`
**Lines**: 116-124
**Problem**: Renaming the `.mtl` file without updating the `mtllib` pointer inside the `.obj` file breaks textures.

### Step 1: Open the file
`e:\SIH\antara\deliverables.py`

### Step 2: Find this code (lines 116-124)

**BEFORE:**
`python
        os.makedirs(os.path.dirname(obj_path) or ".", exist_ok=True)
        with open(sugar_mesh_path, "rb") as src, open(obj_path, "wb") as dst:
            dst.write(src.read())
        mtl_path = os.path.splitext(sugar_mesh_path)[0] + ".mtl"
        if os.path.exists(mtl_path):
            target_mtl = os.path.splitext(obj_path)[0] + ".mtl"
            with open(mtl_path, "rb") as src, open(target_mtl, "wb") as dst:
                dst.write(src.read())
        return obj_path
`

### Step 3: Replace with this

**AFTER:**
`python
        os.makedirs(os.path.dirname(obj_path) or ".", exist_ok=True)
        import shutil
        shutil.copy2(sugar_mesh_path, obj_path)
        mtl_path = os.path.splitext(sugar_mesh_path)[0] + ".mtl"
        if os.path.exists(mtl_path):
            target_mtl = os.path.splitext(obj_path)[0] + ".mtl"
            shutil.copy2(mtl_path, target_mtl)
            # Fix the mtllib pointer inside the OBJ file to match the new MTL filename
            old_mtl_name = os.path.basename(mtl_path)
            new_mtl_name = os.path.basename(target_mtl)
            with open(obj_path, "r", encoding="utf-8", errors="replace") as f:
                obj_content = f.read()
            obj_content = obj_content.replace(f"mtllib {old_mtl_name}", f"mtllib {new_mtl_name}")
            with open(obj_path, "w", encoding="utf-8") as f:
                f.write(obj_content)
        return obj_path
`

### Step 4: Save the file

---

## P1 — Fix 5: Wrong ndim Check for Flattened SLAM3R Output

**File**: `e:\SIH\antara\reconstruct.py`
**Lines**: 143-159
**Problem**: A flattened SLAM3R array with shape `(K*H*W, 3)` has `ndim==2`, not `ndim==3`, causing a false crash.

### Step 1: Open the file
`e:\SIH\antara\reconstruct.py`

### Step 2: Find this code (lines 143-159)

**BEFORE:**
`python
    if reg.ndim == 4 and reg.shape[0] == num_keyframes and reg.shape[-1] == 3:
        reg = reg.reshape(num_keyframes, -1, 3)
    elif reg.ndim == 3 and reg.shape[-1] == 3:
        total = reg.shape[0]
        if total % num_keyframes != 0:
            raise ReconstructionError(
                f"registered_pcds has {total} points, not divisible by "
                f"{num_keyframes} keyframes. SLAM3R keyframe count differs "
                "from the extracted frames — the pipeline cannot align cameras "
                "to GPS."
            )
        reg = reg.reshape(num_keyframes, -1, 3)
    else:
        raise ReconstructionError(
            f"Unexpected registered_pcds shape {reg.shape}; expected "
            f"({num_keyframes}, H, W, 3) or (K*H*W, 3)."
        )
`

### Step 3: Replace with this

**AFTER:**
`python
    if reg.ndim == 4 and reg.shape[0] == num_keyframes and reg.shape[-1] == 3:
        reg = reg.reshape(num_keyframes, -1, 3)
    elif reg.ndim == 3 and reg.shape[-1] == 3:
        total = reg.shape[0]
        if total % num_keyframes != 0:
            raise ReconstructionError(
                f"registered_pcds has {total} points, not divisible by "
                f"{num_keyframes} keyframes. SLAM3R keyframe count differs "
                "from the extracted frames — the pipeline cannot align cameras "
                "to GPS."
            )
        reg = reg.reshape(num_keyframes, -1, 3)
    elif reg.ndim == 2 and reg.shape[-1] == 3:
        # Flattened array: (K*H*W, 3)
        total = reg.shape[0]
        if total % num_keyframes != 0:
            raise ReconstructionError(
                f"registered_pcds has {total} points (ndim=2), not divisible by "
                f"{num_keyframes} keyframes."
            )
        reg = reg.reshape(num_keyframes, -1, 3)
    else:
        raise ReconstructionError(
            f"Unexpected registered_pcds shape {reg.shape}; expected "
            f"({num_keyframes}, H, W, 3) or (K*H*W, 3)."
        )
`

### Step 4: Save the file

---

## P1 — Fix 6: Wrong Python Binary in Subprocesses

**File 1**: `e:\SIH\antara\gaussians.py` (Line 55)
**File 2**: `e:\SIH\antara\sugar.py` (Line 52)
**Problem**: `shutil.which("python")` may invoke the wrong Python (system Python instead of conda env).

### Step 1: Fix gaussians.py

**Open**: `e:\SIH\antara\gaussians.py`

**At line 1-6**, change imports:

**BEFORE (line 1-6):**
`python
from __future__ import annotations

import os
import shutil
import subprocess
from typing import Callable
`

**AFTER:**
`python
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from typing import Callable
`

**Then at line 55**, change:

**BEFORE:**
`python
    python_exe = shutil.which("python") or shutil.which("python3")
`

**AFTER:**
`python
    python_exe = sys.executable
`

### Step 2: Fix sugar.py

**Open**: `e:\SIH\antara\sugar.py`

**At line 1-6**, change imports:

**BEFORE (line 1-6):**
`python
from __future__ import annotations

import os
import shutil
import subprocess
from typing import Callable
`

**AFTER:**
`python
from __future__ import annotations

import os
import subprocess
import sys
from typing import Callable
`

**Then at line 52**, change:

**BEFORE:**
`python
    python_exe = shutil.which("python") or shutil.which("python3")
`

**AFTER:**
`python
    python_exe = sys.executable
`

### Step 3: Save both files

---

## P2 — Fix 7: GeoTIFF Export Not Wrapped in try/except

**File**: `e:\SIH\antara\pipeline.py`
**Lines**: 185-186
**Problem**: If GeoTIFF export fails, the entire pipeline crashes and loses the accuracy report.

### Step 1: Open the file
`e:\SIH\antara\pipeline.py`

### Step 2: Find this code (around line 185-186)

**BEFORE:**
`python
    progress(87, "LAS written. Exporting GeoTIFF orthomosaic...")
    _dl.export_geotiff(corrected_points, world_colors, origin, tif_path)
    progress(90, "GeoTIFF written. Meshing (Poisson or SuGaR fallback)...")
`

### Step 3: Replace with this

**AFTER:**
`python
    progress(87, "LAS written. Exporting GeoTIFF orthomosaic...")
    try:
        _dl.export_geotiff(corrected_points, world_colors, origin, tif_path)
    except Exception as e:
        progress(88, f"GeoTIFF export skipped ({e}). Continuing pipeline...")
        tif_path = None
    progress(90, "GeoTIFF done. Meshing (Poisson or SuGaR fallback)...")
`

### Step 4: Save the file

---

## P2 — Fix 8: GeoTIFF Division-by-Zero on Degenerate Bounds

**File**: `e:\SIH\antara\deliverables.py`
**Lines**: 153-156
**Problem**: If all points have the same X or Y coordinate, rasterio gets zero-width bounds and crashes.

### Step 1: Open the file
`e:\SIH\antara\deliverables.py`

### Step 2: Find this code (lines 153-156)

**BEFORE:**
`python
    min_x, max_x = points[:, 0].min(), points[:, 0].max()
    min_y, max_y = points[:, 1].min(), points[:, 1].max()
    dx = (max_x - min_x) or 1e-5
    dy = (max_y - min_y) or 1e-5
`

### Step 3: Replace with this

**AFTER:**
`python
    min_x, max_x = points[:, 0].min(), points[:, 0].max()
    min_y, max_y = points[:, 1].min(), points[:, 1].max()
    dx = (max_x - min_x) or 1e-5
    dy = (max_y - min_y) or 1e-5
    # Pad degenerate bounds to prevent rasterio division-by-zero
    if max_x == min_x:
        min_x -= 0.5
        max_x += 0.5
    if max_y == min_y:
        min_y -= 0.5
        max_y += 0.5
`

### Step 4: Save the file

---

## P2 — Fix 9: Disk Cleanup for Server

**File**: `e:\SIH\antara_server.py`
**Problem**: Uploaded files are never cleaned up, leading to disk exhaustion.

### Step 1: Open the file
`e:\SIH\antara_server.py`

### Step 2: Add this import at the top (with other imports)

`python
import time as _time
`

### Step 3: Add the following cleanup function BEFORE the first @app.route line

`python
def _cleanup_old_files(directory, max_age_hours=24):
    """Delete files older than max_age_hours in the given directory."""
    if not os.path.isdir(directory):
        return
    now = _time.time()
    cutoff = now - (max_age_hours * 3600)
    for fname in os.listdir(directory):
        fpath = os.path.join(directory, fname)
        try:
            if os.path.isfile(fpath) and os.path.getmtime(fpath) < cutoff:
                os.remove(fpath)
                print(f"[Cleanup] Removed old file: {fpath}")
        except OSError:
            pass
`

### Step 4: Find the upload/processing route and add a cleanup call at the start

**Find the line where a new job starts** (where the upload is saved) and add this BEFORE saving:
`python
    _cleanup_old_files(UPLOAD_DIR, max_age_hours=24)
`

### Step 5: Save the file

---

## P2 — Fix 10: Telemetry File Size Limit

**File**: `e:\SIH\antara_server.py`
**Problem**: A massive JSON telemetry upload can OOM the server.

### Step 1: Open the file
`e:\SIH\antara_server.py`

### Step 2: Find where the GPS/telemetry file is saved in the upload route

**ADD this check right AFTER saving the telemetry file** (before json.load is called):
`python
    # Enforce telemetry file size limit (10 MB max)
    MAX_TELEMETRY_BYTES = 10 * 1024 * 1024  # 10 MB
    if gps_path and os.path.isfile(gps_path):
        if os.path.getsize(gps_path) > MAX_TELEMETRY_BYTES:
            os.remove(gps_path)
            return jsonify({"error": "Telemetry file too large (max 10 MB)"}), 400
`

### Step 3: Save the file

---

## FINAL VERIFICATION — Run All Import Checks

After applying ALL fixes, run this single command to verify nothing is broken:

`powershell
cd e:\SIH
python -c "from antara.config import PipelineConfig; from antara.deliverables import export_ply, export_las, export_mesh, export_geotiff, export_glb, export_fbx; from antara.colmap_sfm import run_colmap_sfm; from antara.reconstruct import reconstruct; from antara.georeference import georeference; from antara.pipeline import run_pipeline; from antara.gaussians import train_gaussian_splat; from antara.sugar import extract_sugar_mesh; from videos.srt_to_json import parse_srt_file, auto_detect_gps_for_video; print('All 10 fixes verified - all modules import successfully!')"
`

**Expected output**: `All 10 fixes verified - all modules import successfully!`

---

## Summary Checklist

| No. | Fix | File | Done? |
|:----|:----|:-----|:------|
| 1 | COLMAP DB cleanup | colmap_sfm.py | |
| 2 | GPS hemisphere swap | srt_to_json.py | |
| 3 | GLB/FBX export | deliverables.py + pipeline.py | |
| 4 | OBJ mtllib fix | deliverables.py | |
| 5 | ndim==2 check | reconstruct.py | |
| 6 | sys.executable | gaussians.py + sugar.py | |
| 7 | GeoTIFF try/except | pipeline.py | |
| 8 | GeoTIFF bounds padding | deliverables.py | |
| 9 | Disk cleanup | antara_server.py | |
| 10 | Telemetry size limit | antara_server.py | |
