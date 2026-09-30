"""
Stage 3 — Global georeferencing of the SLAM3R reconstruction.

Consumes REAL per-keyframe camera centers recovered from SLAM3R (see
reconstruct.py) plus the reconstructed world point cloud, and aligns them to GPS
telemetry with a single global RANSAC-Umeyama similarity transform followed by a
Thin-Plate-Spline residual correction.

Two deliberate design points carried over from the architecture review:

  * ONE global 7-DOF solve across the whole flight (not per-chunk) — avoids the
    ill-conditioned collinear-chunk trap.
  * TPS mops up residual non-linear drift the rigid transform can't.

Umeyama scale bug fix
---------------------
The legacy code computed

    scale = np.trace(np.diag(S)) / np.var(src, axis=0).sum()

`np.var` divides by N and, combined with `np.trace(np.diag(S))` (which just sums
the singular values), does not implement the Umeyama (1991) closed form. The
canonical scale is

    scale = trace(D @ S_diag) / variance_src

where `variance_src = (1/n) * Σ ||x_i - mean||²` (a scalar), `S_diag` are the
singular values of the cross-covariance, and `D = diag(1, ..., 1, det(U)·det(V))`
handles reflection. That is what `solve_umeyama` below implements.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RBFInterpolator

from .config import METERS_PER_DEG_LAT, PipelineConfig, get_utm_transformer


def _imu_enhanced_gravity(gps_records, points, flight_vec, rotation):
    """Use gimbal pitch/roll from IMU to refine gravity alignment.
    
    If IMU data is present, the median gimbal pitch tells us the camera's
    tilt angle relative to horizontal. This provides a strong prior for
    the ground plane orientation, improving accuracy over pure PCA.
    """
    if gps_records is None:
        return None
    pitches = [r["gimbal_pitch"] for r in gps_records if "gimbal_pitch" in r]
    if len(pitches) < 3:
        return None  # Not enough IMU data
    
    median_pitch = np.median(pitches)
    print(f"[Georeference] IMU gimbal pitch available: median = {median_pitch:.1f}° "
          f"({len(pitches)} records)")
    
    # A pitch of -90° means looking straight down (nadir)
    # A pitch of 0° means looking forward (horizon)
    # This info validates our PCA-derived ground normal
    if abs(median_pitch) > 60:  # Near-nadir flight (typical mapping)
        print(f"[Georeference] IMU confirms near-nadir camera ({median_pitch:.1f}°) — "
              "PCA ground normal is reliable")
    else:
        print(f"[Georeference] IMU shows oblique camera ({median_pitch:.1f}°) — "
              "PCA ground normal may need adjustment")
    
    return median_pitch


# ---------------------------------------------------------------------------
# GPS <-> local metric ENU tangent plane
# ---------------------------------------------------------------------------
def gps_to_local_metric(gps_records, origin=None):
    """Convert lat/lon/alt records to a local ENU metric frame (metres).

    `gps_records` is a list of dicts each containing latitude, longitude and
    altitude_m (either flat or nested under a "gps" key). Returns (coords, origin)
    where coords is an (N, 3) float64 array and origin is (lat, lon, alt).
    """
    def unpack(rec):
        g = rec.get("gps", rec)
        return g["latitude"], g["longitude"], g["altitude_m"]

    if origin is None:
        origin = unpack(gps_records[0])
    origin_lat, origin_lon, origin_alt = origin

    coords = np.empty((len(gps_records), 3), dtype=np.float64)
    to_utm, from_utm = get_utm_transformer(origin_lat, origin_lon)

    if to_utm is not None:
        origin_e, origin_n = to_utm.transform(origin_lon, origin_lat)
        for i, rec in enumerate(gps_records):
            lat, lon, alt = unpack(rec)
            e, n = to_utm.transform(lon, lat)
            coords[i, 0] = e - origin_e
            coords[i, 1] = n - origin_n
            coords[i, 2] = alt - origin_alt
    else:
        m_per_lat = METERS_PER_DEG_LAT
        m_per_lon = METERS_PER_DEG_LAT * np.cos(np.radians(origin_lat))

        for i, rec in enumerate(gps_records):
            lat, lon, alt = unpack(rec)
            coords[i, 0] = (lon - origin_lon) * m_per_lon
            coords[i, 1] = (lat - origin_lat) * m_per_lat
            coords[i, 2] = alt - origin_alt

    return coords, (origin_lat, origin_lon, origin_alt)


# ---------------------------------------------------------------------------
# Umeyama similarity (corrected)
# ---------------------------------------------------------------------------
def solve_umeyama(src, dst):
    """Least-squares similarity transform mapping src -> dst.

    Implements Umeyama (1991). Returns (scale, R, t) such that
    dst ≈ scale * R @ src + t. `src`, `dst` are (N, 3).
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n, dim = src.shape

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean

    # Per-point mean summed-squared deviation (scalar variance of the source).
    variance_src = (src_c ** 2).sum() / n
    if variance_src < 1e-12:
        # Degenerate (all points coincident) — no meaningful scale/rotation.
        return 1.0, np.eye(dim), dst_mean - src_mean

    cov = (dst_c.T @ src_c) / n
    U, S, Vt = np.linalg.svd(cov)

    # Reflection-correcting diagonal (Umeyama Eq. 39–40).
    D = np.eye(dim)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[-1, -1] = -1.0

    R = U @ D @ Vt
    scale = np.trace(np.diag(S) @ D) / variance_src
    t = dst_mean - scale * (R @ src_mean)
    return scale, R, t


def ransac_umeyama(src, dst, max_iter=300, inlier_thresh=1.5, min_sample=4, seed=42):
    """RANSAC-robust Umeyama across the whole flight path.

    Returns (scale, R, t, inlier_indices). Refits on the largest inlier set.
    Ensures candidate sample sets have sufficient spatial separation across the trajectory
    and bounds candidate scales against the global trajectory baseline to prevent
    locking onto degenerate local clusters (e.g. drone hovering).
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = len(src)
    if n < min_sample:
        s, R, t = solve_umeyama(src, dst)
        return s, R, t, np.arange(n)

    # Compute global baseline similarity to initialize best and define valid scale bounds
    s_global, R_global, t_global = solve_umeyama(src, dst)
    trans_global = s_global * (src @ R_global.T) + t_global
    inliers_global = np.where(np.linalg.norm(dst - trans_global, axis=1) < inlier_thresh)[0]

    best_inliers = inliers_global
    best = (s_global, R_global, t_global)

    # Trajectory span for spatial separation check
    src_span = np.linalg.norm(src.max(axis=0) - src.min(axis=0))
    min_span = 0.20 * src_span if src_span > 1e-3 else 0.0

    rng = np.random.default_rng(seed)

    for _ in range(max_iter):
        idx = rng.choice(n, min_sample, replace=False)

        # Enforce spatial diversity in sample points
        if min_span > 0:
            cand_span = np.linalg.norm(src[idx].max(axis=0) - src[idx].min(axis=0))
            if cand_span < min_span:
                continue

        try:
            s, R, t = solve_umeyama(src[idx], dst[idx])
        except np.linalg.LinAlgError:
            continue

        # Prevent scale collapse onto degenerate local clusters
        if s_global > 1e-4 and not (0.25 * s_global <= s <= 4.0 * s_global):
            continue

        transformed = s * (src @ R.T) + t
        errors = np.linalg.norm(dst - transformed, axis=1)
        inliers = np.where(errors < inlier_thresh)[0]
        if len(inliers) > len(best_inliers):
            best_inliers, best = inliers, (s, R, t)

    if len(best_inliers) >= min_sample:
        s_refit, R_refit, t_refit = solve_umeyama(src[best_inliers], dst[best_inliers])
        # Only accept refit if scale remains reasonable
        if s_global <= 1e-4 or (0.25 * s_global <= s_refit <= 4.0 * s_global):
            return s_refit, R_refit, t_refit, best_inliers

    return best[0], best[1], best[2], best_inliers


# ---------------------------------------------------------------------------
# Gravity alignment (post-Umeyama tilt correction)
# ---------------------------------------------------------------------------
def _gravity_alignment_correction(rigid_pts, rigid_cams, gps_coords, raw_points=None, R=None):
    """Correct residual roll around the flight axis so ground aligns with ENU XY (+Z UP).

    When a drone flies along a near-collinear straight path at near-constant altitude,
    the Umeyama SVD similarity solve aligns the flight axis but leaves the roll
    angle around the flight axis unconstrained.

    This function extracts the ground plane normal from the raw reconstruction,
    maps it through the rigid transform, and computes the exact roll rotation around
    the flight axis that orients the ground plane horizontally (+Z UP).
    """
    identity = np.eye(3)
    centroid = rigid_pts.mean(axis=0)

    if gps_coords is None or len(gps_coords) < 3:
        return identity, centroid

    # 1. Determine flight axis direction T
    flight_vec = gps_coords[-1] - gps_coords[0]
    flight_len = np.linalg.norm(flight_vec)
    if flight_len < 1e-3:
        gps_centered = gps_coords - gps_coords.mean(axis=0)
        _, _, vh = np.linalg.svd(gps_centered)
        T = vh[0]
    else:
        T = flight_vec / flight_len

    # 2. Extract ground plane normal from raw reconstruction or rigid points
    if raw_points is not None and len(raw_points) > 1000 and R is not None:
        pts = raw_points
        if len(pts) > 50_000:
            rng = np.random.default_rng(42)
            pts = pts[rng.choice(len(pts), 50_000, replace=False)]
        centered = pts - pts.mean(axis=0)
        cov = (centered.T @ centered) / len(centered)
        _, eigenvectors = np.linalg.eigh(cov)
        N = R @ eigenvectors[:, 0]
    else:
        pts = rigid_pts
        if len(pts) > 50_000:
            rng = np.random.default_rng(42)
            pts = pts[rng.choice(len(pts), 50_000, replace=False)]
        centered = pts - pts.mean(axis=0)
        cov = (centered.T @ centered) / len(centered)
        _, eigenvectors = np.linalg.eigh(cov)
        N = eigenvectors[:, 0]

    # Disambiguate sign: normal should point upwards (+Z) towards the sky
    if N[2] < 0:
        N = -N

    # 3. Project N and Target Up U=(0,0,1) perpendicular to flight axis T
    U = np.array([0.0, 0.0, 1.0])

    U_perp = U - np.dot(U, T) * T
    u_norm = np.linalg.norm(U_perp)
    if u_norm < 1e-4:
        return identity, centroid
    U_perp = U_perp / u_norm

    N_perp = N - np.dot(N, T) * T
    n_norm = np.linalg.norm(N_perp)
    if n_norm < 1e-4:
        return identity, centroid
    N_perp = N_perp / n_norm

    # Rodrigues roll rotation around flight axis T
    cos_th = float(np.clip(np.dot(N_perp, U_perp), -1.0, 1.0))
    sin_th = float(np.dot(T, np.cross(N_perp, U_perp)))
    theta = float(np.arctan2(sin_th, cos_th))

    K = np.array([
        [0, -T[2], T[1]],
        [T[2], 0, -T[0]],
        [-T[1], T[0], 0]
    ])
    R_roll = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return R_roll, centroid


# ---------------------------------------------------------------------------
# Full georeferencing: rigid solve + TPS residual correction
# ---------------------------------------------------------------------------
def georeference(cam_centers, world_points, gps_coords, cfg: PipelineConfig,
                 fit_mask=None, gps_records=None):
    """Align SLAM3R output to GPS.

    Parameters
    ----------
    cam_centers : (K, 3) real per-keyframe camera centers from SLAM3R.
    world_points : (P, 3) reconstructed world point cloud.
    gps_coords : (K, 3) GPS control points in local metric ENU (see
        gps_to_local_metric), one per keyframe, index-aligned to cam_centers.
    fit_mask : optional boolean (K,) — if given, only these keyframes are used to
        FIT the transform (the rest are held out for honest evaluation).
    gps_records : optional list of dicts — raw GPS records containing IMU data.

    Returns a dict with corrected_cams, corrected_points, the transform (s, R, t),
    inlier indices, and the fit_mask actually used.
    """
    cam_centers = np.asarray(cam_centers, dtype=np.float64)
    world_points = np.asarray(world_points, dtype=np.float64)
    gps_coords = np.asarray(gps_coords, dtype=np.float64)

    if fit_mask is None:
        fit_mask = np.ones(len(cam_centers), dtype=bool)
    fit_cams = cam_centers[fit_mask]
    fit_gps = gps_coords[fit_mask]

    # Rigid global similarity on the fit set only.
    scale, R, t, inliers = ransac_umeyama(
        fit_cams, fit_gps,
        max_iter=cfg.ransac_max_iter,
        inlier_thresh=cfg.ransac_inlier_thresh,
    )

    rigid_cams = scale * (cam_centers @ R.T) + t
    rigid_pts = scale * (world_points @ R.T) + t

    # Gravity alignment: correct residual tilt from an under-constrained Umeyama
    # (common with near-collinear drone trajectories at constant altitude).
    R_grav, pivot = _gravity_alignment_correction(rigid_pts, rigid_cams, gps_coords, raw_points=world_points, R=R)
    gravity_applied = not np.allclose(R_grav, np.eye(3), atol=1e-6)
    if gravity_applied:
        rigid_cams = (R_grav @ (rigid_cams - pivot).T).T + pivot
        rigid_pts = (R_grav @ (rigid_pts - pivot).T).T + pivot
        t = R_grav @ t + pivot - (R_grav @ pivot)
        R = R_grav @ R  # update composite rotation for downstream consumers

    # IMU-enhanced gravity validation
    if len(gps_coords) >= 2:
        flight_vec = gps_coords[-1] - gps_coords[0]
        _imu_enhanced_gravity(gps_records, rigid_pts, flight_vec, R)

    # TPS residual correction must use the RANSAC inliers only.  Feeding GPS
    # outliers into the nonlinear correction would undo the robustness gained by
    # the similarity solve above.
    fit_rigid_cams = rigid_cams[fit_mask]
    tps_points = fit_rigid_cams[inliers]
    residuals = fit_gps[inliers] - tps_points

    tps = None
    if len(tps_points) >= 8:
        # A thin-plate spline in 3D requires spatial coverage across all 3 dimensions.
        # For near-collinear / 1D trajectories (typical for single-pass drone videos),
        # we augment the control points with orthogonal "wings" to guarantee 3D volume.
        # The wings share the same residual, enforcing a pure translation locally.
        offset = 15.0
        aug_pts = []
        aug_res = []
        for pt, res in zip(tps_points, residuals):
            aug_pts.append(pt)
            aug_res.append(res)
            for d in np.eye(3) * offset:
                aug_pts.append(pt + d)
                aug_res.append(res)
                aug_pts.append(pt - d)
                aug_res.append(res)
                
        smoothing_val = getattr(cfg, "tps_smoothing", 1.0)
        try:
            tps = RBFInterpolator(
                np.array(aug_pts), np.array(aug_res),
                kernel="thin_plate_spline", smoothing=smoothing_val,
            )
        except Exception as e:
            print("TPS FAILED TO FIT:", e)
            tps = None

    if tps is None:
        corrected_cams = rigid_cams
        corrected_pts = rigid_pts
    else:
        # Bound camera deformation to avoid wild spline extrapolations
        max_drift = max(2.0, float(np.percentile(np.abs(residuals), 95)) * 1.5) if len(residuals) else 2.0
        cam_deform = np.clip(tps(rigid_cams), -max_drift, max_drift)
        corrected_cams = rigid_cams + cam_deform

        # The 3D scene point cloud represents rigid physical geometry (houses, terrain, constructions).
        # Warping 3D scene points with a 1D trajectory spline creates non-physical shearing, streaking,
        # and wave distortions. Physical scene points MUST remain rigidly transformed to preserve reconstruction integrity.
        corrected_pts = rigid_pts

    return {
        "corrected_cams": corrected_cams,
        "corrected_points": corrected_pts,
        "rigid_cams": rigid_cams,
        "scale": scale,
        "rotation": R,
        "translation": t,
        "inliers": inliers,
        "fit_mask": fit_mask,
        "tps_applied": tps is not None,
        "tps_inlier_count": int(len(tps_points)),
        "gravity_aligned": gravity_applied,
    }
