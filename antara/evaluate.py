"""
Stage 5 — Honest metric spatial-accuracy evaluation.

Why this module is careful about what it claims
------------------------------------------------
The legacy evaluate_accuracy.py computed RMSE between the reconstructed camera
positions and the GPS points. But the georeferencing step (TPS) explicitly SNAPS
the cameras onto those same GPS points. Measuring the distance afterwards is
circular: it reports how well the fit fit itself, which is ~0 by construction and
says nothing about reconstruction accuracy.

We therefore report two clearly separated numbers:

  1. Alignment residual — error on the keyframes USED to fit the transform.
     Expected near zero. Labelled as a fit diagnostic, NOT accuracy.

  2. Hold-out ATE — a fraction of GPS keyframes are withheld from the Umeyama +
     TPS fit; error is measured on those unseen points. THIS is the number that
     speaks to real georeferencing accuracy.

Honesty caveat (also printed at runtime): with the synthetic straight-line test
GPS, even the hold-out set is trivially predictable, so a low hold-out ATE there
still does NOT substantiate a sub-metre claim. A genuine claim requires real drone
GPS with real noise. This tool measures; it does not manufacture a pass.
"""

from __future__ import annotations

import json
import os

import numpy as np

from .config import PipelineConfig
from .georeference import georeference, gps_to_local_metric


def _stats(errors: np.ndarray) -> dict:
    if len(errors) == 0:
        return {
            "rmse_m": 0.0,
            "mean_error_m": 0.0,
            "median_error_m": 0.0,
            "p95_error_m": 0.0,
            "max_error_m": 0.0,
            "std_dev_m": 0.0,
            "sample_points_count": 0,
        }
    return {
        "rmse_m": round(float(np.sqrt(np.mean(errors ** 2))), 4),
        "mean_error_m": round(float(np.mean(errors)), 4),
        "median_error_m": round(float(np.median(errors)), 4),
        "p95_error_m": round(float(np.percentile(errors, 95)), 4),
        "max_error_m": round(float(np.max(errors)), 4),
        "std_dev_m": round(float(np.std(errors)), 4),
        "sample_points_count": int(len(errors)),
    }


def evaluate(cam_centers, world_points, gps_records, cfg: PipelineConfig,
             output_report: str | None = None, seed: int = 42,
             gps_source: str = "unknown") -> dict:
    """Run alignment-residual + hold-out ATE evaluation.

    cam_centers : (K, 3) real SLAM3R camera centers (pre-georeferencing).
    world_points : (P, 3) reconstructed point cloud (only used so georeference()
        has something to transform; not scored).
    gps_records : list of per-keyframe GPS dicts, index-aligned to cam_centers.
    """
    if len(cam_centers) == 0 or len(gps_records) == 0:
        k = 0
    else:
        k = min(len(cam_centers), len(gps_records))

    if k < 4:
        report = {
            "target_accuracy_m": cfg.target_accuracy_m,
            "keyframes": k,
            "gps_source": gps_source,
            "accuracy_claim_allowed": False,
            "alignment_residual": {
                "_meaning": f"Only {k} keyframes — too few to evaluate."
            },
            "holdout_ate": {
                "_meaning": f"Only {k} keyframes — too few to evaluate."
            },
            "honesty_caveat": "Evaluation aborted due to insufficient data."
        }
        _print_report(report)
        if output_report:
            os.makedirs(os.path.dirname(output_report) or ".", exist_ok=True)
            with open(output_report, "w") as f:
                json.dump(report, f, indent=2)
            print(f"\nAccuracy report saved to: {output_report}")
        return report

    cam_centers = np.asarray(cam_centers, dtype=np.float64)[:k]
    gps_coords, _ = gps_to_local_metric(gps_records[:k])

    # ---- 1. Full-fit alignment residual (diagnostic only) --------------------
    full = georeference(cam_centers, world_points, gps_coords, cfg)
    resid_err = np.linalg.norm(full["corrected_cams"] - gps_coords, axis=1)
    alignment_residual = _stats(resid_err)

    # ---- 2. Hold-out ATE (the real accuracy signal) --------------------------
    rng = np.random.default_rng(seed)
    n_holdout = max(1, int(round(k * cfg.holdout_fraction)))
    if k - n_holdout >= 4:  # need >=4 points to fit a similarity transform
        holdout_idx = rng.choice(k, n_holdout, replace=False)
        fit_mask = np.ones(k, dtype=bool)
        fit_mask[holdout_idx] = False

        held = georeference(cam_centers, world_points, gps_coords, cfg,
                            fit_mask=fit_mask)
        ho_err = np.linalg.norm(
            held["corrected_cams"][holdout_idx] - gps_coords[holdout_idx], axis=1
        )
        holdout_ate = _stats(ho_err)
        holdout_ate["passed"] = bool(holdout_ate["rmse_m"] <= cfg.target_accuracy_m)
    else:
        holdout_ate = {
            "_meaning": (f"Only {k} keyframes — too few to hold out {n_holdout} "
                         f"and still fit a 4-point similarity. Hold-out ATE skipped."),
        }

    report = {
        "target_accuracy_m": cfg.target_accuracy_m,
        "keyframes": k,
        "gps_source": gps_source,
        "accuracy_claim_allowed": gps_source == "verified",
        "alignment_residual": {
            **alignment_residual,
            "_meaning": "Fit diagnostic ONLY. Cameras were snapped to these GPS "
                        "points by TPS; near-zero is expected and is NOT accuracy.",
        },
        "holdout_ate": holdout_ate,
        "honesty_caveat": (
            "Hold-out ATE is the real accuracy signal. Accuracy claims are "
            "disabled unless telemetry is explicitly marked verified. Synthetic "
            "or unverified test telemetry does not substantiate a sub-metre claim."
        ),
    }

    _print_report(report)

    if output_report:
        os.makedirs(os.path.dirname(output_report) or ".", exist_ok=True)
        with open(output_report, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nAccuracy report saved to: {output_report}")

    return report


def _print_report(report: dict) -> None:
    print("=" * 66)
    print(" PROJECT ANTARA — METRIC SPATIAL ACCURACY (NTRO PS-17)")
    print("=" * 66)
    print(f"  Keyframes analysed:       {report['keyframes']}")
    print(f"  Target accuracy:          <= {report['target_accuracy_m']:.2f} m")
    print("  " + "-" * 60)

    ar = report["alignment_residual"]
    print("  [1] Alignment residual (FIT DIAGNOSTIC — not accuracy):")
    if "rmse_m" in ar:
        print(f"        RMSE {ar['rmse_m']:.4f} m | mean {ar['mean_error_m']:.4f} m")
    else:
        print(f"        {ar.get('_meaning', '[Not available]')}")

    ho = report["holdout_ate"]
    print("  [2] Hold-out ATE (REAL accuracy signal):")
    if "rmse_m" in ho:
        verdict = "[PASSED]" if ho.get("passed") else "[FAILED]"
        print(f"        RMSE {ho['rmse_m']:.4f} m | mean {ho['mean_error_m']:.4f} m "
              f"| p95 {ho['p95_error_m']:.4f} m  {verdict}")
    else:
        print(f"        {ho.get('_meaning', '[Not available]')}")
    print("  " + "-" * 60)
    print("  NOTE:", report["honesty_caveat"])
    print("=" * 66)
