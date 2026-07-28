from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.signal import find_peaks


DEFAULT_WINDOW_SECONDS = 4.0


def _as_batch_bt(array: Any) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float64)
    if arr.ndim == 1:
        return arr[None, :]
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return arr[..., 0]
    raise ValueError(f"Expected waveform array shaped [T], [B,T], or [B,T,1], got {arr.shape}.")


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 2 or y.size < 2:
        return None
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def infer_window_fs(waveform: np.ndarray, window_seconds: float = DEFAULT_WINDOW_SECONDS) -> float:
    if waveform.size == 0:
        raise ValueError("Cannot infer sample rate from an empty waveform.")
    if window_seconds <= 0.0:
        raise ValueError("window_seconds must be positive.")
    return float(waveform.size / window_seconds)


def detect_reference_beat_segments(
    waveform: np.ndarray,
    *,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    min_peak_distance_seconds: float = 0.30,
    prominence_scale: float = 0.10,
) -> list[tuple[int, int, int]]:
    x = np.asarray(waveform, dtype=np.float64).reshape(-1)
    if x.size < 8 or not np.isfinite(x).all():
        return []

    fs = infer_window_fs(x, window_seconds=window_seconds)
    min_distance = max(1, int(round(min_peak_distance_seconds * fs)))
    prominence = max(1e-6, float(np.nanstd(x) * prominence_scale))
    peaks, _ = find_peaks(x, distance=min_distance, prominence=prominence)

    if peaks.size == 0:
        return []

    boundaries = [0]
    for left, right in zip(peaks[:-1], peaks[1:]):
        boundaries.append(int((left + right) // 2))
    boundaries.append(int(x.size))

    segments: list[tuple[int, int, int]] = []
    for idx, peak in enumerate(peaks):
        start = int(boundaries[idx])
        end = int(boundaries[idx + 1])
        if end - start < 3:
            continue
        if peak < start or peak >= end:
            continue
        segments.append((start, end, int(peak)))
    return segments


def beat_statistics_from_segments(
    waveform: np.ndarray,
    segments: list[tuple[int, int, int]],
) -> dict[str, np.ndarray]:
    x = np.asarray(waveform, dtype=np.float64).reshape(-1)
    sbp: list[float] = []
    dbp: list[float] = []
    map_values: list[float] = []
    peak_idx: list[int] = []

    for start, end, peak in segments:
        beat = x[start:end]
        if beat.size == 0 or not np.isfinite(beat).all():
            continue
        peak_local = peak - start
        if peak_local < 0 or peak_local >= beat.size:
            continue
        sbp.append(float(beat[peak_local]))
        dbp.append(float(np.min(beat)))
        map_values.append(float(np.mean(beat)))
        peak_idx.append(int(peak))

    return {
        "sbp": np.asarray(sbp, dtype=np.float64),
        "dbp": np.asarray(dbp, dtype=np.float64),
        "map": np.asarray(map_values, dtype=np.float64),
        "peak_idx": np.asarray(peak_idx, dtype=np.int64),
    }


def bp_from_waveforms(
    waveforms: Any,
    *,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
) -> dict[str, Any]:
    batch = _as_batch_bt(waveforms)
    segments_per_sample: list[list[tuple[int, int, int]]] = []
    sbp_per_sample: list[np.ndarray] = []
    dbp_per_sample: list[np.ndarray] = []
    map_per_sample: list[np.ndarray] = []

    for waveform in batch:
        segments = detect_reference_beat_segments(waveform, window_seconds=window_seconds)
        stats = beat_statistics_from_segments(waveform, segments)
        segments_per_sample.append(segments)
        sbp_per_sample.append(stats["sbp"])
        dbp_per_sample.append(stats["dbp"])
        map_per_sample.append(stats["map"])

    return {
        "waveforms": batch,
        "segments": segments_per_sample,
        "sbp": sbp_per_sample,
        "dbp": dbp_per_sample,
        "map": map_per_sample,
        "window_seconds": float(window_seconds),
    }


def waveform_summary(y_true: Any, y_pred: Any) -> dict[str, float | int | None]:
    true_bt = _as_batch_bt(y_true)
    pred_bt = _as_batch_bt(y_pred)
    if true_bt.shape != pred_bt.shape:
        raise ValueError(f"Shape mismatch: {true_bt.shape} vs {pred_bt.shape}")

    mask = np.isfinite(true_bt) & np.isfinite(pred_bt)
    x = true_bt[mask]
    y = pred_bt[mask]
    if x.size == 0:
        return {
            "n_points": 0,
            "waveform_mae": None,
            "waveform_rmse": None,
            "waveform_corr": None,
        }

    return {
        "n_points": int(x.size),
        "waveform_mae": float(np.mean(np.abs(y - x))),
        "waveform_rmse": float(np.sqrt(np.mean((y - x) ** 2))),
        "waveform_corr": _safe_corr(x, y),
    }


def bland_altman_summary(errors: np.ndarray) -> dict[str, float | None]:
    err = np.asarray(errors, dtype=np.float64).reshape(-1)
    err = err[np.isfinite(err)]
    if err.size == 0:
        return {"bias": None, "std": None, "loa_low": None, "loa_high": None}
    bias = float(np.mean(err))
    std = float(np.std(err))
    return {
        "bias": bias,
        "std": std,
        "loa_low": float(bias - 1.96 * std),
        "loa_high": float(bias + 1.96 * std),
    }


def aami_summary(errors: np.ndarray) -> dict[str, float | bool | None]:
    err = np.asarray(errors, dtype=np.float64).reshape(-1)
    err = err[np.isfinite(err)]
    if err.size == 0:
        return {"mean_error": None, "std_error": None, "compliant": None}
    mean_error = float(np.mean(err))
    std_error = float(np.std(err))
    compliant = abs(mean_error) <= 5.0 and std_error <= 8.0
    return {
        "mean_error": mean_error,
        "std_error": std_error,
        "compliant": bool(compliant),
    }


def bhs_grade(errors: np.ndarray) -> dict[str, float | str | None]:
    err = np.asarray(errors, dtype=np.float64).reshape(-1)
    err = np.abs(err[np.isfinite(err)])
    if err.size == 0:
        return {
            "within_5": None,
            "within_10": None,
            "within_15": None,
            "grade": None,
        }

    within_5 = float(np.mean(err <= 5.0) * 100.0)
    within_10 = float(np.mean(err <= 10.0) * 100.0)
    within_15 = float(np.mean(err <= 15.0) * 100.0)

    if within_5 >= 60.0 and within_10 >= 85.0 and within_15 >= 95.0:
        grade = "A"
    elif within_5 >= 50.0 and within_10 >= 75.0 and within_15 >= 90.0:
        grade = "B"
    elif within_5 >= 40.0 and within_10 >= 65.0 and within_15 >= 85.0:
        grade = "C"
    else:
        grade = "D"

    return {
        "within_5": within_5,
        "within_10": within_10,
        "within_15": within_15,
        "grade": grade,
    }


def scalar_metric_summary(errors: np.ndarray) -> dict[str, Any]:
    err = np.asarray(errors, dtype=np.float64).reshape(-1)
    err = err[np.isfinite(err)]
    if err.size == 0:
        return {
            "n_beats": 0,
            "mae": None,
            "rmse": None,
            "bias": None,
            "std": None,
            "loa_low": None,
            "loa_high": None,
            "aami_mean_error": None,
            "aami_std_error": None,
            "aami_compliant": None,
            "bhs_within_5": None,
            "bhs_within_10": None,
            "bhs_within_15": None,
            "bhs_grade": None,
        }

    ba = bland_altman_summary(err)
    aami = aami_summary(err)
    bhs = bhs_grade(err)
    return {
        "n_beats": int(err.size),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "bias": ba["bias"],
        "std": ba["std"],
        "loa_low": ba["loa_low"],
        "loa_high": ba["loa_high"],
        "aami_mean_error": aami["mean_error"],
        "aami_std_error": aami["std_error"],
        "aami_compliant": aami["compliant"],
        "bhs_within_5": bhs["within_5"],
        "bhs_within_10": bhs["within_10"],
        "bhs_within_15": bhs["within_15"],
        "bhs_grade": bhs["grade"],
    }
