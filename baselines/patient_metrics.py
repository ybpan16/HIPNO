from __future__ import annotations

import csv
import gzip
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .metrics import beat_statistics_from_segments, detect_reference_beat_segments, scalar_metric_summary


SUPPORTED_METHODS = ("auto", "inn_par", "ppg2abp", "unicardio", "ptt_based", "papagei", "operator")
DEFAULT_SHIFT_MS = 250.0
DEFAULT_BOOTSTRAP_SAMPLES = 2000
DEFAULT_BOOTSTRAP_SEED = 2025
DEFAULT_CHUNK_SIZE = 2048
DEFAULT_QC_MIN_ABP = 20.0
DEFAULT_QC_MAX_ABP = 250.0
DEFAULT_QC_MIN_STD = 1.0
DEFAULT_QC_MIN_RANGE = 10.0
DEFAULT_QC_MAX_RANGE = 200.0

PATIENT_TABLE_COLUMNS = [
    "method",
    "MAE",
    "MAE ci_low",
    "MAE ci_high",
    "MAE [95% CI]",
    "RMSE",
    "RMSE ci_low",
    "RMSE ci_high",
    "RMSE [95% CI]",
    "Min-MAE",
    "Min-MAE ci_low",
    "Min-MAE ci_high",
    "Min-MAE [95% CI]",
    "Min-RMSE",
    "Min-RMSE ci_low",
    "Min-RMSE ci_high",
    "Min-RMSE [95% CI]",
    "NRMSE",
    "NRMSE ci_low",
    "NRMSE ci_high",
    "NRMSE [95% CI]",
    "Corr",
    "Corr ci_low",
    "Corr ci_high",
    "Corr [95% CI]",
    "SBP err",
    "SBP err ci_low",
    "SBP err ci_high",
    "SBP err [95% CI]",
    "DBP err",
    "DBP err ci_low",
    "DBP err ci_high",
    "DBP err [95% CI]",
    "MAP err",
    "MAP err ci_low",
    "MAP err ci_high",
    "MAP err [95% CI]",
    "SBP ME",
    "SBP error SD",
    "SBP ME +/- error SD",
    "DBP ME",
    "DBP error SD",
    "DBP ME +/- error SD",
    "MAP ME",
    "MAP error SD",
    "MAP ME +/- error SD",
    "n_patients",
    "metrics_json",
]

PATIENT_TABLE_CI_METRICS = [
    ("MAE", "waveform", "mae_mmHg", 2),
    ("RMSE", "waveform", "rmse_mmHg", 2),
    ("Min-MAE", "waveform", "min_mae_mmHg", 2),
    ("Min-RMSE", "waveform", "min_rmse_mmHg", 2),
    ("NRMSE", "waveform", "nrmse", 3),
    ("Corr", "waveform", "corr", 3),
    ("SBP err", "scalar", "sbp_mae_mmHg", 2),
    ("DBP err", "scalar", "dbp_mae_mmHg", 2),
    ("MAP err", "scalar", "map_mae_mmHg", 2),
]

PATIENT_TABLE_SIGNED_SCALARS = [
    ("SBP", "sbp", 2),
    ("DBP", "dbp", 2),
    ("MAP", "map", 2),
]

SEED_AGGREGATE_METRICS = [
    ("MAE", ("patient_summary", "waveform", "mae_mmHg", "mean"), 2),
    ("RMSE", ("patient_summary", "waveform", "rmse_mmHg", "mean"), 2),
    ("Min-MAE", ("patient_summary", "waveform", "min_mae_mmHg", "mean"), 2),
    ("Min-RMSE", ("patient_summary", "waveform", "min_rmse_mmHg", "mean"), 2),
    ("NRMSE", ("patient_summary", "waveform", "nrmse", "mean"), 3),
    ("Corr", ("patient_summary", "waveform", "corr", "mean"), 3),
    ("SBP err", ("patient_summary", "scalar", "sbp_mae_mmHg", "mean"), 2),
    ("DBP err", ("patient_summary", "scalar", "dbp_mae_mmHg", "mean"), 2),
    ("MAP err", ("patient_summary", "scalar", "map_mae_mmHg", "mean"), 2),
    ("SBP ME", ("pooled_scalar", "sbp", "bias"), 2),
    ("SBP error SD", ("pooled_scalar", "sbp", "std"), 2),
    ("DBP ME", ("pooled_scalar", "dbp", "bias"), 2),
    ("DBP error SD", ("pooled_scalar", "dbp", "std"), 2),
    ("MAP ME", ("pooled_scalar", "map", "bias"), 2),
    ("MAP error SD", ("pooled_scalar", "map", "std"), 2),
]

SEED_AGGREGATE_TABLE_COLUMNS = [
    "method",
    "metric",
    "estimate_mean",
    "seed_sd",
    "estimate +/- seed SD",
    "n_seeds",
    "metric_path",
    "seed_values_json",
    "metrics_jsons",
]


def _to_abs_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _as_batch_time(array: Any) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 1:
        return arr[None, :]
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return arr[..., 0]
    raise ValueError(f"Expected waveform array shaped [T], [B,T], or [B,T,1], got {arr.shape}.")


def _finite_values(values: Iterable[float | None]) -> np.ndarray:
    arr = np.asarray([value for value in values if value is not None], dtype=np.float64)
    return arr[np.isfinite(arr)]


def _finite_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return result


def _csv_number(value: Any) -> float | str:
    result = _finite_float_or_none(value)
    return "" if result is None else result


def _format_number(value: Any, digits: int) -> str:
    result = _finite_float_or_none(value)
    return "" if result is None else f"{result:.{int(digits)}f}"


def _format_ci(mean: Any, low: Any, high: Any, digits: int) -> str:
    if any(_finite_float_or_none(value) is None for value in (mean, low, high)):
        return ""
    return (
        f"{_format_number(mean, digits)} "
        f"[{_format_number(low, digits)}, {_format_number(high, digits)}]"
    )


def _format_mean_sd(mean: Any, sd: Any, digits: int) -> str:
    if _finite_float_or_none(mean) is None or _finite_float_or_none(sd) is None:
        return ""
    return f"{_format_number(mean, digits)} +/- {_format_number(sd, digits)}"


def _nested_get(mapping: dict[str, Any], path: Iterable[str]) -> Any:
    value: Any = mapping
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _bootstrap_ci(values: np.ndarray, *, n_bootstrap: int, seed: int) -> tuple[float | None, float | None]:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None, None
    if x.size == 1 or n_bootstrap <= 0:
        value = float(x[0] if x.size == 1 else np.mean(x))
        return value, value
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, x.size, size=(int(n_bootstrap), x.size))
    means = np.mean(x[sample_indices], axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def _distribution_summary(
    values: Iterable[float | None],
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    x = _finite_values(values)
    if x.size == 0:
        return {
            "n_patients": 0,
            "mean": None,
            "bootstrap_ci_low": None,
            "bootstrap_ci_high": None,
            "median": None,
            "iqr_low": None,
            "iqr_high": None,
        }
    ci_low, ci_high = _bootstrap_ci(x, n_bootstrap=n_bootstrap, seed=seed)
    q25, median, q75 = np.percentile(x, [25, 50, 75])
    return {
        "n_patients": int(x.size),
        "mean": float(np.mean(x)),
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "median": float(median),
        "iqr_low": float(q25),
        "iqr_high": float(q75),
    }


@dataclass
class ScalarRunning:
    n: int = 0
    sum_abs: float = 0.0
    sum_sq: float = 0.0
    sum_err: float = 0.0
    sum_err2: float = 0.0
    within_5: int = 0
    within_10: int = 0
    within_15: int = 0

    def update(self, errors: Any) -> None:
        err = np.asarray(errors, dtype=np.float64).reshape(-1)
        err = err[np.isfinite(err)]
        if err.size == 0:
            return
        abs_err = np.abs(err)
        self.n += int(err.size)
        self.sum_abs += float(np.sum(abs_err))
        self.sum_sq += float(np.sum(err**2))
        self.sum_err += float(np.sum(err))
        self.sum_err2 += float(np.sum(err**2))
        self.within_5 += int(np.sum(abs_err <= 5.0))
        self.within_10 += int(np.sum(abs_err <= 10.0))
        self.within_15 += int(np.sum(abs_err <= 15.0))

    def merge(self, other: "ScalarRunning") -> None:
        self.n += other.n
        self.sum_abs += other.sum_abs
        self.sum_sq += other.sum_sq
        self.sum_err += other.sum_err
        self.sum_err2 += other.sum_err2
        self.within_5 += other.within_5
        self.within_10 += other.within_10
        self.within_15 += other.within_15

    def finalize(self) -> dict[str, Any]:
        if self.n == 0:
            return scalar_metric_summary(np.asarray([], dtype=np.float64))
        n = float(self.n)
        bias = self.sum_err / n
        variance = max(0.0, (self.sum_err2 / n) - bias**2)
        std = float(np.sqrt(variance))
        within_5 = float(self.within_5 / n * 100.0)
        within_10 = float(self.within_10 / n * 100.0)
        within_15 = float(self.within_15 / n * 100.0)
        if within_5 >= 60.0 and within_10 >= 85.0 and within_15 >= 95.0:
            grade = "A"
        elif within_5 >= 50.0 and within_10 >= 75.0 and within_15 >= 90.0:
            grade = "B"
        elif within_5 >= 40.0 and within_10 >= 65.0 and within_15 >= 85.0:
            grade = "C"
        else:
            grade = "D"
        return {
            "n_beats": int(self.n),
            "mae": float(self.sum_abs / n),
            "rmse": float(np.sqrt(self.sum_sq / n)),
            "bias": float(bias),
            "std": std,
            "loa_low": float(bias - 1.96 * std),
            "loa_high": float(bias + 1.96 * std),
            "aami_mean_error": float(bias),
            "aami_std_error": std,
            "aami_compliant": bool(abs(bias) <= 5.0 and std <= 8.0),
            "bhs_within_5": within_5,
            "bhs_within_10": within_10,
            "bhs_within_15": within_15,
            "bhs_grade": grade,
        }


@dataclass
class PatientAccumulator:
    subject_id: str
    n_windows: int = 0
    n_points: int = 0
    sum_abs: float = 0.0
    sum_sq: float = 0.0
    sum_x: float = 0.0
    sum_y: float = 0.0
    sum_x2: float = 0.0
    sum_y2: float = 0.0
    sum_xy: float = 0.0
    min_mae_sum: float = 0.0
    min_rmse_sum: float = 0.0
    min_metric_windows: int = 0
    true_p01: float | None = None
    true_p99: float | None = None
    scalar: dict[str, ScalarRunning] = field(
        default_factory=lambda: {
            "sbp": ScalarRunning(),
            "dbp": ScalarRunning(),
            "map": ScalarRunning(),
        }
    )

    def update_waveform(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        true_bt = _as_batch_time(y_true).astype(np.float64, copy=False)
        pred_bt = _as_batch_time(y_pred).astype(np.float64, copy=False)
        if true_bt.shape != pred_bt.shape:
            raise ValueError(f"Shape mismatch: {true_bt.shape} vs {pred_bt.shape}.")
        mask = np.isfinite(true_bt) & np.isfinite(pred_bt)
        x = true_bt[mask]
        y = pred_bt[mask]
        if x.size == 0:
            return
        diff = y - x
        self.n_windows += int(true_bt.shape[0])
        self.n_points += int(x.size)
        self.sum_abs += float(np.sum(np.abs(diff)))
        self.sum_sq += float(np.sum(diff**2))
        self.sum_x += float(np.sum(x))
        self.sum_y += float(np.sum(y))
        self.sum_x2 += float(np.sum(x**2))
        self.sum_y2 += float(np.sum(y**2))
        self.sum_xy += float(np.sum(x * y))

    def update_min_shift(self, min_mae: np.ndarray, min_rmse: np.ndarray) -> None:
        mae = np.asarray(min_mae, dtype=np.float64).reshape(-1)
        rmse = np.asarray(min_rmse, dtype=np.float64).reshape(-1)
        finite = np.isfinite(mae) & np.isfinite(rmse)
        if not np.any(finite):
            return
        self.min_mae_sum += float(np.sum(mae[finite]))
        self.min_rmse_sum += float(np.sum(rmse[finite]))
        self.min_metric_windows += int(np.sum(finite))

    def update_scalar(self, metric: str, errors: Any) -> None:
        self.scalar[metric].update(errors)

    def finalize(self) -> dict[str, Any]:
        if self.n_points == 0:
            waveform = {
                "n_windows": int(self.n_windows),
                "n_points": 0,
                "mae_mmHg": None,
                "rmse_mmHg": None,
                "corr": None,
                "min_mae_mmHg": None,
                "min_rmse_mmHg": None,
                "nrmse": None,
                "true_p01": self.true_p01,
                "true_p99": self.true_p99,
            }
        else:
            n = float(self.n_points)
            mean_x = self.sum_x / n
            mean_y = self.sum_y / n
            var_x = (self.sum_x2 / n) - mean_x**2
            var_y = (self.sum_y2 / n) - mean_y**2
            cov_xy = (self.sum_xy / n) - mean_x * mean_y
            corr = None if var_x <= 1e-12 or var_y <= 1e-12 else float(cov_xy / np.sqrt(var_x * var_y))
            rmse = float(np.sqrt(self.sum_sq / n))
            range_1_99 = None
            nrmse = None
            if self.true_p01 is not None and self.true_p99 is not None:
                range_1_99 = float(self.true_p99 - self.true_p01)
                if np.isfinite(range_1_99) and range_1_99 > 1e-12:
                    nrmse = float(rmse / range_1_99)
            waveform = {
                "n_windows": int(self.n_windows),
                "n_points": int(self.n_points),
                "mae_mmHg": float(self.sum_abs / n),
                "rmse_mmHg": rmse,
                "corr": corr,
                "min_mae_mmHg": (
                    None if self.min_metric_windows == 0 else float(self.min_mae_sum / self.min_metric_windows)
                ),
                "min_rmse_mmHg": (
                    None if self.min_metric_windows == 0 else float(self.min_rmse_sum / self.min_metric_windows)
                ),
                "nrmse": nrmse,
                "true_p01": self.true_p01,
                "true_p99": self.true_p99,
                "true_p99_minus_p01": range_1_99,
            }

        scalar = {metric: accumulator.finalize() for metric, accumulator in self.scalar.items()}
        return {
            "subject_id": self.subject_id,
            "waveform": waveform,
            "scalar": scalar,
        }


@dataclass
class WaveformRunData:
    method: str
    run_dir: Path
    y_true: Any
    y_pred: Any
    metadata: pd.DataFrame
    window_seconds: float
    predictions_ref: str
    true_scale: float = 1.0
    true_offset: float = 0.0
    pred_scale: float = 1.0
    pred_offset: float = 0.0

    @property
    def n_windows(self) -> int:
        return int(self.y_true.shape[0])

    @property
    def signal_length(self) -> int:
        return int(_as_batch_time(self.y_true[:1]).shape[1])

    def true_chunk(self, start: int, end: int) -> np.ndarray:
        return _as_batch_time(self.y_true[start:end]) * self.true_scale + self.true_offset

    def pred_chunk(self, start: int, end: int) -> np.ndarray:
        return _as_batch_time(self.y_pred[start:end]) * self.pred_scale + self.pred_offset

    def true_indices(self, indices: np.ndarray) -> np.ndarray:
        return _as_batch_time(self.y_true[indices]) * self.true_scale + self.true_offset


@dataclass(frozen=True)
class ReferenceQCConfig:
    min_abp: float = DEFAULT_QC_MIN_ABP
    max_abp: float = DEFAULT_QC_MAX_ABP
    min_std: float = DEFAULT_QC_MIN_STD
    min_range: float = DEFAULT_QC_MIN_RANGE
    max_range: float = DEFAULT_QC_MAX_RANGE

    def __post_init__(self) -> None:
        if self.min_abp >= self.max_abp:
            raise ValueError("reference QC requires min_abp < max_abp.")
        if self.min_std < 0.0:
            raise ValueError("reference QC requires min_std >= 0.")
        if self.min_range < 0.0 or self.max_range <= 0.0:
            raise ValueError("reference QC requires non-negative min_range and positive max_range.")
        if self.min_range >= self.max_range:
            raise ValueError("reference QC requires min_range < max_range.")


def _qc_config_dict(config: ReferenceQCConfig) -> dict[str, float]:
    return {
        "min_abp": float(config.min_abp),
        "max_abp": float(config.max_abp),
        "min_std": float(config.min_std),
        "min_range": float(config.min_range),
        "max_range": float(config.max_range),
    }


def _reference_qc_for_waveform_chunk(y_true: np.ndarray, config: ReferenceQCConfig) -> tuple[np.ndarray, dict[str, int]]:
    true_bt = _as_batch_time(y_true).astype(np.float64, copy=False)
    batch = int(true_bt.shape[0])
    finite = np.isfinite(true_bt).all(axis=1)
    min_values = np.full(batch, np.nan, dtype=np.float64)
    max_values = np.full(batch, np.nan, dtype=np.float64)
    std_values = np.full(batch, np.nan, dtype=np.float64)
    range_values = np.full(batch, np.nan, dtype=np.float64)

    if np.any(finite):
        finite_values = true_bt[finite]
        min_values[finite] = np.min(finite_values, axis=1)
        max_values[finite] = np.max(finite_values, axis=1)
        std_values[finite] = np.std(finite_values, axis=1)
        range_values[finite] = max_values[finite] - min_values[finite]

    low = min_values < float(config.min_abp)
    high = max_values > float(config.max_abp)
    flat = std_values < float(config.min_std)
    tiny_range = range_values < float(config.min_range)
    huge_range = range_values > float(config.max_range)
    valid = finite & ~low & ~high & ~flat & ~tiny_range & ~huge_range
    return valid, {
        "nonfinite": int(np.sum(~finite)),
        "below_min_abp": int(np.sum(low)),
        "above_max_abp": int(np.sum(high)),
        "below_min_std": int(np.sum(flat)),
        "below_min_range": int(np.sum(tiny_range)),
        "above_max_range": int(np.sum(huge_range)),
        "valid": int(np.sum(valid)),
        "invalid": int(batch - np.sum(valid)),
    }


def _compute_reference_qc_mask(
    run: WaveformRunData,
    *,
    config: ReferenceQCConfig,
    chunk_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    mask = np.zeros(int(run.n_windows), dtype=bool)
    reason_counts = {
        "nonfinite": 0,
        "below_min_abp": 0,
        "above_max_abp": 0,
        "below_min_std": 0,
        "below_min_range": 0,
        "above_max_range": 0,
    }
    for start in range(0, run.n_windows, int(chunk_size)):
        end = min(start + int(chunk_size), run.n_windows)
        valid, counts = _reference_qc_for_waveform_chunk(run.true_chunk(start, end), config)
        mask[start:end] = valid
        for key in reason_counts:
            reason_counts[key] += int(counts[key])

    by_subject = pd.DataFrame(
        {
            "subject_id": run.metadata["subject_id"].astype(str).to_numpy(),
            "valid": mask,
        }
    )
    per_subject = by_subject.groupby("subject_id", sort=False)["valid"].agg(["size", "sum"])
    total = int(run.n_windows)
    valid_count = int(np.sum(mask))
    return mask, {
        "enabled": True,
        "selection_basis": "reference ABP waveform only; predictions are not used",
        "thresholds": _qc_config_dict(config),
        "n_windows_total": total,
        "n_windows_valid": valid_count,
        "n_windows_excluded": int(total - valid_count),
        "valid_fraction": None if total == 0 else float(valid_count / total),
        "reason_counts": reason_counts,
        "n_patients_total": int(per_subject.shape[0]),
        "n_patients_with_valid_windows": int(np.sum(per_subject["sum"].to_numpy(dtype=np.int64) > 0)),
        "per_patient_valid_window_summary": {
            "min": int(per_subject["sum"].min()) if not per_subject.empty else 0,
            "median": float(per_subject["sum"].median()) if not per_subject.empty else 0.0,
            "max": int(per_subject["sum"].max()) if not per_subject.empty else 0,
        },
    }


def _detect_method(run_dir: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    prepared = run_dir / "prepared_data"
    if (prepared / "inn_par_benchmark_config.json").exists():
        return "inn_par"
    if (prepared / "ppg2abp_benchmark_config.json").exists() or (prepared / "meta_benchmark.p").exists():
        return "ppg2abp"
    if (prepared / "unicardio_benchmark_config.json").exists():
        return "unicardio"
    if (run_dir / "predictions" / "test_scalar_predictions.csv.gz").exists():
        return "ptt_based"
    if (prepared / "papagei_benchmark_config.json").exists():
        return "papagei"
    if (run_dir / "predictions" / "test_predictions_manifest.json").exists() and (
        prepared / "windows_manifest.csv"
    ).exists():
        return "operator"
    raise FileNotFoundError(f"Could not infer baseline method from run directory: {run_dir}")


def _load_prediction_manifest(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "predictions" / "test_predictions_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing prediction manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _load_test_metadata(
    run_dir: Path,
    *,
    method: str,
    n_windows: int,
    prediction_manifest: dict[str, Any] | None = None,
) -> pd.DataFrame:
    manifest_path = run_dir / "prepared_data" / "windows_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing windows manifest for patient-level waveform metrics: {manifest_path}")
    meta = pd.read_csv(manifest_path)
    required = {"split", "subject_id", "memmap_index"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"Windows manifest is missing required columns: {sorted(missing)}")
    test = meta[meta["split"].astype(str) == "test"].copy()
    if method == "unicardio":
        start_index = 0
        if prediction_manifest is not None:
            start_index = int(prediction_manifest.get("window_start", 0))
        stop_index = start_index + int(n_windows)
        test = test[
            (test["memmap_index"].astype(int) >= start_index)
            & (test["memmap_index"].astype(int) < stop_index)
        ].copy()
    test = test.sort_values("memmap_index").reset_index(drop=True)
    if len(test) != int(n_windows):
        raise ValueError(
            "Prediction count does not match test windows metadata: "
            f"{n_windows} predictions vs {len(test)} metadata rows for method={method}."
        )
    return test


def _load_inn_par_run(run_dir: Path) -> WaveformRunData:
    config_path = run_dir / "prepared_data" / "inn_par_benchmark_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing INN-PAR prepared config: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = _load_prediction_manifest(run_dir)
    y_true = np.load(manifest["y_true"], mmap_mode="r")
    y_pred = np.load(manifest["y_pred"], mmap_mode="r")
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Prediction shape mismatch: {y_true.shape} vs {y_pred.shape}")
    metadata = _load_test_metadata(run_dir, method="inn_par", n_windows=int(y_true.shape[0]))
    return WaveformRunData(
        method="inn_par",
        run_dir=run_dir,
        y_true=y_true,
        y_pred=y_pred,
        metadata=metadata,
        window_seconds=float(config["window_sec"]),
        predictions_ref=str(run_dir / "predictions" / "test_predictions_manifest.json"),
    )


def _load_unicardio_run(run_dir: Path) -> WaveformRunData:
    config_path = run_dir / "prepared_data" / "unicardio_benchmark_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing UniCardio prepared config: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = _load_prediction_manifest(run_dir)
    y_true = np.load(manifest["y_true"], mmap_mode="r")
    y_pred = np.load(manifest["y_pred"], mmap_mode="r")
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Prediction shape mismatch: {y_true.shape} vs {y_pred.shape}")
    metadata = _load_test_metadata(
        run_dir,
        method="unicardio",
        n_windows=int(y_true.shape[0]),
        prediction_manifest=manifest,
    )
    return WaveformRunData(
        method="unicardio",
        run_dir=run_dir,
        y_true=y_true,
        y_pred=y_pred,
        metadata=metadata,
        window_seconds=float(config["window_sec"]),
        predictions_ref=str(run_dir / "predictions" / "test_predictions_manifest.json"),
    )


def _load_ppg2abp_run(run_dir: Path) -> WaveformRunData:
    from .ppg2abp_launcher import _load_ppg2abp_prediction_array, _load_prepared_ppg2abp_data

    _, _, test, meta = _load_prepared_ppg2abp_data(run_dir)
    y_true = test["Y_test"]
    y_pred = _load_ppg2abp_prediction_array(run_dir)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Prediction shape mismatch: {y_true.shape} vs {y_pred.shape}")
    metadata = _load_test_metadata(run_dir, method="ppg2abp", n_windows=int(y_true.shape[0]))
    abp_span = float(meta["max_abp"] - meta["min_abp"])
    abp_min = float(meta["min_abp"])
    return WaveformRunData(
        method="ppg2abp",
        run_dir=run_dir,
        y_true=y_true,
        y_pred=y_pred,
        metadata=metadata,
        window_seconds=float(meta["window_seconds_for_model"]),
        predictions_ref=str(run_dir / "predictions" / "test_predictions_manifest.json"),
        true_scale=abp_span,
        true_offset=abp_min,
        pred_scale=abp_span,
        pred_offset=abp_min,
    )


def _load_operator_run(run_dir: Path) -> WaveformRunData:
    manifest = _load_prediction_manifest(run_dir)
    y_true = np.load(manifest["y_true"], mmap_mode="r")
    y_pred = np.load(manifest["y_pred"], mmap_mode="r")
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Prediction shape mismatch: {y_true.shape} vs {y_pred.shape}")

    window_seconds = manifest.get("window_seconds", manifest.get("window_sec"))
    if window_seconds is None:
        for config_name in ("operator_benchmark_config.json", "generic_waveform_config.json"):
            config_path = run_dir / "prepared_data" / config_name
            if config_path.exists():
                config = json.loads(config_path.read_text(encoding="utf-8"))
                window_seconds = config.get("window_seconds", config.get("window_sec"))
                break
    if window_seconds is None:
        raise FileNotFoundError(
            "Operator/generic waveform runs must provide window_seconds in "
            "predictions/test_predictions_manifest.json or prepared_data/operator_benchmark_config.json."
        )

    metadata = _load_test_metadata(run_dir, method="operator", n_windows=int(y_true.shape[0]))
    return WaveformRunData(
        method="operator",
        run_dir=run_dir,
        y_true=y_true,
        y_pred=y_pred,
        metadata=metadata,
        window_seconds=float(window_seconds),
        predictions_ref=str(run_dir / "predictions" / "test_predictions_manifest.json"),
    )


def _load_waveform_run(run_dir: Path, method: str) -> WaveformRunData:
    if method == "inn_par":
        return _load_inn_par_run(run_dir)
    if method == "ppg2abp":
        return _load_ppg2abp_run(run_dir)
    if method == "unicardio":
        return _load_unicardio_run(run_dir)
    if method == "operator":
        return _load_operator_run(run_dir)
    raise ValueError(f"Method {method!r} does not have waveform predictions.")


def _shifted_error_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    max_shift_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    true_bt = _as_batch_time(y_true).astype(np.float64, copy=False)
    pred_bt = _as_batch_time(y_pred).astype(np.float64, copy=False)
    if true_bt.shape != pred_bt.shape:
        raise ValueError(f"Shape mismatch: {true_bt.shape} vs {pred_bt.shape}")
    batch, length = true_bt.shape
    best_mae = np.full(batch, np.inf, dtype=np.float64)
    best_rmse = np.full(batch, np.inf, dtype=np.float64)

    max_shift = min(int(max_shift_samples), max(0, length - 1))
    for shift in range(-max_shift, max_shift + 1):
        if shift < 0:
            yt = true_bt[:, -shift:]
            yp = pred_bt[:, : length + shift]
        elif shift > 0:
            yt = true_bt[:, : length - shift]
            yp = pred_bt[:, shift:]
        else:
            yt = true_bt
            yp = pred_bt
        if yt.shape[1] == 0:
            continue
        diff = yp - yt
        finite = np.isfinite(diff)
        counts = np.sum(finite, axis=1)
        valid = counts > 0
        if not np.any(valid):
            continue
        abs_sum = np.sum(np.where(finite, np.abs(diff), 0.0), axis=1)
        sq_sum = np.sum(np.where(finite, diff**2, 0.0), axis=1)
        mae = np.full(batch, np.inf, dtype=np.float64)
        rmse = np.full(batch, np.inf, dtype=np.float64)
        mae[valid] = abs_sum[valid] / counts[valid]
        rmse[valid] = np.sqrt(sq_sum[valid] / counts[valid])
        best_mae = np.minimum(best_mae, mae)
        best_rmse = np.minimum(best_rmse, rmse)

    best_mae[~np.isfinite(best_mae)] = np.nan
    best_rmse[~np.isfinite(best_rmse)] = np.nan
    return best_mae, best_rmse


def _update_scalar_from_waveforms(
    accumulators: dict[str, PatientAccumulator],
    subjects: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    window_seconds: float,
) -> None:
    true_bt = _as_batch_time(y_true).astype(np.float64, copy=False)
    pred_bt = _as_batch_time(y_pred).astype(np.float64, copy=False)
    for idx, subject_id in enumerate(subjects):
        segments = detect_reference_beat_segments(true_bt[idx], window_seconds=window_seconds)
        if not segments:
            continue
        true_stats = beat_statistics_from_segments(true_bt[idx], segments)
        pred_stats = beat_statistics_from_segments(pred_bt[idx], segments)
        accumulator = accumulators[str(subject_id)]
        for metric in ("sbp", "dbp", "map"):
            true_values = np.asarray(true_stats[metric], dtype=np.float64)
            pred_values = np.asarray(pred_stats[metric], dtype=np.float64)
            n = min(true_values.size, pred_values.size)
            if n == 0:
                continue
            accumulator.update_scalar(metric, pred_values[:n] - true_values[:n])


def _compute_patient_true_ranges(
    run: WaveformRunData,
    accumulators: dict[str, PatientAccumulator],
    *,
    valid_window_mask: np.ndarray | None = None,
) -> None:
    grouped_indices = run.metadata.groupby(run.metadata["subject_id"].astype(str), sort=False).indices
    for subject_id, positions in grouped_indices.items():
        indices = np.asarray(positions, dtype=np.int64)
        if valid_window_mask is not None:
            indices = indices[valid_window_mask[indices]]
        if indices.size == 0:
            continue
        values = run.true_indices(indices).reshape(-1)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        p01, p99 = np.percentile(values, [1, 99])
        accumulators[str(subject_id)].true_p01 = float(p01)
        accumulators[str(subject_id)].true_p99 = float(p99)


def _summarize_patient_records(
    patient_records: list[dict[str, Any]],
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    waveform_metrics = {
        "mae_mmHg": "mae_mmHg",
        "rmse_mmHg": "rmse_mmHg",
        "min_mae_mmHg": "min_mae_mmHg",
        "min_rmse_mmHg": "min_rmse_mmHg",
        "nrmse": "nrmse",
        "corr": "corr",
    }
    waveform = {
        output_key: _distribution_summary(
            [record["waveform"].get(source_key) for record in patient_records],
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        for output_key, source_key in waveform_metrics.items()
    }
    scalar: dict[str, Any] = {}
    for metric in ("sbp", "dbp", "map"):
        scalar[f"{metric}_mae_mmHg"] = _distribution_summary(
            [record["scalar"][metric].get("mae") for record in patient_records],
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        scalar[f"{metric}_rmse_mmHg"] = _distribution_summary(
            [record["scalar"][metric].get("rmse") for record in patient_records],
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
    return {
        "waveform": waveform,
        "scalar": scalar,
    }


def _pooled_scalar_from_patients(patient_records: list[dict[str, Any]]) -> dict[str, Any]:
    pooled = {
        "sbp": ScalarRunning(),
        "dbp": ScalarRunning(),
        "map": ScalarRunning(),
    }
    for record in patient_records:
        subject_id = str(record["subject_id"])
        if "_accumulator" not in record:
            continue
        accumulator = record["_accumulator"]
        if not isinstance(accumulator, PatientAccumulator):
            raise TypeError(f"Unexpected accumulator payload for subject {subject_id}")
        for metric in ("sbp", "dbp", "map"):
            pooled[metric].merge(accumulator.scalar[metric])
    return {metric: accumulator.finalize() for metric, accumulator in pooled.items()}


def _strip_internal_patient_fields(patient_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned = []
    for record in patient_records:
        item = dict(record)
        item.pop("_accumulator", None)
        cleaned.append(item)
    return cleaned


def _compute_waveform_patient_metrics(
    run: WaveformRunData,
    *,
    shift_ms: float,
    n_bootstrap: int,
    seed: int,
    chunk_size: int,
    compute_scalar: bool,
    progress_every_chunks: int,
    reference_qc: ReferenceQCConfig | None,
) -> dict[str, Any]:
    if run.n_windows != len(run.metadata):
        raise ValueError(f"Metadata length mismatch: {len(run.metadata)} vs {run.n_windows}")

    subjects = run.metadata["subject_id"].astype(str).to_numpy()
    unique_subjects = list(dict.fromkeys(subjects.tolist()))
    accumulators = {subject_id: PatientAccumulator(subject_id=subject_id) for subject_id in unique_subjects}
    max_shift_samples = int(round(float(shift_ms) / 1000.0 * run.signal_length / float(run.window_seconds)))
    valid_window_mask: np.ndarray | None = None
    reference_qc_summary: dict[str, Any] = {"enabled": False}
    if reference_qc is not None:
        valid_window_mask, reference_qc_summary = _compute_reference_qc_mask(
            run,
            config=reference_qc,
            chunk_size=chunk_size,
        )

    chunk_index = 0
    for start in range(0, run.n_windows, int(chunk_size)):
        end = min(start + int(chunk_size), run.n_windows)
        true_chunk = run.true_chunk(start, end)
        pred_chunk = run.pred_chunk(start, end)
        subject_chunk = subjects[start:end]
        if valid_window_mask is not None:
            include = valid_window_mask[start:end]
            if not np.any(include):
                chunk_index += 1
                if progress_every_chunks > 0 and chunk_index % int(progress_every_chunks) == 0:
                    print(
                        f"[patient-metrics] processed {end}/{run.n_windows} windows",
                        file=sys.stderr,
                        flush=True,
                    )
                continue
            true_chunk = true_chunk[include]
            pred_chunk = pred_chunk[include]
            subject_chunk = subject_chunk[include]
        min_mae, min_rmse = _shifted_error_summary(
            true_chunk,
            pred_chunk,
            max_shift_samples=max_shift_samples,
        )
        for subject_id in np.unique(subject_chunk):
            positions = np.flatnonzero(subject_chunk == subject_id)
            accumulator = accumulators[str(subject_id)]
            accumulator.update_waveform(true_chunk[positions], pred_chunk[positions])
            accumulator.update_min_shift(min_mae[positions], min_rmse[positions])
        if compute_scalar:
            _update_scalar_from_waveforms(
                accumulators,
                subject_chunk,
                true_chunk,
                pred_chunk,
                window_seconds=run.window_seconds,
            )
        chunk_index += 1
        if progress_every_chunks > 0 and chunk_index % int(progress_every_chunks) == 0:
            print(
                f"[patient-metrics] processed {end}/{run.n_windows} windows",
                file=sys.stderr,
                flush=True,
            )

    _compute_patient_true_ranges(run, accumulators, valid_window_mask=valid_window_mask)
    patient_records = []
    for accumulator in accumulators.values():
        record = accumulator.finalize()
        record["_accumulator"] = accumulator
        patient_records.append(record)

    patient_summary = _summarize_patient_records(patient_records, n_bootstrap=n_bootstrap, seed=seed)
    pooled_scalar = _pooled_scalar_from_patients(patient_records) if compute_scalar else {}
    patient_records_clean = _strip_internal_patient_fields(patient_records)
    return {
        "patient_summary": patient_summary,
        "pooled_scalar": pooled_scalar,
        "patients": patient_records_clean,
        "n_patients": int(len(patient_records_clean)),
        "n_windows": int(sum(record["waveform"]["n_windows"] for record in patient_records_clean)),
        "n_windows_original": int(run.n_windows),
        "n_points": int(sum(record["waveform"]["n_points"] for record in patient_records_clean)),
        "max_shift_samples": int(max_shift_samples),
        "reference_qc": reference_qc_summary,
    }


def _load_ptt_scalar_predictions(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "predictions" / "test_scalar_predictions.csv.gz"
    if not path.exists():
        raise FileNotFoundError(f"Missing PTT scalar predictions: {path}")
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return pd.read_csv(handle)


def _load_papagei_scalar_predictions(run_dir: Path) -> pd.DataFrame:
    predictions_path = run_dir / "predictions" / "test_scalar_predictions.csv"
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing PaPaGei scalar predictions: {predictions_path}")
    pred = pd.read_csv(predictions_path)
    labels_path = run_dir / "prepared_data" / "case_labels.csv"
    if labels_path.exists() and "subject_id" not in pred.columns:
        labels = pd.read_csv(labels_path)
        columns = ["case_key", "subject_id", "record_id", "source"]
        pred = pred.merge(labels[[column for column in columns if column in labels.columns]], on="case_key", how="left")
    if "subject_id" not in pred.columns:
        pred["subject_id"] = pred["case_key"].astype(str)
    return pred


def _filter_scalar_reference_qc(
    frame: pd.DataFrame,
    *,
    config: ReferenceQCConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {"sbp_true", "dbp_true", "map_true"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Scalar prediction file is missing required QC columns: {sorted(missing)}")

    sbp = frame["sbp_true"].to_numpy(dtype=np.float64)
    dbp = frame["dbp_true"].to_numpy(dtype=np.float64)
    map_values = frame["map_true"].to_numpy(dtype=np.float64)
    pulse_pressure = sbp - dbp
    finite = np.isfinite(sbp) & np.isfinite(dbp) & np.isfinite(map_values)
    low = dbp < float(config.min_abp)
    high = sbp > float(config.max_abp)
    map_low = map_values < float(config.min_abp)
    map_high = map_values > float(config.max_abp)
    tiny_range = pulse_pressure < float(config.min_range)
    huge_range = pulse_pressure > float(config.max_range)
    valid = finite & ~low & ~high & ~map_low & ~map_high & ~tiny_range & ~huge_range
    total = int(len(frame))
    valid_count = int(np.sum(valid))
    return frame.loc[valid].copy(), {
        "enabled": True,
        "selection_basis": "reference scalar BP values only; predictions are not used",
        "thresholds": _qc_config_dict(config),
        "n_scalar_total": total,
        "n_scalar_valid": valid_count,
        "n_scalar_excluded": int(total - valid_count),
        "valid_fraction": None if total == 0 else float(valid_count / total),
        "reason_counts": {
            "nonfinite": int(np.sum(~finite)),
            "below_min_abp": int(np.sum(low | map_low)),
            "above_max_abp": int(np.sum(high | map_high)),
            "below_min_range": int(np.sum(tiny_range)),
            "above_max_range": int(np.sum(huge_range)),
        },
    }


def _compute_scalar_patient_metrics(
    frame: pd.DataFrame,
    *,
    n_bootstrap: int,
    seed: int,
    reference_qc: ReferenceQCConfig | None,
) -> dict[str, Any]:
    required = {"subject_id", "sbp_true", "dbp_true", "map_true", "sbp_pred", "dbp_pred", "map_pred"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Scalar prediction file is missing required columns: {sorted(missing)}")
    n_scalar_original = int(len(frame))
    reference_qc_summary: dict[str, Any] = {"enabled": False}
    if reference_qc is not None:
        frame, reference_qc_summary = _filter_scalar_reference_qc(frame, config=reference_qc)

    patient_records: list[dict[str, Any]] = []
    pooled = {
        "sbp": ScalarRunning(),
        "dbp": ScalarRunning(),
        "map": ScalarRunning(),
    }
    for subject_id, group in frame.groupby(frame["subject_id"].astype(str), sort=False):
        scalar_records = {}
        for metric in ("sbp", "dbp", "map"):
            errors = group[f"{metric}_pred"].to_numpy(dtype=np.float64) - group[f"{metric}_true"].to_numpy(
                dtype=np.float64
            )
            accumulator = ScalarRunning()
            accumulator.update(errors)
            pooled[metric].merge(accumulator)
            scalar_records[metric] = accumulator.finalize()
        patient_records.append(
            {
                "subject_id": str(subject_id),
                "waveform": {
                    "n_windows": 0,
                    "n_points": 0,
                    "mae_mmHg": None,
                    "rmse_mmHg": None,
                    "corr": None,
                    "min_mae_mmHg": None,
                    "min_rmse_mmHg": None,
                    "nrmse": None,
                },
                "scalar": scalar_records,
            }
        )

    return {
        "patient_summary": _summarize_patient_records(patient_records, n_bootstrap=n_bootstrap, seed=seed),
        "pooled_scalar": {metric: accumulator.finalize() for metric, accumulator in pooled.items()},
        "patients": patient_records,
        "n_patients": int(len(patient_records)),
        "n_scalar_predictions": int(len(frame)),
        "n_scalar_predictions_original": n_scalar_original,
        "reference_qc": reference_qc_summary,
    }


def _patient_table_row(summary: dict[str, Any], metrics_json: Path) -> dict[str, Any]:
    patient_summary = summary.get("patient_summary", {})
    waveform = patient_summary.get("waveform", {})
    scalar = patient_summary.get("scalar", {})

    n_patients = (
        waveform.get("mae_mmHg", {}).get("n_patients")
        or scalar.get("sbp_mae_mmHg", {}).get("n_patients")
        or summary.get("n_patients", "")
    )
    row: dict[str, Any] = {
        "method": summary.get("method_label") or summary.get("method", ""),
        "n_patients": n_patients,
        "metrics_json": str(metrics_json),
    }
    sections = {
        "waveform": waveform,
        "scalar": scalar,
    }
    for column, section_name, metric_key, digits in PATIENT_TABLE_CI_METRICS:
        metric_summary = sections.get(section_name, {}).get(metric_key, {})
        mean = metric_summary.get("mean")
        ci_low = metric_summary.get("bootstrap_ci_low")
        ci_high = metric_summary.get("bootstrap_ci_high")
        row[column] = _csv_number(mean)
        row[f"{column} ci_low"] = _csv_number(ci_low)
        row[f"{column} ci_high"] = _csv_number(ci_high)
        row[f"{column} [95% CI]"] = _format_ci(mean, ci_low, ci_high, digits)

    pooled_scalar = summary.get("pooled_scalar", {})
    for prefix, metric_key, digits in PATIENT_TABLE_SIGNED_SCALARS:
        metric_summary = pooled_scalar.get(metric_key, {})
        mean_error = metric_summary.get("bias")
        sd_error = metric_summary.get("std")
        row[f"{prefix} ME"] = _csv_number(mean_error)
        row[f"{prefix} error SD"] = _csv_number(sd_error)
        row[f"{prefix} ME +/- error SD"] = _format_mean_sd(mean_error, sd_error, digits)

    return row


def _write_patient_table_row(row: dict[str, Any], output_csv: str | Path, *, append: bool) -> dict[str, Any]:
    output_path = _to_abs_path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing_nonempty = output_path.exists() and output_path.stat().st_size > 0
    mode = "a" if append and existing_nonempty else "w"
    if mode == "a":
        with output_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
        if header != PATIENT_TABLE_COLUMNS:
            raise ValueError(
                f"Existing CSV header does not match the patient metrics table schema: {output_path}. "
                "Use --overwrite-csv or write to a new output path."
            )
    write_header = mode == "w"
    with output_path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PATIENT_TABLE_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in PATIENT_TABLE_COLUMNS})
    return {
        "output_csv": str(output_path),
        "appended": mode == "a",
        "row": row,
    }


def _seed_estimate_summary(values: Iterable[Any], *, digits: int) -> dict[str, Any]:
    x = _finite_values(_finite_float_or_none(value) for value in values)
    if x.size == 0:
        return {
            "n_seeds": 0,
            "estimate_mean": None,
            "seed_sd": None,
            "estimate_plus_minus_seed_sd": "",
            "min": None,
            "max": None,
        }
    mean = float(np.mean(x))
    seed_sd = None if x.size < 2 else float(np.std(x, ddof=1))
    return {
        "n_seeds": int(x.size),
        "estimate_mean": mean,
        "seed_sd": seed_sd,
        "estimate_plus_minus_seed_sd": _format_mean_sd(mean, seed_sd, digits),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def _write_seed_aggregate_rows(rows: list[dict[str, Any]], output_csv: str | Path, *, append: bool) -> dict[str, Any]:
    output_path = _to_abs_path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing_nonempty = output_path.exists() and output_path.stat().st_size > 0
    mode = "a" if append and existing_nonempty else "w"
    if mode == "a":
        with output_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
        if header != SEED_AGGREGATE_TABLE_COLUMNS:
            raise ValueError(
                f"Existing CSV header does not match the seed aggregate schema: {output_path}. "
                "Use --overwrite-csv or write to a new output path."
            )
    write_header = mode == "w"
    with output_path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SEED_AGGREGATE_TABLE_COLUMNS)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in SEED_AGGREGATE_TABLE_COLUMNS})
    return {
        "output_csv": str(output_path),
        "appended": mode == "a",
        "n_rows": int(len(rows)),
    }


def aggregate_patient_metric_seeds(
    metrics_jsons: Iterable[str | Path],
    *,
    seed_labels: Iterable[str] | None = None,
    method_label: str | None = None,
    output_json: str | Path | None = None,
    output_csv: str | Path | None = None,
    overwrite_csv: bool = False,
) -> dict[str, Any]:
    paths = [_to_abs_path(path) for path in metrics_jsons]
    if not paths:
        raise ValueError("At least one metrics JSON is required.")
    labels = None if seed_labels is None else [str(label) for label in seed_labels]
    if labels is not None and len(labels) != len(paths):
        raise ValueError("Number of --seed-label values must match number of --metrics-json values.")

    summaries = []
    for idx, path in enumerate(paths):
        summary = json.loads(path.read_text(encoding="utf-8"))
        seed_label = labels[idx] if labels is not None else str(idx)
        summaries.append(
            {
                "seed_label": seed_label,
                "metrics_json": str(path),
                "method": summary.get("method"),
                "method_label": summary.get("method_label"),
                "run_dir": summary.get("run_dir"),
                "summary": summary,
            }
        )

    if method_label is None:
        method_names = {
            str(item.get("method_label") or item.get("method") or "").strip()
            for item in summaries
            if str(item.get("method_label") or item.get("method") or "").strip()
        }
        if len(method_names) != 1:
            raise ValueError("Pass --method-label when aggregating metrics with missing or non-identical method labels.")
        method_label = next(iter(method_names))

    metric_summaries: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for metric_name, metric_path, digits in SEED_AGGREGATE_METRICS:
        values = [
            {
                "seed_label": item["seed_label"],
                "value": _finite_float_or_none(_nested_get(item["summary"], metric_path)),
                "metrics_json": item["metrics_json"],
            }
            for item in summaries
        ]
        estimate_summary = _seed_estimate_summary((item["value"] for item in values), digits=digits)
        metric_path_text = ".".join(metric_path)
        metric_summaries[metric_name] = {
            "metric_path": metric_path_text,
            **estimate_summary,
            "seed_values": values,
        }
        rows.append(
            {
                "method": method_label,
                "metric": metric_name,
                "estimate_mean": _csv_number(estimate_summary["estimate_mean"]),
                "seed_sd": _csv_number(estimate_summary["seed_sd"]),
                "estimate +/- seed SD": estimate_summary["estimate_plus_minus_seed_sd"],
                "n_seeds": estimate_summary["n_seeds"],
                "metric_path": metric_path_text,
                "seed_values_json": json.dumps(_json_safe(values), sort_keys=True),
                "metrics_jsons": ";".join(str(path) for path in paths),
            }
        )

    result: dict[str, Any] = {
        "method_label": method_label,
        "n_seed_runs": int(len(paths)),
        "protocol": {
            "unit": "one completed run seed",
            "input": "patient-metrics JSON files",
            "estimate_mean": "mean of seed-level point estimates",
            "seed_sd": "sample standard deviation across seed-level point estimates; not the BP error-distribution SD",
            "requires_multiple_seeds_for_sd": True,
        },
        "seed_runs": [
            {
                "seed_label": item["seed_label"],
                "metrics_json": item["metrics_json"],
                "method": item["method"],
                "method_label": item["method_label"],
                "run_dir": item["run_dir"],
            }
            for item in summaries
        ],
        "metrics": metric_summaries,
        "table_rows": rows,
    }

    if output_csv is not None:
        result["table_csv"] = _write_seed_aggregate_rows(rows, output_csv, append=not overwrite_csv)

    if output_json is not None:
        output_json_path = _to_abs_path(output_json)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        result["output_json"] = str(output_json_path)
        output_json_path.write_text(
            json.dumps(_json_safe(result), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    return _json_safe(result)


def compute_patient_metrics(
    run_dir: str | Path,
    *,
    method: str = "auto",
    method_label: str | None = None,
    output_json: str | Path | None = None,
    output_csv: str | Path | None = None,
    overwrite_csv: bool = False,
    shift_ms: float = DEFAULT_SHIFT_MS,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    compute_scalar: bool = True,
    progress_every_chunks: int = 25,
    reference_qc: bool = False,
    qc_min_abp: float = DEFAULT_QC_MIN_ABP,
    qc_max_abp: float = DEFAULT_QC_MAX_ABP,
    qc_min_std: float = DEFAULT_QC_MIN_STD,
    qc_min_range: float = DEFAULT_QC_MIN_RANGE,
    qc_max_range: float = DEFAULT_QC_MAX_RANGE,
) -> dict[str, Any]:
    if float(shift_ms) < 0.0:
        raise ValueError("shift_ms must be non-negative.")
    if int(n_bootstrap) < 0:
        raise ValueError("n_bootstrap must be non-negative.")
    if int(chunk_size) <= 0:
        raise ValueError("chunk_size must be positive.")
    if int(progress_every_chunks) < 0:
        raise ValueError("progress_every_chunks must be non-negative.")

    run_dir = _to_abs_path(run_dir)
    method = _detect_method(run_dir, method)
    if method not in SUPPORTED_METHODS or method == "auto":
        raise ValueError(f"Unsupported method: {method!r}")
    qc_config = (
        ReferenceQCConfig(
            min_abp=float(qc_min_abp),
            max_abp=float(qc_max_abp),
            min_std=float(qc_min_std),
            min_range=float(qc_min_range),
            max_range=float(qc_max_range),
        )
        if reference_qc
        else None
    )

    protocol = {
        "aggregation": "per-patient metrics first, then average across held-out patients",
        "bootstrap": {
            "unit": "patient",
            "n_bootstrap": int(n_bootstrap),
            "seed": int(seed),
            "ci": "2.5th-97.5th percentile of patient-resampled means",
        },
        "table_reporting": {
            "performance": "MAE/RMSE/correlation table fields are patient means with 95% patient-bootstrap CIs",
            "signed_bp_error": "ME +/- error SD fields use pooled signed scalar errors for AAMI-style bias and error-distribution SD; this is not a seed SD",
        },
        "temporal_alignment": {
            "metric": "Min-MAE/Min-RMSE",
            "fixed_shift_ms": float(shift_ms),
            "selection_scope": "per window, fixed before reporting",
        },
        "nrmse": "patient RMSE divided by patient ABP P99-P1 range",
        "reference_qc": {
            "enabled": bool(reference_qc),
            "selection_basis": "reference ABP/input-derived scalar truth only; predictions and prediction errors are never used",
            "thresholds": None if qc_config is None else _qc_config_dict(qc_config),
        },
    }

    if method in {"inn_par", "ppg2abp", "unicardio", "operator"}:
        run = _load_waveform_run(run_dir, method)
        result = _compute_waveform_patient_metrics(
            run,
            shift_ms=shift_ms,
            n_bootstrap=n_bootstrap,
            seed=seed,
            chunk_size=chunk_size,
            compute_scalar=compute_scalar,
            progress_every_chunks=progress_every_chunks,
            reference_qc=qc_config,
        )
        summary: dict[str, Any] = {
            "method": method,
            "method_label": method_label,
            "run_dir": str(run_dir),
            "predictions": run.predictions_ref,
            "window_seconds": float(run.window_seconds),
            "signal_length": int(run.signal_length),
            "protocol": protocol,
            **result,
        }
    elif method == "ptt_based":
        frame = _load_ptt_scalar_predictions(run_dir)
        result = _compute_scalar_patient_metrics(
            frame,
            n_bootstrap=n_bootstrap,
            seed=seed,
            reference_qc=qc_config,
        )
        summary = {
            "method": method,
            "method_label": method_label,
            "run_dir": str(run_dir),
            "protocol": protocol,
            **result,
            "waveform_reason": "PTT-based regression produces scalar SBP/DBP/MAP only.",
        }
    elif method == "papagei":
        frame = _load_papagei_scalar_predictions(run_dir)
        result = _compute_scalar_patient_metrics(
            frame,
            n_bootstrap=n_bootstrap,
            seed=seed,
            reference_qc=qc_config,
        )
        summary = {
            "method": method,
            "method_label": method_label,
            "run_dir": str(run_dir),
            "protocol": protocol,
            **result,
            "waveform_reason": "PaPaGei frozen feature extractor plus scalar regression head does not produce ABP waveforms.",
        }
    else:
        raise ValueError(f"Unsupported method: {method!r}")

    if output_json is None:
        metrics_dir = run_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        output_json_path = metrics_dir / f"{method}_patient_metrics.json"
    else:
        output_json_path = _to_abs_path(output_json)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)

    table_row = _patient_table_row(summary, output_json_path)
    summary["metrics_json"] = str(output_json_path)
    summary["table_row"] = table_row

    csv_summary = None
    if output_csv is not None:
        csv_summary = _write_patient_table_row(table_row, output_csv, append=not overwrite_csv)
        summary["table_csv"] = csv_summary

    output_json_path.write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return _json_safe(summary)
