from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from .manifest import load_standardized_record, validate_unified_manifest
from .metrics import scalar_metric_summary


PTT_BASELINE_KEY = "ptt_based"
METRICS_DIRNAME = "metrics"
PREDICTIONS_DIRNAME = "predictions"
RUN_CONFIG_JSON = "ptt_benchmark_config.json"
FIT_SUMMARY_JSON = "ptt_fit_summary.json"
METRICS_JSON = "ptt_metrics.json"
PREDICTIONS_CSV_GZ = "test_scalar_predictions.csv.gz"

SCALAR_METRICS = ("sbp", "dbp", "map")


@dataclass(frozen=True)
class PTTExtractionConfig:
    min_ptt_sec: float = 0.08
    max_ptt_sec: float = 0.50
    min_peak_distance_sec: float = 0.30
    ecg_prominence_scale: float = 0.50
    ppg_prominence_scale: float = 0.10
    min_finite_segment_sec: float = 8.0


@dataclass(frozen=True)
class PTTFitConfig:
    feature_transform: str = "inverse_ptt"
    include_val_in_fit: bool = False


def _to_abs_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _require_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}.")


def _validate_extraction_config(config: PTTExtractionConfig) -> None:
    _require_positive("min_ptt_sec", config.min_ptt_sec)
    _require_positive("max_ptt_sec", config.max_ptt_sec)
    _require_positive("min_peak_distance_sec", config.min_peak_distance_sec)
    _require_positive("ecg_prominence_scale", config.ecg_prominence_scale)
    _require_positive("ppg_prominence_scale", config.ppg_prominence_scale)
    _require_positive("min_finite_segment_sec", config.min_finite_segment_sec)
    if config.min_ptt_sec >= config.max_ptt_sec:
        raise ValueError("min_ptt_sec must be smaller than max_ptt_sec.")


def _load_valid_manifest(manifest_csv: str | Path, *, limit_records: int | None = None) -> pd.DataFrame:
    summary = validate_unified_manifest(manifest_csv, limit=limit_records)
    if not summary["valid"]:
        raise ValueError(
            "Unified manifest is not valid for PTT baseline evaluation. "
            f"Validation summary:\n{json.dumps(summary, indent=2, sort_keys=True)}"
        )
    manifest_csv = _to_abs_path(manifest_csv)
    frame = pd.read_csv(manifest_csv)
    if limit_records is not None and limit_records > 0:
        frame = frame.head(limit_records).copy()
    return frame


def _finite_segments(mask: np.ndarray, *, min_length: int) -> list[tuple[int, int]]:
    valid = np.asarray(mask, dtype=bool).reshape(-1)
    if valid.size == 0:
        return []

    padded = np.concatenate([[False], valid, [False]])
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    segments: list[tuple[int, int]] = []
    for start, end in zip(changes[0::2], changes[1::2]):
        if int(end - start) >= min_length:
            segments.append((int(start), int(end)))
    return segments


def _standardize_for_peak_detection(signal: np.ndarray) -> np.ndarray:
    values = np.asarray(signal, dtype=np.float64)
    std = float(values.std())
    if std <= 1e-12:
        return np.zeros_like(values)
    return (values - float(values.mean())) / std


def _detect_ecg_r_peaks(ecg: np.ndarray, *, fs: float, config: PTTExtractionConfig) -> np.ndarray:
    x = _standardize_for_peak_detection(ecg)
    if np.all(x == 0):
        return np.asarray([], dtype=np.int64)
    min_distance = max(1, int(round(config.min_peak_distance_sec * fs)))
    peaks, _ = find_peaks(
        x,
        distance=min_distance,
        prominence=config.ecg_prominence_scale,
    )
    return peaks.astype(np.int64, copy=False)


def _detect_ppg_feet(ppg: np.ndarray, *, fs: float, config: PTTExtractionConfig) -> np.ndarray:
    x = _standardize_for_peak_detection(ppg)
    if np.all(x == 0):
        return np.asarray([], dtype=np.int64)
    min_distance = max(1, int(round(config.min_peak_distance_sec * fs)))
    feet, _ = find_peaks(
        -x,
        distance=min_distance,
        prominence=config.ppg_prominence_scale,
    )
    return feet.astype(np.int64, copy=False)


def _extract_ptt_beats_from_segment(
    *,
    ecg: np.ndarray,
    ppg: np.ndarray,
    abp: np.ndarray,
    fs: float,
    source_start: int,
    config: PTTExtractionConfig,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    r_peaks = _detect_ecg_r_peaks(ecg, fs=fs, config=config)
    ppg_feet = _detect_ppg_feet(ppg, fs=fs, config=config)

    stats = {
        "finite_segments": 1,
        "r_peaks": int(r_peaks.size),
        "ppg_feet": int(ppg_feet.size),
        "candidate_r_peaks": 0,
        "paired_beats": 0,
        "dropped_no_ppg_foot": 0,
        "dropped_no_next_ppg_foot": 0,
        "dropped_empty_abp_interval": 0,
    }
    if r_peaks.size == 0 or ppg_feet.size < 2:
        stats["candidate_r_peaks"] = int(r_peaks.size)
        stats["dropped_no_ppg_foot"] = int(r_peaks.size)
        return [], stats

    min_offset = int(round(config.min_ptt_sec * fs))
    max_offset = int(round(config.max_ptt_sec * fs))
    rows: list[dict[str, Any]] = []

    for r_peak in r_peaks:
        stats["candidate_r_peaks"] += 1
        earliest = int(r_peak + min_offset)
        latest = int(r_peak + max_offset)
        foot_positions = np.flatnonzero((ppg_feet >= earliest) & (ppg_feet <= latest))
        if foot_positions.size == 0:
            stats["dropped_no_ppg_foot"] += 1
            continue

        foot_pos = int(foot_positions[0])
        foot = int(ppg_feet[foot_pos])
        if foot_pos + 1 >= ppg_feet.size:
            stats["dropped_no_next_ppg_foot"] += 1
            continue

        next_foot = int(ppg_feet[foot_pos + 1])
        beat_abp = abp[foot:next_foot]
        if beat_abp.size == 0:
            stats["dropped_empty_abp_interval"] += 1
            continue

        ptt_sec = float((foot - int(r_peak)) / fs)
        rows.append(
            {
                "source_start_index": int(source_start + foot),
                "source_end_index": int(source_start + next_foot),
                "ecg_r_index": int(source_start + int(r_peak)),
                "ppg_foot_index": int(source_start + foot),
                "next_ppg_foot_index": int(source_start + next_foot),
                "ptt_sec": ptt_sec,
                "sbp_true": float(np.max(beat_abp)),
                "dbp_true": float(np.min(beat_abp)),
                "map_true": float(np.mean(beat_abp)),
            }
        )
        stats["paired_beats"] += 1

    return rows, stats


def _empty_extraction_stats() -> dict[str, int]:
    return {
        "records": 0,
        "records_with_beats": 0,
        "finite_segments": 0,
        "r_peaks": 0,
        "ppg_feet": 0,
        "candidate_r_peaks": 0,
        "paired_beats": 0,
        "dropped_no_ppg_foot": 0,
        "dropped_no_next_ppg_foot": 0,
        "dropped_empty_abp_interval": 0,
    }


def _add_stats(target: dict[str, int], source: dict[str, int]) -> None:
    for key, value in source.items():
        target[key] = int(target.get(key, 0) + int(value))


def _extract_manifest_beats(
    manifest: pd.DataFrame,
    *,
    config: PTTExtractionConfig,
) -> tuple[pd.DataFrame, dict[str, int]]:
    _validate_extraction_config(config)
    all_rows: list[dict[str, Any]] = []
    summary = _empty_extraction_stats()

    for row in manifest.to_dict(orient="records"):
        summary["records"] += 1
        fs = float(row["fs"])
        if fs <= 0:
            raise ValueError(f"Invalid sampling rate for record_id={row.get('record_id')}: {fs}")

        frame = load_standardized_record(row)
        required = ["II", "PLETH", "ABP"]
        missing = [col for col in required if col not in frame.columns]
        if missing:
            raise KeyError(f"Record {row.get('record_id')} is missing columns: {missing}")

        ecg = frame["II"].to_numpy(dtype=np.float64)
        ppg = frame["PLETH"].to_numpy(dtype=np.float64)
        abp = frame["ABP"].to_numpy(dtype=np.float64)
        finite = np.isfinite(ecg) & np.isfinite(ppg) & np.isfinite(abp)
        min_len = max(1, int(round(config.min_finite_segment_sec * fs)))
        record_rows: list[dict[str, Any]] = []

        for start, end in _finite_segments(finite, min_length=min_len):
            beat_rows, stats = _extract_ptt_beats_from_segment(
                ecg=ecg[start:end],
                ppg=ppg[start:end],
                abp=abp[start:end],
                fs=fs,
                source_start=start,
                config=config,
            )
            _add_stats(summary, stats)
            for beat_index, beat in enumerate(beat_rows):
                beat.update(
                    {
                        "split": row.get("split"),
                        "source": row.get("source"),
                        "subject_id": row.get("subject_id"),
                        "record_id": row.get("record_id"),
                        "hadm_id": row.get("hadm_id"),
                        "fs": fs,
                        "beat_index_in_record": int(len(record_rows) + beat_index),
                    }
                )
            record_rows.extend(beat_rows)

        if record_rows:
            summary["records_with_beats"] += 1
            all_rows.extend(record_rows)

    return pd.DataFrame(all_rows), summary


def _design_matrix(ptt_sec: np.ndarray, *, feature_transform: str) -> np.ndarray:
    ptt = np.asarray(ptt_sec, dtype=np.float64).reshape(-1)
    if not np.isfinite(ptt).all() or np.any(ptt <= 0):
        raise ValueError("PTT values must be finite and positive.")

    if feature_transform == "inverse_ptt":
        feature = 1.0 / ptt
    elif feature_transform == "log_ptt":
        feature = np.log(ptt)
    else:
        raise ValueError(f"Unsupported PTT feature transform: {feature_transform}")

    return np.column_stack([np.ones_like(feature), feature])


def _fit_linear_ptt_model(
    fit_beats: pd.DataFrame,
    *,
    fit_config: PTTFitConfig,
) -> dict[str, Any]:
    if fit_beats.empty:
        raise ValueError("No PTT beats were extracted for calibration.")

    x = _design_matrix(fit_beats["ptt_sec"].to_numpy(dtype=np.float64), feature_transform=fit_config.feature_transform)
    rank = int(np.linalg.matrix_rank(x))
    if rank < x.shape[1]:
        raise ValueError(
            "PTT calibration design matrix is rank deficient. "
            "The train split must contain variable PTT values for a linear calibration."
        )

    models: dict[str, dict[str, float]] = {}
    for metric in SCALAR_METRICS:
        y = fit_beats[f"{metric}_true"].to_numpy(dtype=np.float64)
        if not np.isfinite(y).all():
            raise ValueError(f"Non-finite {metric.upper()} calibration labels were extracted.")
        beta, residuals, _, _ = np.linalg.lstsq(x, y, rcond=None)
        models[metric] = {
            "intercept": float(beta[0]),
            "slope": float(beta[1]),
            "residual_sum_squares": float(residuals[0]) if residuals.size else 0.0,
        }

    return {
        "baseline_key": PTT_BASELINE_KEY,
        "fit_config": asdict(fit_config),
        "n_fit_beats": int(len(fit_beats)),
        "ptt_feature_mean": float(np.mean(x[:, 1])),
        "ptt_feature_std": float(np.std(x[:, 1])),
        "models": models,
    }


def _predict_ptt_model(
    beats: pd.DataFrame,
    *,
    model: dict[str, Any],
) -> pd.DataFrame:
    if beats.empty:
        raise ValueError("No PTT beats were extracted for testing.")

    feature_transform = str(model["fit_config"]["feature_transform"])
    x = _design_matrix(beats["ptt_sec"].to_numpy(dtype=np.float64), feature_transform=feature_transform)
    pred = beats.copy()
    for metric in SCALAR_METRICS:
        params = model["models"][metric]
        pred[f"{metric}_pred"] = float(params["intercept"]) + float(params["slope"]) * x[:, 1]
    return pred


def _scalar_summary_from_predictions(predictions: pd.DataFrame) -> dict[str, Any]:
    summaries = {
        metric: scalar_metric_summary(
            predictions[f"{metric}_pred"].to_numpy(dtype=np.float64)
            - predictions[f"{metric}_true"].to_numpy(dtype=np.float64)
        )
        for metric in SCALAR_METRICS
    }
    return {
        "n_beats": int(max(summaries["sbp"]["n_beats"], summaries["dbp"]["n_beats"], summaries["map"]["n_beats"])),
        "n_scalar_predictions": int(len(predictions)),
        "sbp_mae": summaries["sbp"]["mae"],
        "dbp_mae": summaries["dbp"]["mae"],
        "map_mae": summaries["map"]["mae"],
        "sbp_rmse": summaries["sbp"]["rmse"],
        "dbp_rmse": summaries["dbp"]["rmse"],
        "map_rmse": summaries["map"]["rmse"],
        "sbp_bias": summaries["sbp"]["bias"],
        "dbp_bias": summaries["dbp"]["bias"],
        "map_bias": summaries["map"]["bias"],
        "sbp_std": summaries["sbp"]["std"],
        "dbp_std": summaries["dbp"]["std"],
        "map_std": summaries["map"]["std"],
        "sbp_loa_low": summaries["sbp"]["loa_low"],
        "sbp_loa_high": summaries["sbp"]["loa_high"],
        "dbp_loa_low": summaries["dbp"]["loa_low"],
        "dbp_loa_high": summaries["dbp"]["loa_high"],
        "map_loa_low": summaries["map"]["loa_low"],
        "map_loa_high": summaries["map"]["loa_high"],
        "sbp_aami_compliant": summaries["sbp"]["aami_compliant"],
        "dbp_aami_compliant": summaries["dbp"]["aami_compliant"],
        "map_aami_compliant": summaries["map"]["aami_compliant"],
        "sbp_bhs_grade": summaries["sbp"]["bhs_grade"],
        "dbp_bhs_grade": summaries["dbp"]["bhs_grade"],
        "map_bhs_grade": summaries["map"]["bhs_grade"],
        "standards": summaries,
    }


def _write_predictions_csv_gz(predictions: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "split",
        "source",
        "subject_id",
        "record_id",
        "hadm_id",
        "fs",
        "beat_index_in_record",
        "source_start_index",
        "source_end_index",
        "ecg_r_index",
        "ppg_foot_index",
        "next_ppg_foot_index",
        "ptt_sec",
        "sbp_true",
        "dbp_true",
        "map_true",
        "sbp_pred",
        "dbp_pred",
        "map_pred",
    ]
    predictions.loc[:, columns].to_csv(output_path, index=False, compression="gzip")


def run_ptt_benchmark(
    *,
    train_manifest: str | Path,
    test_manifest: str | Path,
    output_dir: str | Path,
    val_manifest: str | Path | None = None,
    include_val_in_fit: bool = False,
    feature_transform: str = "inverse_ptt",
    min_ptt_sec: float = PTTExtractionConfig.min_ptt_sec,
    max_ptt_sec: float = PTTExtractionConfig.max_ptt_sec,
    min_peak_distance_sec: float = PTTExtractionConfig.min_peak_distance_sec,
    ecg_prominence_scale: float = PTTExtractionConfig.ecg_prominence_scale,
    ppg_prominence_scale: float = PTTExtractionConfig.ppg_prominence_scale,
    min_finite_segment_sec: float = PTTExtractionConfig.min_finite_segment_sec,
    limit_records: int | None = None,
) -> dict[str, Any]:
    output_dir = _to_abs_path(output_dir)
    metrics_dir = output_dir / METRICS_DIRNAME
    predictions_dir = output_dir / PREDICTIONS_DIRNAME
    metrics_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    extraction_config = PTTExtractionConfig(
        min_ptt_sec=min_ptt_sec,
        max_ptt_sec=max_ptt_sec,
        min_peak_distance_sec=min_peak_distance_sec,
        ecg_prominence_scale=ecg_prominence_scale,
        ppg_prominence_scale=ppg_prominence_scale,
        min_finite_segment_sec=min_finite_segment_sec,
    )
    fit_config = PTTFitConfig(
        feature_transform=feature_transform,
        include_val_in_fit=include_val_in_fit,
    )

    train_manifest_path = _to_abs_path(train_manifest)
    test_manifest_path = _to_abs_path(test_manifest)
    val_manifest_path = _to_abs_path(val_manifest) if val_manifest is not None else None

    train_df = _load_valid_manifest(train_manifest_path, limit_records=limit_records)
    test_df = _load_valid_manifest(test_manifest_path, limit_records=limit_records)
    val_df = _load_valid_manifest(val_manifest_path, limit_records=limit_records) if val_manifest_path is not None else None

    train_beats, train_extraction = _extract_manifest_beats(train_df, config=extraction_config)
    fit_parts = [train_beats]
    val_extraction = None
    if include_val_in_fit:
        if val_df is None:
            raise ValueError("--include-val-in-fit requires --val-manifest.")
        val_beats, val_extraction = _extract_manifest_beats(val_df, config=extraction_config)
        fit_parts.append(val_beats)

    fit_beats = pd.concat(fit_parts, ignore_index=True)
    model = _fit_linear_ptt_model(fit_beats, fit_config=fit_config)

    test_beats, test_extraction = _extract_manifest_beats(test_df, config=extraction_config)
    predictions = _predict_ptt_model(test_beats, model=model)

    predictions_path = predictions_dir / PREDICTIONS_CSV_GZ
    _write_predictions_csv_gz(predictions, predictions_path)

    fit_summary_path = output_dir / FIT_SUMMARY_JSON
    fit_summary = {
        **model,
        "train_extraction": train_extraction,
        "val_extraction": val_extraction,
    }
    fit_summary_path.write_text(json.dumps(fit_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    run_config_path = output_dir / RUN_CONFIG_JSON
    run_config = {
        "baseline_key": PTT_BASELINE_KEY,
        "input_columns": ["II", "PLETH"],
        "target_column": "ABP",
        "output_type": "scalar_only",
        "train_manifest": str(train_manifest_path),
        "val_manifest": str(val_manifest_path) if val_manifest_path is not None else None,
        "test_manifest": str(test_manifest_path),
        "extraction_config": asdict(extraction_config),
        "fit_config": asdict(fit_config),
        "limit_records": limit_records,
        "protocol": (
            "Detect ECG R peaks and subsequent PPG foot points, compute beat-level PTT, "
            "fit BP = beta0 + beta1 / PTT on the calibration split, and evaluate scalar BP "
            "on the held-out test split. No ABP waveform is generated."
        ),
    }
    run_config_path.write_text(json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    metrics_summary = {
        "baseline_key": PTT_BASELINE_KEY,
        "run_dir": str(output_dir),
        "run_config": str(run_config_path),
        "fit_summary": str(fit_summary_path),
        "predictions": str(predictions_path),
        "output_type": "scalar_only",
        "waveform": {
            "n_points": 0,
            "waveform_mae": "nan",
            "waveform_rmse": "nan",
            "waveform_corr": "nan",
            "reason": "PTT-based regression produces scalar SBP/DBP/MAP estimates only and does not generate ABP waveforms.",
        },
        "scalar": _scalar_summary_from_predictions(predictions),
        "extraction": {
            "train": train_extraction,
            "val": val_extraction,
            "test": test_extraction,
        },
        "calibration": {
            "feature_transform": feature_transform,
            "include_val_in_fit": include_val_in_fit,
            "n_fit_beats": int(model["n_fit_beats"]),
            "models": model["models"],
        },
    }
    metrics_path = metrics_dir / METRICS_JSON
    metrics_path.write_text(json.dumps(metrics_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics_summary
