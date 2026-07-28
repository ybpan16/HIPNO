from __future__ import annotations

import json
from fractions import Fraction
from math import gcd
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .metrics import scalar_metric_summary


PAPAGEI_BASELINE_KEY = "papagei"
PAPAGEI_TARGET_FS_HZ = 125.0
PAPAGEI_SEGMENT_SEC = 10.0
PAPAGEI_STEP_SEC = 10.0
PAPAGEI_SEGMENT_LENGTH = 1250
PAPAGEI_EMBEDDING_DIM = 512
PAPAGEI_OUTPUT_IDX = 0
PAPAGEI_BATCH_SIZE = 256
PAPAGEI_SEED = 2025
PAPAGEI_MODEL_CONFIG = {
    "base_filters": 32,
    "kernel_size": 3,
    "stride": 2,
    "groups": 1,
    "n_block": 18,
    "n_classes": 512,
    "n_experts": 3,
}
PAPAGEI_RIDGE_ALPHA_GRID = [0.1, 1.0, 10.0, 100.0, 1000.0]
PAPAGEI_RIDGE_SOLVER = "svd_closed_form"
PAPAGEI_GRID_CV = 4

PREPARED_DATA_DIRNAME = "prepared_data"
SEGMENTS_DIRNAME = "segments"
FEATURES_DIRNAME = "features"
MODELS_DIRNAME = "models"
PREDICTIONS_DIRNAME = "predictions"
METRICS_DIRNAME = "metrics"
PREPARED_SEGMENTS_MANIFEST = "segments_manifest.csv"
PREPARED_CASE_LABELS = "case_labels.csv"
PREPARED_CONFIG_JSON = "papagei_benchmark_config.json"
PREPARED_SUMMARY_JSON = "papagei_prepare_summary.json"
RUN_RESULTS_JSON = "papagei_run_summary.json"
METRICS_JSON = "papagei_metrics.json"

EXPORT_MANIFEST_REQUIRED_COLUMNS = [
    "baseline_key",
    "split",
    "source",
    "subject_id",
    "record_id",
    "hadm_id",
    "fs",
    "exported_parquet",
    "source_manifest",
    "source_row_index",
]
PREPARED_SEGMENTS_COLUMNS = [
    "split",
    "case_key",
    "segment_index",
    "source",
    "subject_id",
    "record_id",
    "hadm_id",
    "source_record_rows",
    "source_fs_hz",
    "target_fs_hz",
    "segment_sec",
    "step_sec",
    "target_start_index",
    "target_end_index",
    "target_start_sec",
    "target_end_sec",
    "segment_length",
    "segment_path",
    "source_finite_start_index",
    "source_finite_end_index",
    "sbp",
    "dbp",
    "map",
    "source_exported_parquet",
]
CASE_LABEL_COLUMNS = [
    "split",
    "case_key",
    "source",
    "subject_id",
    "record_id",
    "hadm_id",
    "n_segments",
    "sbp",
    "dbp",
    "map",
    "source_exported_parquet",
]
EMBEDDING_SPLITS = ("train", "val", "test")
SCALAR_METRICS = ("sbp", "dbp", "map")


def _to_abs_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _safe_filename_token(text: Any) -> str:
    token = "".join(ch if str(ch).isalnum() else "_" for ch in str(text).strip())
    token = token.strip("_")
    return token or "record"


def _require_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}.")


def _require_split_coverage(frame: pd.DataFrame) -> None:
    required = {"train", "val", "test"}
    present = set(frame["split"].dropna().astype(str))
    extra = sorted(present - required)
    if extra:
        raise ValueError(
            "PaPaGei benchmark preparation only accepts train, val, and test splits. "
            f"Unexpected split labels: {', '.join(extra)}"
        )
    missing = sorted(required - present)
    if missing:
        raise ValueError(
            "PaPaGei benchmark preparation requires train, val, and test splits. "
            f"Missing: {', '.join(missing)}"
        )


def _load_export_manifests(export_manifests: list[str | Path]) -> pd.DataFrame:
    if not export_manifests:
        raise ValueError("At least one export manifest is required.")

    parts: list[pd.DataFrame] = []
    for manifest in export_manifests:
        manifest_path = _to_abs_path(manifest)
        frame = pd.read_csv(manifest_path)
        missing = [col for col in EXPORT_MANIFEST_REQUIRED_COLUMNS if col not in frame.columns]
        if missing:
            raise ValueError(
                f"Export manifest {manifest_path} is missing required columns: {', '.join(missing)}"
            )
        frame = frame.copy()
        frame["export_manifest_path"] = str(manifest_path)
        parts.append(frame)

    merged = pd.concat(parts, ignore_index=True)
    unique_keys = set(merged["baseline_key"].astype(str))
    if unique_keys != {PAPAGEI_BASELINE_KEY}:
        raise ValueError(
            "PaPaGei launcher only accepts export manifests for baseline_key='papagei'. "
            f"Found: {sorted(unique_keys)}"
        )

    if merged.duplicated(subset=["split", "record_id", "exported_parquet"], keep=False).any():
        raise ValueError("Export manifests contain duplicate split/record/exported_parquet rows.")

    _require_split_coverage(merged)
    return merged

def _zscore(signal: np.ndarray) -> np.ndarray:
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"Expected 1D PPG signal, got shape {values.shape}.")
    if not np.isfinite(values).all():
        raise ValueError("PPG signal contains non-finite values.")
    std = float(values.std())
    if std <= 1e-12:
        raise ValueError("PPG signal has near-zero variance and cannot be z-score normalized.")
    return ((values - float(values.mean())) / std).astype(np.float64, copy=False)


def _resample_1d_papagei(signal: np.ndarray, *, fs_original: float, fs_target: float) -> np.ndarray:
    from scipy.signal import resample_poly

    values = np.asarray(signal, dtype=np.float64).reshape(-1)
    if values.size < 2:
        raise ValueError("Signal must contain at least two samples for PaPaGei resampling.")
    if not np.isfinite(values).all():
        raise ValueError("Signal contains non-finite values before PaPaGei resampling.")
    fs_original_frac = Fraction(float(fs_original)).limit_denominator()
    fs_target_frac = Fraction(float(fs_target)).limit_denominator()
    lcm_denominator = np.lcm(fs_original_frac.denominator, fs_target_frac.denominator)
    fs_original_scaled = fs_original_frac * int(lcm_denominator)
    fs_target_scaled = fs_target_frac * int(lcm_denominator)
    gcd_value = gcd(fs_original_scaled.numerator, fs_target_scaled.numerator)
    up = fs_target_scaled.numerator // gcd_value
    down = fs_original_scaled.numerator // gcd_value
    resampled = resample_poly(values, up, down, axis=0)
    return np.asarray(resampled, dtype=np.float32).reshape(-1)


def _preprocess_papagei_ppg_signal(
    waveform: np.ndarray,
    frequency: float,
    fL: float = 0.5,
    fH: float = 12.0,
    order: int = 4,
    smoothing_windows: dict[str, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from scipy import signal as scipy_signal
    from scipy.signal import filtfilt

    sm_wins = smoothing_windows or {"ppg": 50, "vpg": 10, "apg": 10, "jpg": 10}
    values = np.asarray(waveform, dtype=np.float64).reshape(-1)
    fs = float(frequency)
    if values.size < 2:
        raise ValueError("PPG signal must contain at least two samples.")
    if fs <= 0:
        raise ValueError("PPG sampling frequency must be positive.")

    if fL == 0:
        b, a = scipy_signal.cheby2(order, 20, [float(fH)], "low", fs=fs)
    else:
        b, a = scipy_signal.cheby2(order, 20, [float(fL), float(fH)], "bandpass", fs=fs)

    ppg_cb2 = filtfilt(b, a, values)
    if fs >= 75:
        win = round(fs * sm_wins["ppg"] / 1000)
        B = 1 / win * np.ones(win)
        ppg = filtfilt(B, 1, ppg_cb2)
    else:
        ppg = ppg_cb2

    if fs >= 150:
        win = round(fs * sm_wins["vpg"] / 1000)
        B1 = 1 / win * np.ones(win)
        dx = np.gradient(ppg)
        vpg = filtfilt(B1, 1, dx)

        win = round(fs * sm_wins["apg"] / 1000)
        B2 = 1 / win * np.ones(win)
        ddx = np.gradient(vpg)
        apg = filtfilt(B2, 1, ddx)

        win = round(fs * sm_wins["jpg"] / 1000)
        B3 = 1 / win * np.ones(win)
        dddx = np.gradient(apg)
        jpg = filtfilt(B3, 1, dddx)
    else:
        vpg = np.gradient(ppg)
        apg = np.gradient(vpg)
        jpg = np.gradient(apg)

    return (
        np.asarray(ppg, dtype=np.float32),
        np.asarray(vpg, dtype=np.float32),
        np.asarray(apg, dtype=np.float32),
        np.asarray(jpg, dtype=np.float32),
    )


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


def _write_segment(path: Path, segment: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(segment, dtype=np.float32))


def _extract_preprocessed_segments(
    frame: pd.DataFrame,
    *,
    source_fs_hz: float,
    target_fs_hz: float,
    segment_sec: float,
    step_sec: float,
    segment_length: int,
) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    ppg_raw = np.asarray(frame["PLETH"], dtype=np.float64)
    abp_raw = np.asarray(frame["ABP"], dtype=np.float64)
    finite_mask = np.isfinite(ppg_raw) & np.isfinite(abp_raw)
    target_step = int(round(target_fs_hz * step_sec))
    if target_step <= 0:
        raise ValueError("step_sec is too small for PaPaGei target sampling rate.")
    if segment_length <= 1:
        raise ValueError("segment_length must be greater than 1.")
    min_source_length = int(np.ceil(segment_length * float(source_fs_hz) / float(target_fs_hz)))

    rows: list[dict[str, Any]] = []
    segments: list[np.ndarray] = []
    source_record_rows = int(len(frame))
    for finite_start, finite_end in _finite_segments(finite_mask, min_length=min_source_length):
        ppg_raw_segment = ppg_raw[finite_start:finite_end]
        abp_raw_segment = abp_raw[finite_start:finite_end]
        try:
            ppg_norm = _zscore(ppg_raw_segment)
            ppg_processed, _, _, _ = _preprocess_papagei_ppg_signal(ppg_norm, frequency=float(source_fs_hz))
            ppg_target = _resample_1d_papagei(
                ppg_processed,
                fs_original=source_fs_hz,
                fs_target=target_fs_hz,
            )
            abp_target = _resample_1d_papagei(
                abp_raw_segment,
                fs_original=source_fs_hz,
                fs_target=target_fs_hz,
            )
        except ValueError:
            continue

        total_length = min(int(ppg_target.size), int(abp_target.size))
        for start in range(0, total_length - segment_length + 1, target_step):
            end = start + segment_length
            ppg_segment = ppg_target[start:end]
            abp_segment = abp_target[start:end]
            if ppg_segment.size != segment_length or abp_segment.size != segment_length:
                continue
            if not np.isfinite(ppg_segment).all() or not np.isfinite(abp_segment).all():
                continue
            source_start = finite_start + int(round(start * float(source_fs_hz) / float(target_fs_hz)))
            source_end = finite_start + int(round(end * float(source_fs_hz) / float(target_fs_hz)))
            segments.append(ppg_segment.astype(np.float32, copy=False))
            rows.append(
                {
                    "source_record_rows": source_record_rows,
                    "source_fs_hz": float(source_fs_hz),
                    "target_fs_hz": float(target_fs_hz),
                    "segment_sec": float(segment_sec),
                    "step_sec": float(step_sec),
                    "target_start_index": int(start),
                    "target_end_index": int(end),
                    "target_start_sec": float(source_start / source_fs_hz),
                    "target_end_sec": float(source_end / source_fs_hz),
                    "segment_length": int(segment_length),
                    "source_finite_start_index": int(finite_start),
                    "source_finite_end_index": int(finite_end),
                    "sbp": float(np.max(abp_segment)),
                    "dbp": float(np.min(abp_segment)),
                    "map": float(np.mean(abp_segment)),
                }
            )

    return rows, segments


def _prepared_config_path(run_dir: str | Path) -> Path:
    return _to_abs_path(run_dir) / PREPARED_DATA_DIRNAME / PREPARED_CONFIG_JSON


def _load_prepared_config(run_dir: str | Path) -> dict[str, Any]:
    config_path = _prepared_config_path(run_dir)
    if not config_path.exists():
        raise FileNotFoundError(f"Missing prepared PaPaGei config: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def prepare_papagei_run(
    export_manifests: list[str | Path],
    output_dir: str | Path,
    upstream_repo: str | Path,
    *,
    target_fs_hz: float = PAPAGEI_TARGET_FS_HZ,
    segment_sec: float = PAPAGEI_SEGMENT_SEC,
    step_sec: float = PAPAGEI_STEP_SEC,
    segment_length: int | None = None,
    drop_zero_segment_records: bool = False,
) -> dict[str, Any]:
    _require_positive("target_fs_hz", target_fs_hz)
    _require_positive("segment_sec", segment_sec)
    _require_positive("step_sec", step_sec)
    if segment_length is None:
        segment_length = int(round(target_fs_hz * segment_sec))
    if segment_length <= 1:
        raise ValueError("segment_length must be greater than 1.")

    export_frame = _load_export_manifests(export_manifests)
    output_dir = _to_abs_path(output_dir)
    prepared_dir = output_dir / PREPARED_DATA_DIRNAME
    segments_root = prepared_dir / SEGMENTS_DIRNAME
    prepared_dir.mkdir(parents=True, exist_ok=True)
    segments_root.mkdir(parents=True, exist_ok=True)

    segment_rows: list[dict[str, Any]] = []
    split_record_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    split_dropped_record_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    split_segment_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    zero_segment_records: list[dict[str, Any]] = []
    case_counter = 0

    for row in export_frame.to_dict(orient="records"):
        split = str(row["split"])
        record_path = _to_abs_path(row["exported_parquet"])
        if not record_path.exists():
            raise FileNotFoundError(f"Exported parquet does not exist: {record_path}")

        frame = pd.read_parquet(record_path)
        expected_cols = {"PLETH", "ABP", "fs", "subject_id", "record_id", "split", "source"}
        missing_cols = expected_cols - set(frame.columns)
        if missing_cols:
            raise ValueError(
                f"Exported parquet {record_path} is missing required columns: {sorted(missing_cols)}"
            )

        case_key = (
            f"{split}_{case_counter:08d}_"
            f"{_safe_filename_token(row['subject_id'])}_"
            f"{_safe_filename_token(row['record_id'])}"
        )
        case_counter += 1
        source_fs_hz = float(row["fs"])
        metadata_rows, segments = _extract_preprocessed_segments(
            frame,
            source_fs_hz=source_fs_hz,
            target_fs_hz=target_fs_hz,
            segment_sec=segment_sec,
            step_sec=step_sec,
            segment_length=segment_length,
        )
        if not segments:
            zero_segment_record = {
                "split": split,
                "source": row["source"],
                "subject_id": row["subject_id"],
                "record_id": row["record_id"],
                "hadm_id": row.get("hadm_id", None),
                "source_exported_parquet": str(record_path),
                "reason": "PaPaGei preprocessing produced zero finite 10 s PPG segments",
            }
            zero_segment_records.append(zero_segment_record)
            split_dropped_record_counts[split] += 1
            if drop_zero_segment_records:
                continue
            raise ValueError(
                "PaPaGei preparation produced zero 10 s segments for "
                f"record {row['record_id']}. Rerun with --drop-zero-segment-records "
                "only if this pre-registered input eligibility exclusion is intended."
            )

        split_record_counts[split] += 1
        split_segment_counts[split] += len(segments)
        case_dir = segments_root / split / case_key
        for segment_index, (metadata, segment) in enumerate(zip(metadata_rows, segments)):
            segment_path = case_dir / f"{segment_index}.npy"
            _write_segment(segment_path, segment)
            segment_rows.append(
                {
                    "split": split,
                    "case_key": case_key,
                    "segment_index": int(segment_index),
                    "source": row["source"],
                    "subject_id": row["subject_id"],
                    "record_id": row["record_id"],
                    "hadm_id": row.get("hadm_id", None),
                    **metadata,
                    "segment_path": str(segment_path),
                    "source_exported_parquet": str(record_path),
                }
            )

    zero_segment_records_path = prepared_dir / "zero_segment_records.csv"
    pd.DataFrame(
        zero_segment_records,
        columns=["split", "source", "subject_id", "record_id", "hadm_id", "source_exported_parquet", "reason"],
    ).to_csv(zero_segment_records_path, index=False)

    empty_splits = [split for split, count in split_segment_counts.items() if count == 0]
    if empty_splits:
        raise ValueError(
            "PaPaGei preparation produced empty split(s): "
            f"{', '.join(empty_splits)}. Check record duration and split manifests."
        )

    segment_manifest_path = prepared_dir / PREPARED_SEGMENTS_MANIFEST
    segments_df = pd.DataFrame(segment_rows, columns=PREPARED_SEGMENTS_COLUMNS)
    segments_df.to_csv(segment_manifest_path, index=False)

    case_labels = (
        segments_df.groupby(
            [
                "split",
                "case_key",
                "source",
                "subject_id",
                "record_id",
                "hadm_id",
                "source_exported_parquet",
            ],
            dropna=False,
        )
        .agg(n_segments=("segment_index", "count"), sbp=("sbp", "mean"), dbp=("dbp", "mean"), map=("map", "mean"))
        .reset_index()
    )
    case_labels = case_labels[CASE_LABEL_COLUMNS]
    case_labels_path = prepared_dir / PREPARED_CASE_LABELS
    case_labels.to_csv(case_labels_path, index=False)

    config = {
        "baseline_key": PAPAGEI_BASELINE_KEY,
        "method": "official_PaPaGei_S_frozen_feature_extractor_plus_ridge_linear_probe",
        "upstream_repo": str(_to_abs_path(upstream_repo)),
        "export_manifests": [str(_to_abs_path(path)) for path in export_manifests],
        "output_dir": str(output_dir),
        "prepared_data_dir": str(prepared_dir),
        "segments_root": str(segments_root),
        "segment_manifest": str(segment_manifest_path),
        "case_labels": str(case_labels_path),
        "target_fs_hz": float(target_fs_hz),
        "segment_sec": float(segment_sec),
        "step_sec": float(step_sec),
        "segment_length": int(segment_length),
        "preprocessing": [
            "per finite source span z-score normalization of PLETH",
            "PaPaGei/pyPPG Preprocess-equivalent Chebyshev Type II zero-phase PPG filtering",
            "PaPaGei resample_poly frequency-ratio resampling to 125 Hz",
            "non-overlapping 10 s segments saved as NumPy .npy files",
        ],
        "case_embedding_aggregation": "mean of segment embeddings, matching PaPaGei patient-level linear probing",
        "labels": "case-level mean of segment SBP=max(ABP), DBP=min(ABP), MAP=mean(ABP)",
        "zero_segment_record_policy": {
            "drop_zero_segment_records": bool(drop_zero_segment_records),
            "selection_basis": "input/reference eligibility only; predictions are not used",
            "zero_segment_records_csv": str(zero_segment_records_path),
            "reason": "official PaPaGei preprocessing produced zero finite 10 s PPG segments",
        },
        "split_records": split_record_counts,
        "split_dropped_records": split_dropped_record_counts,
        "split_segments": split_segment_counts,
        "zero_segment_records": zero_segment_records,
        "model_config": PAPAGEI_MODEL_CONFIG,
    }
    config_path = prepared_dir / PREPARED_CONFIG_JSON
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary = {
        "baseline_key": PAPAGEI_BASELINE_KEY,
        "output_dir": str(output_dir),
        "prepared_data_dir": str(prepared_dir),
        "config_json": str(config_path),
        "segment_manifest": str(segment_manifest_path),
        "case_labels": str(case_labels_path),
        "split_records": split_record_counts,
        "split_dropped_records": split_dropped_record_counts,
        "split_segments": split_segment_counts,
        "zero_segment_records": zero_segment_records,
        "zero_segment_records_csv": str(zero_segment_records_path),
        "drop_zero_segment_records": bool(drop_zero_segment_records),
        "target_fs_hz": float(target_fs_hz),
        "segment_sec": float(segment_sec),
        "step_sec": float(step_sec),
        "segment_length": int(segment_length),
    }
    summary_path = output_dir / PREPARED_SUMMARY_JSON
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _build_papagei_s_model(upstream_repo: str | Path, weights_path: str | Path, device: Any) -> Any:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class MyConv1dPadSame(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, groups: int = 1):
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.kernel_size = kernel_size
            self.stride = stride
            self.groups = groups
            self.conv = nn.Conv1d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                groups=self.groups,
            )

        def forward(self, x: Any) -> Any:
            in_dim = x.shape[-1]
            out_dim = (in_dim + self.stride - 1) // self.stride
            pad = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
            pad_left = pad // 2
            pad_right = pad - pad_left
            return self.conv(F.pad(x, (pad_left, pad_right), "constant", 0))

    class MyMaxPool1dPadSame(nn.Module):
        def __init__(self, kernel_size: int):
            super().__init__()
            self.kernel_size = kernel_size
            self.stride = 1
            self.max_pool = nn.MaxPool1d(kernel_size=self.kernel_size)

        def forward(self, x: Any) -> Any:
            in_dim = x.shape[-1]
            out_dim = (in_dim + self.stride - 1) // self.stride
            pad = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
            pad_left = pad // 2
            pad_right = pad - pad_left
            return self.max_pool(F.pad(x, (pad_left, pad_right), "constant", 0))

    class BasicBlock(nn.Module):
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int,
            groups: int,
            downsample: bool,
            use_bn: bool,
            use_do: bool,
            is_first_block: bool = False,
        ):
            super().__init__()
            self.in_channels = in_channels
            self.kernel_size = kernel_size
            self.out_channels = out_channels
            self.stride = stride if downsample else 1
            self.groups = groups
            self.downsample = downsample
            self.is_first_block = is_first_block
            self.use_bn = use_bn
            self.use_do = use_do
            self.bn1 = nn.BatchNorm1d(in_channels)
            self.relu1 = nn.ReLU()
            self.do1 = nn.Dropout(p=0.5)
            self.conv1 = MyConv1dPadSame(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=self.stride,
                groups=groups,
            )
            self.bn2 = nn.BatchNorm1d(out_channels)
            self.relu2 = nn.ReLU()
            self.do2 = nn.Dropout(p=0.5)
            self.conv2 = MyConv1dPadSame(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                groups=groups,
            )
            self.max_pool = MyMaxPool1dPadSame(kernel_size=self.stride)

        def forward(self, x: Any) -> Any:
            identity = x
            out = x
            if not self.is_first_block:
                if self.use_bn:
                    out = self.bn1(out)
                out = self.relu1(out)
                if self.use_do:
                    out = self.do1(out)
            out = self.conv1(out)
            if self.use_bn:
                out = self.bn2(out)
            out = self.relu2(out)
            if self.use_do:
                out = self.do2(out)
            out = self.conv2(out)
            if self.downsample:
                identity = self.max_pool(identity)
            if self.out_channels != self.in_channels:
                identity = identity.transpose(-1, -2)
                ch1 = (self.out_channels - self.in_channels) // 2
                ch2 = self.out_channels - self.in_channels - ch1
                identity = F.pad(identity, (ch1, ch2), "constant", 0)
                identity = identity.transpose(-1, -2)
            out += identity
            return out

    class ResNet1DMoE(nn.Module):
        def __init__(
            self,
            in_channels: int,
            base_filters: int,
            kernel_size: int,
            stride: int,
            groups: int,
            n_block: int,
            n_classes: int,
            n_experts: int = 2,
            downsample_gap: int = 2,
            increasefilter_gap: int = 4,
            use_bn: bool = True,
            use_do: bool = True,
            verbose: bool = False,
            use_projection: bool = False,
        ):
            super().__init__()
            self.verbose = verbose
            self.n_block = n_block
            self.kernel_size = kernel_size
            self.stride = stride
            self.groups = groups
            self.use_bn = use_bn
            self.use_do = use_do
            self.use_projection = use_projection
            self.downsample_gap = downsample_gap
            self.increasefilter_gap = increasefilter_gap
            self.n_experts = n_experts
            self.first_block_conv = MyConv1dPadSame(
                in_channels=in_channels,
                out_channels=base_filters,
                kernel_size=self.kernel_size,
                stride=1,
            )
            self.first_block_bn = nn.BatchNorm1d(base_filters)
            self.first_block_relu = nn.ReLU()
            out_channels = base_filters
            self.basicblock_list = nn.ModuleList()
            for i_block in range(self.n_block):
                is_first_block = i_block == 0
                downsample = i_block % self.downsample_gap == 1
                in_channels_i = (
                    base_filters
                    if is_first_block
                    else int(base_filters * 2 ** ((i_block - 1) // self.increasefilter_gap))
                )
                out_channels = (
                    in_channels_i * 2
                    if (i_block % self.increasefilter_gap == 0 and i_block != 0)
                    else in_channels_i
                )
                self.basicblock_list.append(
                    BasicBlock(
                        in_channels=in_channels_i,
                        out_channels=out_channels,
                        kernel_size=self.kernel_size,
                        stride=self.stride,
                        groups=self.groups,
                        downsample=downsample,
                        use_bn=self.use_bn,
                        use_do=self.use_do,
                        is_first_block=is_first_block,
                    )
                )
            if self.use_projection:
                self.projector = nn.Sequential(
                    nn.Linear(out_channels, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                )
            self.final_bn = nn.BatchNorm1d(out_channels)
            self.final_relu = nn.ReLU(inplace=True)
            self.dense = nn.Linear(out_channels, n_classes)
            self.expert_layers_1 = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(out_channels, out_channels // 2),
                        nn.ReLU(),
                        nn.Linear(out_channels // 2, 1),
                    )
                    for _ in range(self.n_experts)
                ]
            )
            self.gating_network_1 = nn.Sequential(nn.Linear(out_channels, self.n_experts), nn.Softmax(dim=1))
            self.expert_layers_2 = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(out_channels, out_channels // 2),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(out_channels // 2, 1),
                    )
                    for _ in range(self.n_experts)
                ]
            )
            self.gating_network_2 = nn.Sequential(nn.Linear(out_channels, self.n_experts), nn.Softmax(dim=1))

        def forward(self, x: Any) -> tuple[Any, Any, Any, Any]:
            out = self.first_block_conv(x)
            if self.use_bn:
                out = self.first_block_bn(out)
            out = self.first_block_relu(out)
            for block in self.basicblock_list:
                out = block(out)
            if self.use_bn:
                out = self.final_bn(out)
            out = self.final_relu(out)
            out = out.mean(-1)
            out_class = self.projector(out) if self.use_projection else self.dense(out)
            expert_outputs_1 = torch.stack([expert(out) for expert in self.expert_layers_1], dim=1)
            gate_weights_1 = self.gating_network_1(out)
            out_moe1 = torch.sum(gate_weights_1.unsqueeze(2) * expert_outputs_1, dim=1)
            expert_outputs_2 = torch.stack([expert(out) for expert in self.expert_layers_2], dim=1)
            gate_weights_2 = self.gating_network_2(out)
            out_moe2 = torch.sum(gate_weights_2.unsqueeze(2) * expert_outputs_2, dim=1)
            return out_class, out_moe1, out_moe2, out

    config = PAPAGEI_MODEL_CONFIG
    model = ResNet1DMoE(
        in_channels=1,
        base_filters=config["base_filters"],
        kernel_size=config["kernel_size"],
        stride=config["stride"],
        groups=config["groups"],
        n_block=config["n_block"],
        n_classes=config["n_classes"],
        n_experts=config["n_experts"],
    )
    checkpoint = torch.load(str(_to_abs_path(weights_path)), map_location=device)
    state_dict = {
        (key[7:] if str(key).startswith("module.") else key): value
        for key, value in checkpoint.items()
    }
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _load_case_segments(case_dir: Path) -> np.ndarray:
    segment_paths = sorted(case_dir.glob("*.npy"), key=lambda path: int(path.stem))
    if not segment_paths:
        raise ValueError(f"PaPaGei case directory has no segment files: {case_dir}")
    segments = [np.asarray(np.load(path), dtype=np.float32).reshape(-1) for path in segment_paths]
    lengths = {segment.size for segment in segments}
    if len(lengths) != 1:
        raise ValueError(f"PaPaGei segment length mismatch in {case_dir}: {sorted(lengths)}")
    return np.vstack(segments).astype(np.float32, copy=False)


def _extract_embeddings_for_split(
    *,
    model: Any,
    run_dir: Path,
    split: str,
    batch_size: int,
    device: Any,
    output_idx: int,
) -> dict[str, Any]:
    import torch

    prepared_dir = run_dir / PREPARED_DATA_DIRNAME
    labels_path = prepared_dir / PREPARED_CASE_LABELS
    labels = pd.read_csv(labels_path)
    split_labels = labels[labels["split"].astype(str) == split].copy()
    if split_labels.empty:
        raise ValueError(f"PaPaGei has no case labels for split {split!r}.")

    case_keys: list[str] = []
    embeddings: list[np.ndarray] = []
    segment_counts: list[int] = []
    segment_root = prepared_dir / SEGMENTS_DIRNAME / split

    with torch.inference_mode():
        for row in split_labels.to_dict(orient="records"):
            case_key = str(row["case_key"])
            case_segments = _load_case_segments(segment_root / case_key)
            segment_embeddings: list[np.ndarray] = []
            for start in range(0, case_segments.shape[0], int(batch_size)):
                batch = case_segments[start : start + int(batch_size)]
                tensor = torch.tensor(batch, dtype=torch.float32, device=device).unsqueeze(1)
                outputs = model(tensor)
                output = outputs[int(output_idx)]
                segment_embeddings.append(output.detach().cpu().numpy().astype(np.float32, copy=False))
            case_embedding = np.vstack(segment_embeddings).mean(axis=0)
            case_keys.append(case_key)
            embeddings.append(case_embedding.astype(np.float32, copy=False))
            segment_counts.append(int(case_segments.shape[0]))

    X = np.vstack(embeddings).astype(np.float32, copy=False)
    if X.shape[1] != PAPAGEI_EMBEDDING_DIM:
        raise ValueError(f"Expected PaPaGei embeddings with dim 512, got {X.shape}.")

    aligned = split_labels.set_index("case_key").loc[case_keys].reset_index()
    y = aligned[["sbp", "dbp", "map"]].to_numpy(dtype=np.float64)
    return {
        "split": split,
        "case_keys": np.asarray(case_keys, dtype=object),
        "embeddings": X,
        "targets": y,
        "segment_counts": np.asarray(segment_counts, dtype=np.int64),
    }


def _save_embedding_split(features_dir: Path, split_payload: dict[str, Any]) -> str:
    features_dir.mkdir(parents=True, exist_ok=True)
    path = features_dir / f"{split_payload['split']}_case_embeddings.npz"
    np.savez_compressed(
        path,
        case_keys=split_payload["case_keys"],
        embeddings=split_payload["embeddings"],
        targets=split_payload["targets"],
        segment_counts=split_payload["segment_counts"],
    )
    return str(path)


def _load_embedding_split(run_dir: str | Path, split: str) -> dict[str, Any]:
    path = _to_abs_path(run_dir) / FEATURES_DIRNAME / f"{split}_case_embeddings.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing PaPaGei embedding split: {path}")
    data = np.load(path, allow_pickle=True)
    return {
        "case_keys": data["case_keys"].astype(str),
        "embeddings": data["embeddings"].astype(np.float32),
        "targets": data["targets"].astype(np.float64),
        "segment_counts": data["segment_counts"].astype(np.int64),
    }


def _standard_scaler_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X64 = np.asarray(X, dtype=np.float64)
    mean = X64.mean(axis=0)
    scale = X64.std(axis=0)
    scale[~np.isfinite(scale)] = 1.0
    scale[scale == 0.0] = 1.0
    return mean, scale


def _standard_scaler_transform(X: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (np.asarray(X, dtype=np.float64) - mean) / scale


def _ridge_fit_svd(X: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    if alpha <= 0:
        raise ValueError("Ridge alpha must be positive.")
    X64 = np.asarray(X, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64)
    intercept = float(y64.mean())
    y_centered = y64 - intercept
    U, singular_values, Vt = np.linalg.svd(X64, full_matrices=False)
    shrink = singular_values / (singular_values * singular_values + float(alpha))
    coef = Vt.T @ (shrink * (U.T @ y_centered))
    return coef.astype(np.float64, copy=False), intercept


def _ridge_predict(X: np.ndarray, coef: np.ndarray, intercept: float) -> np.ndarray:
    return np.asarray(X, dtype=np.float64) @ np.asarray(coef, dtype=np.float64) + float(intercept)


def _kfold_validation_indices(n_samples: int, n_splits: int) -> list[np.ndarray]:
    if n_splits <= 1:
        raise ValueError("n_splits must be greater than 1.")
    if n_samples < n_splits:
        raise ValueError(f"Cannot split {n_samples} samples into {n_splits} folds.")
    indices = np.arange(n_samples)
    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
    fold_sizes[: n_samples % n_splits] += 1
    folds: list[np.ndarray] = []
    start = 0
    for fold_size in fold_sizes:
        stop = start + int(fold_size)
        folds.append(indices[start:stop])
        start = stop
    return folds


def _select_ridge_alpha_cv(X: np.ndarray, y: np.ndarray, alpha_grid: list[float], n_splits: int) -> dict[str, Any]:
    folds = _kfold_validation_indices(X.shape[0], n_splits)
    best_alpha: float | None = None
    best_mse = float("inf")
    cv_results: list[dict[str, Any]] = []
    for alpha in alpha_grid:
        fold_mse: list[float] = []
        for val_idx in folds:
            train_mask = np.ones(X.shape[0], dtype=bool)
            train_mask[val_idx] = False
            X_train = X[train_mask]
            y_train = y[train_mask]
            X_val = X[val_idx]
            y_val = y[val_idx]
            mean, scale = _standard_scaler_fit(X_train)
            X_train_scaled = _standard_scaler_transform(X_train, mean, scale)
            X_val_scaled = _standard_scaler_transform(X_val, mean, scale)
            coef, intercept = _ridge_fit_svd(X_train_scaled, y_train, float(alpha))
            pred = _ridge_predict(X_val_scaled, coef, intercept)
            fold_mse.append(float(np.mean((pred - y_val) ** 2)))
        mean_mse = float(np.mean(fold_mse))
        cv_results.append(
            {
                "alpha": float(alpha),
                "mean_test_mse": mean_mse,
                "fold_test_mse": fold_mse,
            }
        )
        if mean_mse < best_mse:
            best_mse = mean_mse
            best_alpha = float(alpha)
    if best_alpha is None:
        raise RuntimeError("Ridge alpha selection did not evaluate any candidates.")
    return {
        "best_alpha": best_alpha,
        "best_mean_test_mse": best_mse,
        "cv_results": cv_results,
    }


def _fit_predict_ridge(
    run_dir: Path,
    *,
    include_val_in_fit: bool,
    n_jobs: int,
    seed: int,
) -> dict[str, Any]:
    np.random.seed(int(seed))
    train = _load_embedding_split(run_dir, "train")
    test = _load_embedding_split(run_dir, "test")
    fit_X_parts = [train["embeddings"]]
    fit_y_parts = [train["targets"]]
    fit_splits = ["train"]
    if include_val_in_fit:
        val = _load_embedding_split(run_dir, "val")
        fit_X_parts.append(val["embeddings"])
        fit_y_parts.append(val["targets"])
        fit_splits.append("val")

    X_fit = np.vstack(fit_X_parts)
    y_fit = np.vstack(fit_y_parts)
    X_test = test["embeddings"]
    y_test = test["targets"]
    if X_fit.shape[0] < PAPAGEI_GRID_CV:
        raise ValueError(
            f"PaPaGei Ridge cross-validation requires at least {PAPAGEI_GRID_CV} fit cases; got {X_fit.shape[0]}."
        )

    models_dir = run_dir / MODELS_DIRNAME
    models_dir.mkdir(parents=True, exist_ok=True)
    predictions: dict[str, np.ndarray] = {}
    best_params: dict[str, Any] = {}
    cv_results: dict[str, Any] = {}
    model_paths: dict[str, str] = {}

    for metric_idx, metric_name in enumerate(SCALAR_METRICS):
        alpha_summary = _select_ridge_alpha_cv(
            X_fit,
            y_fit[:, metric_idx],
            alpha_grid=PAPAGEI_RIDGE_ALPHA_GRID,
            n_splits=PAPAGEI_GRID_CV,
        )
        mean, scale = _standard_scaler_fit(X_fit)
        X_fit_scaled = _standard_scaler_transform(X_fit, mean, scale)
        X_test_scaled = _standard_scaler_transform(X_test, mean, scale)
        coef, intercept = _ridge_fit_svd(X_fit_scaled, y_fit[:, metric_idx], alpha_summary["best_alpha"])
        predictions[metric_name] = _ridge_predict(X_test_scaled, coef, intercept).astype(np.float64)
        best_params[metric_name] = {
            "ridge__alpha": float(alpha_summary["best_alpha"]),
            "ridge__solver": PAPAGEI_RIDGE_SOLVER,
        }
        cv_results[metric_name] = alpha_summary
        model_path = models_dir / f"{metric_name}_ridge.npz"
        np.savez_compressed(
            model_path,
            mean=mean,
            scale=scale,
            coef=coef,
            intercept=np.asarray(intercept, dtype=np.float64),
            alpha=np.asarray(alpha_summary["best_alpha"], dtype=np.float64),
            solver=np.asarray(PAPAGEI_RIDGE_SOLVER),
        )
        model_paths[metric_name] = str(model_path)

    predictions_dir = run_dir / PREDICTIONS_DIRNAME
    predictions_dir.mkdir(parents=True, exist_ok=True)
    pred_frame = pd.DataFrame(
        {
            "case_key": test["case_keys"],
            "sbp_true": y_test[:, 0],
            "dbp_true": y_test[:, 1],
            "map_true": y_test[:, 2],
            "sbp_pred": predictions["sbp"],
            "dbp_pred": predictions["dbp"],
            "map_pred": predictions["map"],
            "n_segments": test["segment_counts"],
        }
    )
    predictions_path = predictions_dir / "test_scalar_predictions.csv"
    pred_frame.to_csv(predictions_path, index=False)

    return {
        "fit_splits": fit_splits,
        "n_fit_cases": int(X_fit.shape[0]),
        "n_test_cases": int(X_test.shape[0]),
        "predictions": str(predictions_path),
        "best_params": best_params,
        "cv_results": cv_results,
        "model_paths": model_paths,
        "ridge_implementation": {
            "scaler": "StandardScaler",
            "estimator": "Ridge",
            "solver": PAPAGEI_RIDGE_SOLVER,
            "cv": PAPAGEI_GRID_CV,
            "n_jobs_requested": int(n_jobs),
        },
    }


def evaluate_papagei_run(run_dir: str | Path) -> dict[str, Any]:
    run_dir = _to_abs_path(run_dir)
    predictions_path = run_dir / PREDICTIONS_DIRNAME / "test_scalar_predictions.csv"
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing PaPaGei scalar predictions: {predictions_path}")

    pred = pd.read_csv(predictions_path)
    required = {
        "sbp_true",
        "dbp_true",
        "map_true",
        "sbp_pred",
        "dbp_pred",
        "map_pred",
    }
    missing = required - set(pred.columns)
    if missing:
        raise ValueError(f"PaPaGei predictions are missing columns: {sorted(missing)}")

    summaries = {
        metric: scalar_metric_summary(
            pred[f"{metric}_pred"].to_numpy(dtype=np.float64)
            - pred[f"{metric}_true"].to_numpy(dtype=np.float64)
        )
        for metric in SCALAR_METRICS
    }
    scalar = {
        "n_beats": int(max(summaries["sbp"]["n_beats"], summaries["dbp"]["n_beats"], summaries["map"]["n_beats"])),
        "n_scalar_predictions": int(len(pred)),
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

    metrics_dir = run_dir / METRICS_DIRNAME
    metrics_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "baseline_key": PAPAGEI_BASELINE_KEY,
        "run_dir": str(run_dir),
        "predictions": str(predictions_path),
        "output_type": "scalar_only",
        "waveform": {
            "n_points": 0,
            "waveform_mae": "nan",
            "waveform_rmse": "nan",
            "waveform_corr": "nan",
            "reason": "PaPaGei is used as a frozen PPG feature extractor with scalar Ridge heads and does not generate ABP waveforms.",
        },
        "scalar": scalar,
    }
    metrics_path = metrics_dir / METRICS_JSON
    metrics_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def run_papagei_benchmark(
    run_dir: str | Path,
    upstream_repo: str | Path,
    weights_path: str | Path,
    *,
    batch_size: int = PAPAGEI_BATCH_SIZE,
    device: str = "cuda",
    output_idx: int = PAPAGEI_OUTPUT_IDX,
    include_val_in_fit: bool = True,
    n_jobs: int = -1,
    seed: int = PAPAGEI_SEED,
) -> dict[str, Any]:
    import torch

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if output_idx != PAPAGEI_OUTPUT_IDX:
        raise ValueError("PaPaGei-S official embedding output_idx is 0; this launcher does not expose other outputs.")

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    run_dir = _to_abs_path(run_dir)
    weights_path = _to_abs_path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing PaPaGei weights: {weights_path}")
    _load_prepared_config(run_dir)

    features_dir = run_dir / FEATURES_DIRNAME
    feature_paths: dict[str, str] = {}
    torch_device = torch.device(device)
    model = _build_papagei_s_model(upstream_repo, weights_path, torch_device)
    for split in EMBEDDING_SPLITS:
        payload = _extract_embeddings_for_split(
            model=model,
            run_dir=run_dir,
            split=split,
            batch_size=int(batch_size),
            device=torch_device,
            output_idx=int(output_idx),
        )
        feature_paths[split] = _save_embedding_split(features_dir, payload)

    regression_summary = _fit_predict_ridge(
        run_dir,
        include_val_in_fit=bool(include_val_in_fit),
        n_jobs=int(n_jobs),
        seed=int(seed),
    )
    metrics_summary = evaluate_papagei_run(run_dir)

    summary = {
        "baseline_key": PAPAGEI_BASELINE_KEY,
        "run_dir": str(run_dir),
        "upstream_repo": str(_to_abs_path(upstream_repo)),
        "weights_path": str(weights_path),
        "features": feature_paths,
        "regression": regression_summary,
        "metrics": metrics_summary,
        "metrics_json": str(run_dir / METRICS_DIRNAME / METRICS_JSON),
        "batch_size": int(batch_size),
        "device": str(torch_device),
        "output_idx": int(output_idx),
        "include_val_in_fit": bool(include_val_in_fit),
        "seed": int(seed),
        "linear_probe": {
            "estimator": "Ridge",
            "scaler": "StandardScaler",
            "alpha_grid": PAPAGEI_RIDGE_ALPHA_GRID,
            "solver": PAPAGEI_RIDGE_SOLVER,
            "cv": PAPAGEI_GRID_CV,
        },
    }
    summary_path = run_dir / RUN_RESULTS_JSON
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary
