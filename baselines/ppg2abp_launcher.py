from __future__ import annotations

import csv
import importlib
import json
import pickle
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .evaluate import RunningScalarErrorMetrics, RunningWaveformMetrics
from .metrics import bp_from_waveforms


PPG2ABP_BASELINE_KEY = "ppg2abp"
PPG2ABP_TARGET_FS_HZ = 125.0
PPG2ABP_WINDOW_SEC = 10.0
PPG2ABP_STEP_SEC = 5.0
PPG2ABP_MODEL_INPUT_LENGTH = 1024
PPG2ABP_LABEL_CHUNK_SIZE = 8192
PPG2ABP_PREDICT_CHUNK_SIZE = 8192
PPG2ABP_EVAL_CHUNK_SIZE = 2048

PREPARED_DATA_DIRNAME = "prepared_data"
MEMMAP_DATA_DIRNAME = "memmap"
LABEL_DATA_DIRNAME = "labels"
STAGE2_DATA_DIRNAME = "stage2"
PREPARED_WINDOWS_MANIFEST = "windows_manifest.csv"
PREPARED_CONFIG_JSON = "ppg2abp_benchmark_config.json"
PREPARED_SUMMARY_JSON = "ppg2abp_prepare_summary.json"
META_PICKLE_NAME = "meta_benchmark.p"
UPSTREAM_RUN_NOTES = "UPSTREAM_PPG2ABP_NOTES.txt"
TRAIN_HISTORY_DIRNAME = "History"
TRAIN_MODELS_DIRNAME = "models"
PREDICTIONS_DIRNAME = "predictions"
METRICS_DIRNAME = "metrics"
TRAIN_RESULTS_JSON = "ppg2abp_train_summary.json"
METRICS_JSON = "ppg2abp_metrics.json"
PREDICTIONS_MANIFEST_JSON = "test_predictions_manifest.json"

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
PREPARED_WINDOWS_COLUMNS = [
    "split",
    "window_id",
    "memmap_index",
    "source",
    "subject_id",
    "record_id",
    "hadm_id",
    "source_record_rows",
    "source_fs_hz",
    "target_fs_hz",
    "window_sec",
    "step_sec",
    "source_start_index",
    "source_end_index",
    "source_start_sec",
    "source_end_sec",
    "resampled_window_length",
    "model_input_length",
    "source_exported_parquet",
    "ppg_memmap",
    "abp_memmap",
]


def _to_abs_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _require_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")


def _require_split_coverage(frame: pd.DataFrame) -> None:
    required = {"train", "val", "test"}
    present = set(frame["split"].dropna().astype(str))
    missing = sorted(required - present)
    if missing:
        raise ValueError(
            "PPG2ABP benchmark preparation requires train, val, and test splits. "
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
    if unique_keys != {PPG2ABP_BASELINE_KEY}:
        raise ValueError(
            "PPG2ABP launcher only accepts export manifests for baseline_key='ppg2abp'. "
            f"Found: {sorted(unique_keys)}"
        )

    if merged.duplicated(subset=["split", "record_id", "exported_parquet"], keep=False).any():
        raise ValueError("Export manifests contain duplicate split/record/exported_parquet rows.")

    _require_split_coverage(merged)
    return merged


def _resample_window(signal: np.ndarray, target_length: int) -> np.ndarray:
    if signal.ndim != 1:
        raise ValueError(f"Expected a 1D signal window, got shape {signal.shape}.")
    if signal.size < 2:
        raise ValueError("Signal window must contain at least 2 samples for resampling.")
    if not np.isfinite(signal).all():
        raise ValueError("Signal window contains non-finite values.")

    source_pos = np.linspace(0.0, 1.0, num=signal.size, endpoint=False, dtype=np.float64)
    target_pos = np.linspace(0.0, 1.0, num=target_length, endpoint=False, dtype=np.float64)
    return np.interp(target_pos, source_pos, signal).astype(np.float32, copy=False)


def _memmap_split_paths(prepared_dir: Path, split: str) -> tuple[Path, Path]:
    memmap_dir = prepared_dir / MEMMAP_DATA_DIRNAME
    return memmap_dir / f"{split}_X.npy", memmap_dir / f"{split}_Y.npy"


def _scan_record_windows(
    record_frame: pd.DataFrame,
    *,
    source_fs_hz: float,
    target_fs_hz: float,
    window_sec: float,
    step_sec: float,
    model_input_length: int,
    update_minmax: bool,
) -> dict[str, Any]:
    source_window_length = int(round(source_fs_hz * window_sec))
    source_step_length = int(round(source_fs_hz * step_sec))
    target_window_length = int(round(target_fs_hz * window_sec))

    if source_window_length <= 1:
        raise ValueError("window_sec is too small for the source sampling rate.")
    if source_step_length <= 0:
        raise ValueError("step_sec is too small for the source sampling rate.")
    if model_input_length > target_window_length:
        raise ValueError(
            "model_input_length cannot exceed the resampled window length. "
            f"Got model_input_length={model_input_length}, target_window_length={target_window_length}."
        )

    ppg = np.asarray(record_frame["PLETH"], dtype=np.float64)
    abp = np.asarray(record_frame["ABP"], dtype=np.float64)
    total_length = int(len(record_frame))

    stats: dict[str, Any] = {
        "candidate_windows": 0,
        "valid_windows": 0,
        "dropped_nonfinite_windows": 0,
        "min_ppg": None,
        "max_ppg": None,
        "min_abp": None,
        "max_abp": None,
    }

    for start in range(0, total_length - source_window_length + 1, source_step_length):
        end = start + source_window_length
        stats["candidate_windows"] += 1
        ppg_window = ppg[start:end]
        abp_window = abp[start:end]
        if ppg_window.size != source_window_length or abp_window.size != source_window_length:
            continue
        if not np.isfinite(ppg_window).all() or not np.isfinite(abp_window).all():
            stats["dropped_nonfinite_windows"] += 1
            continue

        stats["valid_windows"] += 1
        if update_minmax:
            ppg_model = _resample_window(ppg_window, target_window_length)[:model_input_length]
            abp_model = _resample_window(abp_window, target_window_length)[:model_input_length]
            ppg_min = float(np.min(ppg_model))
            ppg_max = float(np.max(ppg_model))
            abp_min = float(np.min(abp_model))
            abp_max = float(np.max(abp_model))
            stats["min_ppg"] = ppg_min if stats["min_ppg"] is None else min(stats["min_ppg"], ppg_min)
            stats["max_ppg"] = ppg_max if stats["max_ppg"] is None else max(stats["max_ppg"], ppg_max)
            stats["min_abp"] = abp_min if stats["min_abp"] is None else min(stats["min_abp"], abp_min)
            stats["max_abp"] = abp_max if stats["max_abp"] is None else max(stats["max_abp"], abp_max)

    return stats


def _write_record_windows(
    record_frame: pd.DataFrame,
    *,
    source_fs_hz: float,
    target_fs_hz: float,
    window_sec: float,
    step_sec: float,
    model_input_length: int,
    ppg_memmap: np.memmap,
    abp_memmap: np.memmap,
    start_memmap_index: int,
    min_ppg: float,
    max_ppg: float,
    min_abp: float,
    max_abp: float,
    metadata_base: dict[str, Any],
    ppg_memmap_path: Path,
    abp_memmap_path: Path,
    manifest_writer: csv.DictWriter,
) -> int:
    source_window_length = int(round(source_fs_hz * window_sec))
    source_step_length = int(round(source_fs_hz * step_sec))
    target_window_length = int(round(target_fs_hz * window_sec))
    ppg = np.asarray(record_frame["PLETH"], dtype=np.float64)
    abp = np.asarray(record_frame["ABP"], dtype=np.float64)
    total_length = int(len(record_frame))
    ppg_span = float(max_ppg - min_ppg)
    abp_span = float(max_abp - min_abp)
    if ppg_span <= 0.0 or abp_span <= 0.0:
        raise ValueError("PPG2ABP normalization span must be positive.")

    write_index = int(start_memmap_index)
    for start in range(0, total_length - source_window_length + 1, source_step_length):
        end = start + source_window_length
        ppg_window = ppg[start:end]
        abp_window = abp[start:end]
        if ppg_window.size != source_window_length or abp_window.size != source_window_length:
            continue
        if not np.isfinite(ppg_window).all() or not np.isfinite(abp_window).all():
            continue

        ppg_model = _resample_window(ppg_window, target_window_length)[:model_input_length]
        abp_model = _resample_window(abp_window, target_window_length)[:model_input_length]
        ppg_memmap[write_index, :, 0] = _normalize_pair(ppg_model, lower=min_ppg, upper=max_ppg)
        abp_memmap[write_index, :, 0] = _normalize_pair(abp_model, lower=min_abp, upper=max_abp)

        row = {
            **metadata_base,
            "window_id": f"{metadata_base['split']}_{write_index:08d}",
            "memmap_index": int(write_index),
            "source_record_rows": total_length,
            "source_fs_hz": float(source_fs_hz),
            "target_fs_hz": float(target_fs_hz),
            "window_sec": float(window_sec),
            "step_sec": float(step_sec),
            "source_start_index": int(start),
            "source_end_index": int(end),
            "source_start_sec": float(start / source_fs_hz),
            "source_end_sec": float(end / source_fs_hz),
            "resampled_window_length": int(target_window_length),
            "model_input_length": int(model_input_length),
            "ppg_memmap": str(ppg_memmap_path),
            "abp_memmap": str(abp_memmap_path),
        }
        manifest_writer.writerow(row)
        write_index += 1

    return write_index - int(start_memmap_index)


def _normalize_pair(
    values: np.ndarray,
    *,
    lower: float,
    upper: float,
) -> np.ndarray:
    span = upper - lower
    if span <= 0:
        raise ValueError("Normalization span must be positive.")
    return ((values - lower) / span).astype(np.float32, copy=False)


def _save_pickle(path: Path, payload: Any) -> None:
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def prepare_ppg2abp_run(
    export_manifests: list[str | Path],
    output_dir: str | Path,
    *,
    target_fs_hz: float = PPG2ABP_TARGET_FS_HZ,
    window_sec: float = PPG2ABP_WINDOW_SEC,
    step_sec: float = PPG2ABP_STEP_SEC,
    model_input_length: int = PPG2ABP_MODEL_INPUT_LENGTH,
) -> dict[str, Any]:
    _require_positive("target_fs_hz", target_fs_hz)
    _require_positive("window_sec", window_sec)
    _require_positive("step_sec", step_sec)
    if model_input_length <= 1:
        raise ValueError("model_input_length must be greater than 1.")
    if model_input_length % 16 != 0:
        raise ValueError("model_input_length must be divisible by 16 for PPG2ABP deep supervision labels.")

    export_frame = _load_export_manifests(export_manifests)
    output_dir = _to_abs_path(output_dir)
    prepared_dir = output_dir / PREPARED_DATA_DIRNAME
    prepared_dir.mkdir(parents=True, exist_ok=True)

    split_candidate_window_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    split_valid_window_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    split_dropped_nonfinite_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    split_record_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    min_ppg: float | None = None
    max_ppg: float | None = None
    min_abp: float | None = None
    max_abp: float | None = None

    # First pass: count finite windows and compute the train/val min-max normalization.
    for row in export_frame.to_dict(orient="records"):
        split = str(row["split"])
        record_path = _to_abs_path(row["exported_parquet"])
        if not record_path.exists():
            raise FileNotFoundError(f"Exported parquet does not exist: {record_path}")

        frame = pd.read_parquet(record_path, columns=["PLETH", "ABP"])
        stats = _scan_record_windows(
            frame,
            source_fs_hz=float(row["fs"]),
            target_fs_hz=target_fs_hz,
            window_sec=window_sec,
            step_sec=step_sec,
            model_input_length=model_input_length,
            update_minmax=(split in {"train", "val"}),
        )

        split_record_counts[split] += 1
        split_candidate_window_counts[split] += int(stats["candidate_windows"])
        split_valid_window_counts[split] += int(stats["valid_windows"])
        split_dropped_nonfinite_counts[split] += int(stats["dropped_nonfinite_windows"])
        if split in {"train", "val"} and stats["valid_windows"]:
            min_ppg = stats["min_ppg"] if min_ppg is None else min(min_ppg, stats["min_ppg"])
            max_ppg = stats["max_ppg"] if max_ppg is None else max(max_ppg, stats["max_ppg"])
            min_abp = stats["min_abp"] if min_abp is None else min(min_abp, stats["min_abp"])
            max_abp = stats["max_abp"] if max_abp is None else max(max_abp, stats["max_abp"])

    if any(split_valid_window_counts[split] == 0 for split in ("train", "val", "test")):
        raise ValueError(
            "PPG2ABP preparation produced an empty split. "
            f"train={split_valid_window_counts['train']}, "
            f"val={split_valid_window_counts['val']}, "
            f"test={split_valid_window_counts['test']}"
        )
    if min_ppg is None or max_ppg is None or min_abp is None or max_abp is None:
        raise ValueError("PPG2ABP preparation could not compute train/val normalization statistics.")
    if max_ppg <= min_ppg or max_abp <= min_abp:
        raise ValueError("PPG2ABP normalization span must be positive.")

    memmap_paths: dict[str, dict[str, str]] = {}
    ppg_memmaps: dict[str, np.memmap] = {}
    abp_memmaps: dict[str, np.memmap] = {}
    for split in ("train", "val", "test"):
        ppg_path, abp_path = _memmap_split_paths(prepared_dir, split)
        ppg_path.parent.mkdir(parents=True, exist_ok=True)
        ppg_memmaps[split] = np.lib.format.open_memmap(
            ppg_path,
            mode="w+",
            dtype=np.float32,
            shape=(split_valid_window_counts[split], int(model_input_length), 1),
        )
        abp_memmaps[split] = np.lib.format.open_memmap(
            abp_path,
            mode="w+",
            dtype=np.float32,
            shape=(split_valid_window_counts[split], int(model_input_length), 1),
        )
        memmap_paths[split] = {"X": str(ppg_path), "Y": str(abp_path)}

    windows_manifest_path = prepared_dir / PREPARED_WINDOWS_MANIFEST
    write_indices: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    with windows_manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREPARED_WINDOWS_COLUMNS)
        writer.writeheader()
        for row in export_frame.to_dict(orient="records"):
            split = str(row["split"])
            record_path = _to_abs_path(row["exported_parquet"])
            frame = pd.read_parquet(record_path, columns=["PLETH", "ABP"])
            ppg_path, abp_path = _memmap_split_paths(prepared_dir, split)
            written = _write_record_windows(
                frame,
                source_fs_hz=float(row["fs"]),
                target_fs_hz=target_fs_hz,
                window_sec=window_sec,
                step_sec=step_sec,
                model_input_length=model_input_length,
                ppg_memmap=ppg_memmaps[split],
                abp_memmap=abp_memmaps[split],
                start_memmap_index=write_indices[split],
                min_ppg=float(min_ppg),
                max_ppg=float(max_ppg),
                min_abp=float(min_abp),
                max_abp=float(max_abp),
                metadata_base={
                    "split": split,
                    "source": row["source"],
                    "subject_id": row["subject_id"],
                    "record_id": row["record_id"],
                    "hadm_id": row.get("hadm_id", np.nan),
                    "source_exported_parquet": str(record_path),
                },
                ppg_memmap_path=ppg_path,
                abp_memmap_path=abp_path,
                manifest_writer=writer,
            )
            write_indices[split] += int(written)

    for split in ("train", "val", "test"):
        ppg_memmaps[split].flush()
        abp_memmaps[split].flush()
        if write_indices[split] != split_valid_window_counts[split]:
            raise ValueError(
                f"PPG2ABP memmap write count mismatch for {split}: "
                f"expected {split_valid_window_counts[split]}, wrote {write_indices[split]}"
            )

    meta_pickle_path = prepared_dir / META_PICKLE_NAME
    meta_payload = {
        "max_ppg": float(max_ppg),
        "min_ppg": float(min_ppg),
        "max_abp": float(max_abp),
        "min_abp": float(min_abp),
        "target_fs_hz": float(target_fs_hz),
        "window_sec": float(window_sec),
        "step_sec": float(step_sec),
        "model_input_length": int(model_input_length),
        "window_seconds_for_model": float(model_input_length / target_fs_hz),
        "normalization_reference_splits": ["train", "val"],
        "storage_format": "memmap_npy",
        "memmap_files": memmap_paths,
    }
    _save_pickle(meta_pickle_path, meta_payload)

    config = {
        "baseline_key": PPG2ABP_BASELINE_KEY,
        "export_manifests": [str(_to_abs_path(path)) for path in export_manifests],
        "output_dir": str(output_dir),
        "prepared_data_dir": str(prepared_dir),
        "storage_format": "memmap_npy",
        "memmap_files": memmap_paths,
        "target_fs_hz": float(target_fs_hz),
        "window_sec": float(window_sec),
        "step_sec": float(step_sec),
        "model_input_length": int(model_input_length),
        "normalization_reference_splits": ["train", "val"],
        "normalization": {
            "min_ppg": float(min_ppg),
            "max_ppg": float(max_ppg),
            "min_abp": float(min_abp),
            "max_abp": float(max_abp),
        },
        "candidate_windows": split_candidate_window_counts,
        "split_windows": split_valid_window_counts,
        "dropped_nonfinite_windows": split_dropped_nonfinite_counts,
        "prepared_files": {
            "meta_pickle": str(meta_pickle_path),
            "windows_manifest": str(windows_manifest_path),
        },
    }
    config_path = output_dir / PREPARED_CONFIG_JSON
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    notes = [
        "baseline_key=ppg2abp",
        "upstream_code_policy=keep_original_code_untouched",
        f"target_fs_hz={target_fs_hz}",
        f"window_sec={window_sec}",
        f"step_sec={step_sec}",
        f"model_input_length={model_input_length}",
        "source_backed_defaults=fs125_t10_dt5_length1024",
        "rule=window extraction happens outside the upstream repo",
        "rule=only PLETH is exposed as model input",
        "rule=ABP remains the target waveform",
        "rule=windows with non-finite samples are dropped instead of being imputed",
        "storage_format=memmap_npy",
        f"prepared_data_dir={prepared_dir}",
        f"config_json={config_path}",
    ]
    (output_dir / UPSTREAM_RUN_NOTES).write_text("\n".join(notes) + "\n", encoding="utf-8")

    summary = {
        "baseline_key": PPG2ABP_BASELINE_KEY,
        "output_dir": str(output_dir),
        "prepared_data_dir": str(prepared_dir),
        "storage_format": "memmap_npy",
        "memmap_files": memmap_paths,
        "candidate_windows": split_candidate_window_counts,
        "window_counts": split_valid_window_counts,
        "split_windows": split_valid_window_counts,
        "dropped_nonfinite_windows": split_dropped_nonfinite_counts,
        "record_counts": split_record_counts,
        "target_fs_hz": float(target_fs_hz),
        "window_sec": float(window_sec),
        "step_sec": float(step_sec),
        "model_input_length": int(model_input_length),
        "prepared_files": config["prepared_files"],
    }
    summary_path = output_dir / PREPARED_SUMMARY_JSON
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


@contextmanager
def _temporary_upstream_imports(codes_dir: Path):
    module_names = ["metrics", "helper_functions", "models"]
    original_modules = {name: sys.modules.get(name) for name in module_names}
    codes_dir_str = str(codes_dir)
    sys.path.insert(0, codes_dir_str)
    for name in module_names:
        sys.modules.pop(name, None)
    try:
        yield
    finally:
        if codes_dir_str in sys.path:
            sys.path.remove(codes_dir_str)
        for name in module_names:
            sys.modules.pop(name, None)
            original = original_modules[name]
            if original is not None:
                sys.modules[name] = original


def _load_prepared_ppg2abp_data(
    run_dir: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    run_dir = _to_abs_path(run_dir)
    prepared_dir = run_dir / PREPARED_DATA_DIRNAME
    with (prepared_dir / META_PICKLE_NAME).open("rb") as handle:
        meta = pickle.load(handle)

    if meta.get("storage_format") == "memmap_npy":
        memmap_files = meta.get("memmap_files", {})
        train = {
            "X_train": np.load(memmap_files["train"]["X"], mmap_mode="r"),
            "Y_train": np.load(memmap_files["train"]["Y"], mmap_mode="r"),
        }
        val = {
            "X_val": np.load(memmap_files["val"]["X"], mmap_mode="r"),
            "Y_val": np.load(memmap_files["val"]["Y"], mmap_mode="r"),
        }
        test = {
            "X_test": np.load(memmap_files["test"]["X"], mmap_mode="r"),
            "Y_test": np.load(memmap_files["test"]["Y"], mmap_mode="r"),
        }
        return train, val, test, meta

    raise ValueError(
        "PPG2ABP prepared data must use storage_format='memmap_npy'. "
        "Re-run prepare-ppg2abp-run with the current launcher."
    )


def _label_memmap_paths(run_dir: Path, split: str) -> dict[str, Path]:
    labels_dir = run_dir / PREPARED_DATA_DIRNAME / LABEL_DATA_DIRNAME
    return {
        "level1": labels_dir / f"{split}_level1.npy",
        "level2": labels_dir / f"{split}_level2.npy",
        "level3": labels_dir / f"{split}_level3.npy",
        "level4": labels_dir / f"{split}_level4.npy",
    }


def _load_or_create_label_memmaps(
    run_dir: Path,
    split: str,
    y: np.ndarray,
    *,
    length: int,
    chunk_size: int = PPG2ABP_LABEL_CHUNK_SIZE,
) -> dict[str, np.ndarray]:
    paths = _label_memmap_paths(run_dir, split)
    paths["level1"].parent.mkdir(parents=True, exist_ok=True)
    level_shapes = {
        "level1": (len(y), length // 2, 1),
        "level2": (len(y), length // 4, 1),
        "level3": (len(y), length // 8, 1),
        "level4": (len(y), length // 16, 1),
    }
    if all(path.exists() for path in paths.values()):
        loaded = {level: np.load(path, mmap_mode="r") for level, path in paths.items()}
        for level, array in loaded.items():
            if tuple(array.shape) != level_shapes[level]:
                raise ValueError(
                    f"Existing PPG2ABP label memmap has wrong shape for {split}/{level}: "
                    f"{array.shape} vs {level_shapes[level]}"
                )
        return {
            "out": y,
            **loaded,
        }

    memmaps = {}
    for level, path in paths.items():
        memmaps[level] = np.lib.format.open_memmap(
            path,
            mode="w+",
            dtype=np.float32,
            shape=level_shapes[level],
        )
    for start in range(0, len(y), int(chunk_size)):
        end = min(start + int(chunk_size), len(y))
        chunk = np.asarray(y[start:end], dtype=np.float32).reshape(end - start, length)
        memmaps["level1"][start:end, :, 0] = chunk.reshape(end - start, length // 2, 2).mean(axis=2)
        memmaps["level2"][start:end, :, 0] = chunk.reshape(end - start, length // 4, 4).mean(axis=2)
        memmaps["level3"][start:end, :, 0] = chunk.reshape(end - start, length // 8, 8).mean(axis=2)
        memmaps["level4"][start:end, :, 0] = chunk.reshape(end - start, length // 16, 16).mean(axis=2)
    for memmap in memmaps.values():
        memmap.flush()
    return {
        "out": y,
        **{level: np.load(path, mmap_mode="r") for level, path in paths.items()},
    }


def _stage2_memmap_path(run_dir: Path, split: str) -> Path:
    return run_dir / PREPARED_DATA_DIRNAME / STAGE2_DATA_DIRNAME / f"{split}_stage2_X.npy"


def _predict_to_memmap(
    model: Any,
    x: np.ndarray,
    output_path: Path,
    *,
    batch_size: int,
    chunk_size: int = PPG2ABP_PREDICT_CHUNK_SIZE,
    output_index: int | None = None,
) -> np.memmap:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_memmap: np.memmap | None = None
    for start in range(0, len(x), int(chunk_size)):
        end = min(start + int(chunk_size), len(x))
        pred = model.predict(x[start:end], batch_size=int(batch_size), verbose=1)
        if output_index is not None:
            if not isinstance(pred, (list, tuple)):
                raise ValueError("Expected a multi-output PPG2ABP stage-1 prediction.")
            pred = pred[output_index]
        pred = np.asarray(pred, dtype=np.float32)
        if output_memmap is None:
            output_memmap = np.lib.format.open_memmap(
                output_path,
                mode="w+",
                dtype=np.float32,
                shape=(len(x),) + tuple(pred.shape[1:]),
            )
        output_memmap[start:end] = pred
    if output_memmap is None:
        raise ValueError("Cannot predict an empty PPG2ABP array.")
    output_memmap.flush()
    return output_memmap


def _load_ppg2abp_prediction_array(run_dir: Path) -> np.ndarray:
    predictions_dir = run_dir / PREDICTIONS_DIRNAME
    manifest_path = predictions_dir / PREDICTIONS_MANIFEST_JSON
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing PPG2ABP prediction manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return np.load(manifest["test_output"], mmap_mode="r")


def evaluate_ppg2abp_run(run_dir: str | Path) -> dict[str, Any]:
    run_dir = _to_abs_path(run_dir)
    metrics_dir = run_dir / METRICS_DIRNAME
    metrics_dir.mkdir(parents=True, exist_ok=True)

    _, _, test, meta = _load_prepared_ppg2abp_data(run_dir)
    y_true_norm = test["Y_test"]
    y_pred_norm = _load_ppg2abp_prediction_array(run_dir)

    if y_true_norm.shape != y_pred_norm.shape:
        raise ValueError(f"Prediction shape mismatch: {y_true_norm.shape} vs {y_pred_norm.shape}")

    abp_span = float(meta["max_abp"] - meta["min_abp"])
    abp_min = float(meta["min_abp"])
    window_seconds = float(meta["window_seconds_for_model"])

    waveform_metrics = RunningWaveformMetrics()
    scalar_metrics = RunningScalarErrorMetrics()
    for start in range(0, len(y_true_norm), PPG2ABP_EVAL_CHUNK_SIZE):
        end = min(start + PPG2ABP_EVAL_CHUNK_SIZE, len(y_true_norm))
        y_true = np.asarray(y_true_norm[start:end], dtype=np.float32) * abp_span + abp_min
        y_pred = np.asarray(y_pred_norm[start:end], dtype=np.float32) * abp_span + abp_min
        waveform_metrics.update(y_true, y_pred)
        true_bp = bp_from_waveforms(y_true, window_seconds=window_seconds)
        pred_bp = {"waveforms": y_pred.reshape(y_true.shape[0], y_true.shape[1])}
        scalar_metrics.update(true_bp, pred_bp)

    waveform = waveform_metrics.finalize()
    scalar_summary = scalar_metrics.finalize()

    summary = {
        "baseline_key": PPG2ABP_BASELINE_KEY,
        "run_dir": str(run_dir),
        "window_seconds": window_seconds,
        "n_test_windows": int(len(y_true_norm)),
        "waveform": waveform,
        "scalar": scalar_summary,
    }
    metrics_path = metrics_dir / METRICS_JSON
    metrics_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def run_ppg2abp_benchmark(
    run_dir: str | Path,
    upstream_repo: str | Path,
    *,
    epochs_stage1: int = 100,
    epochs_stage2: int = 100,
    batch_size_stage1: int = 256,
    batch_size_stage2: int = 192,
) -> dict[str, Any]:
    run_dir = _to_abs_path(run_dir)
    upstream_repo = _to_abs_path(upstream_repo)
    codes_dir = upstream_repo / "codes"
    if not codes_dir.exists():
        raise FileNotFoundError(f"Expected PPG2ABP codes directory at {codes_dir}")

    train, val, test, meta = _load_prepared_ppg2abp_data(run_dir)
    history_dir = run_dir / TRAIN_HISTORY_DIRNAME
    model_dir = run_dir / TRAIN_MODELS_DIRNAME
    predictions_dir = run_dir / PREDICTIONS_DIRNAME
    history_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    X_train = train["X_train"]
    Y_train = train["Y_train"]
    X_val = val["X_val"]
    Y_val = val["Y_val"]
    X_test = test["X_test"]

    length = int(meta["model_input_length"])

    with _temporary_upstream_imports(codes_dir):
        models = importlib.import_module("models")
        from keras.callbacks import ModelCheckpoint

        Y_train_labels = _load_or_create_label_memmaps(run_dir, "train", Y_train, length=length)
        Y_val_labels = _load_or_create_label_memmaps(run_dir, "val", Y_val, length=length)

        mdl1 = models.UNetDS64(length)
        mdl1.compile(
            loss=["mean_absolute_error"] * 5,
            optimizer="adam",
            metrics=None,
            loss_weights=[1.0, 0.9, 0.8, 0.7, 0.6],
            jit_compile=False,
        )

        stage1_weights = model_dir / "UNetDS64_model1_benchmark.weights.h5"
        checkpoint1 = ModelCheckpoint(
            str(stage1_weights),
            verbose=1,
            monitor="val_out_loss",
            save_best_only=True,
            save_weights_only=True,
            mode="min",
        )
        history1 = mdl1.fit(
            X_train,
            {
                "out": Y_train_labels["out"],
                "level1": Y_train_labels["level1"],
                "level2": Y_train_labels["level2"],
                "level3": Y_train_labels["level3"],
                "level4": Y_train_labels["level4"],
            },
            epochs=int(epochs_stage1),
            batch_size=int(batch_size_stage1),
            validation_data=(
                X_val,
                {
                    "out": Y_val_labels["out"],
                    "level1": Y_val_labels["level1"],
                    "level2": Y_val_labels["level2"],
                    "level3": Y_val_labels["level3"],
                    "level4": Y_val_labels["level4"],
                },
            ),
            callbacks=[checkpoint1],
            verbose=1,
        )
        _save_pickle(history_dir / "UNetDS64_model1_benchmark_history.p", history1.history)

        mdl1.load_weights(str(stage1_weights))
        X_train_stage2 = _predict_to_memmap(
            mdl1,
            X_train,
            _stage2_memmap_path(run_dir, "train"),
            batch_size=int(batch_size_stage1),
            output_index=0,
        )
        X_val_stage2 = _predict_to_memmap(
            mdl1,
            X_val,
            _stage2_memmap_path(run_dir, "val"),
            batch_size=int(batch_size_stage1),
            output_index=0,
        )

        mdl2 = models.MultiResUNet1D(length)
        mdl2.compile(
            loss="mean_squared_error",
            optimizer="adam",
            metrics=["mean_absolute_error"],
            jit_compile=False,
        )

        stage2_weights = model_dir / "MultiResUNet1D_model2_benchmark.weights.h5"
        checkpoint2 = ModelCheckpoint(
            str(stage2_weights),
            verbose=1,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            mode="min",
        )
        history2 = mdl2.fit(
            X_train_stage2,
            Y_train_labels["out"],
            epochs=int(epochs_stage2),
            batch_size=int(batch_size_stage2),
            validation_data=(X_val_stage2, Y_val_labels["out"]),
            callbacks=[checkpoint2],
            verbose=1,
        )
        _save_pickle(history_dir / "MultiResUNet1D_model2_benchmark_history.p", history2.history)

        mdl2.load_weights(str(stage2_weights))
        y_test_pred_approximate = _predict_to_memmap(
            mdl1,
            X_test,
            predictions_dir / "test_output_approximate_out.npy",
            batch_size=int(batch_size_stage1),
            output_index=0,
        )
        y_test_pred = _predict_to_memmap(
            mdl2,
            y_test_pred_approximate,
            predictions_dir / "test_output.npy",
            batch_size=int(batch_size_stage2),
            output_index=None,
        )

    predictions_manifest = {
        "storage_format": "memmap_npy",
        "test_output_approximate_out": str(predictions_dir / "test_output_approximate_out.npy"),
        "test_output": str(predictions_dir / "test_output.npy"),
        "prediction_shape": list(y_test_pred.shape),
    }
    (predictions_dir / PREDICTIONS_MANIFEST_JSON).write_text(
        json.dumps(predictions_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    metrics_summary = evaluate_ppg2abp_run(run_dir)
    train_summary = {
        "baseline_key": PPG2ABP_BASELINE_KEY,
        "run_dir": str(run_dir),
        "upstream_repo": str(upstream_repo),
        "epochs_stage1": int(epochs_stage1),
        "epochs_stage2": int(epochs_stage2),
        "batch_size_stage1": int(batch_size_stage1),
        "batch_size_stage2": int(batch_size_stage2),
        "model_input_length": length,
        "artifacts": {
            "stage1_weights": str(stage1_weights),
            "stage2_weights": str(stage2_weights),
            "stage1_history": str(history_dir / "UNetDS64_model1_benchmark_history.p"),
            "stage2_history": str(history_dir / "MultiResUNet1D_model2_benchmark_history.p"),
            "stage2_train_input": str(_stage2_memmap_path(run_dir, "train")),
            "stage2_val_input": str(_stage2_memmap_path(run_dir, "val")),
            "test_output_approximate": str(predictions_dir / "test_output_approximate_out.npy"),
            "test_output": str(predictions_dir / "test_output.npy"),
            "predictions_manifest": str(predictions_dir / PREDICTIONS_MANIFEST_JSON),
            "metrics_json": str(run_dir / METRICS_DIRNAME / METRICS_JSON),
        },
        "metrics": metrics_summary,
    }
    (run_dir / TRAIN_RESULTS_JSON).write_text(
        json.dumps(train_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return train_summary
