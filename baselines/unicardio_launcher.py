from __future__ import annotations

import csv
import importlib
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .evaluate import RunningScalarErrorMetrics, RunningWaveformMetrics
from .metrics import bp_from_waveforms


UNICARDIO_BASELINE_KEY = "unicardio"
UNICARDIO_SLOT_LENGTH = 500
UNICARDIO_TOTAL_LENGTH = UNICARDIO_SLOT_LENGTH * 4
UNICARDIO_WINDOW_SEC = 4.0
UNICARDIO_STEP_SEC = 4.0
UNICARDIO_BP_CENTER = 100.0
UNICARDIO_BP_SCALE = 50.0
UNICARDIO_PPG_TO_ECG_MODEL_FLAG = "02"
UNICARDIO_PPG_TO_ECG_BORROW_MODE = 2
UNICARDIO_ECG_TO_BP_MODEL_FLAG = "21"
UNICARDIO_ECG_TO_BP_BORROW_MODE = 2
UNICARDIO_N_SAMPLES = 50
UNICARDIO_DDIM_FLAG = 0
UNICARDIO_SAMPLE_STEPS = 6
UNICARDIO_DEFAULT_FINETUNE_LR = 1.0e-4

PREPARED_DATA_DIRNAME = "prepared_data"
MEMMAP_DATA_DIRNAME = "memmap"
PREPARED_WINDOWS_MANIFEST = "windows_manifest.csv"
PREPARED_CONFIG_JSON = "unicardio_benchmark_config.json"
PREPARED_SUMMARY_JSON = "unicardio_prepare_summary.json"
TRAIN_RESULTS_JSON = "unicardio_train_summary.json"
GENERATION_RESULTS_JSON = "unicardio_generation_summary.json"
PREDICTIONS_DIRNAME = "predictions"
METRICS_DIRNAME = "metrics"
TRAIN_MODELS_DIRNAME = "models"
TRAIN_HISTORY_DIRNAME = "history"
METRICS_JSON = "unicardio_metrics.json"
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
    "window_sec",
    "step_sec",
    "slot_length",
    "total_length",
    "source_start_index",
    "source_end_index",
    "source_start_sec",
    "source_end_sec",
    "source_exported_parquet",
    "full_signal_memmap",
]


def _to_abs_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _require_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}.")


def _require_split_coverage(frame: pd.DataFrame) -> None:
    required = {"train", "val", "test"}
    present = set(frame["split"].dropna().astype(str))
    missing = sorted(required - present)
    if missing:
        raise ValueError(
            "UniCardio benchmark preparation requires train, val, and test splits. "
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
    if unique_keys != {UNICARDIO_BASELINE_KEY}:
        raise ValueError(
            "UniCardio launcher only accepts export manifests for baseline_key='unicardio'. "
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


def _window_to_unicardio_unit_range(signal: np.ndarray, *, name: str) -> np.ndarray:
    values = np.asarray(signal, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values.")

    std = float(values.std())
    if std <= 1e-12:
        raise ValueError(f"{name} has near-zero variance and cannot be normalized for UniCardio.")

    standardized = (values - float(values.mean())) / std
    lower = float(standardized.min())
    upper = float(standardized.max())
    span = upper - lower
    if span <= 1e-12:
        raise ValueError(f"{name} has near-zero normalized range for UniCardio.")

    return (-1.0 + 2.0 * (standardized - lower) / span).astype(np.float32, copy=False)


def _memmap_split_path(prepared_dir: Path, split: str) -> Path:
    return prepared_dir / MEMMAP_DATA_DIRNAME / f"{split}_full_signal.npy"


def _source_window_lengths(source_fs_hz: float, window_sec: float, step_sec: float) -> tuple[int, int]:
    source_window_length = int(round(source_fs_hz * window_sec))
    source_step_length = int(round(source_fs_hz * step_sec))
    if source_window_length <= 1:
        raise ValueError("window_sec is too small for the source sampling rate.")
    if source_step_length <= 0:
        raise ValueError("step_sec is too small for the source sampling rate.")
    return source_window_length, source_step_length


def _scan_record_windows(
    record_frame: pd.DataFrame,
    *,
    source_fs_hz: float,
    window_sec: float,
    step_sec: float,
    slot_length: int,
) -> dict[str, int]:
    source_window_length, source_step_length = _source_window_lengths(source_fs_hz, window_sec, step_sec)
    ppg = np.asarray(record_frame["PLETH"], dtype=np.float64)
    ecg = np.asarray(record_frame["II"], dtype=np.float64)
    abp = np.asarray(record_frame["ABP"], dtype=np.float64)
    total_length = int(len(record_frame))
    counts = {
        "candidate_windows": 0,
        "valid_windows": 0,
        "dropped_nonfinite_windows": 0,
        "dropped_low_variance_windows": 0,
    }

    for start in range(0, total_length - source_window_length + 1, source_step_length):
        end = start + source_window_length
        counts["candidate_windows"] += 1
        ppg_window = ppg[start:end]
        ecg_window = ecg[start:end]
        abp_window = abp[start:end]
        if (
            ppg_window.size != source_window_length
            or ecg_window.size != source_window_length
            or abp_window.size != source_window_length
        ):
            continue
        if (
            not np.isfinite(ppg_window).all()
            or not np.isfinite(ecg_window).all()
            or not np.isfinite(abp_window).all()
        ):
            counts["dropped_nonfinite_windows"] += 1
            continue

        try:
            _window_to_unicardio_unit_range(_resample_window(ppg_window, slot_length), name="PLETH")
            _window_to_unicardio_unit_range(_resample_window(ecg_window, slot_length), name="II")
        except ValueError:
            counts["dropped_low_variance_windows"] += 1
            continue
        counts["valid_windows"] += 1

    return counts


def _write_record_windows(
    record_frame: pd.DataFrame,
    *,
    source_fs_hz: float,
    window_sec: float,
    step_sec: float,
    slot_length: int,
    full_signal_memmap: np.memmap,
    start_memmap_index: int,
    metadata_base: dict[str, Any],
    full_signal_memmap_path: Path,
    manifest_writer: csv.DictWriter,
) -> int:
    source_window_length, source_step_length = _source_window_lengths(source_fs_hz, window_sec, step_sec)
    ppg = np.asarray(record_frame["PLETH"], dtype=np.float64)
    ecg = np.asarray(record_frame["II"], dtype=np.float64)
    abp = np.asarray(record_frame["ABP"], dtype=np.float64)
    total_rows = int(len(record_frame))
    total_length = int(slot_length * 4)
    write_index = int(start_memmap_index)

    for start in range(0, total_rows - source_window_length + 1, source_step_length):
        end = start + source_window_length
        ppg_window = ppg[start:end]
        ecg_window = ecg[start:end]
        abp_window = abp[start:end]
        if (
            ppg_window.size != source_window_length
            or ecg_window.size != source_window_length
            or abp_window.size != source_window_length
        ):
            continue
        if (
            not np.isfinite(ppg_window).all()
            or not np.isfinite(ecg_window).all()
            or not np.isfinite(abp_window).all()
        ):
            continue

        ppg_resampled = _resample_window(ppg_window, slot_length)
        ecg_resampled = _resample_window(ecg_window, slot_length)
        abp_resampled = _resample_window(abp_window, slot_length)
        try:
            ppg_unit = _window_to_unicardio_unit_range(ppg_resampled, name="PLETH")
            ecg_unit = _window_to_unicardio_unit_range(ecg_resampled, name="II")
        except ValueError:
            continue

        full_signal_memmap[write_index, :] = 0.0
        full_signal_memmap[write_index, 0:slot_length] = ppg_unit
        full_signal_memmap[write_index, slot_length : 2 * slot_length] = (
            abp_resampled.astype(np.float32, copy=False) - UNICARDIO_BP_CENTER
        ) / UNICARDIO_BP_SCALE
        full_signal_memmap[write_index, 2 * slot_length : 3 * slot_length] = ecg_unit

        row = {
            **metadata_base,
            "window_id": f"{metadata_base['split']}_{write_index:08d}",
            "memmap_index": int(write_index),
            "source_record_rows": total_rows,
            "source_fs_hz": float(source_fs_hz),
            "window_sec": float(window_sec),
            "step_sec": float(step_sec),
            "slot_length": int(slot_length),
            "total_length": total_length,
            "source_start_index": int(start),
            "source_end_index": int(end),
            "source_start_sec": float(start / source_fs_hz),
            "source_end_sec": float(end / source_fs_hz),
            "full_signal_memmap": str(full_signal_memmap_path),
        }
        manifest_writer.writerow(row)
        write_index += 1

    return write_index - int(start_memmap_index)


def prepare_unicardio_run(
    export_manifests: list[str | Path],
    output_dir: str | Path,
    *,
    window_sec: float = UNICARDIO_WINDOW_SEC,
    step_sec: float = UNICARDIO_STEP_SEC,
    slot_length: int = UNICARDIO_SLOT_LENGTH,
) -> dict[str, Any]:
    _require_positive("window_sec", window_sec)
    _require_positive("step_sec", step_sec)
    if slot_length <= 1:
        raise ValueError("slot_length must be greater than 1.")

    export_frame = _load_export_manifests(export_manifests)
    output_dir = _to_abs_path(output_dir)
    prepared_dir = output_dir / PREPARED_DATA_DIRNAME
    prepared_dir.mkdir(parents=True, exist_ok=True)

    split_record_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    split_candidate_window_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    split_window_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    split_dropped_nonfinite_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    split_dropped_low_variance_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}

    # First pass: count valid windows so each split can be allocated as one memmap.
    for row in export_frame.to_dict(orient="records"):
        split = str(row["split"])
        record_path = _to_abs_path(row["exported_parquet"])
        if not record_path.exists():
            raise FileNotFoundError(f"Exported parquet does not exist: {record_path}")

        frame = pd.read_parquet(record_path, columns=["PLETH", "II", "ABP"])
        counts = _scan_record_windows(
            frame,
            source_fs_hz=float(row["fs"]),
            window_sec=window_sec,
            step_sec=step_sec,
            slot_length=slot_length,
        )
        split_record_counts[split] += 1
        split_candidate_window_counts[split] += counts["candidate_windows"]
        split_window_counts[split] += counts["valid_windows"]
        split_dropped_nonfinite_counts[split] += counts["dropped_nonfinite_windows"]
        split_dropped_low_variance_counts[split] += counts["dropped_low_variance_windows"]

    if any(split_window_counts[split] == 0 for split in ("train", "val", "test")):
        raise ValueError(
            "UniCardio preparation produced an empty split after strict window-level QC. "
            f"train={split_window_counts['train']}, "
            f"val={split_window_counts['val']}, "
            f"test={split_window_counts['test']}"
        )

    total_length = int(slot_length * 4)
    split_paths: dict[str, str] = {}
    full_signal_memmaps: dict[str, np.memmap] = {}
    for split in ("train", "val", "test"):
        split_path = _memmap_split_path(prepared_dir, split)
        split_path.parent.mkdir(parents=True, exist_ok=True)
        full_signal_memmaps[split] = np.lib.format.open_memmap(
            split_path,
            mode="w+",
            dtype=np.float32,
            shape=(split_window_counts[split], total_length),
        )
        split_paths[split] = str(split_path)

    windows_manifest_path = prepared_dir / PREPARED_WINDOWS_MANIFEST
    write_indices: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    with windows_manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREPARED_WINDOWS_COLUMNS)
        writer.writeheader()
        for row in export_frame.to_dict(orient="records"):
            split = str(row["split"])
            record_path = _to_abs_path(row["exported_parquet"])
            frame = pd.read_parquet(record_path, columns=["PLETH", "II", "ABP"])
            split_path = _memmap_split_path(prepared_dir, split)
            written = _write_record_windows(
                frame,
                source_fs_hz=float(row["fs"]),
                window_sec=window_sec,
                step_sec=step_sec,
                slot_length=slot_length,
                full_signal_memmap=full_signal_memmaps[split],
                start_memmap_index=write_indices[split],
                metadata_base={
                    "split": split,
                    "source": row["source"],
                    "subject_id": row["subject_id"],
                    "record_id": row["record_id"],
                    "hadm_id": row.get("hadm_id", None),
                    "source_exported_parquet": str(record_path),
                },
                full_signal_memmap_path=split_path,
                manifest_writer=writer,
            )
            write_indices[split] += int(written)

    for split in ("train", "val", "test"):
        full_signal_memmaps[split].flush()
        if write_indices[split] != split_window_counts[split]:
            raise ValueError(
                f"UniCardio memmap write count mismatch for {split}: "
                f"expected {split_window_counts[split]}, wrote {write_indices[split]}"
            )

    config = {
        "baseline_key": UNICARDIO_BASELINE_KEY,
        "pipeline": "official_two_stage_ppg_to_ecg_then_ecg_to_bp",
        "storage_format": "memmap_npy",
        "slot_layout": {
            "slot_0": "PLETH / PPG condition",
            "slot_1": "ABP / BP target normalized as (ABP - 100) / 50",
            "slot_2": "II / ECG intermediate",
            "slot_3": "zero placeholder",
        },
        "stage_1": {
            "name": "PPG_to_ECG",
            "model_flag": UNICARDIO_PPG_TO_ECG_MODEL_FLAG,
            "borrow_mode": UNICARDIO_PPG_TO_ECG_BORROW_MODE,
        },
        "stage_2": {
            "name": "ECG_to_ABP",
            "model_flag": UNICARDIO_ECG_TO_BP_MODEL_FLAG,
            "borrow_mode": UNICARDIO_ECG_TO_BP_BORROW_MODE,
        },
        "window_sec": float(window_sec),
        "step_sec": float(step_sec),
        "slot_length": int(slot_length),
        "total_length": int(slot_length * 4),
        "bp_normalization": {
            "formula": "(ABP_mmHg - center) / scale",
            "center": UNICARDIO_BP_CENTER,
            "scale": UNICARDIO_BP_SCALE,
        },
        "ppg_ecg_normalization": "per-window z-score followed by min-max scaling to [-1, 1], matching UniCardio downstream preprocessing style",
        "memmap_files": {split: {"full_signal": path} for split, path in split_paths.items()},
        "candidate_windows": split_candidate_window_counts,
        "split_windows": split_window_counts,
        "dropped_nonfinite_windows": split_dropped_nonfinite_counts,
        "dropped_low_variance_windows": split_dropped_low_variance_counts,
        "windows_manifest": str(windows_manifest_path),
        "source_export_manifests": [str(_to_abs_path(path)) for path in export_manifests],
    }
    config_path = prepared_dir / PREPARED_CONFIG_JSON
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary = {
        "baseline_key": UNICARDIO_BASELINE_KEY,
        "output_dir": str(output_dir),
        "prepared_dir": str(prepared_dir),
        "config_json": str(config_path),
        "windows_manifest": str(windows_manifest_path),
        "split_records": split_record_counts,
        "candidate_windows": split_candidate_window_counts,
        "split_windows": split_window_counts,
        "dropped_nonfinite_windows": split_dropped_nonfinite_counts,
        "dropped_low_variance_windows": split_dropped_low_variance_counts,
        "slot_length": int(slot_length),
        "window_sec": float(window_sec),
        "step_sec": float(step_sec),
        "pipeline": "PPG->ECG->ABP",
        "storage_format": "memmap_npy",
        "memmap_files": {split: {"full_signal": path} for split, path in split_paths.items()},
    }
    summary_path = output_dir / PREPARED_SUMMARY_JSON
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


@contextmanager
def _temporary_sys_path(path: Path):
    path_str = str(path)
    sys.path.insert(0, path_str)
    try:
        yield
    finally:
        try:
            sys.path.remove(path_str)
        except ValueError:
            pass


def _load_module_from_dir(module_dir: Path, module_name: str, required_file: str):
    module_file = module_dir / required_file
    if not module_file.exists():
        raise FileNotFoundError(f"Missing UniCardio module file: {module_file}")

    sys.modules.pop(module_name, None)
    importlib.invalidate_caches()
    with _temporary_sys_path(module_dir):
        return importlib.import_module(module_name)


def _load_base_module(upstream_repo: str | Path):
    base_model_dir = _to_abs_path(upstream_repo) / "base_model"
    return _load_module_from_dir(
        base_model_dir,
        "diffusion_model_no_compress_final",
        "diffusion_model_no_compress_final.py",
    )


def _load_finetune_module(upstream_repo: str | Path):
    mimic_dir = _resolve_mimic_downstream_dir(upstream_repo)
    return _load_module_from_dir(
        mimic_dir,
        "diffusion_model_no_compress_finetune",
        "diffusion_model_no_compress_finetune.py",
    )


def _resolve_mimic_downstream_dir(upstream_repo: str | Path) -> Path:
    repo = _to_abs_path(upstream_repo)
    candidates = [
        repo / "downstream_Code" / "MIMIC",
        repo / "down_stream_code" / "MIMIC",
        repo / "downstream_code" / "MIMIC",
        repo / "downstream_Code" / "Mimic",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Missing UniCardio downstream MIMIC directory. Expected one of: "
        + ", ".join(str(path) for path in candidates)
    )


def _load_unicardio_config(upstream_repo: str | Path, config_path: str | Path | None = None) -> dict[str, Any]:
    import yaml

    if config_path is None:
        base_config = _to_abs_path(upstream_repo) / "base_model" / "base_no_compress_original.yaml"
        try:
            mimic_config = _resolve_mimic_downstream_dir(upstream_repo) / "base_no_compress_original.yaml"
        except FileNotFoundError:
            mimic_config = base_config
        config_path = mimic_config if mimic_config.exists() else base_config
    config_path = _to_abs_path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Missing UniCardio config file: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _default_pretrained_checkpoint(upstream_repo: str | Path) -> Path:
    return _to_abs_path(upstream_repo) / "base_model" / "no_compress799.pth"


def _resolve_pretrained_checkpoint(
    upstream_repo: str | Path,
    checkpoint_path: str | Path | None,
) -> Path:
    resolved = _default_pretrained_checkpoint(upstream_repo) if checkpoint_path is None else _to_abs_path(checkpoint_path)
    if not resolved.exists():
        raise FileNotFoundError(
            "Missing UniCardio pretrained checkpoint. "
            f"Expected: {resolved}. Download the official checkpoint and pass --pretrained-checkpoint/--base-checkpoint-path."
        )
    return resolved


def _load_state_dict_strict(model: Any, checkpoint_path: str | Path, device: Any) -> str:
    import torch

    checkpoint_path = _to_abs_path(checkpoint_path)
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint {checkpoint_path} does not contain a PyTorch state dict.")

    try:
        model.load_state_dict(state, strict=True)
        return "as_saved"
    except RuntimeError:
        if state and all(str(key).startswith("module.") for key in state):
            stripped = {str(key)[7:]: value for key, value in state.items()}
            model.load_state_dict(stripped, strict=True)
            return "stripped_dataparallel_module_prefix"
        raise


def _load_prepared_config(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / PREPARED_DATA_DIRNAME / PREPARED_CONFIG_JSON
    if not config_path.exists():
        raise FileNotFoundError(f"Missing prepared UniCardio config: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def _load_split_data(run_dir: Path, split: str, prepared_config: dict[str, Any] | None = None) -> dict[str, Any]:
    if prepared_config is None:
        prepared_config = _load_prepared_config(run_dir)

    if prepared_config.get("storage_format") == "memmap_npy":
        split_files = prepared_config.get("memmap_files", {}).get(split)
        if not split_files or "full_signal" not in split_files:
            raise ValueError(f"Prepared UniCardio config does not contain full_signal memmap for split {split!r}.")
        return {
            "full_signal": np.load(split_files["full_signal"], mmap_mode="r"),
            "storage_format": "memmap_npy",
        }

    raise ValueError(
        "UniCardio prepared data must use storage_format='memmap_npy'. "
        "Re-run prepare-unicardio-run with the current launcher."
    )


def _slot_view(full_signal: np.ndarray, slot_index: int, slot_length: int) -> np.ndarray:
    start = int(slot_index * slot_length)
    end = int(start + slot_length)
    return full_signal[:, start:end]


def _denormalize_abp_slot(abp_norm: np.ndarray) -> np.ndarray:
    return (np.asarray(abp_norm, dtype=np.float32) * UNICARDIO_BP_SCALE + UNICARDIO_BP_CENTER).astype(
        np.float32,
        copy=False,
    )


class _UniCardioMemmapDataset:
    def __init__(self, full_signal: np.ndarray, *, slot_length: int):
        if full_signal.ndim != 2:
            raise ValueError(f"Expected UniCardio full_signal shaped [N,L], got {full_signal.shape}.")
        self.full_signal = full_signal
        self.slot_length = int(slot_length)

    def __len__(self) -> int:
        return int(self.full_signal.shape[0])

    def __getitem__(self, idx: int):
        import torch

        signal = torch.from_numpy(np.asarray(self.full_signal[int(idx)], dtype=np.float32).copy()).unsqueeze(0)
        mask = torch.ones((1, self.slot_length), dtype=torch.float32)
        return signal, signal.clone(), signal.clone(), mask


def train_unicardio_benchmark(
    run_dir: str | Path,
    upstream_repo: str | Path,
    *,
    config_path: str | Path | None = None,
    pretrained_checkpoint: str | Path | None = None,
    epochs: int = 200,
    batch_size: int = 128,
    lr: float | None = None,
    device: str = "cuda",
    seed: int = 2025,
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    if epochs <= 0:
        raise ValueError("epochs must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    run_dir = _to_abs_path(run_dir)
    models_dir = run_dir / TRAIN_MODELS_DIRNAME
    history_dir = run_dir / TRAIN_HISTORY_DIRNAME
    models_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)

    prepared_config = _load_prepared_config(run_dir)
    total_length = int(prepared_config["total_length"])
    slot_length = int(prepared_config["slot_length"])
    train_data = _load_split_data(run_dir, "train", prepared_config)
    train_dataset = _UniCardioMemmapDataset(train_data["full_signal"], slot_length=slot_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(batch_size),
        shuffle=True,
        pin_memory=(device.startswith("cuda")),
    )

    module = _load_finetune_module(upstream_repo)
    config = _load_unicardio_config(upstream_repo, config_path=config_path)
    torch_device = torch.device(device)
    model = module.CSDI_base(config, torch_device, L=total_length).to(torch_device)
    checkpoint = _resolve_pretrained_checkpoint(upstream_repo, pretrained_checkpoint)
    checkpoint_load_mode = _load_state_dict_strict(model, checkpoint, torch_device)

    if lr is None:
        lr = UNICARDIO_DEFAULT_FINETUNE_LR
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 200], gamma=0.1)

    history_path = history_dir / "loss.csv"
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "loss_noise", "loss_fft", "loss_l1", "lr"])
        writer.writeheader()
        for epoch in range(int(epochs)):
            model.train()
            loss_sum = 0.0
            loss1_sum = 0.0
            loss2_sum = 0.0
            loss3_sum = 0.0
            batch_count = 0
            for observed, sig_impute, sig_denoise, batch_mask in train_loader:
                optimizer.zero_grad()
                loss, loss1, loss2, loss3 = model(
                    observed.to(torch_device),
                    sig_impute=sig_impute.to(torch_device),
                    sig_denoise=sig_denoise.to(torch_device),
                    mask=batch_mask.to(torch_device),
                    task_dice=np.random.rand(1),
                    dirty_dice=np.random.rand(1),
                    condition_dice=np.random.rand(1),
                    train_threshold=0,
                    stage=1,
                    is_train=1,
                    train_gen_flag=0,
                )
                loss = loss.mean()
                loss1 = loss1.mean()
                loss2 = loss2.mean()
                loss3 = loss3.mean()
                loss.backward()
                optimizer.step()

                loss_sum += float(loss.detach().cpu())
                loss1_sum += float(loss1.detach().cpu())
                loss2_sum += float(loss2.detach().cpu())
                loss3_sum += float(loss3.detach().cpu())
                batch_count += 1

            scheduler.step()
            denom = max(1, batch_count)
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": loss_sum / denom,
                    "loss_noise": loss1_sum / denom,
                    "loss_fft": loss2_sum / denom,
                    "loss_l1": loss3_sum / denom,
                    "lr": float(scheduler.get_last_lr()[0]),
                }
            )

    checkpoint_path = models_dir / "final.pth"
    torch.save(model.state_dict(), checkpoint_path)

    summary = {
        "baseline_key": UNICARDIO_BASELINE_KEY,
        "run_dir": str(run_dir),
        "upstream_repo": str(_to_abs_path(upstream_repo)),
        "config_path": str(_to_abs_path(config_path)) if config_path is not None else str(_to_abs_path(upstream_repo) / "base_model" / "base_no_compress_original.yaml"),
        "pretrained_checkpoint": str(checkpoint),
        "pretrained_checkpoint_load_mode": checkpoint_load_mode,
        "checkpoint": str(checkpoint_path),
        "history_csv": str(history_path),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "device": device,
        "seed": int(seed),
        "slot_length": slot_length,
        "total_length": total_length,
        "train_objective": "official MIMIC fine-tune objective: ECG(slot 2)->BP(slot 1), initialized from the UniCardio pretrained checkpoint",
    }
    summary_path = run_dir / TRAIN_RESULTS_JSON
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _generate_slot_from_source_values(
    model: Any,
    source_values: np.ndarray,
    *,
    fill_slot_index: int,
    total_length: int,
    slot_length: int,
    output_path: Path,
    model_flag: str,
    borrow_mode: int,
    batch_size: int,
    n_samples: int,
    device: str,
    aggregation: str,
) -> np.memmap:
    import torch

    if aggregation not in {"mean", "median"}:
        raise ValueError(f"Unsupported aggregation {aggregation!r}. Use 'mean' or 'median'.")
    if source_values.ndim != 2 or source_values.shape[1] != slot_length:
        raise ValueError(f"Expected source values shaped [N,{slot_length}], got {source_values.shape}.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(int(source_values.shape[0]), int(slot_length)),
    )
    fill_start = int(fill_slot_index * slot_length)
    fill_end = int(fill_start + slot_length)
    torch_device = torch.device(device)

    with torch.no_grad():
        for start in range(0, int(source_values.shape[0]), int(batch_size)):
            end = min(start + int(batch_size), int(source_values.shape[0]))
            batch_signal_np = np.zeros((end - start, int(total_length)), dtype=np.float32)
            batch_signal_np[:, fill_start:fill_end] = np.asarray(source_values[start:end], dtype=np.float32)
            batch_signal = torch.from_numpy(batch_signal_np)[:, None, :].to(torch_device)
            samples = model(
                batch_signal,
                n_samples=int(n_samples),
                model_flag=model_flag,
                borrow_mode=int(borrow_mode),
                DDIM_flag=UNICARDIO_DDIM_FLAG,
                sample_steps=UNICARDIO_SAMPLE_STEPS,
                train_gen_flag=1,
            )
            if aggregation == "mean":
                reduced = samples.mean(dim=1)
            else:
                reduced = samples.median(dim=1).values
            output[start:end, :] = reduced.detach().cpu().numpy()[:, 0, :].astype(np.float32, copy=False)

    output.flush()
    return output


def _write_slot_memmap(
    source_values: np.ndarray,
    output_path: Path,
    *,
    transform: str = "identity",
    chunk_size: int = 8192,
) -> np.memmap:
    if source_values.ndim != 2:
        raise ValueError(f"Expected source values shaped [N,L], got {source_values.shape}.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=tuple(source_values.shape),
    )
    for start in range(0, int(source_values.shape[0]), int(chunk_size)):
        end = min(start + int(chunk_size), int(source_values.shape[0]))
        values = np.asarray(source_values[start:end], dtype=np.float32)
        if transform == "denormalize_abp":
            values = _denormalize_abp_slot(values)
        elif transform != "identity":
            raise ValueError(f"Unsupported transform {transform!r}.")
        output[start:end, :] = values
    output.flush()
    return output


def generate_unicardio_run(
    run_dir: str | Path,
    upstream_repo: str | Path,
    *,
    checkpoint_path: str | Path | None = None,
    base_checkpoint_path: str | Path | None = None,
    config_path: str | Path | None = None,
    batch_size: int = 64,
    n_samples: int = UNICARDIO_N_SAMPLES,
    device: str = "cuda",
    seed: int = 2025,
    aggregation: str = "mean",
    inference_only: bool = False,
    window_start: int = 0,
    limit_windows: int | None = None,
) -> dict[str, Any]:
    import torch

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if inference_only and checkpoint_path is not None:
        raise ValueError(
            "UniCardio inference-only generation uses the official base checkpoint for both stages. "
            "Do not pass --checkpoint-path."
        )

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    run_dir = _to_abs_path(run_dir)
    predictions_dir = run_dir / PREDICTIONS_DIRNAME
    predictions_dir.mkdir(parents=True, exist_ok=True)
    prepared_config = _load_prepared_config(run_dir)
    test_data = _load_split_data(run_dir, "test", prepared_config)
    test_full_signal = test_data["full_signal"]
    source_test_windows = int(test_full_signal.shape[0])
    if window_start < 0:
        raise ValueError("window_start must be non-negative.")
    if window_start >= source_test_windows:
        raise ValueError(
            f"window_start={window_start} exceeds available test windows ({source_test_windows})."
        )
    window_end = source_test_windows
    if limit_windows is not None:
        if limit_windows <= 0:
            raise ValueError("limit_windows must be positive when provided.")
        window_end = int(window_start + limit_windows)
        if window_end > source_test_windows:
            raise ValueError(
                "Requested UniCardio window slice exceeds available test windows: "
                f"window_start={window_start}, limit_windows={limit_windows}, "
                f"available={source_test_windows}."
            )
    test_full_signal = test_full_signal[int(window_start) : int(window_end)]
    total_length = int(prepared_config["total_length"])
    slot_length = int(prepared_config["slot_length"])
    ppg_values = _slot_view(test_full_signal, 0, slot_length)
    truth_abp_norm = _slot_view(test_full_signal, 1, slot_length)
    truth_ecg = _slot_view(test_full_signal, 2, slot_length)

    base_checkpoint = _resolve_pretrained_checkpoint(upstream_repo, base_checkpoint_path)
    if inference_only:
        checkpoint_path = base_checkpoint
    elif checkpoint_path is None:
        checkpoint_path = run_dir / TRAIN_MODELS_DIRNAME / "final.pth"
    checkpoint_path = _to_abs_path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing UniCardio ECG->BP checkpoint: {checkpoint_path}")

    config = _load_unicardio_config(upstream_repo, config_path=config_path)
    torch_device = torch.device(device)

    base_module = _load_base_module(upstream_repo)
    base_model = base_module.CSDI_base(config, torch_device, L=total_length).to(torch_device)
    base_load_mode = _load_state_dict_strict(base_model, base_checkpoint, torch_device)
    base_model.eval()
    generated_ecg_path = predictions_dir / "generated_ecg.npy"
    generated_ecg = _generate_slot_from_source_values(
        base_model,
        ppg_values,
        fill_slot_index=0,
        total_length=total_length,
        slot_length=slot_length,
        output_path=generated_ecg_path,
        model_flag=UNICARDIO_PPG_TO_ECG_MODEL_FLAG,
        borrow_mode=UNICARDIO_PPG_TO_ECG_BORROW_MODE,
        batch_size=batch_size,
        n_samples=n_samples,
        device=device,
        aggregation=aggregation,
    )
    del base_model
    if torch_device.type == "cuda":
        torch.cuda.empty_cache()

    if inference_only:
        bp_module = base_module
        bp_model_source = "official_base_pretrained_checkpoint_no_finetune"
    else:
        bp_module = _load_finetune_module(upstream_repo)
        bp_model_source = "mimic_finetune_module_checkpoint"
    bp_model = bp_module.CSDI_base(config, torch_device, L=total_length).to(torch_device)
    bp_load_mode = _load_state_dict_strict(bp_model, checkpoint_path, torch_device)
    bp_model.eval()
    pred_abp_norm_path = predictions_dir / "pred_abp_norm.npy"
    pred_abp_norm = _generate_slot_from_source_values(
        bp_model,
        generated_ecg,
        fill_slot_index=2,
        total_length=total_length,
        slot_length=slot_length,
        output_path=pred_abp_norm_path,
        model_flag=UNICARDIO_ECG_TO_BP_MODEL_FLAG,
        borrow_mode=UNICARDIO_ECG_TO_BP_BORROW_MODE,
        batch_size=batch_size,
        n_samples=n_samples,
        device=device,
        aggregation=aggregation,
    )
    y_true_path = predictions_dir / "y_true.npy"
    y_pred_path = predictions_dir / "y_pred.npy"
    truth_ecg_path = predictions_dir / "truth_ecg.npy"
    y_true = _write_slot_memmap(truth_abp_norm, y_true_path, transform="denormalize_abp")
    y_pred = _write_slot_memmap(pred_abp_norm, y_pred_path, transform="denormalize_abp")
    _write_slot_memmap(truth_ecg, truth_ecg_path)
    if y_pred.shape != y_true.shape:
        raise ValueError(f"Prediction shape mismatch: {y_pred.shape} vs {y_true.shape}.")

    predictions_manifest_path = predictions_dir / PREDICTIONS_MANIFEST_JSON
    predictions_manifest = {
        "storage_format": "memmap_npy",
        "y_true": str(y_true_path),
        "y_pred": str(y_pred_path),
        "generated_ecg": str(generated_ecg_path),
        "truth_ecg": str(truth_ecg_path),
        "pred_abp_norm": str(pred_abp_norm_path),
        "prediction_shape": list(y_pred.shape),
        "source_test_windows": int(source_test_windows),
        "window_start": int(window_start),
        "limit_windows": None if limit_windows is None else int(limit_windows),
    }
    predictions_manifest_path.write_text(
        json.dumps(predictions_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    summary = {
        "baseline_key": UNICARDIO_BASELINE_KEY,
        "run_dir": str(run_dir),
        "upstream_repo": str(_to_abs_path(upstream_repo)),
        "base_checkpoint": str(base_checkpoint),
        "base_checkpoint_load_mode": base_load_mode,
        "ecg_to_bp_checkpoint": str(checkpoint_path),
        "ecg_to_bp_checkpoint_load_mode": bp_load_mode,
        "ecg_to_bp_model_source": bp_model_source,
        "inference_only": bool(inference_only),
        "predictions_manifest": str(predictions_manifest_path),
        "prediction_storage_format": "memmap_npy",
        "source_test_windows": int(source_test_windows),
        "n_test_windows": int(y_true.shape[0]),
        "window_start": int(window_start),
        "limit_windows": None if limit_windows is None else int(limit_windows),
        "window_sec": float(prepared_config["window_sec"]),
        "slot_length": slot_length,
        "batch_size": int(batch_size),
        "n_samples": int(n_samples),
        "aggregation": aggregation,
        "stage_1": {
            "name": "PPG_to_ECG",
            "model_flag": UNICARDIO_PPG_TO_ECG_MODEL_FLAG,
            "borrow_mode": UNICARDIO_PPG_TO_ECG_BORROW_MODE,
        },
        "stage_2": {
            "name": "generated_ECG_to_ABP",
            "model_flag": UNICARDIO_ECG_TO_BP_MODEL_FLAG,
            "borrow_mode": UNICARDIO_ECG_TO_BP_BORROW_MODE,
        },
        "device": device,
        "seed": int(seed),
    }
    summary_path = run_dir / GENERATION_RESULTS_JSON
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def evaluate_unicardio_run(run_dir: str | Path) -> dict[str, Any]:
    run_dir = _to_abs_path(run_dir)
    prepared_config = _load_prepared_config(run_dir)
    predictions_manifest_path = run_dir / PREDICTIONS_DIRNAME / PREDICTIONS_MANIFEST_JSON
    if not predictions_manifest_path.exists():
        raise FileNotFoundError(f"Missing UniCardio prediction manifest: {predictions_manifest_path}")
    predictions_manifest = json.loads(predictions_manifest_path.read_text(encoding="utf-8"))
    y_true = np.load(predictions_manifest["y_true"], mmap_mode="r")
    y_pred = np.load(predictions_manifest["y_pred"], mmap_mode="r")
    predictions_ref = str(predictions_manifest_path)
    prediction_storage_format = "memmap_npy"
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Prediction shape mismatch: {y_true.shape} vs {y_pred.shape}.")

    window_seconds = float(prepared_config["window_sec"])
    waveform_metrics = RunningWaveformMetrics()
    scalar_metrics = RunningScalarErrorMetrics()
    for start in range(0, int(y_true.shape[0]), 2048):
        end = min(start + 2048, int(y_true.shape[0]))
        true_chunk = np.asarray(y_true[start:end], dtype=np.float32)
        pred_chunk = np.asarray(y_pred[start:end], dtype=np.float32)
        waveform_metrics.update(true_chunk, pred_chunk)
        true_bp = bp_from_waveforms(true_chunk, window_seconds=window_seconds)
        pred_bp = {"waveforms": pred_chunk}
        scalar_metrics.update(true_bp, pred_bp)

    waveform = waveform_metrics.finalize()
    scalar_summary = scalar_metrics.finalize()

    metrics_dir = run_dir / METRICS_DIRNAME
    metrics_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "baseline_key": UNICARDIO_BASELINE_KEY,
        "run_dir": str(run_dir),
        "predictions": predictions_ref,
        "prediction_storage_format": prediction_storage_format,
        "pipeline": "PPG->ECG->ABP",
        "window_seconds": window_seconds,
        "n_test_windows": int(y_true.shape[0]),
        "waveform": waveform,
        "scalar": scalar_summary,
    }
    metrics_path = metrics_dir / METRICS_JSON
    metrics_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def run_unicardio_benchmark(
    run_dir: str | Path,
    upstream_repo: str | Path,
    *,
    config_path: str | Path | None = None,
    pretrained_checkpoint: str | Path | None = None,
    epochs: int = 200,
    train_batch_size: int = 128,
    generation_batch_size: int = 64,
    lr: float | None = None,
    n_samples: int = UNICARDIO_N_SAMPLES,
    device: str = "cuda",
    seed: int = 2025,
    aggregation: str = "mean",
    window_start: int = 0,
    limit_windows: int | None = None,
) -> dict[str, Any]:
    train_summary = train_unicardio_benchmark(
        run_dir,
        upstream_repo,
        config_path=config_path,
        pretrained_checkpoint=pretrained_checkpoint,
        epochs=epochs,
        batch_size=train_batch_size,
        lr=lr,
        device=device,
        seed=seed,
    )
    generation_summary = generate_unicardio_run(
        run_dir,
        upstream_repo,
        checkpoint_path=train_summary["checkpoint"],
        base_checkpoint_path=pretrained_checkpoint,
        config_path=config_path,
        batch_size=generation_batch_size,
        n_samples=n_samples,
        device=device,
        seed=seed,
        aggregation=aggregation,
        window_start=window_start,
        limit_windows=limit_windows,
    )
    metrics_summary = evaluate_unicardio_run(run_dir)
    return {
        "baseline_key": UNICARDIO_BASELINE_KEY,
        "run_dir": str(_to_abs_path(run_dir)),
        "pipeline": "PPG->ECG->ABP",
        "train": train_summary,
        "generation": generation_summary,
        "metrics": metrics_summary,
        "metrics_json": str(_to_abs_path(run_dir) / METRICS_DIRNAME / METRICS_JSON),
    }


def run_unicardio_inference_benchmark(
    run_dir: str | Path,
    upstream_repo: str | Path,
    *,
    config_path: str | Path | None = None,
    pretrained_checkpoint: str | Path | None = None,
    generation_batch_size: int = 64,
    n_samples: int = UNICARDIO_N_SAMPLES,
    device: str = "cuda",
    seed: int = 2025,
    aggregation: str = "mean",
    window_start: int = 0,
    limit_windows: int | None = None,
) -> dict[str, Any]:
    generation_summary = generate_unicardio_run(
        run_dir,
        upstream_repo,
        checkpoint_path=None,
        base_checkpoint_path=pretrained_checkpoint,
        config_path=config_path,
        batch_size=generation_batch_size,
        n_samples=n_samples,
        device=device,
        seed=seed,
        aggregation=aggregation,
        inference_only=True,
        window_start=window_start,
        limit_windows=limit_windows,
    )
    metrics_summary = evaluate_unicardio_run(run_dir)
    return {
        "baseline_key": UNICARDIO_BASELINE_KEY,
        "run_dir": str(_to_abs_path(run_dir)),
        "pipeline": "PPG->ECG->ABP",
        "protocol": "inference_only_official_pretrained_checkpoint_no_finetune",
        "generation": generation_summary,
        "metrics": metrics_summary,
        "metrics_json": str(_to_abs_path(run_dir) / METRICS_DIRNAME / METRICS_JSON),
    }
