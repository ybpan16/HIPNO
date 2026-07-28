from __future__ import annotations

import csv
import importlib
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .evaluate import RunningScalarErrorMetrics, RunningWaveformMetrics
from .metrics import bp_from_waveforms


INN_PAR_BASELINE_KEY = "inn_par"
INN_PAR_TARGET_FS_HZ = 125.0
INN_PAR_WINDOW_SEC = 5.0
INN_PAR_STEP_SEC = 5.0
INN_PAR_SIGNAL_LENGTH = 625
INN_PAR_MAX_STEP = 500
INN_PAR_BATCH_SIZE = 128
INN_PAR_TEST_BATCH_SIZE = 1
INN_PAR_LR = 1.0e-4
INN_PAR_NUM_WORKERS = 12
INN_PAR_CHECKPOINT_INTERVAL = 10
INN_PAR_SEED = 1

PREPARED_DATA_DIRNAME = "prepared_data"
MEMMAP_DATA_DIRNAME = "memmap"
PREPARED_WINDOWS_MANIFEST = "windows_manifest.csv"
PREPARED_CONFIG_JSON = "inn_par_benchmark_config.json"
PREPARED_SUMMARY_JSON = "inn_par_prepare_summary.json"
TRAIN_RESULTS_JSON = "inn_par_train_summary.json"
GENERATION_RESULTS_JSON = "inn_par_generation_summary.json"
TRAIN_MODELS_DIRNAME = "models"
TRAIN_HISTORY_DIRNAME = "history"
PREDICTIONS_DIRNAME = "predictions"
METRICS_DIRNAME = "metrics"
METRICS_JSON = "inn_par_metrics.json"

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
    "signal_length",
    "memmap_index",
    "ppg_memmap",
    "abp_memmap",
    "source_exported_parquet",
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
            "INN-PAR benchmark preparation requires train, val, and test splits. "
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
    if unique_keys != {INN_PAR_BASELINE_KEY}:
        raise ValueError(
            "INN-PAR launcher only accepts export manifests for baseline_key='inn_par'. "
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


def _count_record_windows(
    record_frame: pd.DataFrame,
    *,
    source_fs_hz: float,
    window_sec: float,
    step_sec: float,
) -> dict[str, int]:
    source_window_length = int(round(source_fs_hz * window_sec))
    source_step_length = int(round(source_fs_hz * step_sec))
    if source_window_length <= 1:
        raise ValueError("window_sec is too small for the source sampling rate.")
    if source_step_length <= 0:
        raise ValueError("step_sec is too small for the source sampling rate.")

    ppg = np.asarray(record_frame["PLETH"], dtype=np.float64)
    abp = np.asarray(record_frame["ABP"], dtype=np.float64)
    total_length = int(len(record_frame))
    counts = {
        "candidate_windows": 0,
        "kept_windows": 0,
        "dropped_nonfinite_windows": 0,
    }
    for start in range(0, total_length - source_window_length + 1, source_step_length):
        end = start + source_window_length
        counts["candidate_windows"] += 1
        if np.isfinite(ppg[start:end]).all() and np.isfinite(abp[start:end]).all():
            counts["kept_windows"] += 1
        else:
            counts["dropped_nonfinite_windows"] += 1
    return counts


def _iter_record_windows(
    record_frame: pd.DataFrame,
    *,
    source_fs_hz: float,
    target_fs_hz: float,
    window_sec: float,
    step_sec: float,
    signal_length: int,
):
    source_window_length = int(round(source_fs_hz * window_sec))
    source_step_length = int(round(source_fs_hz * step_sec))
    target_window_length = int(round(target_fs_hz * window_sec))
    if source_window_length <= 1:
        raise ValueError("window_sec is too small for the source sampling rate.")
    if source_step_length <= 0:
        raise ValueError("step_sec is too small for the source sampling rate.")
    if signal_length > target_window_length:
        raise ValueError(
            "signal_length cannot exceed the resampled target window length. "
            f"Got signal_length={signal_length}, target_window_length={target_window_length}."
        )

    ppg = np.asarray(record_frame["PLETH"], dtype=np.float64)
    abp = np.asarray(record_frame["ABP"], dtype=np.float64)
    total_length = int(len(record_frame))
    for start in range(0, total_length - source_window_length + 1, source_step_length):
        end = start + source_window_length
        ppg_window = ppg[start:end]
        abp_window = abp[start:end]
        if ppg_window.size != source_window_length or abp_window.size != source_window_length:
            continue
        if not np.isfinite(ppg_window).all() or not np.isfinite(abp_window).all():
            continue

        ppg_resampled = _resample_window(ppg_window, target_window_length)[:signal_length]
        abp_resampled = _resample_window(abp_window, target_window_length)[:signal_length]
        metadata = {
            "source_record_rows": total_length,
            "source_fs_hz": float(source_fs_hz),
            "target_fs_hz": float(target_fs_hz),
            "window_sec": float(window_sec),
            "step_sec": float(step_sec),
            "source_start_index": int(start),
            "source_end_index": int(end),
            "source_start_sec": float(start / source_fs_hz),
            "source_end_sec": float(end / source_fs_hz),
            "signal_length": int(signal_length),
        }
        yield metadata, ppg_resampled.astype(np.float32, copy=False), abp_resampled.astype(np.float32, copy=False)


def _memmap_split_paths(prepared_dir: Path, split: str) -> tuple[Path, Path]:
    memmap_dir = prepared_dir / MEMMAP_DATA_DIRNAME
    return memmap_dir / f"{split}_ppg.npy", memmap_dir / f"{split}_abp.npy"


def prepare_inn_par_run(
    export_manifests: list[str | Path],
    output_dir: str | Path,
    *,
    target_fs_hz: float = INN_PAR_TARGET_FS_HZ,
    window_sec: float = INN_PAR_WINDOW_SEC,
    step_sec: float = INN_PAR_STEP_SEC,
    signal_length: int = INN_PAR_SIGNAL_LENGTH,
) -> dict[str, Any]:
    _require_positive("target_fs_hz", target_fs_hz)
    _require_positive("window_sec", window_sec)
    _require_positive("step_sec", step_sec)
    if signal_length <= 1:
        raise ValueError("signal_length must be greater than 1.")

    export_frame = _load_export_manifests(export_manifests)
    output_dir = _to_abs_path(output_dir)
    prepared_dir = output_dir / PREPARED_DATA_DIRNAME
    prepared_dir.mkdir(parents=True, exist_ok=True)
    (prepared_dir / MEMMAP_DATA_DIRNAME).mkdir(parents=True, exist_ok=True)

    split_record_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    split_window_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    split_candidate_window_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    split_dropped_nonfinite_window_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}

    # First pass: count finite windows so we can allocate one memmap array per split/signal.
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

        source_fs_hz = float(row["fs"])
        counts = _count_record_windows(
            frame,
            source_fs_hz=source_fs_hz,
            window_sec=window_sec,
            step_sec=step_sec,
        )
        split_record_counts[split] += 1
        split_candidate_window_counts[split] += counts["candidate_windows"]
        split_window_counts[split] += counts["kept_windows"]
        split_dropped_nonfinite_window_counts[split] += counts["dropped_nonfinite_windows"]

    empty_splits = [split for split, count in split_window_counts.items() if count == 0]
    if empty_splits:
        raise ValueError(
            "INN-PAR preparation produced empty split(s): "
            f"{', '.join(empty_splits)}. Check record duration and split manifests."
        )

    ppg_memmaps: dict[str, np.memmap] = {}
    abp_memmaps: dict[str, np.memmap] = {}
    memmap_paths: dict[str, dict[str, str]] = {}
    for split in ("train", "val", "test"):
        ppg_path, abp_path = _memmap_split_paths(prepared_dir, split)
        ppg_memmaps[split] = np.lib.format.open_memmap(
            ppg_path,
            mode="w+",
            dtype=np.float32,
            shape=(split_window_counts[split], int(signal_length)),
        )
        abp_memmaps[split] = np.lib.format.open_memmap(
            abp_path,
            mode="w+",
            dtype=np.float32,
            shape=(split_window_counts[split], int(signal_length)),
        )
        memmap_paths[split] = {"ppg": str(ppg_path), "abp": str(abp_path)}

    write_indices: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    windows_manifest_path = prepared_dir / PREPARED_WINDOWS_MANIFEST
    with windows_manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREPARED_WINDOWS_COLUMNS)
        writer.writeheader()
        for row in export_frame.to_dict(orient="records"):
            split = str(row["split"])
            record_path = _to_abs_path(row["exported_parquet"])
            frame = pd.read_parquet(record_path)
            source_fs_hz = float(row["fs"])
            ppg_path, abp_path = _memmap_split_paths(prepared_dir, split)
            for metadata, ppg_window, abp_window in _iter_record_windows(
                frame,
                source_fs_hz=source_fs_hz,
                target_fs_hz=target_fs_hz,
                window_sec=window_sec,
                step_sec=step_sec,
                signal_length=signal_length,
            ):
                memmap_index = write_indices[split]
                ppg_memmaps[split][memmap_index, :] = ppg_window
                abp_memmaps[split][memmap_index, :] = abp_window
                writer.writerow(
                    {
                        "split": split,
                        "window_id": f"{split}_{memmap_index:08d}",
                        "source": row["source"],
                        "subject_id": row["subject_id"],
                        "record_id": row["record_id"],
                        "hadm_id": row.get("hadm_id", None),
                        **metadata,
                        "memmap_index": int(memmap_index),
                        "ppg_memmap": str(ppg_path),
                        "abp_memmap": str(abp_path),
                        "source_exported_parquet": str(record_path),
                    }
                )
                write_indices[split] += 1

    for split in ("train", "val", "test"):
        ppg_memmaps[split].flush()
        abp_memmaps[split].flush()
        if write_indices[split] != split_window_counts[split]:
            raise RuntimeError(
                f"INN-PAR memmap write count mismatch for {split}: "
                f"expected {split_window_counts[split]}, wrote {write_indices[split]}."
            )

    config = {
        "baseline_key": INN_PAR_BASELINE_KEY,
        "method": "official_INNPAR_PPG_gradient_to_ABP_gradient",
        "export_manifests": [str(_to_abs_path(path)) for path in export_manifests],
        "output_dir": str(output_dir),
        "prepared_data_dir": str(prepared_dir),
        "storage_format": "memmap_npy",
        "memmap_files": memmap_paths,
        "target_fs_hz": float(target_fs_hz),
        "window_sec": float(window_sec),
        "step_sec": float(step_sec),
        "signal_length": int(signal_length),
        "padding_rule": (
            "The local memmap Dataset mirrors official data.py: it builds PPG/ABP gradient channels with "
            "[-1,0,1] conv1d and pads two-channel tensors with zeros until length is divisible by 4; "
            "evaluation slices predictions back to signal_length."
        ),
        "input_channels_after_official_loader": ["PLETH", "Grad(PLETH)"],
        "target_channels_after_official_loader": ["ABP", "Grad(ABP)"],
        "normalization": "none added by launcher; memmap arrays store raw float32 finite windows directly",
        "finite_window_qc": (
            "Windows containing non-finite PLETH or ABP samples are excluded before writing memmap arrays. "
            "No interpolation, imputation, smoothing, clipping, or postprocessing is applied."
        ),
        "windows_manifest": str(windows_manifest_path),
        "candidate_windows": split_candidate_window_counts,
        "split_windows": split_window_counts,
        "dropped_nonfinite_windows": split_dropped_nonfinite_window_counts,
        "split_records": split_record_counts,
    }
    config_path = prepared_dir / PREPARED_CONFIG_JSON
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary = {
        "baseline_key": INN_PAR_BASELINE_KEY,
        "output_dir": str(output_dir),
        "prepared_data_dir": str(prepared_dir),
        "config_json": str(config_path),
        "windows_manifest": str(windows_manifest_path),
        "storage_format": "memmap_npy",
        "memmap_files": memmap_paths,
        "split_records": split_record_counts,
        "candidate_windows": split_candidate_window_counts,
        "split_windows": split_window_counts,
        "dropped_nonfinite_windows": split_dropped_nonfinite_window_counts,
        "target_fs_hz": float(target_fs_hz),
        "window_sec": float(window_sec),
        "step_sec": float(step_sec),
        "signal_length": int(signal_length),
    }
    summary_path = output_dir / PREPARED_SUMMARY_JSON
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


@contextmanager
def _temporary_upstream_imports(upstream_repo: str | Path):
    upstream_repo = _to_abs_path(upstream_repo)
    required_files = ["network.py", "data.py", "losses.py", "utils.py"]
    missing = [name for name in required_files if not (upstream_repo / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing INN-PAR official source files in {upstream_repo}: {', '.join(missing)}"
        )

    module_names = ["network", "data", "losses", "utils"]
    original_modules = {name: sys.modules.get(name) for name in module_names}
    repo_str = str(upstream_repo)
    sys.path.insert(0, repo_str)
    for name in module_names:
        sys.modules.pop(name, None)
    try:
        yield
    finally:
        try:
            sys.path.remove(repo_str)
        except ValueError:
            pass
        for name in module_names:
            sys.modules.pop(name, None)
            original = original_modules[name]
            if original is not None:
                sys.modules[name] = original


def _load_prepared_config(run_dir: str | Path) -> dict[str, Any]:
    run_dir = _to_abs_path(run_dir)
    config_path = run_dir / PREPARED_DATA_DIRNAME / PREPARED_CONFIG_JSON
    if not config_path.exists():
        raise FileNotFoundError(f"Missing prepared INN-PAR config: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def _resolve_torch_device(device: str | None):
    import torch

    torch_device = torch.device("cuda" if device is None else device)
    if torch_device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA device but torch.cuda.is_available() is false.")
        try:
            torch.empty(1, device=torch_device)
        except Exception as exc:
            raise RuntimeError(
                f"Requested CUDA device {torch_device} cannot allocate a test tensor. "
                "Check that this shell owns an active GPU allocation and that the cluster MPS/CUDA "
                "environment is visible inside the session."
            ) from exc
    return torch_device


def _pad_to_multiple_of_four(signal: Any):
    import torch.nn.functional as F

    length = int(signal.shape[-1])
    remainder = length % 4
    if remainder == 0:
        return signal
    return F.pad(signal, pad=(0, 4 - remainder), mode="constant", value=0)


def _signal_and_gradient(signal_1d: np.ndarray):
    import torch
    import torch.nn.functional as F

    signal = torch.from_numpy(np.asarray(signal_1d, dtype=np.float32).copy()).unsqueeze(0)
    kernel = torch.FloatTensor([[-1, 0, 1]]).unsqueeze(0)
    grad = F.conv1d(signal.unsqueeze(0), kernel, padding=1).squeeze(0)
    return torch.cat((signal, grad), dim=0)


class _INNParMemmapDataset:
    def __init__(self, ppg_path: str | Path, abp_path: str | Path):
        self.ppg_path = str(_to_abs_path(ppg_path))
        self.abp_path = str(_to_abs_path(abp_path))
        self.ppg = np.load(self.ppg_path, mmap_mode="r")
        self.abp = np.load(self.abp_path, mmap_mode="r")
        if self.ppg.shape != self.abp.shape:
            raise ValueError(f"INN-PAR memmap shape mismatch: {self.ppg.shape} vs {self.abp.shape}.")
        if self.ppg.ndim != 2:
            raise ValueError(f"Expected INN-PAR memmaps shaped [N,L], got {self.ppg.shape}.")

    def __len__(self) -> int:
        return int(self.ppg.shape[0])

    def __getitem__(self, idx: int):
        ppg = _signal_and_gradient(self.ppg[idx])
        abp = _signal_and_gradient(self.abp[idx])
        ppg = _pad_to_multiple_of_four(ppg)
        abp = _pad_to_multiple_of_four(abp)
        return ppg, abp


def _load_memmap_dataset(run_dir: str | Path, split: str, prepared_config: dict[str, Any]) -> _INNParMemmapDataset:
    memmap_files = prepared_config.get("memmap_files", {})
    split_files = memmap_files.get(split)
    if not split_files:
        raise ValueError(
            f"Prepared INN-PAR config does not contain memmap files for split {split!r}. "
            "Re-run prepare-inn-par-run with the current launcher."
        )
    return _INNParMemmapDataset(split_files["ppg"], split_files["abp"])


def _require_dataset_signal_length(dataset: _INNParMemmapDataset, *, split: str, signal_length: int) -> None:
    actual = int(dataset.ppg.shape[1])
    expected = int(signal_length)
    if actual != expected:
        raise ValueError(
            f"Prepared INN-PAR {split} memmap length does not match config signal_length: "
            f"{actual} vs {expected}. Re-run prepare-inn-par-run for this run directory."
        )


def train_inn_par_benchmark(
    run_dir: str | Path,
    upstream_repo: str | Path,
    *,
    max_step: int = INN_PAR_MAX_STEP,
    batch_size: int = INN_PAR_BATCH_SIZE,
    lr: float = INN_PAR_LR,
    num_workers: int = INN_PAR_NUM_WORKERS,
    checkpoint_interval: int = INN_PAR_CHECKPOINT_INTERVAL,
    pin_memory: bool | None = None,
    device: str | None = None,
    seed: int = INN_PAR_SEED,
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    if max_step < 0:
        raise ValueError("max_step must be non-negative.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if lr <= 0:
        raise ValueError("lr must be positive.")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative.")
    if checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be positive.")

    run_dir = _to_abs_path(run_dir)
    prepared_config = _load_prepared_config(run_dir)
    train_dataset = _load_memmap_dataset(run_dir, "train", prepared_config)
    if len(train_dataset) == 0:
        raise ValueError("Prepared INN-PAR train memmap dataset is empty.")
    signal_length = int(prepared_config["signal_length"])
    _require_dataset_signal_length(train_dataset, split="train", signal_length=signal_length)

    models_dir = run_dir / TRAIN_MODELS_DIRNAME
    history_dir = run_dir / TRAIN_HISTORY_DIRNAME
    models_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)

    torch_device = _resolve_torch_device(device)
    pin_memory_enabled = (torch_device.type == "cuda") if pin_memory is None else bool(pin_memory)
    with _temporary_upstream_imports(upstream_repo):
        losses_module = importlib.import_module("losses")
        network_module = importlib.import_module("network")
        utils_module = importlib.import_module("utils")

        utils_module.seed_everything(int(seed))
        model = network_module.INNPAR().to(torch_device)
        loss_fn = losses_module.model_loss()
        train_loader = DataLoader(
            train_dataset,
            batch_size=int(batch_size),
            shuffle=True,
            num_workers=int(num_workers),
            pin_memory=pin_memory_enabled,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))

        history_path = history_dir / "loss.csv"
        with history_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss"])
            writer.writeheader()
            for epoch in range(int(max_step) + 1):
                model.train()
                train_loss = 0.0
                for ppg, abp in train_loader:
                    ppg = ppg.to(torch_device, non_blocking=pin_memory_enabled)
                    abp = abp.to(torch_device, non_blocking=pin_memory_enabled)
                    model.zero_grad()
                    optimizer.zero_grad()
                    abp_re = model(ppg)
                    loss = loss_fn(abp_re, abp)
                    loss.backward()
                    optimizer.step()
                    train_loss += float(loss.detach().cpu()) * int(ppg.size(0))

                train_loss /= float(len(train_loader.sampler))
                writer.writerow({"epoch": epoch, "train_loss": train_loss})
                print(f"[Epoch:{epoch}\tTraining Loss:{train_loss:.3f}]")

                if epoch % int(checkpoint_interval) == 0:
                    torch.save(model.state_dict(), models_dir / f"checkpoint_{epoch}.pth")

        final_checkpoint_path = models_dir / "final.pth"
        torch.save(model.state_dict(), final_checkpoint_path)
        n_parameters = int(sum(param.numel() for param in model.parameters() if param.requires_grad))

    summary = {
        "baseline_key": INN_PAR_BASELINE_KEY,
        "run_dir": str(run_dir),
        "upstream_repo": str(_to_abs_path(upstream_repo)),
        "checkpoint": str(final_checkpoint_path),
        "history_csv": str(history_path),
        "max_step": int(max_step),
        "epochs_run": int(max_step) + 1,
        "batch_size": int(batch_size),
        "lr": float(lr),
        "num_workers": int(num_workers),
        "checkpoint_interval": int(checkpoint_interval),
        "pin_memory": bool(pin_memory_enabled),
        "device": str(torch_device),
        "seed": int(seed),
        "n_parameters": n_parameters,
        "signal_length": int(prepared_config["signal_length"]),
        "storage_format": prepared_config.get("storage_format", "memmap_npy"),
        "train_objective": (
            "official model_loss L1 over both output channels: reconstructed ABP and reconstructed ABP gradient"
        ),
    }
    summary_path = run_dir / TRAIN_RESULTS_JSON
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _load_state_dict_strict(model: Any, checkpoint_path: str | Path) -> None:
    import torch

    checkpoint_path = _to_abs_path(checkpoint_path)
    state = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint {checkpoint_path} does not contain a PyTorch state dict.")
    model.load_state_dict(state, strict=True)


def generate_inn_par_run(
    run_dir: str | Path,
    upstream_repo: str | Path,
    *,
    checkpoint_path: str | Path | None = None,
    batch_size: int = INN_PAR_TEST_BATCH_SIZE,
    num_workers: int = INN_PAR_NUM_WORKERS,
    pin_memory: bool | None = None,
    device: str | None = None,
    seed: int = INN_PAR_SEED,
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative.")

    run_dir = _to_abs_path(run_dir)
    prepared_config = _load_prepared_config(run_dir)
    test_dataset = _load_memmap_dataset(run_dir, "test", prepared_config)
    if len(test_dataset) == 0:
        raise ValueError("Prepared INN-PAR test memmap dataset is empty.")
    signal_length = int(prepared_config["signal_length"])
    _require_dataset_signal_length(test_dataset, split="test", signal_length=signal_length)
    if checkpoint_path is None:
        checkpoint_path = run_dir / TRAIN_MODELS_DIRNAME / "final.pth"
    checkpoint_path = _to_abs_path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing INN-PAR checkpoint: {checkpoint_path}")

    predictions_dir = run_dir / PREDICTIONS_DIRNAME
    predictions_dir.mkdir(parents=True, exist_ok=True)
    torch_device = _resolve_torch_device(device)
    pin_memory_enabled = (torch_device.type == "cuda") if pin_memory is None else bool(pin_memory)

    y_true_path = predictions_dir / "test_y_true.npy"
    y_pred_path = predictions_dir / "test_y_pred.npy"
    temp_suffix = f".tmp.{os.getpid()}.npy"
    y_true_tmp_path = predictions_dir / f"test_y_true{temp_suffix}"
    y_pred_tmp_path = predictions_dir / f"test_y_pred{temp_suffix}"
    for path in (y_true_tmp_path, y_pred_tmp_path):
        path.unlink(missing_ok=True)

    try:
        with _temporary_upstream_imports(upstream_repo):
            network_module = importlib.import_module("network")
            utils_module = importlib.import_module("utils")

            utils_module.seed_everything(int(seed))
            model = network_module.INNPAR()
            _load_state_dict_strict(model, checkpoint_path)
            model.to(torch_device)
            model.eval()

            test_loader = DataLoader(
                test_dataset,
                batch_size=int(batch_size),
                shuffle=False,
                num_workers=int(num_workers),
                pin_memory=pin_memory_enabled,
            )

            y_true = np.lib.format.open_memmap(
                y_true_tmp_path,
                mode="w+",
                dtype=np.float32,
                shape=(len(test_dataset), signal_length),
            )
            y_pred = np.lib.format.open_memmap(
                y_pred_tmp_path,
                mode="w+",
                dtype=np.float32,
                shape=(len(test_dataset), signal_length),
            )

            offset = 0
            with torch.no_grad():
                for ppg, abp in test_loader:
                    ppg = ppg.to(torch_device, non_blocking=pin_memory_enabled)
                    prediction = model(ppg)
                    batch_size_actual = int(prediction.shape[0])
                    end = offset + batch_size_actual
                    y_pred[offset:end, :] = (
                        prediction[:, 0, 0:signal_length].detach().cpu().numpy().astype(np.float32, copy=False)
                    )
                    y_true[offset:end, :] = abp[:, 0, 0:signal_length].numpy().astype(np.float32, copy=False)
                    offset = end
            if offset != len(test_dataset):
                raise RuntimeError(f"INN-PAR prediction count mismatch: expected {len(test_dataset)}, wrote {offset}.")

            y_true.flush()
            y_pred.flush()
            del y_true
            del y_pred
    except Exception:
        y_true_tmp_path.unlink(missing_ok=True)
        y_pred_tmp_path.unlink(missing_ok=True)
        raise

    y_true_tmp_path.replace(y_true_path)
    y_pred_tmp_path.replace(y_pred_path)
    predictions_manifest_path = predictions_dir / "test_predictions_manifest.json"
    predictions_manifest = {
        "storage_format": "memmap_npy",
        "y_true": str(y_true_path),
        "y_pred": str(y_pred_path),
        "shape": [int(len(test_dataset)), int(signal_length)],
        "dtype": "float32",
    }
    predictions_manifest_path.write_text(
        json.dumps(predictions_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    summary = {
        "baseline_key": INN_PAR_BASELINE_KEY,
        "run_dir": str(run_dir),
        "upstream_repo": str(_to_abs_path(upstream_repo)),
        "checkpoint": str(checkpoint_path),
        "predictions": str(predictions_manifest_path),
        "prediction_files": predictions_manifest,
        "n_test_windows": int(len(test_dataset)),
        "signal_length": signal_length,
        "window_sec": float(prepared_config["window_sec"]),
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory_enabled),
        "device": str(torch_device),
        "seed": int(seed),
        "prediction_channel_used_for_metrics": "channel_0_reconstructed_ABP",
        "gradient_channel_saved_for_audit": False,
    }
    summary_path = run_dir / GENERATION_RESULTS_JSON
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def evaluate_inn_par_run(run_dir: str | Path) -> dict[str, Any]:
    run_dir = _to_abs_path(run_dir)
    prepared_config = _load_prepared_config(run_dir)
    predictions_manifest_path = run_dir / PREDICTIONS_DIRNAME / "test_predictions_manifest.json"
    if not predictions_manifest_path.exists():
        raise FileNotFoundError(f"Missing INN-PAR prediction manifest: {predictions_manifest_path}")
    predictions_manifest = json.loads(predictions_manifest_path.read_text(encoding="utf-8"))
    y_true = np.load(predictions_manifest["y_true"], mmap_mode="r")
    y_pred = np.load(predictions_manifest["y_pred"], mmap_mode="r")
    predictions_path: Path | str = predictions_manifest_path
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Prediction shape mismatch: {y_true.shape} vs {y_pred.shape}.")
    manifest_shape = tuple(int(v) for v in predictions_manifest.get("shape", []))
    if manifest_shape and tuple(y_true.shape) != manifest_shape:
        raise ValueError(f"Prediction manifest shape {manifest_shape} does not match arrays {tuple(y_true.shape)}.")
    signal_length = int(prepared_config["signal_length"])
    if int(y_true.shape[1]) != signal_length:
        raise ValueError(
            f"Prediction length does not match prepared signal_length: {int(y_true.shape[1])} vs {signal_length}."
        )

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
        "baseline_key": INN_PAR_BASELINE_KEY,
        "run_dir": str(run_dir),
        "predictions": str(predictions_path),
        "prediction_storage_format": "memmap_npy",
        "window_seconds": window_seconds,
        "n_test_windows": int(y_true.shape[0]),
        "waveform": waveform,
        "scalar": scalar_summary,
    }
    metrics_path = metrics_dir / METRICS_JSON
    metrics_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def run_inn_par_benchmark(
    run_dir: str | Path,
    upstream_repo: str | Path,
    *,
    max_step: int = INN_PAR_MAX_STEP,
    train_batch_size: int = INN_PAR_BATCH_SIZE,
    test_batch_size: int = INN_PAR_TEST_BATCH_SIZE,
    lr: float = INN_PAR_LR,
    num_workers: int = INN_PAR_NUM_WORKERS,
    checkpoint_interval: int = INN_PAR_CHECKPOINT_INTERVAL,
    device: str | None = None,
    pin_memory: bool | None = None,
    seed: int = INN_PAR_SEED,
) -> dict[str, Any]:
    train_summary = train_inn_par_benchmark(
        run_dir,
        upstream_repo,
        max_step=max_step,
        batch_size=train_batch_size,
        lr=lr,
        num_workers=num_workers,
        checkpoint_interval=checkpoint_interval,
        pin_memory=pin_memory,
        device=device,
        seed=seed,
    )
    generation_summary = generate_inn_par_run(
        run_dir,
        upstream_repo,
        checkpoint_path=train_summary["checkpoint"],
        batch_size=test_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        device=device,
        seed=seed,
    )
    metrics_summary = evaluate_inn_par_run(run_dir)
    return {
        "baseline_key": INN_PAR_BASELINE_KEY,
        "run_dir": str(_to_abs_path(run_dir)),
        "train": train_summary,
        "generation": generation_summary,
        "metrics": metrics_summary,
        "metrics_json": str(_to_abs_path(run_dir) / METRICS_DIRNAME / METRICS_JSON),
    }
