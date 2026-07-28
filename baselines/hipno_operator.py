from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from baselines.patient_metrics import compute_patient_metrics


MODEL_MODULE = "models.training_v9_tauA"


def _to_abs_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _import_v9_model_module() -> tuple[Any, Any]:
    module = importlib.import_module(MODEL_MODULE)
    if not hasattr(module, "DirectWaveformDeepONet1DFlowPINN"):
        raise AttributeError(f"{MODEL_MODULE} is missing DirectWaveformDeepONet1DFlowPINN.")
    api = getattr(module, "v9c", None)
    required = ["_broadcast_dt", "_calibrate_dt", "_to_bt2d", "_align_bt"]
    missing = [name for name in required if api is None or not hasattr(api, name)]
    if missing:
        raise AttributeError(f"{MODEL_MODULE}.v9c is missing required operator helpers: {missing}")
    if not hasattr(module, "_nibp_features_from_batch"):
        raise AttributeError(f"{MODEL_MODULE} is missing _nibp_features_from_batch.")
    return module, api


def _build_v9_model(module: Any, *, nx: int, encoder_temporal_stride: int) -> torch.nn.Module:
    cls = module.DirectWaveformDeepONet1DFlowPINN
    signature = inspect.signature(cls)
    kwargs: dict[str, Any] = {"nx": int(nx)}
    if "encoder_temporal_stride" in signature.parameters:
        kwargs["encoder_temporal_stride"] = int(encoder_temporal_stride)
    return cls(**kwargs)


def _build_dataset(
    manifest_csv: Path,
    *,
    chunk_duration: float,
    grid_points: int,
    min_valid_ratio: float,
    buffer_size: int,
) -> Any:
    from data.dataset_vitaldb4 import MaterializedVitalDBWaveformDataset

    return MaterializedVitalDBWaveformDataset(
        metadata_csv=str(manifest_csv),
        input_signals=["II", "PLETH"],
        output_signals=["ABP"],
        meta_fields=("subject_id", "record_id", "hadm_id"),
        chunk_duration=float(chunk_duration),
        forecast_duration=0.0,
        M=int(grid_points),
        normalize_t=True,
        shuffle=False,
        buffer_size=int(buffer_size),
        min_valid_ratio=float(min_valid_ratio),
        emit_t=False,
        emit_x_locs=False,
        emit_coords=False,
        emit_phase_targets=False,
        emit_beat_bounds=False,
        emit_matched_events=False,
        arrow_use_threads=False,
    )


def _build_loader(dataset: Any, *, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        collate_fn=dataset.collate_fn,
        num_workers=int(num_workers),
        pin_memory=False,
    )


def _count_windows(loader: DataLoader) -> int:
    total = 0
    for batch in loader:
        if batch:
            total += int(batch["input"].shape[0])
    return int(total)


def _checkpoint_field(checkpoint: dict[str, Any], key: str) -> dict[str, torch.Tensor]:
    if key not in checkpoint:
        raise KeyError(f"Checkpoint does not contain {key!r}. Available keys: {sorted(checkpoint.keys())}")
    selected = checkpoint[key]
    if not isinstance(selected, dict):
        raise TypeError(f"Checkpoint field {key!r} is not a state_dict.")
    return selected


def _select_checkpoint_state(checkpoint: dict[str, Any], state: str) -> dict[str, torch.Tensor]:
    if state not in {"model", "ema"}:
        raise ValueError(f"Unsupported checkpoint state {state!r}; expected 'model' or 'ema'.")
    model_state = _checkpoint_field(checkpoint, "model_state")
    if state == "model":
        return dict(model_state)

    ema_state = _checkpoint_field(checkpoint, "ema_state")
    merged = dict(model_state)
    for key, value in ema_state.items():
        if key not in merged:
            raise KeyError(f"EMA state contains key absent from model_state: {key}")
        if tuple(merged[key].shape) != tuple(value.shape):
            raise ValueError(
                f"EMA state shape mismatch for {key}: {tuple(value.shape)} vs model_state {tuple(merged[key].shape)}"
            )
        merged[key] = value
    return merged


def _load_v9_model(
    checkpoint_path: Path,
    *,
    state: str,
    device: torch.device,
    nx: int,
    encoder_temporal_stride: int,
) -> tuple[torch.nn.Module, Any, Any, dict[str, Any]]:
    module, api = _import_v9_model_module()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint dict, got {type(checkpoint)!r}.")

    model = _build_v9_model(module, nx=nx, encoder_temporal_stride=encoder_temporal_stride)
    state_dict = _select_checkpoint_state(checkpoint, state)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model, module, api, checkpoint


def _batch_subjects(batch: dict[str, Any], take: int) -> list[Any]:
    values = batch.get("subject_id")
    if values is None:
        raise KeyError("Dataset batch is missing subject_id; cannot compute subject-level metrics.")
    return list(values[:take])


def _batch_field(batch: dict[str, Any], key: str, take: int, default: Any = None) -> list[Any]:
    values = batch.get(key)
    if values is None:
        return [default for _ in range(take)]
    return list(values[:take])


def export_v9_operator_predictions(
    *,
    checkpoint_path: str | Path,
    test_manifest: str | Path,
    output_dir: str | Path,
    state: str,
    device: str,
    batch_size: int,
    num_workers: int,
    chunk_duration: float,
    grid_points: int,
    operator_time_points: int,
    nx: int,
    encoder_temporal_stride: int,
    min_valid_ratio: float,
    buffer_size: int,
) -> dict[str, Any]:
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive.")
    if int(num_workers) < 0:
        raise ValueError("num_workers must be non-negative.")
    if float(chunk_duration) <= 0.0:
        raise ValueError("chunk_duration must be positive.")
    if int(grid_points) <= 1:
        raise ValueError("grid_points must be greater than one.")
    if int(operator_time_points) <= 1:
        raise ValueError("operator_time_points must be greater than one.")
    if int(nx) <= 1:
        raise ValueError("nx must be greater than one.")
    if not (0.0 <= float(min_valid_ratio) <= 1.0):
        raise ValueError("min_valid_ratio must be within [0, 1].")

    checkpoint_path = _to_abs_path(checkpoint_path)
    test_manifest = _to_abs_path(test_manifest)
    output_dir = _to_abs_path(output_dir)
    prepared_dir = output_dir / "prepared_data"
    predictions_dir = output_dir / "predictions"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    if not test_manifest.exists():
        raise FileNotFoundError(f"Missing test manifest: {test_manifest}")

    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device but torch.cuda.is_available() is false.")
    if torch_device.type == "cuda":
        try:
            torch.empty(1, device=torch_device)
        except Exception as exc:
            raise RuntimeError(
                f"Requested CUDA device {torch_device} cannot allocate a test tensor. "
                "Check that this shell owns an active GPU allocation and that the cluster MPS/CUDA "
                "environment is visible inside the session."
            ) from exc

    model, model_module_obj, model_api, checkpoint = _load_v9_model(
        checkpoint_path,
        state=state,
        device=torch_device,
        nx=nx,
        encoder_temporal_stride=encoder_temporal_stride,
    )
    forward_signature = inspect.signature(model.forward)
    if "nibp_features" not in forward_signature.parameters:
        raise TypeError(f"Model {type(model).__name__} does not accept nibp_features.")

    count_dataset = _build_dataset(
        test_manifest,
        chunk_duration=chunk_duration,
        grid_points=grid_points,
        min_valid_ratio=min_valid_ratio,
        buffer_size=buffer_size,
    )
    count_loader = _build_loader(count_dataset, batch_size=batch_size, num_workers=num_workers)
    n_windows = _count_windows(count_loader)
    if n_windows <= 0:
        raise RuntimeError("Operator export found zero eligible test windows.")

    export_dataset = _build_dataset(
        test_manifest,
        chunk_duration=chunk_duration,
        grid_points=grid_points,
        min_valid_ratio=min_valid_ratio,
        buffer_size=buffer_size,
    )
    export_loader = _build_loader(export_dataset, batch_size=batch_size, num_workers=num_workers)

    y_true_path = predictions_dir / "y_true.npy"
    y_pred_path = predictions_dir / "y_pred.npy"
    mask_path = predictions_dir / "output_mask.npy"

    y_true_mm = y_pred_mm = mask_mm = None
    signal_length = None
    rows: list[dict[str, Any]] = []
    offset = 0

    with torch.inference_mode():
        for batch in export_loader:
            if not batch:
                continue
            take = int(batch["input"].shape[0])

            x = batch["input"].to(torch_device, non_blocking=True)
            y = batch["output"].to(torch_device, non_blocking=True)
            mask = batch["output_mask"].to(torch_device, non_blocking=True)
            batch_size_actual, source_length = int(x.shape[0]), int(x.shape[1])

            raw_dt_b = model_api._broadcast_dt(batch["dt"], batch_size_actual, source_length, x.device, x.dtype)
            raw_dt_b = model_api._calibrate_dt(raw_dt_b)
            x, y, mask, dt_b = model_api._resample_to_operator_grid(
                x,
                y,
                mask,
                raw_dt_b,
                int(operator_time_points),
            )
            model_api._assert_operator_grid_dt_consistency(
                raw_dt_b,
                dt_b,
                source_length,
                int(operator_time_points),
            )
            nibp_features = model_module_obj._nibp_features_from_batch(
                batch,
                device=x.device,
                dtype=x.dtype,
                require_all_valid=False,
            )
            pred, _, _ = model(x, dt_seconds=dt_b, nibp_features=nibp_features)
            pred_bt = model_api._to_bt2d(pred)
            true_bt = model_api._to_bt2d(y)
            pred_bt, true_bt, mask_bt = model_api._align_bt(pred_bt, true_bt, mask)

            if signal_length is None:
                signal_length = int(true_bt.shape[1])
                y_true_mm = np.lib.format.open_memmap(
                    y_true_path,
                    mode="w+",
                    dtype=np.float32,
                    shape=(n_windows, signal_length),
                )
                y_pred_mm = np.lib.format.open_memmap(
                    y_pred_path,
                    mode="w+",
                    dtype=np.float32,
                    shape=(n_windows, signal_length),
                )
                mask_mm = np.lib.format.open_memmap(
                    mask_path,
                    mode="w+",
                    dtype=np.bool_,
                    shape=(n_windows, signal_length),
                )
            elif int(true_bt.shape[1]) != signal_length:
                raise RuntimeError(f"Inconsistent signal length: {true_bt.shape[1]} vs {signal_length}.")

            end = offset + batch_size_actual
            true_np = true_bt.detach().cpu().numpy().astype(np.float32, copy=False)
            pred_np = pred_bt.detach().cpu().numpy().astype(np.float32, copy=False)
            mask_np = mask_bt.detach().cpu().numpy().astype(np.bool_, copy=False)
            true_np = np.where(mask_np, true_np, np.nan).astype(np.float32, copy=False)
            pred_np = np.where(mask_np, pred_np, np.nan).astype(np.float32, copy=False)
            y_true_mm[offset:end, :] = true_np
            y_pred_mm[offset:end, :] = pred_np
            mask_mm[offset:end, :] = mask_np

            subjects = _batch_subjects(batch, batch_size_actual)
            record_ids = _batch_field(batch, "record_id", batch_size_actual)
            hadm_ids = _batch_field(batch, "hadm_id", batch_size_actual)
            starts = _batch_field(batch, "start", batch_size_actual)
            ends = _batch_field(batch, "end", batch_size_actual)
            for local_idx in range(batch_size_actual):
                memmap_index = offset + local_idx
                rows.append(
                    {
                        "split": "test",
                        "window_id": f"test_{memmap_index:08d}",
                        "subject_id": subjects[local_idx],
                        "record_id": record_ids[local_idx],
                        "hadm_id": hadm_ids[local_idx],
                        "start": starts[local_idx],
                        "end": ends[local_idx],
                        "memmap_index": int(memmap_index),
                    }
                )

            offset = end
            if offset % max(1, int(batch_size) * 25) == 0:
                print(f"[operator-v9-export] wrote {offset}/{n_windows} windows", flush=True)

    if offset != n_windows:
        raise RuntimeError(f"Prediction count mismatch: expected {n_windows}, wrote {offset}.")
    if y_true_mm is None or y_pred_mm is None or mask_mm is None or signal_length is None:
        raise RuntimeError("No predictions were written.")

    y_true_mm.flush()
    y_pred_mm.flush()
    mask_mm.flush()

    windows_manifest_path = prepared_dir / "windows_manifest.csv"
    pd.DataFrame(rows).to_csv(windows_manifest_path, index=False)

    config_path = prepared_dir / "operator_benchmark_config.json"
    config = {
        "method": "operator_v9_A_nibp_checkpoint_inference",
        "model_module": MODEL_MODULE,
        "dataset_backend": "materialized_vitaldb",
        "checkpoint": str(checkpoint_path),
        "checkpoint_state": state,
        "checkpoint_architecture": checkpoint.get("architecture"),
        "checkpoint_training_variant": checkpoint.get("training_variant"),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_seed": checkpoint.get("seed"),
        "test_manifest": str(test_manifest),
        "output_dir": str(output_dir),
        "pin_memory": False,
        "window_seconds": float(chunk_duration),
        "signal_length": int(signal_length),
        "n_windows": int(n_windows),
        "grid_points": int(grid_points),
        "operator_time_points": int(operator_time_points),
        "nx": int(nx),
        "encoder_temporal_stride": int(encoder_temporal_stride),
        "min_valid_ratio": float(min_valid_ratio),
        "input_signals": ["II", "PLETH"],
        "output_signals": ["ABP"],
        "use_nibp_input": True,
        "require_nibp_input_valid": False,
    }
    config_path.write_text(json.dumps(_json_safe(config), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    predictions_manifest_path = predictions_dir / "test_predictions_manifest.json"
    predictions_manifest = {
        "storage_format": "memmap_npy",
        "y_true": str(y_true_path),
        "y_pred": str(y_pred_path),
        "output_mask": str(mask_path),
        "shape": [int(n_windows), int(signal_length)],
        "dtype": "float32",
        "window_seconds": float(chunk_duration),
        "model_module": MODEL_MODULE,
        "checkpoint": str(checkpoint_path),
        "checkpoint_state": state,
        "use_nibp_input": True,
        "require_nibp_input_valid": False,
    }
    predictions_manifest_path.write_text(
        json.dumps(_json_safe(predictions_manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    summary = {
        "run_dir": str(output_dir),
        "prepared_config": str(config_path),
        "windows_manifest": str(windows_manifest_path),
        "predictions_manifest": str(predictions_manifest_path),
        "n_windows": int(n_windows),
        "signal_length": int(signal_length),
        "model_module": MODEL_MODULE,
        "dataset_backend": "materialized_vitaldb",
        "checkpoint": str(checkpoint_path),
        "checkpoint_state": state,
        "checkpoint_architecture": checkpoint.get("architecture"),
        "checkpoint_training_variant": checkpoint.get("training_variant"),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_seed": checkpoint.get("seed"),
        "use_nibp_input": True,
        "require_nibp_input_valid": False,
    }
    summary_path = output_dir / "operator_export_summary.json"
    summary_path.write_text(json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the VitalDB fold 0 benchmark for the v9-A NIBP operator checkpoint."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--table-csv", required=True)
    parser.add_argument("--label", default="HIPNO v9-A NIBP operator")
    parser.add_argument("--state", choices=["ema", "model"], default="ema")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--chunk-duration", type=float, default=10.0)
    parser.add_argument("--grid-points", type=int, default=250)
    parser.add_argument("--operator-time-points", type=int, default=250)
    parser.add_argument("--nx", type=int, default=16)
    parser.add_argument("--encoder-temporal-stride", type=int, default=1)
    parser.add_argument("--min-valid-ratio", type=float, default=0.98)
    parser.add_argument("--buffer-size", type=int, default=64)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=2025)
    parser.add_argument("--metrics-chunk-size", type=int, default=256)
    parser.add_argument("--overwrite-csv", action="store_true")
    args = parser.parse_args()

    os.environ["PINN_DT_UNITS"] = "seconds"

    checkpoint = _path(args.checkpoint)
    test_manifest = _path(args.test_manifest)
    run_dir = _path(args.run_dir)
    table_csv = _path(args.table_csv)

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    if not test_manifest.is_file():
        raise FileNotFoundError(f"Missing test manifest: {test_manifest}")

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    table_csv.parent.mkdir(parents=True, exist_ok=True)

    export_summary = export_v9_operator_predictions(
        checkpoint_path=checkpoint,
        test_manifest=test_manifest,
        output_dir=run_dir,
        state=args.state,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        chunk_duration=args.chunk_duration,
        grid_points=args.grid_points,
        operator_time_points=args.operator_time_points,
        nx=args.nx,
        encoder_temporal_stride=args.encoder_temporal_stride,
        min_valid_ratio=args.min_valid_ratio,
        buffer_size=args.buffer_size,
    )

    predictions_manifest = run_dir / "predictions" / "test_predictions_manifest.json"
    windows_manifest = run_dir / "prepared_data" / "windows_manifest.csv"
    if not predictions_manifest.is_file():
        raise FileNotFoundError(f"Missing prediction manifest after export: {predictions_manifest}")
    if not windows_manifest.is_file():
        raise FileNotFoundError(f"Missing windows manifest after export: {windows_manifest}")

    metrics_json = run_dir / "metrics" / "patient_metrics.json"
    metrics_summary = compute_patient_metrics(
        run_dir,
        method="operator",
        method_label=args.label,
        output_json=metrics_json,
        output_csv=table_csv,
        overwrite_csv=bool(args.overwrite_csv),
        n_bootstrap=args.bootstrap_samples,
        seed=args.bootstrap_seed,
        chunk_size=args.metrics_chunk_size,
        compute_scalar=True,
        reference_qc=False,
    )

    print(
        json.dumps(
            _json_safe(
                {
                    "export": export_summary,
                    "dt_units": os.environ["PINN_DT_UNITS"],
                    "model_module": MODEL_MODULE,
                    "nibp_input": {
                        "enabled": True,
                        "require_all_valid": False,
                    },
                    "metrics_json": metrics_json,
                    "table_csv": table_csv,
                    "table_row": metrics_summary.get("table_row"),
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
