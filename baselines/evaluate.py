from __future__ import annotations

from typing import Any

import numpy as np

from .metrics import beat_statistics_from_segments, bp_from_waveforms, scalar_metric_summary


class RunningWaveformMetrics:
    def __init__(self):
        self.n_points = 0
        self.sum_abs = 0.0
        self.sum_sq = 0.0
        self.sum_x = 0.0
        self.sum_y = 0.0
        self.sum_x2 = 0.0
        self.sum_y2 = 0.0
        self.sum_xy = 0.0

    def update(self, y_true: Any, y_pred: Any) -> None:
        true_bt = np.asarray(y_true, dtype=np.float64)
        pred_bt = np.asarray(y_pred, dtype=np.float64)
        if true_bt.shape != pred_bt.shape:
            raise ValueError(f"Shape mismatch: {true_bt.shape} vs {pred_bt.shape}")

        mask = np.isfinite(true_bt) & np.isfinite(pred_bt)
        x = true_bt[mask].reshape(-1)
        y = pred_bt[mask].reshape(-1)
        if x.size == 0:
            return

        diff = y - x
        self.n_points += int(x.size)
        self.sum_abs += float(np.sum(np.abs(diff)))
        self.sum_sq += float(np.sum(diff**2))
        self.sum_x += float(np.sum(x))
        self.sum_y += float(np.sum(y))
        self.sum_x2 += float(np.sum(x**2))
        self.sum_y2 += float(np.sum(y**2))
        self.sum_xy += float(np.sum(x * y))

    def finalize(self) -> dict[str, float | int | None]:
        if self.n_points == 0:
            return {
                "n_points": 0,
                "waveform_mae": None,
                "waveform_rmse": None,
                "waveform_corr": None,
            }

        n = float(self.n_points)
        mean_x = self.sum_x / n
        mean_y = self.sum_y / n
        cov = (self.sum_xy / n) - (mean_x * mean_y)
        var_x = (self.sum_x2 / n) - (mean_x**2)
        var_y = (self.sum_y2 / n) - (mean_y**2)
        if var_x <= 1e-12 or var_y <= 1e-12:
            corr = None
        else:
            corr = float(cov / np.sqrt(var_x * var_y))

        return {
            "n_points": int(self.n_points),
            "waveform_mae": float(self.sum_abs / n),
            "waveform_rmse": float(np.sqrt(self.sum_sq / n)),
            "waveform_corr": corr,
        }


class RunningScalarErrorMetrics:
    def __init__(self):
        self.errors = {
            "sbp": [],
            "dbp": [],
            "map": [],
        }

    def update(self, true_bp: dict[str, Any], pred_bp: dict[str, Any]) -> None:
        true_waveforms = np.asarray(true_bp["waveforms"], dtype=np.float64)
        pred_waveforms = np.asarray(pred_bp["waveforms"], dtype=np.float64)
        if true_waveforms.shape != pred_waveforms.shape:
            raise ValueError(f"Shape mismatch: {true_waveforms.shape} vs {pred_waveforms.shape}")

        segments_per_sample = true_bp.get("segments", [])
        for sample_idx, segments in enumerate(segments_per_sample):
            if not segments:
                continue

            true_stats = beat_statistics_from_segments(true_waveforms[sample_idx], segments)
            pred_stats = beat_statistics_from_segments(pred_waveforms[sample_idx], segments)

            for metric in ("sbp", "dbp", "map"):
                true_values = np.asarray(true_stats[metric], dtype=np.float64)
                pred_values = np.asarray(pred_stats[metric], dtype=np.float64)
                if true_values.size == 0 or pred_values.size == 0:
                    continue
                n = min(true_values.size, pred_values.size)
                self.errors[metric].extend((pred_values[:n] - true_values[:n]).tolist())

    def finalize(self) -> dict[str, Any]:
        sbp = scalar_metric_summary(np.asarray(self.errors["sbp"], dtype=np.float64))
        dbp = scalar_metric_summary(np.asarray(self.errors["dbp"], dtype=np.float64))
        map_ = scalar_metric_summary(np.asarray(self.errors["map"], dtype=np.float64))

        return {
            "n_beats": int(max(sbp["n_beats"], dbp["n_beats"], map_["n_beats"])),
            "sbp_mae": sbp["mae"],
            "dbp_mae": dbp["mae"],
            "map_mae": map_["mae"],
            "sbp_rmse": sbp["rmse"],
            "dbp_rmse": dbp["rmse"],
            "map_rmse": map_["rmse"],
            "sbp_bias": sbp["bias"],
            "dbp_bias": dbp["bias"],
            "map_bias": map_["bias"],
            "sbp_std": sbp["std"],
            "dbp_std": dbp["std"],
            "map_std": map_["std"],
            "sbp_loa_low": sbp["loa_low"],
            "sbp_loa_high": sbp["loa_high"],
            "dbp_loa_low": dbp["loa_low"],
            "dbp_loa_high": dbp["loa_high"],
            "map_loa_low": map_["loa_low"],
            "map_loa_high": map_["loa_high"],
            "sbp_aami_compliant": sbp["aami_compliant"],
            "dbp_aami_compliant": dbp["aami_compliant"],
            "map_aami_compliant": map_["aami_compliant"],
            "sbp_bhs_grade": sbp["bhs_grade"],
            "dbp_bhs_grade": dbp["bhs_grade"],
            "map_bhs_grade": map_["bhs_grade"],
            "standards": {
                "sbp": sbp,
                "dbp": dbp,
                "map": map_,
            },
        }
