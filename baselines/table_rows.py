from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


TABLE_ROW_COLUMNS = [
    "method",
    "MAE",
    "RMSE",
    "Corr",
    "SBP err",
    "DBP err",
    "MAP err",
    "metrics_json",
]


def _to_abs_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _metric(summary: dict[str, Any], section: str, key: str) -> Any:
    value = summary.get(section, {}).get(key)
    return "" if value is None else value


def metrics_json_to_table_row(
    metrics_json: str | Path,
    *,
    method: str | None = None,
) -> dict[str, Any]:
    metrics_path = _to_abs_path(metrics_json)
    summary = json.loads(metrics_path.read_text(encoding="utf-8"))
    method_name = method or summary.get("baseline_key") or metrics_path.parent.parent.name

    return {
        "method": method_name,
        "MAE": _metric(summary, "waveform", "waveform_mae"),
        "RMSE": _metric(summary, "waveform", "waveform_rmse"),
        "Corr": _metric(summary, "waveform", "waveform_corr"),
        "SBP err": _metric(summary, "scalar", "sbp_mae"),
        "DBP err": _metric(summary, "scalar", "dbp_mae"),
        "MAP err": _metric(summary, "scalar", "map_mae"),
        "metrics_json": str(metrics_path),
    }


def write_table_row(
    row: dict[str, Any],
    output_csv: str | Path,
    *,
    append: bool = True,
) -> dict[str, Any]:
    output_path = _to_abs_path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append and output_path.exists() else "w"
    write_header = mode == "w" or output_path.stat().st_size == 0

    with output_path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_ROW_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in TABLE_ROW_COLUMNS})

    return {
        "output_csv": str(output_path),
        "appended": mode == "a",
        "row": row,
    }
