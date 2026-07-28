from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from .manifest import build_method_frame, validate_unified_manifest
from .policy import INPUT_POLICIES
from .registry import BASELINE_SPECS


RUN_LAYOUT_VERSION = 1
EXPORT_MANIFEST_COLUMNS = [
    "export_index",
    "baseline_key",
    "split",
    "source",
    "subject_id",
    "record_id",
    "hadm_id",
    "fs",
    "num_rows",
    "input_columns",
    "target_column",
    "source_parquet",
    "exported_parquet",
    "source_manifest",
    "source_row_index",
]


def _safe_filename_token(text: Any) -> str:
    token = "".join(ch if str(ch).isalnum() else "_" for ch in str(text).strip())
    token = token.strip("_")
    return token or "record"


def _ensure_supported_baseline(baseline_key: str) -> None:
    if baseline_key not in INPUT_POLICIES:
        raise KeyError(f"Unsupported baseline key: {baseline_key}")
    if baseline_key not in BASELINE_SPECS:
        raise KeyError(f"Missing baseline spec for key: {baseline_key}")


def _load_valid_unified_manifest(manifest_csv: str | Path, limit: int | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    summary = validate_unified_manifest(manifest_csv, limit=limit)
    if not summary["valid"]:
        raise ValueError(
            "Unified manifest is not valid for export. "
            f"Validation summary:\n{json.dumps(summary, indent=2, sort_keys=True)}"
        )

    manifest_csv = Path(manifest_csv).expanduser().resolve()
    df = pd.read_csv(manifest_csv)
    if limit is not None and limit > 0:
        df = df.head(limit).copy()
    return df, summary


def export_baseline_view(
    manifest_csv: str | Path,
    baseline_key: str,
    output_dir: str | Path,
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    _ensure_supported_baseline(baseline_key)
    df, validation_summary = _load_valid_unified_manifest(manifest_csv, limit=limit)

    manifest_csv = Path(manifest_csv).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    records_dir = output_dir / "records"
    upstream_output_dir = output_dir / "upstream_output"
    metrics_dir = output_dir / "metrics"
    records_dir.mkdir(parents=True, exist_ok=True)
    upstream_output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    policy = INPUT_POLICIES[baseline_key]
    spec = BASELINE_SPECS[baseline_key]
    input_columns = tuple(policy.allowed_input_sets[0])
    target_column = policy.target

    exported_rows: list[dict[str, Any]] = []
    for export_index, row in enumerate(df.to_dict(orient="records")):
        frame = build_method_frame(
            row,
            input_columns=input_columns,
            target_column=target_column,
            include_meta=True,
        )
        filename = (
            f"{export_index:05d}_"
            f"{_safe_filename_token(row['split'])}_"
            f"{_safe_filename_token(row['source'])}_"
            f"{_safe_filename_token(row['record_id'])}.parquet"
        )
        exported_path = records_dir / filename
        frame.to_parquet(exported_path, index=False)

        exported_rows.append(
            {
                "export_index": int(export_index),
                "baseline_key": baseline_key,
                "split": row["split"],
                "source": row["source"],
                "subject_id": row["subject_id"],
                "record_id": row["record_id"],
                "hadm_id": row.get("hadm_id", None),
                "fs": row["fs"],
                "num_rows": int(len(frame)),
                "input_columns": json.dumps(list(input_columns)),
                "target_column": target_column,
                "source_parquet": row["parquet"],
                "exported_parquet": str(exported_path),
                "source_manifest": row["source_manifest"],
                "source_row_index": int(row["source_row_index"]),
            }
        )

    export_manifest_path = output_dir / "export_manifest.csv"
    export_manifest = pd.DataFrame(exported_rows, columns=EXPORT_MANIFEST_COLUMNS)
    export_manifest.to_csv(export_manifest_path, index=False)

    run_config_path = output_dir / "run_config.json"
    run_config = {
        "layout_version": RUN_LAYOUT_VERSION,
        "baseline_key": baseline_key,
        "baseline_spec": asdict(spec),
        "policy": {
            "baseline_key": policy.baseline_key,
            "allowed_input_sets": [list(cols) for cols in policy.allowed_input_sets],
            "target": policy.target,
            "compare_waveform_metrics": policy.compare_waveform_metrics,
            "compare_scalar_metrics": policy.compare_scalar_metrics,
            "keep_upstream_code_untouched": policy.keep_upstream_code_untouched,
        },
        "source_unified_manifest": str(manifest_csv),
        "validation_summary": validation_summary,
        "record_view_level": "record",
        "rows_exported": int(len(export_manifest)),
        "export_manifest": str(export_manifest_path),
        "records_dir": str(records_dir),
        "upstream_output_dir": str(upstream_output_dir),
        "metrics_dir": str(metrics_dir),
    }
    run_config_path.write_text(json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    notes_path = output_dir / "UPSTREAM_NOTES.txt"
    notes_lines = [
        f"baseline_key={baseline_key}",
        f"baseline_name={spec.name}",
        "export_level=record",
        "upstream_code_policy=keep_original_code_untouched",
        f"source_unified_manifest={manifest_csv}",
        f"export_manifest={export_manifest_path}",
        f"records_dir={records_dir}",
        f"allowed_input_columns={','.join(input_columns)}",
        f"target_column={target_column}",
        "rule=consume only the exported record views for this baseline",
        "rule=do not expose extra modalities that are not listed above",
        "rule=windowing, resampling, and batching must stay faithful to the original upstream method",
        "rule=any later launcher script must live outside the upstream repo and must not edit upstream model code",
        "next_step=write a repo-specific launcher that reads export_manifest.csv and converts these record views into the upstream method's original training/evaluation inputs",
    ]
    notes_path.write_text("\n".join(notes_lines) + "\n", encoding="utf-8")

    return {
        "baseline_key": baseline_key,
        "output_dir": str(output_dir),
        "source_unified_manifest": str(manifest_csv),
        "rows_exported": int(len(export_manifest)),
        "record_view_level": "record",
        "export_manifest": str(export_manifest_path),
        "run_config": str(run_config_path),
        "records_dir": str(records_dir),
        "upstream_output_dir": str(upstream_output_dir),
        "metrics_dir": str(metrics_dir),
        "inputs": list(input_columns),
        "target": target_column,
    }
