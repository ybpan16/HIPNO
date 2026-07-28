from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


SIGNAL_ALIASES = {
    "II": ["II"],
    "PLETH": ["PLETH", "PPG"],
    "ABP": ["ABP", "ART"],
}

CANONICAL_SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "test": "test",
    "testing": "test",
}
CANONICAL_SOURCE_ALIASES = {
    "vitaldb": "vitaldb",
    "vital": "vitaldb",
    "mimic4": "mimic4wdb",
    "mimic4wdb": "mimic4wdb",
    "mimiciv": "mimic4wdb",
    "mimicivwdb": "mimic4wdb",
    "mimic2": "mimic2co",
    "mimic2co": "mimic2co",
    "mimiciico": "mimic2co",
}

REQUIRED_SOURCE_MANIFEST_COLUMNS = ["subject_id", "record_id", "fs", "parquet"]
OPTIONAL_SOURCE_MANIFEST_COLUMNS = ["hadm_id"]
STANDARDIZED_SIGNAL_COLUMNS = ["II", "PLETH", "ABP"]
STANDARDIZED_META_COLUMNS = ["timestamp", "fs", "subject_id", "hadm_id", "record_id", "split", "source"]
META_COLUMNS = ["subject_id", "record_id", "hadm_id", "fs", "split", "source"]
UNIFIED_MANIFEST_COLUMNS = [
    "split",
    "source",
    "subject_id",
    "record_id",
    "hadm_id",
    "fs",
    "parquet",
    "ecg_col",
    "ppg_col",
    "abp_col",
    "source_manifest",
    "source_row_index",
]
REQUIRED_UNIFIED_VALUE_COLUMNS = [
    "split",
    "source",
    "subject_id",
    "record_id",
    "fs",
    "parquet",
    "ecg_col",
    "ppg_col",
    "abp_col",
    "source_manifest",
    "source_row_index",
]


def _normalize_token(text: str) -> str:
    return "".join(ch.lower() for ch in str(text) if ch.isalnum())


def _resolve_first_matching_column(columns: list[str], aliases: list[str]) -> str | None:
    normalized = {_normalize_token(col): col for col in columns}
    for alias in aliases:
        match = normalized.get(_normalize_token(alias))
        if match is not None:
            return match
    return None


def resolve_parquet_path(raw_path: str | Path, manifest_csv: str | Path) -> Path:
    path = Path(str(raw_path)).expanduser()
    if path.is_absolute():
        return path.resolve()
    manifest_dir = Path(manifest_csv).expanduser().resolve().parent
    return (manifest_dir / path).resolve()


def _safe_scalar_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    return value


def _first_non_null(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if pd.isna(value):
            continue
        return value
    return None


def normalize_split_label(split: str) -> str:
    token = _normalize_token(split)
    normalized = CANONICAL_SPLIT_ALIASES.get(token)
    if normalized is None:
        allowed = ", ".join(sorted(set(CANONICAL_SPLIT_ALIASES.values())))
        raise ValueError(f"Unsupported split {split!r}. Use one of: {allowed}.")
    return normalized


def normalize_source_label(source: str) -> str:
    text = str(source).strip()
    if not text:
        raise ValueError("Source label must be a non-empty string.")
    token = _normalize_token(text)
    if token in CANONICAL_SOURCE_ALIASES:
        return CANONICAL_SOURCE_ALIASES[token]
    return text.lower()


def validate_source_manifest_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in REQUIRED_SOURCE_MANIFEST_COLUMNS if col not in df.columns]


def validate_unified_manifest_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in UNIFIED_MANIFEST_COLUMNS if col not in df.columns]


def inspect_parquet_schema(parquet_path: str | Path) -> dict[str, Any]:
    parquet_path = Path(parquet_path).expanduser().resolve()
    columns = list(pq.read_schema(parquet_path).names)

    ecg_col = _resolve_first_matching_column(columns, SIGNAL_ALIASES["II"])
    ppg_col = _resolve_first_matching_column(columns, SIGNAL_ALIASES["PLETH"])
    abp_col = _resolve_first_matching_column(columns, SIGNAL_ALIASES["ABP"])

    fs = None
    if "fs" in columns:
        parquet_file = pq.ParquetFile(parquet_path)
        if parquet_file.metadata is not None and parquet_file.metadata.num_rows > 0:
            first_fs = parquet_file.read_row_group(0, columns=["fs"]).to_pandas().iloc[0]["fs"]
            fs = None if pd.isna(first_fs) else first_fs

    return {
        "columns": columns,
        "ecg_col": ecg_col,
        "ppg_col": ppg_col,
        "abp_col": abp_col,
        "fs": None if fs is None else float(fs),
        "has_timestamp": "timestamp" in columns,
        "qualified": all([ecg_col, ppg_col, abp_col]),
    }


def inspect_manifest(manifest_csv: str | Path, limit: int | None = None) -> dict[str, Any]:
    manifest_csv = Path(manifest_csv).expanduser().resolve()
    df = pd.read_csv(manifest_csv)
    if limit is not None and limit > 0:
        df = df.head(limit).copy()

    missing_manifest_columns = validate_source_manifest_columns(df)
    total_rows = int(len(df))
    existing_rows = 0
    qualified_rows = 0
    missing_path_rows = 0
    missing_ecg_rows = 0
    missing_ppg_rows = 0
    missing_abp_rows = 0
    read_error_rows = 0
    fs_values: list[float] = []
    qualified_examples: list[str] = []

    if "parquet" not in df.columns:
        return {
            "manifest_csv": str(manifest_csv),
            "rows_total": total_rows,
            "rows_existing_locally": 0,
            "rows_qualified": 0,
            "rows_missing_path": total_rows,
            "rows_read_error": 0,
            "rows_missing_ii": 0,
            "rows_missing_pleth": 0,
            "rows_missing_abp": 0,
            "sample_rates_hz": [],
            "qualified_examples": [],
            "manifest_missing_columns": missing_manifest_columns,
            "ready_for_unified_build": False,
        }

    for row in df.itertuples(index=False):
        raw_path = getattr(row, "parquet", None)
        if raw_path is None or (isinstance(raw_path, float) and np.isnan(raw_path)):
            missing_path_rows += 1
            continue

        resolved = resolve_parquet_path(raw_path, manifest_csv)
        if not resolved.exists():
            missing_path_rows += 1
            continue

        existing_rows += 1
        try:
            schema = inspect_parquet_schema(resolved)
        except Exception:
            read_error_rows += 1
            continue

        if schema["fs"] is not None and np.isfinite(schema["fs"]):
            fs_values.append(float(schema["fs"]))
        if schema["ecg_col"] is None:
            missing_ecg_rows += 1
        if schema["ppg_col"] is None:
            missing_ppg_rows += 1
        if schema["abp_col"] is None:
            missing_abp_rows += 1
        if schema["qualified"]:
            qualified_rows += 1
            if len(qualified_examples) < 5:
                qualified_examples.append(str(resolved))

    unique_fs = sorted({round(v, 6) for v in fs_values})
    return {
        "manifest_csv": str(manifest_csv),
        "rows_total": total_rows,
        "rows_existing_locally": existing_rows,
        "rows_qualified": qualified_rows,
        "rows_missing_path": missing_path_rows,
        "rows_read_error": read_error_rows,
        "rows_missing_ii": missing_ecg_rows,
        "rows_missing_pleth": missing_ppg_rows,
        "rows_missing_abp": missing_abp_rows,
        "sample_rates_hz": unique_fs,
        "qualified_examples": qualified_examples,
        "manifest_missing_columns": missing_manifest_columns,
        "ready_for_unified_build": not missing_manifest_columns,
    }


def build_unified_manifest(
    manifest_csv: str | Path,
    output_csv: str | Path,
    *,
    split: str,
    source: str,
    limit: int | None = None,
) -> dict[str, Any]:
    manifest_csv = Path(manifest_csv).expanduser().resolve()
    output_csv = Path(output_csv).expanduser().resolve()
    canonical_split = normalize_split_label(split)
    canonical_source = normalize_source_label(source)

    df = pd.read_csv(manifest_csv)
    if limit is not None and limit > 0:
        df = df.head(limit).copy()

    missing_manifest_columns = validate_source_manifest_columns(df)
    if missing_manifest_columns:
        missing_text = ", ".join(missing_manifest_columns)
        raise ValueError(
            f"Source manifest {manifest_csv} is missing required columns: {missing_text}"
        )

    kept_rows: list[dict[str, Any]] = []
    dropped = {
        "missing_path": 0,
        "read_error": 0,
        "missing_ii": 0,
        "missing_pleth": 0,
        "missing_abp": 0,
    }

    for idx, row in df.iterrows():
        raw_path = row.get("parquet", None)
        if raw_path is None or (isinstance(raw_path, float) and np.isnan(raw_path)):
            dropped["missing_path"] += 1
            continue

        resolved = resolve_parquet_path(raw_path, manifest_csv)
        if not resolved.exists():
            dropped["missing_path"] += 1
            continue

        try:
            schema = inspect_parquet_schema(resolved)
        except Exception:
            dropped["read_error"] += 1
            continue

        if schema["ecg_col"] is None:
            dropped["missing_ii"] += 1
            continue
        if schema["ppg_col"] is None:
            dropped["missing_pleth"] += 1
            continue
        if schema["abp_col"] is None:
            dropped["missing_abp"] += 1
            continue

        kept_rows.append(
            {
                "split": canonical_split,
                "source": canonical_source,
                "subject_id": _first_non_null(row.get("subject_id", None), np.nan),
                "record_id": _first_non_null(row.get("record_id", None), resolved.stem),
                "hadm_id": _first_non_null(row.get("hadm_id", None), np.nan),
                "fs": _first_non_null(row.get("fs", None), schema["fs"]),
                "parquet": str(resolved),
                "ecg_col": schema["ecg_col"],
                "ppg_col": schema["ppg_col"],
                "abp_col": schema["abp_col"],
                "source_manifest": str(manifest_csv),
                "source_row_index": int(idx),
            }
        )

    out_df = pd.DataFrame(kept_rows, columns=UNIFIED_MANIFEST_COLUMNS)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)

    return {
        "source_manifest": str(manifest_csv),
        "output_manifest": str(output_csv),
        "split": canonical_split,
        "source": canonical_source,
        "rows_in": int(len(df)),
        "rows_out": int(len(out_df)),
        "dropped": dropped,
    }


def validate_unified_manifest(manifest_csv: str | Path, limit: int | None = None) -> dict[str, Any]:
    manifest_csv = Path(manifest_csv).expanduser().resolve()
    df = pd.read_csv(manifest_csv)
    if limit is not None and limit > 0:
        df = df.head(limit).copy()

    total_rows = int(len(df))
    missing_manifest_columns = validate_unified_manifest_columns(df)
    summary = {
        "manifest_csv": str(manifest_csv),
        "rows_total": total_rows,
        "empty_manifest": total_rows == 0,
        "manifest_missing_columns": missing_manifest_columns,
        "rows_missing_required_values": 0,
        "rows_invalid_split": 0,
        "rows_invalid_source": 0,
        "rows_missing_path": 0,
        "rows_read_error": 0,
        "rows_signal_column_mismatch": 0,
        "rows_fs_mismatch": 0,
        "rows_valid": 0,
        "duplicate_key_rows": 0,
        "sample_rates_hz": [],
        "valid": False,
    }

    if missing_manifest_columns:
        return summary

    dedupe_columns = ["split", "source", "subject_id", "record_id", "parquet"]
    summary["duplicate_key_rows"] = int(df.duplicated(subset=dedupe_columns, keep=False).sum())

    fs_values: list[float] = []
    for row in df.itertuples(index=False):
        row_dict = row._asdict()

        missing_required = any(_safe_scalar_value(row_dict.get(col)) is None for col in REQUIRED_UNIFIED_VALUE_COLUMNS)
        if missing_required:
            summary["rows_missing_required_values"] += 1
            continue

        try:
            normalize_split_label(str(row_dict["split"]))
        except Exception:
            summary["rows_invalid_split"] += 1
            continue

        try:
            normalize_source_label(str(row_dict["source"]))
        except Exception:
            summary["rows_invalid_source"] += 1
            continue

        resolved = resolve_parquet_path(row_dict["parquet"], manifest_csv)
        if not resolved.exists():
            summary["rows_missing_path"] += 1
            continue

        try:
            schema = inspect_parquet_schema(resolved)
        except Exception:
            summary["rows_read_error"] += 1
            continue

        signal_cols = [row_dict["ecg_col"], row_dict["ppg_col"], row_dict["abp_col"]]
        if any(col not in schema["columns"] for col in signal_cols):
            summary["rows_signal_column_mismatch"] += 1
            continue

        manifest_fs = float(row_dict["fs"])
        if schema["fs"] is not None and np.isfinite(schema["fs"]):
            fs_values.append(float(schema["fs"]))
            if not np.isclose(manifest_fs, float(schema["fs"]), atol=1e-6):
                summary["rows_fs_mismatch"] += 1
                continue
        else:
            fs_values.append(manifest_fs)

        summary["rows_valid"] += 1

    summary["sample_rates_hz"] = sorted({round(v, 6) for v in fs_values})
    summary["valid"] = (
        total_rows > 0
        and
        summary["rows_valid"] == total_rows
        and summary["duplicate_key_rows"] == 0
        and not summary["manifest_missing_columns"]
    )
    return summary


def split_unified_manifest(
    manifest_csv: str | Path,
    train_output_csv: str | Path,
    val_output_csv: str | Path,
    *,
    val_fraction: float = 0.2,
    seed: int = 2025,
    group_column: str = "subject_id",
) -> dict[str, Any]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction!r}.")

    manifest_csv = Path(manifest_csv).expanduser().resolve()
    train_output_csv = Path(train_output_csv).expanduser().resolve()
    val_output_csv = Path(val_output_csv).expanduser().resolve()

    summary = validate_unified_manifest(manifest_csv)
    if not summary["valid"]:
        raise ValueError(
            "Cannot split an invalid unified manifest. "
            f"Validation summary:\n{manifest_summary_json(summary)}"
        )

    df = pd.read_csv(manifest_csv)
    if group_column not in df.columns:
        raise ValueError(f"group_column {group_column!r} is not present in {manifest_csv}.")

    groups = pd.Series(df[group_column].dropna().unique()).sample(frac=1.0, random_state=int(seed)).tolist()
    if len(groups) < 2:
        raise ValueError(
            f"Need at least two non-null groups in {group_column!r} to create train/val splits."
        )

    n_val = max(1, int(round(len(groups) * float(val_fraction))))
    n_val = min(n_val, len(groups) - 1)
    val_groups = set(groups[:n_val])

    val_df = df[df[group_column].isin(val_groups)].copy()
    train_df = df[~df[group_column].isin(val_groups)].copy()
    if train_df.empty or val_df.empty:
        raise ValueError("Split produced an empty train or val manifest.")

    train_df["split"] = "train"
    val_df["split"] = "val"

    train_output_csv.parent.mkdir(parents=True, exist_ok=True)
    val_output_csv.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_output_csv, index=False)
    val_df.to_csv(val_output_csv, index=False)

    return {
        "source_manifest": str(manifest_csv),
        "train_output_manifest": str(train_output_csv),
        "val_output_manifest": str(val_output_csv),
        "group_column": group_column,
        "seed": int(seed),
        "val_fraction": float(val_fraction),
        "groups_total": int(len(groups)),
        "groups_train": int(len(groups) - len(val_groups)),
        "groups_val": int(len(val_groups)),
        "rows_total": int(len(df)),
        "rows_train": int(len(train_df)),
        "rows_val": int(len(val_df)),
    }


def load_standardized_record(row: pd.Series | dict[str, Any]) -> pd.DataFrame:
    row = dict(row)
    parquet_path = Path(str(row["parquet"])).expanduser().resolve()
    available_cols = set(pq.read_schema(parquet_path).names)

    read_cols = [row["ecg_col"], row["ppg_col"], row["abp_col"]]
    for optional in ["timestamp", "fs", "subject_id", "hadm_id"]:
        if optional in available_cols and optional not in read_cols:
            read_cols.append(optional)

    frame = pd.read_parquet(parquet_path, columns=read_cols).copy()
    frame = frame.rename(
        columns={
            row["ecg_col"]: "II",
            row["ppg_col"]: "PLETH",
            row["abp_col"]: "ABP",
        }
    )
    if "record_id" not in frame.columns:
        frame["record_id"] = row.get("record_id", parquet_path.stem)
    if "fs" not in frame.columns or frame["fs"].isna().all():
        frame["fs"] = row.get("fs", np.nan)
    if "subject_id" not in frame.columns or frame["subject_id"].isna().all():
        frame["subject_id"] = row.get("subject_id", np.nan)
    if "hadm_id" not in frame.columns or frame["hadm_id"].isna().all():
        frame["hadm_id"] = row.get("hadm_id", np.nan)
    frame["split"] = row.get("split", None)
    frame["source"] = row.get("source", None)
    return frame


def build_method_frame(
    row: pd.Series | dict[str, Any],
    *,
    input_columns: tuple[str, ...] | list[str],
    target_column: str = "ABP",
    include_meta: bool = True,
) -> pd.DataFrame:
    frame = load_standardized_record(row)
    input_columns = tuple(input_columns)

    missing = [col for col in input_columns if col not in frame.columns]
    if missing:
        raise KeyError(f"Requested input columns are missing from the standardized record: {missing}")
    if target_column not in frame.columns:
        raise KeyError(f"Requested target column is missing from the standardized record: {target_column}")

    keep = list(input_columns) + [target_column]
    if include_meta:
        for col in STANDARDIZED_META_COLUMNS:
            if col in frame.columns and col not in keep:
                keep.append(col)
    return frame.loc[:, keep].copy()


def manifest_summary_json(summary: dict[str, Any]) -> str:
    return json.dumps(summary, indent=2, sort_keys=True)
