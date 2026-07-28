from __future__ import annotations

import argparse

from .baseline_views import export_baseline_view
from .inn_par_launcher import (
    evaluate_inn_par_run,
    generate_inn_par_run,
    prepare_inn_par_run,
    run_inn_par_benchmark,
    train_inn_par_benchmark,
)
from .manifest import (
    build_unified_manifest,
    inspect_manifest,
    manifest_summary_json,
    split_unified_manifest,
    validate_unified_manifest,
)
from .papagei_launcher import (
    evaluate_papagei_run,
    prepare_papagei_run,
    run_papagei_benchmark,
)
from .patient_metrics import aggregate_patient_metric_seeds, compute_patient_metrics
from .ppg2abp_launcher import (
    evaluate_ppg2abp_run,
    prepare_ppg2abp_run,
    run_ppg2abp_benchmark,
)
from .ptt_launcher import run_ptt_benchmark
from .table_rows import metrics_json_to_table_row, write_table_row
from .unicardio_launcher import (
    evaluate_unicardio_run,
    generate_unicardio_run,
    prepare_unicardio_run,
    run_unicardio_benchmark,
    run_unicardio_inference_benchmark,
    train_unicardio_benchmark,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="ABP baseline comparison utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect-manifest", help="Inspect a manifest and report schema readiness.")
    inspect_parser.add_argument("--manifest", required=True)
    inspect_parser.add_argument("--limit", type=int, default=None)

    build_parser = subparsers.add_parser("build-unified-manifest", help="Filter a manifest down to qualified ABP benchmark rows.")
    build_parser.add_argument("--manifest", required=True)
    build_parser.add_argument("--output", required=True)
    build_parser.add_argument("--split", required=True, help="Canonical split label: train, val, or test.")
    build_parser.add_argument("--source", required=True, help="Canonical data source label, for example vitaldb or mimic4wdb.")
    build_parser.add_argument("--limit", type=int, default=None)

    validate_parser = subparsers.add_parser(
        "validate-unified-manifest",
        help="Validate that a unified manifest follows the benchmark schema and still matches local parquet files.",
    )
    validate_parser.add_argument("--manifest", required=True)
    validate_parser.add_argument("--limit", type=int, default=None)

    split_parser = subparsers.add_parser(
        "split-unified-manifest",
        help="Create train/val unified manifests from one validated unified training manifest.",
    )
    split_parser.add_argument("--manifest", required=True)
    split_parser.add_argument("--train-output", required=True)
    split_parser.add_argument("--val-output", required=True)
    split_parser.add_argument("--val-fraction", type=float, default=0.2)
    split_parser.add_argument("--seed", type=int, default=2025)
    split_parser.add_argument("--group-column", default="subject_id")

    export_parser = subparsers.add_parser(
        "export-baseline-view",
        help="Materialize one baseline-specific record view from a validated unified manifest.",
    )
    export_parser.add_argument("--manifest", required=True)
    export_parser.add_argument("--baseline", required=True)
    export_parser.add_argument("--output-dir", required=True)
    export_parser.add_argument("--limit", type=int, default=None)

    run_ptt_parser = subparsers.add_parser(
        "run-ptt-benchmark",
        help="Fit and evaluate the classical ECG+PPG pulse-transit-time scalar BP baseline.",
    )
    run_ptt_parser.add_argument("--train-manifest", required=True)
    run_ptt_parser.add_argument("--test-manifest", required=True)
    run_ptt_parser.add_argument("--output-dir", required=True)
    run_ptt_parser.add_argument("--val-manifest", default=None)
    run_ptt_parser.add_argument("--include-val-in-fit", action=argparse.BooleanOptionalAction, default=False)
    run_ptt_parser.add_argument("--feature-transform", choices=["inverse_ptt", "log_ptt"], default="inverse_ptt")
    run_ptt_parser.add_argument("--min-ptt-sec", type=float, default=0.08)
    run_ptt_parser.add_argument("--max-ptt-sec", type=float, default=0.50)
    run_ptt_parser.add_argument("--min-peak-distance-sec", type=float, default=0.30)
    run_ptt_parser.add_argument("--ecg-prominence-scale", type=float, default=0.50)
    run_ptt_parser.add_argument("--ppg-prominence-scale", type=float, default=0.10)
    run_ptt_parser.add_argument("--min-finite-segment-sec", type=float, default=8.0)
    run_ptt_parser.add_argument("--limit-records", type=int, default=None)

    prepare_ppg2abp_parser = subparsers.add_parser(
        "prepare-ppg2abp-run",
        help="Build a benchmark-specific PPG2ABP run package from one or more exported baseline views.",
    )
    prepare_ppg2abp_parser.add_argument("--export-manifest", action="append", required=True)
    prepare_ppg2abp_parser.add_argument("--output-dir", required=True)
    prepare_ppg2abp_parser.add_argument("--target-fs-hz", type=float, default=None)
    prepare_ppg2abp_parser.add_argument("--window-sec", type=float, default=None)
    prepare_ppg2abp_parser.add_argument("--step-sec", type=float, default=None)
    prepare_ppg2abp_parser.add_argument("--model-input-length", type=int, default=None)

    run_ppg2abp_parser = subparsers.add_parser(
        "run-ppg2abp-benchmark",
        help="Train and evaluate the PPG2ABP benchmark launcher against a local upstream repo clone.",
    )
    run_ppg2abp_parser.add_argument("--run-dir", required=True)
    run_ppg2abp_parser.add_argument("--upstream-repo", required=True)
    run_ppg2abp_parser.add_argument("--epochs-stage1", type=int, default=100)
    run_ppg2abp_parser.add_argument("--epochs-stage2", type=int, default=100)
    run_ppg2abp_parser.add_argument("--batch-size-stage1", type=int, default=256)
    run_ppg2abp_parser.add_argument("--batch-size-stage2", type=int, default=192)

    evaluate_ppg2abp_parser = subparsers.add_parser(
        "evaluate-ppg2abp-run",
        help="Evaluate an already-completed PPG2ABP benchmark run using the shared metric stack.",
    )
    evaluate_ppg2abp_parser.add_argument("--run-dir", required=True)

    prepare_inn_par_parser = subparsers.add_parser(
        "prepare-inn-par-run",
        help="Build an INN-PAR datasets/{split}/{ppg,abp} run package from exported baseline views.",
    )
    prepare_inn_par_parser.add_argument("--export-manifest", action="append", required=True)
    prepare_inn_par_parser.add_argument("--output-dir", required=True)
    prepare_inn_par_parser.add_argument("--target-fs-hz", type=float, default=None)
    prepare_inn_par_parser.add_argument("--window-sec", type=float, default=None)
    prepare_inn_par_parser.add_argument("--step-sec", type=float, default=None)
    prepare_inn_par_parser.add_argument("--signal-length", type=int, default=None)

    train_inn_par_parser = subparsers.add_parser(
        "train-inn-par-benchmark",
        help="Train INN-PAR with the official PyTorch modules from an untouched upstream repo clone.",
    )
    train_inn_par_parser.add_argument("--run-dir", required=True)
    train_inn_par_parser.add_argument("--upstream-repo", required=True)
    train_inn_par_parser.add_argument("--max-step", type=int, default=500)
    train_inn_par_parser.add_argument("--batch-size", type=int, default=128)
    train_inn_par_parser.add_argument("--lr", type=float, default=1.0e-4)
    train_inn_par_parser.add_argument("--num-workers", type=int, default=12)
    train_inn_par_parser.add_argument("--checkpoint-interval", type=int, default=10)
    train_inn_par_parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=None)
    train_inn_par_parser.add_argument("--device", default=None)
    train_inn_par_parser.add_argument("--seed", type=int, default=1)

    generate_inn_par_parser = subparsers.add_parser(
        "generate-inn-par-run",
        help="Generate INN-PAR ABP predictions from an already-trained checkpoint.",
    )
    generate_inn_par_parser.add_argument("--run-dir", required=True)
    generate_inn_par_parser.add_argument("--upstream-repo", required=True)
    generate_inn_par_parser.add_argument("--checkpoint-path", default=None)
    generate_inn_par_parser.add_argument("--batch-size", type=int, default=1)
    generate_inn_par_parser.add_argument("--num-workers", type=int, default=12)
    generate_inn_par_parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=None)
    generate_inn_par_parser.add_argument("--device", default=None)
    generate_inn_par_parser.add_argument("--seed", type=int, default=1)

    evaluate_inn_par_parser = subparsers.add_parser(
        "evaluate-inn-par-run",
        help="Evaluate an already-completed INN-PAR benchmark run using the shared metric stack.",
    )
    evaluate_inn_par_parser.add_argument("--run-dir", required=True)

    run_inn_par_parser = subparsers.add_parser(
        "run-inn-par-benchmark",
        help="Train, generate, and evaluate the INN-PAR benchmark launcher.",
    )
    run_inn_par_parser.add_argument("--run-dir", required=True)
    run_inn_par_parser.add_argument("--upstream-repo", required=True)
    run_inn_par_parser.add_argument("--max-step", type=int, default=500)
    run_inn_par_parser.add_argument("--train-batch-size", type=int, default=128)
    run_inn_par_parser.add_argument("--test-batch-size", type=int, default=1)
    run_inn_par_parser.add_argument("--lr", type=float, default=1.0e-4)
    run_inn_par_parser.add_argument("--num-workers", type=int, default=12)
    run_inn_par_parser.add_argument("--checkpoint-interval", type=int, default=10)
    run_inn_par_parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=None)
    run_inn_par_parser.add_argument("--device", default=None)
    run_inn_par_parser.add_argument("--seed", type=int, default=1)

    prepare_papagei_parser = subparsers.add_parser(
        "prepare-papagei-run",
        help="Build a PaPaGei PPG segment package from exported baseline views using official preprocessing.",
    )
    prepare_papagei_parser.add_argument("--export-manifest", action="append", required=True)
    prepare_papagei_parser.add_argument("--output-dir", required=True)
    prepare_papagei_parser.add_argument("--upstream-repo", required=True)
    prepare_papagei_parser.add_argument("--target-fs-hz", type=float, default=None)
    prepare_papagei_parser.add_argument("--segment-sec", type=float, default=None)
    prepare_papagei_parser.add_argument("--step-sec", type=float, default=None)
    prepare_papagei_parser.add_argument("--segment-length", type=int, default=None)
    prepare_papagei_parser.add_argument(
        "--drop-zero-segment-records",
        action="store_true",
        help=(
            "Exclude records for which official PaPaGei preprocessing produces zero finite 10 s PPG segments. "
            "The exclusion is recorded in prepared_data/zero_segment_records.csv."
        ),
    )

    run_papagei_parser = subparsers.add_parser(
        "run-papagei-benchmark",
        help="Extract frozen PaPaGei-S embeddings, train Ridge scalar heads, and evaluate scalar BP metrics.",
    )
    run_papagei_parser.add_argument("--run-dir", required=True)
    run_papagei_parser.add_argument("--upstream-repo", required=True)
    run_papagei_parser.add_argument("--weights-path", required=True)
    run_papagei_parser.add_argument("--batch-size", type=int, default=256)
    run_papagei_parser.add_argument("--device", default="cuda")
    run_papagei_parser.add_argument("--output-idx", type=int, default=0)
    run_papagei_parser.add_argument("--include-val-in-fit", action=argparse.BooleanOptionalAction, default=True)
    run_papagei_parser.add_argument("--n-jobs", type=int, default=-1)
    run_papagei_parser.add_argument("--seed", type=int, default=2025)

    evaluate_papagei_parser = subparsers.add_parser(
        "evaluate-papagei-run",
        help="Evaluate an already-completed PaPaGei scalar benchmark run.",
    )
    evaluate_papagei_parser.add_argument("--run-dir", required=True)

    prepare_unicardio_parser = subparsers.add_parser(
        "prepare-unicardio-run",
        help="Build a UniCardio PPG->ECG->ABP run package from one or more exported baseline views.",
    )
    prepare_unicardio_parser.add_argument("--export-manifest", action="append", required=True)
    prepare_unicardio_parser.add_argument("--output-dir", required=True)
    prepare_unicardio_parser.add_argument("--window-sec", type=float, default=None)
    prepare_unicardio_parser.add_argument("--step-sec", type=float, default=None)
    prepare_unicardio_parser.add_argument("--slot-length", type=int, default=None)

    train_unicardio_parser = subparsers.add_parser(
        "train-unicardio-benchmark",
        help="Fine-tune UniCardio's official ECG->BP branch against a local upstream repo clone.",
    )
    train_unicardio_parser.add_argument("--run-dir", required=True)
    train_unicardio_parser.add_argument("--upstream-repo", required=True)
    train_unicardio_parser.add_argument("--config-path", default=None)
    train_unicardio_parser.add_argument("--pretrained-checkpoint", default=None)
    train_unicardio_parser.add_argument("--epochs", type=int, default=200)
    train_unicardio_parser.add_argument("--batch-size", type=int, default=128)
    train_unicardio_parser.add_argument("--lr", type=float, default=None)
    train_unicardio_parser.add_argument("--device", default="cuda")
    train_unicardio_parser.add_argument("--seed", type=int, default=2025)

    generate_unicardio_parser = subparsers.add_parser(
        "generate-unicardio-run",
        help="Generate UniCardio ABP waveforms with the official PPG->ECG->BP chain.",
    )
    generate_unicardio_parser.add_argument("--run-dir", required=True)
    generate_unicardio_parser.add_argument("--upstream-repo", required=True)
    generate_unicardio_parser.add_argument("--checkpoint-path", default=None)
    generate_unicardio_parser.add_argument("--base-checkpoint-path", default=None)
    generate_unicardio_parser.add_argument("--config-path", default=None)
    generate_unicardio_parser.add_argument("--batch-size", type=int, default=64)
    generate_unicardio_parser.add_argument("--n-samples", type=int, default=50)
    generate_unicardio_parser.add_argument("--aggregation", choices=["mean", "median"], default="mean")
    generate_unicardio_parser.add_argument("--device", default="cuda")
    generate_unicardio_parser.add_argument("--seed", type=int, default=2025)
    generate_unicardio_parser.add_argument("--window-start", type=int, default=0)
    generate_unicardio_parser.add_argument("--limit-windows", type=int, default=None)
    generate_unicardio_parser.add_argument(
        "--inference-only",
        action="store_true",
        help="Use the official base checkpoint for both PPG->ECG and ECG->ABP; do not require a fine-tuned checkpoint.",
    )

    evaluate_unicardio_parser = subparsers.add_parser(
        "evaluate-unicardio-run",
        help="Evaluate an already-completed UniCardio PPG->ECG->ABP benchmark run.",
    )
    evaluate_unicardio_parser.add_argument("--run-dir", required=True)

    run_unicardio_parser = subparsers.add_parser(
        "run-unicardio-benchmark",
        help="Fine-tune, generate, and evaluate the UniCardio PPG->ECG->ABP benchmark launcher.",
    )
    run_unicardio_parser.add_argument("--run-dir", required=True)
    run_unicardio_parser.add_argument("--upstream-repo", required=True)
    run_unicardio_parser.add_argument("--config-path", default=None)
    run_unicardio_parser.add_argument("--pretrained-checkpoint", default=None)
    run_unicardio_parser.add_argument("--epochs", type=int, default=200)
    run_unicardio_parser.add_argument("--train-batch-size", type=int, default=128)
    run_unicardio_parser.add_argument("--generation-batch-size", type=int, default=64)
    run_unicardio_parser.add_argument("--lr", type=float, default=None)
    run_unicardio_parser.add_argument("--n-samples", type=int, default=50)
    run_unicardio_parser.add_argument("--aggregation", choices=["mean", "median"], default="mean")
    run_unicardio_parser.add_argument("--device", default="cuda")
    run_unicardio_parser.add_argument("--seed", type=int, default=2025)
    run_unicardio_parser.add_argument("--window-start", type=int, default=0)
    run_unicardio_parser.add_argument("--limit-windows", type=int, default=None)

    run_unicardio_inference_parser = subparsers.add_parser(
        "run-unicardio-inference-benchmark",
        help="Generate and evaluate UniCardio with the official pretrained checkpoint only; no fine-tuning.",
    )
    run_unicardio_inference_parser.add_argument("--run-dir", required=True)
    run_unicardio_inference_parser.add_argument("--upstream-repo", required=True)
    run_unicardio_inference_parser.add_argument("--config-path", default=None)
    run_unicardio_inference_parser.add_argument("--pretrained-checkpoint", default=None)
    run_unicardio_inference_parser.add_argument("--generation-batch-size", type=int, default=64)
    run_unicardio_inference_parser.add_argument("--n-samples", type=int, default=50)
    run_unicardio_inference_parser.add_argument("--aggregation", choices=["mean", "median"], default="mean")
    run_unicardio_inference_parser.add_argument("--device", default="cuda")
    run_unicardio_inference_parser.add_argument("--seed", type=int, default=2025)
    run_unicardio_inference_parser.add_argument("--window-start", type=int, default=0)
    run_unicardio_inference_parser.add_argument("--limit-windows", type=int, default=None)

    table_row_parser = subparsers.add_parser(
        "metrics-table-row",
        help="Convert one metrics JSON file into the compact benchmark table row fields.",
    )
    table_row_parser.add_argument("--metrics-json", required=True)
    table_row_parser.add_argument("--method", default=None)
    table_row_parser.add_argument("--output-csv", default=None)
    table_row_parser.add_argument("--overwrite", action="store_true")

    patient_metrics_parser = subparsers.add_parser(
        "patient-metrics",
        help="Recompute benchmark metrics by patient, including fixed-shift Min-MAE/Min-RMSE and NRMSE.",
    )
    patient_metrics_parser.add_argument("--run-dir", required=True)
    patient_metrics_parser.add_argument(
        "--method",
        choices=["auto", "inn_par", "ppg2abp", "unicardio", "ptt_based", "papagei", "operator"],
        default="auto",
    )
    patient_metrics_parser.add_argument("--label", default=None, help="Display name for the compact CSV row.")
    patient_metrics_parser.add_argument("--output-json", default=None)
    patient_metrics_parser.add_argument("--output-csv", default=None)
    patient_metrics_parser.add_argument("--overwrite-csv", action="store_true")
    patient_metrics_parser.add_argument("--shift-ms", type=float, default=250.0)
    patient_metrics_parser.add_argument("--bootstrap-samples", type=int, default=2000)
    patient_metrics_parser.add_argument("--seed", type=int, default=2025, help="Bootstrap RNG seed, not model seed.")
    patient_metrics_parser.add_argument("--chunk-size", type=int, default=2048)
    patient_metrics_parser.add_argument("--progress-every-chunks", type=int, default=25)
    patient_metrics_parser.add_argument("--compute-scalar", action=argparse.BooleanOptionalAction, default=True)
    patient_metrics_parser.add_argument(
        "--reference-qc",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply pre-registered physiologic QC based only on reference ABP/scalar truth.",
    )
    patient_metrics_parser.add_argument("--qc-min-abp", type=float, default=20.0)
    patient_metrics_parser.add_argument("--qc-max-abp", type=float, default=250.0)
    patient_metrics_parser.add_argument("--qc-min-std", type=float, default=1.0)
    patient_metrics_parser.add_argument("--qc-min-range", type=float, default=10.0)
    patient_metrics_parser.add_argument("--qc-max-range", type=float, default=200.0)

    aggregate_patient_metrics_parser = subparsers.add_parser(
        "aggregate-patient-metric-seeds",
        help="Aggregate patient-metrics JSON files from multiple completed seeds.",
    )
    aggregate_patient_metrics_parser.add_argument("--metrics-json", action="append", required=True)
    aggregate_patient_metrics_parser.add_argument(
        "--seed-label",
        action="append",
        default=None,
        help="Optional seed label. Repeat once per --metrics-json, in the same order.",
    )
    aggregate_patient_metrics_parser.add_argument("--method-label", default=None)
    aggregate_patient_metrics_parser.add_argument("--output-json", default=None)
    aggregate_patient_metrics_parser.add_argument("--output-csv", default=None)
    aggregate_patient_metrics_parser.add_argument("--overwrite-csv", action="store_true")

    args = parser.parse_args()

    if args.command == "inspect-manifest":
        summary = inspect_manifest(args.manifest, limit=args.limit)
        print(manifest_summary_json(summary))
        return 0

    if args.command == "build-unified-manifest":
        summary = build_unified_manifest(
            args.manifest,
            args.output,
            split=args.split,
            source=args.source,
            limit=args.limit,
        )
        print(manifest_summary_json(summary))
        return 0

    if args.command == "validate-unified-manifest":
        summary = validate_unified_manifest(args.manifest, limit=args.limit)
        print(manifest_summary_json(summary))
        return 0

    if args.command == "split-unified-manifest":
        summary = split_unified_manifest(
            args.manifest,
            args.train_output,
            args.val_output,
            val_fraction=args.val_fraction,
            seed=args.seed,
            group_column=args.group_column,
        )
        print(manifest_summary_json(summary))
        return 0

    if args.command == "export-baseline-view":
        summary = export_baseline_view(
            args.manifest,
            args.baseline,
            args.output_dir,
            limit=args.limit,
        )
        print(manifest_summary_json(summary))
        return 0

    if args.command == "run-ptt-benchmark":
        summary = run_ptt_benchmark(
            train_manifest=args.train_manifest,
            val_manifest=args.val_manifest,
            test_manifest=args.test_manifest,
            output_dir=args.output_dir,
            include_val_in_fit=args.include_val_in_fit,
            feature_transform=args.feature_transform,
            min_ptt_sec=args.min_ptt_sec,
            max_ptt_sec=args.max_ptt_sec,
            min_peak_distance_sec=args.min_peak_distance_sec,
            ecg_prominence_scale=args.ecg_prominence_scale,
            ppg_prominence_scale=args.ppg_prominence_scale,
            min_finite_segment_sec=args.min_finite_segment_sec,
            limit_records=args.limit_records,
        )
        print(manifest_summary_json(summary))
        return 0

    if args.command == "prepare-ppg2abp-run":
        kwargs = {}
        if args.target_fs_hz is not None:
            kwargs["target_fs_hz"] = args.target_fs_hz
        if args.window_sec is not None:
            kwargs["window_sec"] = args.window_sec
        if args.step_sec is not None:
            kwargs["step_sec"] = args.step_sec
        if args.model_input_length is not None:
            kwargs["model_input_length"] = args.model_input_length
        summary = prepare_ppg2abp_run(
            args.export_manifest,
            args.output_dir,
            **kwargs,
        )
        print(manifest_summary_json(summary))
        return 0

    if args.command == "run-ppg2abp-benchmark":
        summary = run_ppg2abp_benchmark(
            args.run_dir,
            args.upstream_repo,
            epochs_stage1=args.epochs_stage1,
            epochs_stage2=args.epochs_stage2,
            batch_size_stage1=args.batch_size_stage1,
            batch_size_stage2=args.batch_size_stage2,
        )
        print(manifest_summary_json(summary))
        return 0

    if args.command == "evaluate-ppg2abp-run":
        summary = evaluate_ppg2abp_run(args.run_dir)
        print(manifest_summary_json(summary))
        return 0

    if args.command == "prepare-inn-par-run":
        kwargs = {}
        if args.target_fs_hz is not None:
            kwargs["target_fs_hz"] = args.target_fs_hz
        if args.window_sec is not None:
            kwargs["window_sec"] = args.window_sec
        if args.step_sec is not None:
            kwargs["step_sec"] = args.step_sec
        if args.signal_length is not None:
            kwargs["signal_length"] = args.signal_length
        summary = prepare_inn_par_run(
            args.export_manifest,
            args.output_dir,
            **kwargs,
        )
        print(manifest_summary_json(summary))
        return 0

    if args.command == "train-inn-par-benchmark":
        summary = train_inn_par_benchmark(
            args.run_dir,
            args.upstream_repo,
            max_step=args.max_step,
            batch_size=args.batch_size,
            lr=args.lr,
            num_workers=args.num_workers,
            checkpoint_interval=args.checkpoint_interval,
            pin_memory=args.pin_memory,
            device=args.device,
            seed=args.seed,
        )
        print(manifest_summary_json(summary))
        return 0

    if args.command == "generate-inn-par-run":
        summary = generate_inn_par_run(
            args.run_dir,
            args.upstream_repo,
            checkpoint_path=args.checkpoint_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            device=args.device,
            seed=args.seed,
        )
        print(manifest_summary_json(summary))
        return 0

    if args.command == "evaluate-inn-par-run":
        summary = evaluate_inn_par_run(args.run_dir)
        print(manifest_summary_json(summary))
        return 0

    if args.command == "run-inn-par-benchmark":
        summary = run_inn_par_benchmark(
            args.run_dir,
            args.upstream_repo,
            max_step=args.max_step,
            train_batch_size=args.train_batch_size,
            test_batch_size=args.test_batch_size,
            lr=args.lr,
            num_workers=args.num_workers,
            checkpoint_interval=args.checkpoint_interval,
            device=args.device,
            pin_memory=args.pin_memory,
            seed=args.seed,
        )
        print(manifest_summary_json(summary))
        return 0

    if args.command == "prepare-papagei-run":
        kwargs = {}
        if args.target_fs_hz is not None:
            kwargs["target_fs_hz"] = args.target_fs_hz
        if args.segment_sec is not None:
            kwargs["segment_sec"] = args.segment_sec
        if args.step_sec is not None:
            kwargs["step_sec"] = args.step_sec
        if args.segment_length is not None:
            kwargs["segment_length"] = args.segment_length
        kwargs["drop_zero_segment_records"] = args.drop_zero_segment_records
        summary = prepare_papagei_run(
            args.export_manifest,
            args.output_dir,
            args.upstream_repo,
            **kwargs,
        )
        print(manifest_summary_json(summary))
        return 0

    if args.command == "run-papagei-benchmark":
        summary = run_papagei_benchmark(
            args.run_dir,
            args.upstream_repo,
            args.weights_path,
            batch_size=args.batch_size,
            device=args.device,
            output_idx=args.output_idx,
            include_val_in_fit=args.include_val_in_fit,
            n_jobs=args.n_jobs,
            seed=args.seed,
        )
        print(manifest_summary_json(summary))
        return 0

    if args.command == "evaluate-papagei-run":
        summary = evaluate_papagei_run(args.run_dir)
        print(manifest_summary_json(summary))
        return 0

    if args.command == "prepare-unicardio-run":
        kwargs = {}
        if args.window_sec is not None:
            kwargs["window_sec"] = args.window_sec
        if args.step_sec is not None:
            kwargs["step_sec"] = args.step_sec
        if args.slot_length is not None:
            kwargs["slot_length"] = args.slot_length
        summary = prepare_unicardio_run(
            args.export_manifest,
            args.output_dir,
            **kwargs,
        )
        print(manifest_summary_json(summary))
        return 0

    if args.command == "train-unicardio-benchmark":
        summary = train_unicardio_benchmark(
            args.run_dir,
            args.upstream_repo,
            config_path=args.config_path,
            pretrained_checkpoint=args.pretrained_checkpoint,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            seed=args.seed,
        )
        print(manifest_summary_json(summary))
        return 0

    if args.command == "generate-unicardio-run":
        summary = generate_unicardio_run(
            args.run_dir,
            args.upstream_repo,
            checkpoint_path=args.checkpoint_path,
            base_checkpoint_path=args.base_checkpoint_path,
            config_path=args.config_path,
            batch_size=args.batch_size,
            n_samples=args.n_samples,
            device=args.device,
            seed=args.seed,
            aggregation=args.aggregation,
            inference_only=args.inference_only,
            window_start=args.window_start,
            limit_windows=args.limit_windows,
        )
        print(manifest_summary_json(summary))
        return 0

    if args.command == "evaluate-unicardio-run":
        summary = evaluate_unicardio_run(args.run_dir)
        print(manifest_summary_json(summary))
        return 0

    if args.command == "run-unicardio-benchmark":
        summary = run_unicardio_benchmark(
            args.run_dir,
            args.upstream_repo,
            config_path=args.config_path,
            pretrained_checkpoint=args.pretrained_checkpoint,
            epochs=args.epochs,
            train_batch_size=args.train_batch_size,
            generation_batch_size=args.generation_batch_size,
            lr=args.lr,
            n_samples=args.n_samples,
            device=args.device,
            seed=args.seed,
            aggregation=args.aggregation,
            window_start=args.window_start,
            limit_windows=args.limit_windows,
        )
        print(manifest_summary_json(summary))
        return 0

    if args.command == "run-unicardio-inference-benchmark":
        summary = run_unicardio_inference_benchmark(
            args.run_dir,
            args.upstream_repo,
            config_path=args.config_path,
            pretrained_checkpoint=args.pretrained_checkpoint,
            generation_batch_size=args.generation_batch_size,
            n_samples=args.n_samples,
            device=args.device,
            seed=args.seed,
            aggregation=args.aggregation,
            window_start=args.window_start,
            limit_windows=args.limit_windows,
        )
        print(manifest_summary_json(summary))
        return 0

    if args.command == "metrics-table-row":
        row = metrics_json_to_table_row(args.metrics_json, method=args.method)
        if args.output_csv is None:
            print(manifest_summary_json({"row": row}))
            return 0
        summary = write_table_row(row, args.output_csv, append=not args.overwrite)
        print(manifest_summary_json(summary))
        return 0

    if args.command == "patient-metrics":
        summary = compute_patient_metrics(
            args.run_dir,
            method=args.method,
            method_label=args.label,
            output_json=args.output_json,
            output_csv=args.output_csv,
            overwrite_csv=args.overwrite_csv,
            shift_ms=args.shift_ms,
            n_bootstrap=args.bootstrap_samples,
            seed=args.seed,
            chunk_size=args.chunk_size,
            compute_scalar=args.compute_scalar,
            progress_every_chunks=args.progress_every_chunks,
            reference_qc=args.reference_qc,
            qc_min_abp=args.qc_min_abp,
            qc_max_abp=args.qc_max_abp,
            qc_min_std=args.qc_min_std,
            qc_min_range=args.qc_min_range,
            qc_max_range=args.qc_max_range,
        )
        print(manifest_summary_json(summary))
        return 0

    if args.command == "aggregate-patient-metric-seeds":
        summary = aggregate_patient_metric_seeds(
            args.metrics_json,
            seed_labels=args.seed_label,
            method_label=args.method_label,
            output_json=args.output_json,
            output_csv=args.output_csv,
            overwrite_csv=args.overwrite_csv,
        )
        print(manifest_summary_json(summary))
        return 0

    raise RuntimeError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
