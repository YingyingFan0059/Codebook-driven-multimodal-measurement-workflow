from __future__ import annotations

import argparse

from cbma import __version__
from cbma.commands import (
    run_baseline_run,
    run_doctor,
    run_eval_run,
    run_init,
    run_report_build_command,
    run_split_create,
    run_train_recommend_n,
    run_train_sweep,
    run_ui_serve,
    run_validate,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cbma",
        description="Workflow-first CLI for codebook-driven multimodal analysis.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Initialize a local CBMA project.")
    init_parser.add_argument("project_path", help="Directory to initialize.")
    init_parser.add_argument("--force", action="store_true", help="Overwrite template files if they exist.")

    doctor_parser = subparsers.add_parser("doctor", help="Check local environment and project readiness.")
    doctor_parser.add_argument("--project", dest="project_path", help="Project directory to inspect.")
    doctor_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON output.")

    validate_parser = subparsers.add_parser("validate", help="Validate project config, codebook, and labels.")
    validate_parser.add_argument("--project", dest="project_path", required=True, help="Project directory to validate.")
    validate_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON output.")

    split_parser = subparsers.add_parser("split", help="Create data split artifacts.")
    split_subparsers = split_parser.add_subparsers(dest="split_command", required=True)
    split_create_parser = split_subparsers.add_parser("create", help="Create train/val/test split artifacts.")
    split_create_parser.add_argument("--project", dest="project_path", required=True, help="Project directory.")
    split_create_parser.add_argument("--force", action="store_true", help="Overwrite existing split artifacts.")
    split_create_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON output.")

    baseline_parser = subparsers.add_parser("baseline", help="Run baseline methods.")
    baseline_subparsers = baseline_parser.add_subparsers(dest="baseline_command", required=True)
    baseline_run_parser = baseline_subparsers.add_parser("run", help="Run qwen2 baseline evaluation.")
    baseline_run_parser.add_argument("--project", dest="project_path", required=True, help="Project directory.")
    baseline_run_parser.add_argument(
        "--methods",
        help="Comma-separated baseline methods. Defaults to project config.",
    )
    baseline_run_parser.add_argument("--model", dest="model_name", help="Override the model alias from project config.")
    baseline_run_parser.add_argument("--model-path", dest="model_path", help="Override the resolved local model path.")
    baseline_run_parser.add_argument("--dry-run", action="store_true", help="Resolve inputs without loading the model.")
    baseline_run_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON output.")

    eval_parser = subparsers.add_parser("eval", help="Run evaluation workflows.")
    eval_subparsers = eval_parser.add_subparsers(dest="eval_command", required=True)
    eval_run_parser = eval_subparsers.add_parser("run", help="Run standardized evaluation on the fixed test split.")
    eval_run_parser.add_argument("--project", dest="project_path", required=True, help="Project directory.")
    eval_run_parser.add_argument("--run-dir", dest="run_dir", help="Train sweep run directory.")
    eval_run_parser.add_argument("--recommend-file", dest="recommend_file", help="Path to recommend_n.json.")
    eval_run_parser.add_argument("--output", dest="output", help="Output directory for evaluation results.")

    report_parser = subparsers.add_parser("report", help="Build report artifacts from existing eval outputs.")
    report_subparsers = report_parser.add_subparsers(dest="report_command", required=True)
    report_build_parser = report_subparsers.add_parser("build", help="Build a standardized report from an eval run.")
    report_build_parser.add_argument("--project", dest="project_path", required=True, help="Project directory.")
    report_build_parser.add_argument("--eval-dir", dest="eval_dir", help="Evaluation run directory.")
    report_build_parser.add_argument("--output", dest="output", help="Output directory for report artifacts.")
    report_build_parser.add_argument("--format", dest="format_name", default="markdown", help="Report format. Only markdown is supported.")
    report_build_parser.add_argument("--include-errors", action="store_true", help="Include error-case export when data is available.")
    report_build_parser.add_argument("--include-confusion", action="store_true", help="Include confusion export when data is available.")

    train_parser = subparsers.add_parser("train", help="Run training workflows.")
    train_subparsers = train_parser.add_subparsers(dest="train_command", required=True)
    train_sweep_parser = train_subparsers.add_parser("sweep", help="Run or plan the Auto-LoRA size sweep.")
    train_sweep_parser.add_argument("--project", dest="project_path", required=True, help="Project directory.")
    train_sweep_parser.add_argument(
        "--sizes",
        help="Comma-separated training sizes. Defaults to defaults.training.candidate_n.",
    )
    train_sweep_parser.add_argument("--model", dest="model_name", help="Override the model alias from project config.")
    train_sweep_parser.add_argument("--model-path", dest="model_path", help="Override the resolved local model path.")
    train_sweep_parser.add_argument("--dry-run", action="store_true", help="Resolve inputs without launching training.")
    train_sweep_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON output.")
    train_recommend_parser = train_subparsers.add_parser("recommend-n", help="Generate recommend_n.json from a train sweep run.")
    train_recommend_parser.add_argument("--project", dest="project_path", required=True, help="Project directory.")
    train_recommend_parser.add_argument("--run-dir", dest="run_dir", help="Train sweep run directory.")
    train_recommend_parser.add_argument("--output", dest="output", help="Output path for recommend_n.json.")

    ui_parser = subparsers.add_parser("ui", help="Serve the local web UI.")
    ui_subparsers = ui_parser.add_subparsers(dest="ui_command", required=True)
    ui_serve_parser = ui_subparsers.add_parser("serve", help="Start the local web UI.")
    ui_serve_parser.add_argument("--project", dest="project_path", required=True, help="Project directory.")
    ui_serve_parser.add_argument("--host", default="127.0.0.1", help="Host to bind the local UI server.")
    ui_serve_parser.add_argument("--port", type=int, default=7860, help="Port to bind the local UI server.")
    ui_serve_parser.add_argument("--reload", action="store_true", help="Enable auto-reload for UI development.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "init":
        return run_init(project_path=args.project_path, force=args.force)
    if args.command == "doctor":
        return run_doctor(project_path=args.project_path, json_output=args.json)
    if args.command == "validate":
        return run_validate(project_path=args.project_path, json_output=args.json)
    if args.command == "split" and args.split_command == "create":
        return run_split_create(
            project_path=args.project_path,
            force=args.force,
            json_output=args.json,
        )
    if args.command == "baseline" and args.baseline_command == "run":
        methods = [part.strip() for part in args.methods.split(",")] if args.methods else None
        return run_baseline_run(
            project_path=args.project_path,
            methods=methods,
            dry_run=args.dry_run,
            json_output=args.json,
            model_name=args.model_name,
            model_path_override=args.model_path,
        )
    if args.command == "eval" and args.eval_command == "run":
        return run_eval_run(
            project_path=args.project_path,
            run_dir=args.run_dir,
            recommend_file=args.recommend_file,
            output=args.output,
        )
    if args.command == "report" and args.report_command == "build":
        return run_report_build_command(
            project_path=args.project_path,
            eval_dir=args.eval_dir,
            output=args.output,
            format_name=args.format_name,
            include_errors=args.include_errors,
            include_confusion=args.include_confusion,
        )
    if args.command == "train" and args.train_command == "sweep":
        sizes = [int(part.strip()) for part in args.sizes.split(",")] if args.sizes else None
        return run_train_sweep(
            project_path=args.project_path,
            sizes=sizes,
            dry_run=args.dry_run,
            json_output=args.json,
            model_name=args.model_name,
            model_path_override=args.model_path,
        )
    if args.command == "train" and args.train_command == "recommend-n":
        return run_train_recommend_n(
            project_path=args.project_path,
            run_dir=args.run_dir,
            output=args.output,
        )
    if args.command == "ui" and args.ui_command == "serve":
        return run_ui_serve(
            project_path=args.project_path,
            host=args.host,
            port=args.port,
            reload=args.reload,
        )

    parser.error(f"Unknown command: {args.command}")
    return 2
