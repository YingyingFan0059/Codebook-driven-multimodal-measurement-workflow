from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


TEST_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = TEST_ROOT / "src"
DEMO_PROJECT_ROOT = TEST_ROOT.parent / "demo" / "demo_project"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cbma.cli import main  # noqa: E402


class CliSmokeTests(unittest.TestCase):
    def _populate_demo_labels(self, project_dir: Path) -> None:
        videos_dir = project_dir / "data" / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)
        rows = ["video_id,video_path,label,split"]
        labels = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
        for index, label in enumerate(labels, start=1):
            video_name = f"{index:04d}.mp4"
            (videos_dir / video_name).write_bytes(b"")
            rows.append(f"{index:04d},{video_name},{label},")
        (project_dir / "data" / "labels.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    def _copy_release_demo_project(self, project_dir: Path) -> None:
        shutil.copytree(DEMO_PROJECT_ROOT, project_dir)

    def test_init_creates_expected_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir) / "demo_project"
            exit_code = main(["init", str(project_dir)])
            self.assertEqual(exit_code, 0)
            self.assertTrue((project_dir / "project.yaml").exists())
            self.assertTrue((project_dir / "data" / "codebook.yaml").exists())
            self.assertTrue((project_dir / "data" / "labels.csv").exists())

    def test_validate_passes_on_fresh_template_project(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir) / "demo_project"
            main(["init", str(project_dir)])
            exit_code = main(["validate", "--project", str(project_dir)])
            self.assertEqual(exit_code, 0)

    def test_split_create_generates_expected_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir) / "demo_project"
            main(["init", str(project_dir)])
            self._populate_demo_labels(project_dir)
            exit_code = main(["split", "create", "--project", str(project_dir)])
            self.assertEqual(exit_code, 0)
            self.assertTrue((project_dir / "splits" / "split_v1" / "test_main.csv").exists())
            self.assertTrue((project_dir / "splits" / "split_v1" / "train_pool.csv").exists())
            self.assertTrue((project_dir / "splits" / "split_v1" / "split_summary.json").exists())

    def test_baseline_run_dry_run_resolves_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir) / "demo_project"
            main(["init", str(project_dir)])
            self._populate_demo_labels(project_dir)
            main(["split", "create", "--project", str(project_dir)])
            exit_code = main(["baseline", "run", "--project", str(project_dir), "--dry-run"])
            self.assertEqual(exit_code, 0)

    def test_train_sweep_dry_run_resolves_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir) / "demo_project"
            main(["init", str(project_dir)])
            self._populate_demo_labels(project_dir)
            main(["split", "create", "--project", str(project_dir)])
            exit_code = main(["train", "sweep", "--project", str(project_dir), "--dry-run"])
            self.assertEqual(exit_code, 0)

    def test_train_recommend_n_generates_file_and_eval_consumes_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir) / "demo_project"
            self._copy_release_demo_project(project_dir)
            main(["split", "create", "--project", str(project_dir), "--force"])

            train_run_dir = project_dir / "runs" / "train-sweep-sample"
            recommend_path = train_run_dir / "recommend_n.json"
            if recommend_path.exists():
                recommend_path.unlink()

            artifact_root = train_run_dir / "artifacts" / "qwen2"
            lora_dir = artifact_root / "runs" / "scaling_experiments" / "lora_5class_5"
            model_dir = project_dir / "models" / "qwen2" / "Qwen2-VL-7B-Instruct"
            lora_dir.mkdir(parents=True, exist_ok=True)
            model_dir.mkdir(parents=True, exist_ok=True)

            train_sweep_payload = json.loads((train_run_dir / "train_sweep.json").read_text(encoding="utf-8"))
            train_sweep_payload.update(
                {
                    "project": str(project_dir),
                    "run_dir": str(train_run_dir),
                    "artifact_root": str(artifact_root),
                    "split_dir": str(project_dir / "splits" / "split_v1"),
                    "model_path": str(model_dir),
                }
            )
            (train_run_dir / "train_sweep.json").write_text(
                json.dumps(train_sweep_payload, indent=2) + "\n",
                encoding="utf-8",
            )

            exit_code = main(
                [
                    "train",
                    "recommend-n",
                    "--project",
                    str(project_dir),
                    "--run-dir",
                    str(train_run_dir),
                ]
            )
            self.assertEqual(exit_code, 0)
            self.assertTrue(recommend_path.exists())

            recommend_payload = json.loads(recommend_path.read_text(encoding="utf-8"))
            self.assertEqual(recommend_payload["recommended_n"], 5)
            self.assertEqual(recommend_payload["metric"], "val_macro_f1")
            self.assertIn("selection_log", recommend_payload)

            fake_eval_script = Path(tmp_dir) / "fake_eval.py"
            fake_eval_script.write_text(
                (
                    "import os\n"
                    "from pathlib import Path\n\n"
                    "def evaluate_model(train_size, test_csv_name='test_main.csv'):\n"
                    "    out_dir = Path(os.environ['QWEN2_EVAL_OUT_DIR'])\n"
                    "    out_dir.mkdir(parents=True, exist_ok=True)\n"
                    "    report = out_dir / f'eval_lora_5class_{train_size}_report.txt'\n"
                    "    report.write_text(\n"
                    "        'Accuracy: 0.7800\\nMacro-F1: 0.7400\\n',\n"
                    "        encoding='utf-8',\n"
                    "    )\n"
                ),
                encoding="utf-8",
            )

            eval_output_dir = project_dir / "runs" / "eval-generated"
            with mock.patch("cbma.eval.run._resolve_eval_script", return_value=fake_eval_script):
                exit_code = main(
                    [
                        "eval",
                        "run",
                        "--project",
                        str(project_dir),
                        "--run-dir",
                        str(train_run_dir),
                        "--output",
                        str(eval_output_dir),
                    ]
                )

            self.assertEqual(exit_code, 0)
            eval_result = json.loads((eval_output_dir / "eval_result.json").read_text(encoding="utf-8"))
            self.assertEqual(eval_result["recommended_n"], 5)

    def test_doctor_runs_on_initialized_project(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir) / "demo_project"
            main(["init", str(project_dir)])
            exit_code = main(["doctor", "--project", str(project_dir)])
            self.assertEqual(exit_code, 0)

    def test_eval_run_uses_recommendation_and_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir) / "demo_project"
            main(["init", str(project_dir)])
            self._populate_demo_labels(project_dir)
            main(["split", "create", "--project", str(project_dir)])

            train_run_dir = project_dir / "runs" / "train-sweep-20260411-120000"
            artifact_root = train_run_dir / "artifacts" / "qwen2"
            lora_dir = artifact_root / "runs" / "scaling_experiments" / "lora_5class_5"
            model_dir = project_dir / "models" / "qwen2" / "Qwen2-VL-7B-Instruct"
            lora_dir.mkdir(parents=True, exist_ok=True)
            model_dir.mkdir(parents=True, exist_ok=True)

            (train_run_dir / "train_sweep.json").write_text(
                json.dumps(
                    {
                        "project": str(project_dir),
                        "run_dir": str(train_run_dir),
                        "artifact_root": str(artifact_root),
                        "split_dir": str(project_dir / "splits" / "split_v1"),
                        "sizes": [5],
                        "model_path": str(model_dir),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            (train_run_dir / "recommend_n.json").write_text(
                '{\n  "recommended_n": 5,\n  "metric": "val_macro_f1"\n}\n',
                encoding="utf-8",
            )

            fake_eval_script = Path(tmp_dir) / "fake_eval.py"
            fake_eval_script.write_text(
                (
                    "import os\n"
                    "from pathlib import Path\n\n"
                    "def evaluate_model(train_size, test_csv_name='test_main.csv'):\n"
                    "    out_dir = Path(os.environ['QWEN2_EVAL_OUT_DIR'])\n"
                    "    out_dir.mkdir(parents=True, exist_ok=True)\n"
                    "    report = out_dir / f'eval_lora_5class_{train_size}_report.txt'\n"
                    "    report.write_text(\n"
                    "        'Accuracy: 0.7800\\nMacro-F1: 0.7400\\n',\n"
                    "        encoding='utf-8',\n"
                    "    )\n"
                ),
                encoding="utf-8",
            )

            with mock.patch("cbma.eval.run._resolve_eval_script", return_value=fake_eval_script):
                exit_code = main(["eval", "run", "--project", str(project_dir), "--run-dir", str(train_run_dir)])

            self.assertEqual(exit_code, 0)

            eval_runs = sorted((project_dir / "runs").glob("eval-*"))
            self.assertTrue(eval_runs)
            eval_result = json.loads((eval_runs[-1] / "eval_result.json").read_text(encoding="utf-8"))
            eval_metadata = json.loads((eval_runs[-1] / "eval_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(eval_result["recommended_n"], 5)
            self.assertEqual(eval_result["test_metric"]["macro_f1"], 0.74)
            self.assertEqual(eval_result["test_metric"]["accuracy"], 0.78)
            self.assertEqual(eval_metadata["source_run_dir"], str(train_run_dir))

    def test_eval_run_fails_without_recommend_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir) / "demo_project"
            main(["init", str(project_dir)])
            train_run_dir = project_dir / "runs" / "train-sweep-20260411-120000"
            train_run_dir.mkdir(parents=True, exist_ok=True)
            exit_code = main(["eval", "run", "--project", str(project_dir), "--run-dir", str(train_run_dir)])
            self.assertEqual(exit_code, 1)

    def test_report_build_succeeds_with_minimal_eval_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir) / "demo_project"
            main(["init", str(project_dir)])

            eval_dir = project_dir / "runs" / "eval-20260411-130000"
            eval_dir.mkdir(parents=True, exist_ok=True)
            (eval_dir / "eval_result.json").write_text(
                json.dumps(
                    {
                        "recommended_n": 16,
                        "test_metric": {"macro_f1": 0.74, "accuracy": 0.78},
                        "source_run": str(project_dir / "runs" / "train-sweep-20260411-120000"),
                        "test_split": "test_main.csv",
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            (eval_dir / "eval_metadata.json").write_text(
                json.dumps(
                    {
                        "project": str(project_dir),
                        "source_run_dir": str(project_dir / "runs" / "train-sweep-20260411-120000"),
                        "test_split_path": str(project_dir / "splits" / "split_v1" / "test_main.csv"),
                        "model_path": str(project_dir / "models" / "qwen2" / "Qwen2-VL-7B-Instruct"),
                        "output_dir": str(eval_dir),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            exit_code = main(["report", "build", "--project", str(project_dir)])
            self.assertEqual(exit_code, 0)

            report_dir = eval_dir / "report"
            self.assertTrue((report_dir / "report.md").exists())
            self.assertTrue((report_dir / "metrics.json").exists())
            self.assertTrue((report_dir / "run_summary.json").exists())

            report_text = (report_dir / "report.md").read_text(encoding="utf-8")
            self.assertIn("CBMA Evaluation Report", report_text)
            self.assertIn("unavailable", report_text)

    def test_report_build_parses_predictions_and_generates_optional_exports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir) / "demo_project"
            main(["init", str(project_dir)])

            eval_dir = project_dir / "runs" / "eval-20260411-130000"
            raw_eval_dir = eval_dir / "raw_eval"
            raw_eval_dir.mkdir(parents=True, exist_ok=True)

            (eval_dir / "eval_result.json").write_text(
                json.dumps(
                    {
                        "recommended_n": 16,
                        "test_metric": {"macro_f1": 0.74, "accuracy": 0.78},
                        "source_run": str(project_dir / "runs" / "train-sweep-20260411-120000"),
                        "test_split": "test_main.csv",
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            (eval_dir / "eval_metadata.json").write_text(
                json.dumps(
                    {
                        "project": str(project_dir),
                        "source_run_dir": str(project_dir / "runs" / "train-sweep-20260411-120000"),
                        "test_split_path": str(project_dir / "splits" / "split_v1" / "test_main.csv"),
                        "model_path": str(project_dir / "models" / "qwen2" / "Qwen2-VL-7B-Instruct"),
                        "output_dir": str(eval_dir),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            (raw_eval_dir / "eval_lora_5class_16_predictions.csv").write_text(
                "\n".join(
                    [
                        "video_id,gold_label,pred_label,raw_output",
                        "0001,0,0,ok",
                        "0002,1,0,wrong",
                        "0003,1,1,ok",
                        "0004,2,2,ok",
                    ]
                )
                + "\n",
                encoding="utf-8-sig",
            )

            exit_code = main(
                [
                    "report",
                    "build",
                    "--project",
                    str(project_dir),
                    "--eval-dir",
                    str(eval_dir),
                    "--include-errors",
                    "--include-confusion",
                ]
            )
            self.assertEqual(exit_code, 0)

            report_dir = eval_dir / "report"
            self.assertTrue((report_dir / "per_class_f1.csv").exists())
            self.assertTrue((report_dir / "confusion_matrix.csv").exists())
            self.assertTrue((report_dir / "error_cases.csv").exists())

            summary = json.loads((report_dir / "run_summary.json").read_text(encoding="utf-8"))
            self.assertTrue(any(path.endswith("per_class_f1.csv") for path in summary["generated_files"]))
            self.assertTrue(any(path.endswith("confusion_matrix.csv") for path in summary["generated_files"]))
            self.assertTrue(any(path.endswith("error_cases.csv") for path in summary["generated_files"]))


if __name__ == "__main__":
    unittest.main()
