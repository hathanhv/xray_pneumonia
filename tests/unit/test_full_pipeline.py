from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from src.pipelines.full_pipeline import (
    FullPipelineRunner,
    PipelineConfigError,
    validate_output,
)
from src.reporting.pipeline_report import write_pipeline_reports


class FullPipelineTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def config(self):
        return {
            "pipeline": {
                "name": "test_pipeline",
                "output_dir": "outputs/pipeline",
                "resume": True,
                "report": {
                    "collect": {
                        "metrics": ["artifacts/metrics.json"],
                        "figures": [],
                        "manifests": [],
                        "artifacts": [],
                    }
                },
            },
            "stages": [
                {
                    "id": "create_metrics",
                    "enabled": True,
                    "command": [
                        sys.executable,
                        "-c",
                        (
                            "from pathlib import Path; "
                            "p=Path('artifacts/metrics.json'); "
                            "p.parent.mkdir(parents=True, exist_ok=True); "
                            "p.write_text('{\"f1\": 0.9}', encoding='utf-8')"
                        ),
                    ],
                    "outputs": [
                        {
                            "path": "artifacts/metrics.json",
                            "kind": "file",
                            "validator": "json",
                        }
                    ],
                }
            ],
        }

    def test_runs_and_resumes_from_valid_output(self):
        config = self.config()
        first = FullPipelineRunner(config, project_root=self.root)
        state = first.run()
        self.assertEqual(state["stages"]["create_metrics"]["status"], "completed")

        second = FullPipelineRunner(config, project_root=self.root)
        state = second.run()
        self.assertEqual(
            state["stages"]["create_metrics"]["status"],
            "skipped_completed",
        )

    def test_invalid_json_is_not_complete(self):
        path = self.root / "broken.json"
        path.write_text("{", encoding="utf-8")
        check = validate_output(
            {"path": "broken.json", "validator": "json"},
            project_root=self.root,
        )
        self.assertFalse(check.valid)

    def test_disabled_stage_does_not_run(self):
        config = self.config()
        config["stages"][0]["enabled"] = False
        config["stages"][0].pop("command")
        runner = FullPipelineRunner(config, project_root=self.root)
        state = runner.run()
        self.assertEqual(state["stages"]["create_metrics"]["status"], "disabled")

    def test_runs_multiple_commands_in_order(self):
        config = self.config()
        config["stages"][0].pop("command")
        config["stages"][0]["commands"] = [
            [
                sys.executable,
                "-c",
                "from pathlib import Path; Path('artifacts').mkdir()",
            ],
            [
                sys.executable,
                "-c",
                (
                    "from pathlib import Path; "
                    "Path('artifacts/metrics.json').write_text("
                    "'{\"f1\": 0.8}', encoding='utf-8')"
                ),
            ],
        ]
        state = FullPipelineRunner(config, project_root=self.root).run()
        self.assertEqual(state["stages"]["create_metrics"]["status"], "completed")

    def test_changed_stage_command_is_not_resumed(self):
        config = self.config()
        runner = FullPipelineRunner(config, project_root=self.root)
        runner.run()
        config["stages"][0]["command"][-1] = config["stages"][0]["command"][
            -1
        ].replace("0.9", "0.7")
        state = FullPipelineRunner(config, project_root=self.root).run()
        self.assertEqual(state["stages"]["create_metrics"]["status"], "completed")
        metrics = json.loads(
            (self.root / "artifacts/metrics.json").read_text(encoding="utf-8")
        )
        self.assertEqual(metrics["f1"], 0.7)

    def test_unknown_dependency_is_rejected(self):
        config = self.config()
        config["stages"][0]["depends_on"] = ["missing"]
        with self.assertRaises(PipelineConfigError):
            FullPipelineRunner(config, project_root=self.root)

    def test_writes_markdown_and_html_reports(self):
        config = self.config()
        runner = FullPipelineRunner(config, project_root=self.root)
        state = runner.run()
        paths = write_pipeline_reports(
            config,
            state,
            project_root=self.root,
            output_dir=runner.output_dir,
        )
        markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
        html = Path(paths["html"]).read_text(encoding="utf-8")
        self.assertIn("create_metrics", markdown)
        self.assertIn("f1=0.9", markdown)
        self.assertIn("<table>", html)
        self.assertTrue(json.loads((self.root / "artifacts/metrics.json").read_text()))


if __name__ == "__main__":
    unittest.main()
