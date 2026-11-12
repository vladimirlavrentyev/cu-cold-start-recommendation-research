import json
from pathlib import Path
from unittest import TestCase


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ColabNotebookOrderTest(TestCase):
    def test_project_is_defined_before_drive_config_uses_it(self) -> None:
        notebook_path = (
            PROJECT_ROOT / "notebooks" / "tanp_gating_film_colab.ipynb"
        )
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        code_cells = [
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        ]

        project_definition = next(
            index
            for index, source in enumerate(code_cells)
            if "PROJECT = Path(" in source
        )
        drive_configuration = next(
            index
            for index, source in enumerate(code_cells)
            if "DRIVE_CONFIG = Path(" in source
        )
        training = next(
            index
            for index, source in enumerate(code_cells)
            if "--skip-existing --resume" in source
        )

        self.assertLess(project_definition, drive_configuration)
        self.assertLess(drive_configuration, training)

    def test_final_cell_audits_baseline_artifacts(self) -> None:
        notebook_path = (
            PROJECT_ROOT / "notebooks" / "tanp_gating_film_colab.ipynb"
        )
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        final_source = "".join(notebook["cells"][-1].get("source", []))

        for required_fragment in (
            "RUN_NAMES",
            ".pt",
            ".log",
            "run_metadata.json",
            "_metrics.csv",
            "_metrics.pdf",
            "len(epochs) == 100",
            "tanp_gating_film_seed2020_summary.csv",
        ):
            self.assertIn(required_fragment, final_source)
