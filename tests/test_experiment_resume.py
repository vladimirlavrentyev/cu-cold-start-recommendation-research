from pathlib import Path
from unittest import TestCase

from src.cold_start_research.config import ExperimentConfig
from src.cold_start_research.experiment import build_training_command


class ExperimentResumeCommandTest(TestCase):
    def test_resume_flag_is_forwarded_to_tanp(self) -> None:
        project_root = Path("/project")
        config = ExperimentConfig(
            run_name="run",
            data_dir=project_root / "data",
            output_dir=project_root / "artifacts",
            results_dir=project_root / "results",
            figures_dir=project_root / "figures",
            device="cuda",
            parameters={"num_epoch": 100, "checkpoint_every": 5},
        )

        command = build_training_command(
            config,
            project_root,
            python_executable="python",
            resume=True,
        )

        self.assertEqual(command.count("--resume"), 1)
        self.assertIn("--checkpoint_every", command)
        self.assertIn("5", command)
