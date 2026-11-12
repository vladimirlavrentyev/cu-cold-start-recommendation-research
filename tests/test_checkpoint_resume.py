import random
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "TaNP"))

from TaNP_training import load_training_checkpoint, save_training_checkpoint
from utils.helper import FileLogger


class TinyTrainer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)


class CheckpointResumeTest(TestCase):
    def test_restores_model_optimizer_rng_and_epoch(self) -> None:
        random.seed(2020)
        np.random.seed(2020)
        torch.manual_seed(2020)
        trainer = TinyTrainer()

        loss = trainer.weight.square().sum()
        loss.backward()
        trainer.optimizer.step()
        saved_weight = trainer.weight.detach().clone()

        with TemporaryDirectory() as directory:
            checkpoint_path = str(Path(directory) / "run.checkpoint.pt")
            save_training_checkpoint(trainer, checkpoint_path, 3, [2, 0, 1])

            expected_random = random.random()
            expected_numpy = np.random.rand()
            expected_torch = torch.rand(1)

            trainer.weight.data.fill_(99.0)
            random.seed(1)
            np.random.seed(1)
            torch.manual_seed(1)

            next_epoch, order = load_training_checkpoint(
                trainer, checkpoint_path
            )

        self.assertEqual(next_epoch, 4)
        self.assertEqual(order, [2, 0, 1])
        self.assertTrue(torch.equal(trainer.weight.detach(), saved_weight))
        self.assertEqual(random.random(), expected_random)
        self.assertEqual(np.random.rand(), expected_numpy)
        self.assertTrue(torch.equal(torch.rand(1), expected_torch))
        self.assertTrue(trainer.optimizer.state)

    def test_logger_appends_without_replacing_existing_epochs(self) -> None:
        with TemporaryDirectory() as directory:
            log_path = Path(directory) / "run.log"
            first = FileLogger(str(log_path), header="header")
            first.log("epoch 0")
            resumed = FileLogger(str(log_path), header="header", append=True)
            resumed.log("epoch 1")
            lines = log_path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(lines, ["header", "epoch 0", "epoch 1"])
