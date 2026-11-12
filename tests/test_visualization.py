from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from src.cold_start_research.visualization import read_tanp_log


class ReadTaNPLogTest(TestCase):
    def test_parses_author_log_format(self) -> None:
        content = "\n".join(
            [
                "# epoch train_loss precision5 NDCG5 MAP5 precision7 NDCG7 MAP7 precision10 NDCG10 MAP10",
                "0 23.659832 TOP-5 0.5576 0.5655 0.4631 TOP-7: 0.6645 0.6021 0.5809 TOP-10: 0.8318 0.6914 0.7881",
            ]
        )
        with TemporaryDirectory() as directory:
            log_path = Path(directory) / "run.log"
            log_path.write_text(content, encoding="utf-8")
            rows = read_tanp_log(log_path)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["epoch"], 0.0)
        self.assertAlmostEqual(rows[0]["ndcg10"], 0.6914)
