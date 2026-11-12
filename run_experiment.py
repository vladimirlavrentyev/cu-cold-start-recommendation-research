#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cold_start_research.config import ExperimentConfig, ExperimentSuite
from cold_start_research.experiment import run_training
from cold_start_research.visualization import build_outputs, save_suite_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Единая точка запуска экспериментов cold-start recommendation."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--config",
        type=Path,
        help="Путь к JSON-конфигурации эксперимента.",
    )
    source.add_argument(
        "--suite",
        type=Path,
        help="Путь к JSON-конфигурации последовательности экспериментов.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Проверить конфигурацию и вывести команду без обучения.",
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Не обучать модель, а заново собрать таблицу и рисунки из существующего лога.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Выполнить вычисления без построения рисунков.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Использовать уже существующий завершённый лог вместо повторного обучения.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Продолжить незавершённый запуск из промежуточного checkpoint.",
    )
    return parser.parse_args()


def run_one(config: ExperimentConfig, args: argparse.Namespace) -> Path:
    log_path = config.output_dir / config.run_name / f"{config.run_name}.log"
    checkpoint_path = config.output_dir / config.run_name / f"{config.run_name}.pt"
    completed_run_exists = log_path.exists() and checkpoint_path.exists()
    reuse_log = args.plots_only or (args.skip_existing and completed_run_exists)
    if reuse_log:
        if not log_path.exists():
            raise FileNotFoundError(f"Не найден лог запуска: {log_path}")
        print(f"Используется существующий запуск: {log_path}")
    else:
        log_path = run_training(
            config,
            PROJECT_ROOT,
            dry_run=args.dry_run,
            resume=args.resume,
        )

    if args.dry_run or args.skip_plots:
        return log_path
    build_outputs(
        log_path=log_path,
        run_name=config.run_name,
        results_dir=config.results_dir,
        figures_dir=config.figures_dir,
    )
    return log_path


def main() -> None:
    args = parse_args()
    if args.config is not None:
        config_path = args.config
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path
        config = ExperimentConfig.from_json(config_path.resolve(), PROJECT_ROOT)
        run_one(config, args)
        return

    suite_path = args.suite
    if not suite_path.is_absolute():
        suite_path = PROJECT_ROOT / suite_path
    suite = ExperimentSuite.from_json(suite_path.resolve(), PROJECT_ROOT)
    summary_runs = []
    for config in suite.experiments:
        print(f"\n=== {config.run_name} ===")
        log_path = run_one(config, args)
        summary_runs.append(
            {
                "run_name": config.run_name,
                "dataset": config.data_dir.name,
                "decoder": config.parameters.get("decoder", "film"),
                "seed": config.parameters.get("seed", ""),
                "log_path": log_path,
            }
        )
    if not args.dry_run and not args.skip_plots:
        save_suite_summary(
            summary_runs,
            suite.results_dir / f"{suite.suite_name}_summary.csv",
            suite.experiments[0].figures_dir,
        )


if __name__ == "__main__":
    main()
