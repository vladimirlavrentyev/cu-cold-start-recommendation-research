from __future__ import annotations

import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from .config import ExperimentConfig


def build_training_command(
    config: ExperimentConfig,
    project_root: Path,
    python_executable: str = sys.executable,
) -> List[str]:
    train_script = project_root / "third_party" / "TaNP" / "train_TaNP.py"
    run_dir = config.output_dir / config.run_name
    command = [
        python_executable,
        str(train_script),
        "--device",
        config.device,
        "--data_dir",
        str(config.data_dir),
        "--model_save_dir",
        str(run_dir),
        "--id",
        config.run_name,
    ]

    for name, value in config.parameters.items():
        flag = f"--{name}"
        if isinstance(value, bool):
            if value:
                command.append(flag)
        else:
            command.extend([flag, str(value)])
    return command


def run_training(
    config: ExperimentConfig,
    project_root: Path,
    dry_run: bool = False,
) -> Path:
    config.validate()
    command = build_training_command(config, project_root)
    run_dir = config.output_dir / config.run_name
    log_path = run_dir / f"{config.run_name}.log"

    printable = shlex.join(command)
    print(printable)
    if dry_run:
        return log_path

    if run_dir.exists():
        raise FileExistsError(
            f"Каталог запуска уже существует: {run_dir}. "
            "Задайте новый run_name, чтобы не перезаписать результаты."
        )

    run_dir.mkdir(parents=True)
    metadata = {
        "run_name": config.run_name,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "parameters": config.parameters,
        "device": config.device,
    }
    with (run_dir / "run_metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)
    (run_dir / "command.txt").write_text(printable + "\n", encoding="utf-8")

    subprocess.run(
        command,
        cwd=project_root / "third_party" / "TaNP",
        check=True,
    )

    if not log_path.exists():
        raise FileNotFoundError(f"TaNP не создал ожидаемый лог: {log_path}")
    return log_path
