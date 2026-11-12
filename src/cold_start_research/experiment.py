from __future__ import annotations

import json
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from .config import ExperimentConfig


def collect_runtime_info() -> dict:
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import torch

        info.update(
            {
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda,
                "gpu": (
                    torch.cuda.get_device_name(0)
                    if torch.cuda.is_available()
                    else None
                ),
            }
        )
    except ImportError:
        info["torch"] = None
    return info


def build_training_command(
    config: ExperimentConfig,
    project_root: Path,
    python_executable: str = sys.executable,
    resume: bool = False,
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
    if resume:
        command.append("--resume")
    return command


def run_training(
    config: ExperimentConfig,
    project_root: Path,
    dry_run: bool = False,
    resume: bool = False,
) -> Path:
    config.validate()
    command = build_training_command(config, project_root, resume=resume)
    run_dir = config.output_dir / config.run_name
    log_path = run_dir / f"{config.run_name}.log"
    checkpoint_path = run_dir / f"{config.run_name}.checkpoint.pt"
    model_path = run_dir / f"{config.run_name}.pt"

    printable = shlex.join(command)
    print(printable)
    if dry_run:
        return log_path

    if run_dir.exists() and not resume:
        raise FileExistsError(
            f"Каталог запуска уже существует: {run_dir}. "
            "Задайте новый run_name или передайте --resume."
        )
    if (
        run_dir.exists()
        and resume
        and not checkpoint_path.exists()
        and not model_path.exists()
    ):
        raise FileNotFoundError(
            f"В каталоге {run_dir} нет checkpoint для возобновления."
        )

    run_dir.mkdir(parents=True, exist_ok=resume)
    metadata_path = run_dir / "run_metadata.json"
    now = datetime.now(timezone.utc).isoformat()
    if resume and metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as stream:
            metadata = json.load(stream)
        metadata.setdefault("resume_history", []).append(
            {"resumed_at_utc": now, "command": command}
        )
        command_path = run_dir / "resume_commands.log"
        with command_path.open("a", encoding="utf-8") as stream:
            stream.write(printable + "\n")
    else:
        metadata = {
            "run_name": config.run_name,
            "started_at_utc": now,
            "command": command,
            "parameters": config.parameters,
            "device": config.device,
            "runtime": collect_runtime_info(),
        }
        (run_dir / "command.txt").write_text(printable + "\n", encoding="utf-8")
    with metadata_path.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)

    subprocess.run(
        command,
        cwd=project_root / "third_party" / "TaNP",
        check=True,
    )

    if not log_path.exists():
        raise FileNotFoundError(f"TaNP не создал ожидаемый лог: {log_path}")
    if config.parameters.get("save", 0) and not model_path.exists():
        raise FileNotFoundError(f"TaNP не создал итоговую модель: {model_path}")
    return log_path
