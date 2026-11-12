from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass(frozen=True)
class ExperimentConfig:
    run_name: str
    data_dir: Path
    output_dir: Path
    results_dir: Path
    figures_dir: Path
    device: str
    parameters: Dict[str, Any]

    @classmethod
    def from_json(cls, path: Path, project_root: Path) -> "ExperimentConfig":
        with path.open("r", encoding="utf-8") as stream:
            raw = json.load(stream)

        required = {
            "run_name",
            "data_dir",
            "output_dir",
            "results_dir",
            "figures_dir",
            "parameters",
        }
        missing = required.difference(raw)
        if missing:
            raise ValueError(f"В конфигурации отсутствуют поля: {sorted(missing)}")

        parameters = raw["parameters"]
        if not isinstance(parameters, dict):
            raise ValueError("Поле parameters должно быть JSON-объектом.")

        return cls(
            run_name=str(raw["run_name"]),
            data_dir=(project_root / raw["data_dir"]).resolve(),
            output_dir=(project_root / raw["output_dir"]).resolve(),
            results_dir=(project_root / raw["results_dir"]).resolve(),
            figures_dir=(project_root / raw["figures_dir"]).resolve(),
            device=str(raw.get("device", "cpu")),
            parameters=parameters,
        )

    def validate(self) -> None:
        if not self.run_name.strip():
            raise ValueError("run_name не может быть пустым.")
        if self.device not in {"cpu", "cuda", "auto"}:
            raise ValueError("device должен быть одним из: cpu, cuda, auto.")
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Не найдены данные: {self.data_dir}. "
                "Запустите scripts/prepare_datasets.py."
            )
        for filename in (
            "interaction_dict_x.json",
            "interaction_dict_y.json",
            "user_list.json",
            "item_list.json",
        ):
            if not (self.data_dir / filename).is_file():
                raise FileNotFoundError(
                    f"В каталоге данных отсутствует {filename}: {self.data_dir}"
                )


@dataclass(frozen=True)
class ExperimentSuite:
    suite_name: str
    experiments: List[ExperimentConfig]
    results_dir: Path

    @classmethod
    def from_json(cls, path: Path, project_root: Path) -> "ExperimentSuite":
        with path.open("r", encoding="utf-8") as stream:
            raw = json.load(stream)

        common = raw.get("common", {})
        common_parameters = common.get("parameters", {})
        datasets = raw.get("datasets", [])
        if not datasets:
            raise ValueError("В suite-конфигурации должен быть непустой список datasets.")

        experiments: List[ExperimentConfig] = []
        for dataset in datasets:
            parameters = dict(common_parameters)
            parameters.update(dataset.get("parameters", {}))
            experiments.append(
                ExperimentConfig(
                    run_name=str(dataset["run_name"]),
                    data_dir=(project_root / dataset["data_dir"]).resolve(),
                    output_dir=(project_root / common["output_dir"]).resolve(),
                    results_dir=(project_root / common["results_dir"]).resolve(),
                    figures_dir=(project_root / common["figures_dir"]).resolve(),
                    device=str(dataset.get("device", common.get("device", "cpu"))),
                    parameters=parameters,
                )
            )

        return cls(
            suite_name=str(raw["suite_name"]),
            experiments=experiments,
            results_dir=(project_root / common["results_dir"]).resolve(),
        )
