#!/usr/bin/env python3
"""Prepare the three TaNP datasets in the generic format used by the author code."""

from __future__ import annotations

import argparse
import json
import random
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
GOWALLA_TRAIN_URL = (
    "https://raw.githubusercontent.com/xiangwang1223/"
    "neural_graph_collaborative_filtering/master/Data/gowalla/train.txt"
)
GOWALLA_TEST_URL = (
    "https://raw.githubusercontent.com/xiangwang1223/"
    "neural_graph_collaborative_filtering/master/Data/gowalla/test.txt"
)


def download(url: str, destination: Path) -> None:
    if destination.exists():
        print(f"Уже загружено: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Загрузка {url}")
    urllib.request.urlretrieve(url, destination)


def write_dataset(
    output_dir: Path,
    interactions: Dict[int, Sequence[Tuple[int, float]]],
    manifest: Dict[str, object],
    force: bool,
) -> None:
    required = [
        output_dir / "interaction_dict_x.json",
        output_dir / "interaction_dict_y.json",
        output_dir / "user_list.json",
        output_dir / "item_list.json",
    ]
    if all(path.exists() for path in required) and not force:
        print(f"Датасет уже подготовлен: {output_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    user_ids = sorted(interactions)
    item_ids = sorted(
        {item_id for values in interactions.values() for item_id, _ in values}
    )
    interaction_x = {
        str(user_id): [item_id for item_id, _ in interactions[user_id]]
        for user_id in user_ids
    }
    interaction_y = {
        str(user_id): [label for _, label in interactions[user_id]]
        for user_id in user_ids
    }

    payloads = {
        "interaction_dict_x.json": interaction_x,
        "interaction_dict_y.json": interaction_y,
        "user_list.json": user_ids,
        "item_list.json": item_ids,
        "manifest.json": manifest,
    }
    for filename, payload in payloads.items():
        with (output_dir / filename).open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
    print(
        f"Подготовлено: {output_dir} "
        f"({len(user_ids)} пользователей, {len(item_ids)} объектов, "
        f"{sum(len(values) for values in interactions.values())} взаимодействий)"
    )


def prepare_lastfm(project_root: Path) -> None:
    archive_path = project_root / "third_party" / "TaNP" / "data.zip"
    destination = project_root / "third_party" / "TaNP"
    dataset_dir = destination / "data" / "lastfm_20"
    if (dataset_dir / "interaction_dict_x.json").exists():
        print(f"LastFM уже распакован: {dataset_dir}")
        return
    if not archive_path.exists():
        raise FileNotFoundError(f"Не найден авторский архив: {archive_path}")
    with zipfile.ZipFile(archive_path) as archive:
        root = destination.resolve()
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if root not in target.parents and target != root:
                raise ValueError(f"Небезопасный путь в архиве: {member.filename}")
        archive.extractall(destination)
    print(f"LastFM распакован: {dataset_dir}")


def prepare_movielens(project_root: Path, force: bool) -> None:
    archive_path = project_root / "data" / "raw" / "ml-1m.zip"
    output_dir = project_root / "data" / "processed" / "movielens_1m"
    download(MOVIELENS_URL, archive_path)

    rows: List[Tuple[int, int, float]] = []
    with zipfile.ZipFile(archive_path) as archive:
        content = archive.read("ml-1m/ratings.dat").decode("latin-1")
    for line in content.splitlines():
        raw_user, raw_item, rating, _ = line.split("::")
        rows.append((int(raw_user), int(raw_item), float(rating)))

    user_map = {value: index for index, value in enumerate(sorted({r[0] for r in rows}))}
    item_map = {value: index for index, value in enumerate(sorted({r[1] for r in rows}))}
    interactions: Dict[int, List[Tuple[int, float]]] = {
        index: [] for index in user_map.values()
    }
    for raw_user, raw_item, rating in rows:
        interactions[user_map[raw_user]].append((item_map[raw_item], rating))

    write_dataset(
        output_dir,
        interactions,
        {
            "name": "MovieLens-1M",
            "source": MOVIELENS_URL,
            "protocol": "Full explicit ratings, contiguous zero-based ids.",
            "warning": (
                "The TaNP authors did not publish their prepared MovieLens episodes; "
                "user splitting is reproduced later by the TaNP loader with seed 2020."
            ),
        },
        force,
    )


def read_ngcf(path: Path) -> Dict[int, Set[int]]:
    interactions: Dict[int, Set[int]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        values = [int(value) for value in line.split()]
        if not values:
            continue
        interactions.setdefault(values[0], set()).update(values[1:])
    return interactions


def merge_interactions(
    first: Dict[int, Set[int]], second: Dict[int, Set[int]]
) -> Dict[int, Set[int]]:
    result = {user_id: set(items) for user_id, items in first.items()}
    for user_id, items in second.items():
        result.setdefault(user_id, set()).update(items)
    return result


def prepare_gowalla(
    project_root: Path,
    force: bool,
    seed: int,
    max_users: int,
) -> None:
    raw_dir = project_root / "data" / "raw" / "gowalla_ngcf"
    train_path = raw_dir / "train.txt"
    test_path = raw_dir / "test.txt"
    output_dir = project_root / "data" / "processed" / "gowalla_reconstructed"
    download(GOWALLA_TRAIN_URL, train_path)
    download(GOWALLA_TEST_URL, test_path)

    positives = merge_interactions(read_ngcf(train_path), read_ngcf(test_path))
    # With one negative per positive, 20..100 positives become 40..200
    # labelled interactions, matching the history-length condition in TaNP.
    eligible_users = sorted(
        user_id for user_id, items in positives.items() if 20 <= len(items) <= 100
    )
    rng = random.Random(seed)
    if max_users and len(eligible_users) > max_users:
        eligible_users = sorted(rng.sample(eligible_users, max_users))

    positive_item_pool = sorted(
        {item_id for user_id in eligible_users for item_id in positives[user_id]}
    )
    positive_item_set = set(positive_item_pool)
    raw_interactions: Dict[int, List[Tuple[int, float]]] = {}
    for raw_user in eligible_users:
        user_positives = sorted(positives[raw_user])
        candidates = list(positive_item_set.difference(user_positives))
        negatives = rng.sample(candidates, len(user_positives))
        labelled = [(item_id, 1.0) for item_id in user_positives]
        labelled.extend((item_id, 0.0) for item_id in negatives)
        rng.shuffle(labelled)
        raw_interactions[raw_user] = labelled

    user_map = {
        value: index for index, value in enumerate(sorted(raw_interactions))
    }
    used_items = sorted(
        {item_id for values in raw_interactions.values() for item_id, _ in values}
    )
    item_map = {value: index for index, value in enumerate(used_items)}
    interactions = {
        user_map[raw_user]: [
            (item_map[raw_item], label) for raw_item, label in values
        ]
        for raw_user, values in raw_interactions.items()
    }

    write_dataset(
        output_dir,
        interactions,
        {
            "name": "Gowalla (reconstructed TaNP protocol)",
            "sources": [GOWALLA_TRAIN_URL, GOWALLA_TEST_URL],
            "seed": seed,
            "max_users": max_users,
            "positive_history_filter": [20, 100],
            "negative_sampling": "One unseen item per positive interaction.",
            "warning": (
                "The TaNP authors did not publish their prepared Gowalla data or "
                "exact filtering and negative-sampling code. This deterministic "
                "reconstruction is suitable for our experiments but is not claimed "
                "to reproduce the paper split exactly."
            ),
        },
        force,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("all", "lastfm", "movielens", "gowalla"),
        default="all",
    )
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument(
        "--gowalla-max-users",
        type=int,
        default=2692,
        help="Limit reconstructed Gowalla to the user count reported in TaNP; 0 keeps all.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset in {"all", "lastfm"}:
        prepare_lastfm(PROJECT_ROOT)
    if args.dataset in {"all", "movielens"}:
        prepare_movielens(PROJECT_ROOT, args.force)
    if args.dataset in {"all", "gowalla"}:
        prepare_gowalla(
            PROJECT_ROOT,
            args.force,
            args.seed,
            args.gowalla_max_users,
        )


if __name__ == "__main__":
    main()
