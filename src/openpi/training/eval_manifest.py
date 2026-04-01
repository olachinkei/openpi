from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import Any
from typing import Literal

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]


@dataclasses.dataclass(frozen=True)
class ManifestSelection:
    policy: str
    source: str
    seed: int | None = None
    notes: str | None = None


@dataclasses.dataclass(frozen=True)
class ManifestRecord:
    example_id: str
    episode_id: str | None = None
    prompt: str | None = None
    hf_episode_index: int | None = None
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class EvalManifest:
    manifest_version: str
    name: str
    environment_family: str
    task_name: str
    config_name: str
    dataset_name: str
    split_name: Literal["subsample", "full"]
    prompt: str
    selection: ManifestSelection
    records: list[ManifestRecord]


def resolve_repo_path(path_str: str | pathlib.Path | None) -> pathlib.Path | None:
    if path_str is None:
        return None
    path = pathlib.Path(path_str)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_eval_manifest(path: pathlib.Path) -> EvalManifest:
    payload = json.loads(path.read_text())
    selection = ManifestSelection(**payload["selection"])
    records = [ManifestRecord(**record) for record in payload["records"]]
    return EvalManifest(
        manifest_version=payload["manifest_version"],
        name=payload["name"],
        environment_family=payload["environment_family"],
        task_name=payload["task_name"],
        config_name=payload["config_name"],
        dataset_name=payload["dataset_name"],
        split_name=payload["split_name"],
        prompt=payload["prompt"],
        selection=selection,
        records=records,
    )


def load_manifest_episode_indices(path: pathlib.Path) -> tuple[int, ...]:
    manifest = load_eval_manifest(path)
    episode_indices: list[int] = []
    for record in manifest.records:
        if record.hf_episode_index is None:
            raise ValueError(f"Manifest record {record.example_id!r} is missing hf_episode_index: {path}")
        episode_indices.append(int(record.hf_episode_index))
    return tuple(dict.fromkeys(episode_indices))
