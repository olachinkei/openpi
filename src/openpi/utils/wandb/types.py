from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass(frozen=True)
class LeaderboardRow:
    eval_run_id: str
    source_train_run_id: str | None = None
    source_train_run_name: str | None = None
    eval_name: str | None = None
    eval_split: str | None = None
    model_family: str | None = None
    config_name: str | None = None
    task_name: str | None = None
    dataset_name: str | None = None
    checkpoint_alias: str | None = None
    checkpoint_step: int | None = None
    primary_score: float | None = None
    success_rate: float | None = None
    mean_max_reward: float | None = None
    eval_loss: float | None = None
    num_examples: int | None = None
    artifact_ref_results: str | None = None
    artifact_ref_media: str | None = None
    created_at: str | None = None
    notes: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class ExampleRecord:
    example_id: str
    prompt: str | None = None
    task_name: str | None = None
    split: str | None = None
    metric_primary: float | None = None
    metric_aux_json: str | None = None
    checkpoint_step: int | None = None
    video: Any | None = None
    artifact_ref_video: str | None = None
    metadata_json: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class MediaRecord:
    key: str
    caption: str | None = None
    path: str | None = None
    artifact_ref: str | None = None
    step: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class VideoRecord:
    path: str
    name: str | None = None
    caption: str | None = None
    fps: int = 20
    format: str = "mp4"

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class ArtifactRecord:
    name: str
    type: str
    path: str
    description: str | None = None
    aliases: tuple[str, ...] = ()
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)
