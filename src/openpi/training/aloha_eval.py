from __future__ import annotations

import csv
import dataclasses
import json
import logging
import pathlib
import re
import sys
from typing import Any

import imageio
import numpy as np
import wandb

from openpi.training import eval_manifest as _eval_manifest
from openpi.utils.wandb.types import ExampleRecord
from openpi.utils.wandb.types import VideoRecord
from openpi_client import action_chunk_broker
from openpi_client import base_policy as _base_policy

ALOHA_SIM_EXAMPLES_ROOT = pathlib.Path(__file__).resolve().parents[3] / "examples" / "aloha_sim"
if str(ALOHA_SIM_EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(ALOHA_SIM_EXAMPLES_ROOT))

import env as _aloha_env

DEFAULT_TASK = "gym_aloha/AlohaTransferCube-v0"
EXAMPLE_COLUMNS = tuple(field.name for field in dataclasses.fields(ExampleRecord))


@dataclasses.dataclass(frozen=True)
class EpisodeResult:
    example_id: str
    prompt: str
    seed: int
    success: bool
    max_reward: float
    num_steps: int
    video_path: str
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "prompt": self.prompt,
            "seed": self.seed,
            "success": self.success,
            "max_reward": self.max_reward,
            "num_steps": self.num_steps,
            "video_path": self.video_path,
            "metadata": self.metadata,
        }


@dataclasses.dataclass(frozen=True)
class EvalBundle:
    metrics: dict[str, Any]
    results: list[EpisodeResult]
    video_records: list[VideoRecord]
    example_rows: list[ExampleRecord]


def slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value.strip()).strip("_") or "example"


def load_manifest(path: pathlib.Path) -> _eval_manifest.EvalManifest:
    return _eval_manifest.load_eval_manifest(path)


def build_eval_records(
    manifest: _eval_manifest.EvalManifest, num_examples: int
) -> list[_eval_manifest.ManifestRecord]:
    if manifest.records:
        return manifest.records[:num_examples] if num_examples > 0 else list(manifest.records)

    if num_examples <= 0:
        raise ValueError("num_examples must be greater than 0 when the manifest has no explicit records.")

    base_seed = manifest.selection.seed or 0
    return [
        _eval_manifest.ManifestRecord(
            example_id=f"{manifest.split_name}_{index:03d}",
            metadata={"seed": base_seed + index},
        )
        for index in range(num_examples)
    ]


def resolve_seed(manifest: _eval_manifest.EvalManifest, record: _eval_manifest.ManifestRecord, index: int) -> int:
    if isinstance(record.metadata.get("seed"), int):
        return int(record.metadata["seed"])
    if record.hf_episode_index is not None:
        return int(record.hf_episode_index)
    if manifest.selection.seed is not None:
        return int(manifest.selection.seed + index)
    return index


def write_results(
    path: pathlib.Path,
    *,
    config_name: str,
    checkpoint_dir: str,
    checkpoint_step: int | None,
    checkpoint_alias: str,
    manifest_path: pathlib.Path,
    split_name: str,
    metrics: dict[str, Any],
    results: list[EpisodeResult],
) -> None:
    payload = {
        "config_name": config_name,
        "checkpoint_dir": checkpoint_dir,
        "checkpoint_step": checkpoint_step,
        "checkpoint_alias": checkpoint_alias,
        "manifest": str(manifest_path),
        "split_name": split_name,
        "metrics": metrics,
        "results": [result.as_dict() for result in results],
    }
    _write_results_json(path, payload)
    _write_results_csv(path, results)


def evaluate_policy(
    *,
    policy: _base_policy.BasePolicy,
    manifest: _eval_manifest.EvalManifest,
    split_name: str,
    checkpoint_step: int | None,
    num_examples: int,
    video_dir: pathlib.Path,
    media_artifact_ref: str,
    task: str = DEFAULT_TASK,
    action_horizon: int = 10,
    max_episode_steps: int = 0,
    success_reward_threshold: float = 4.0,
    fps: int = 50,
    render_mode: str | None = None,
    visualization_width: int = 640,
    visualization_height: int = 336,
    visualization_camera_id: str = "angle",
) -> EvalBundle:
    broker = action_chunk_broker.ActionChunkBroker(policy=policy, action_horizon=action_horizon)
    eval_records = build_eval_records(manifest, num_examples)

    results: list[EpisodeResult] = []
    video_records: list[VideoRecord] = []
    example_rows: list[ExampleRecord] = []

    for index, record in enumerate(eval_records):
        seed = resolve_seed(manifest, record, index)
        prompt = record.prompt or manifest.prompt
        logging.info("Starting eval example %s (seed=%s)", record.example_id, seed)

        env = _aloha_env.AlohaSimEnvironment(
            task=task,
            seed=seed,
            render_mode=render_mode,
            visualization_width=visualization_width,
            visualization_height=visualization_height,
            visualization_camera_id=visualization_camera_id,
        )

        broker.reset()
        env.reset()
        frames: list[np.ndarray] = [np.asarray(env.get_video_frame()).copy()]
        num_steps = 0

        while True:
            observation = dict(env.get_observation())
            observation["prompt"] = prompt
            action = broker.infer(observation)
            env.apply_action(action)
            frames.append(np.asarray(env.get_video_frame()).copy())
            num_steps += 1

            if env.is_episode_complete() or (max_episode_steps > 0 and num_steps >= max_episode_steps):
                break

        success = env.is_success(success_reward_threshold)
        video_path = video_dir / f"{index:03d}_{slugify(record.example_id)}_{'success' if success else 'failure'}.mp4"
        _save_video(video_path, frames, fps=fps)

        result = EpisodeResult(
            example_id=record.example_id,
            prompt=prompt,
            seed=seed,
            success=success,
            max_reward=env.episode_reward,
            num_steps=num_steps,
            video_path=str(video_path),
            metadata=dict(record.metadata),
        )
        results.append(result)
        logging.info(
            "Finished eval example %s: success=%s max_reward=%.2f steps=%s",
            record.example_id,
            success,
            env.episode_reward,
            num_steps,
        )

        caption = (
            f"step={checkpoint_step} | {record.example_id} | "
            f"success={int(success)} | reward={env.episode_reward:.2f}"
        )
        video_records.append(
            VideoRecord(
                path=str(video_path),
                name=record.example_id,
                caption=caption,
                fps=fps,
            )
        )
        example_rows.append(
            ExampleRecord(
                example_id=record.example_id,
                prompt=prompt,
                task_name=manifest.task_name,
                split=split_name,
                metric_primary=1.0 if success else 0.0,
                metric_aux_json=json.dumps(
                    {
                        "max_reward": env.episode_reward,
                        "num_steps": num_steps,
                        "seed": seed,
                    },
                    sort_keys=True,
                ),
                checkpoint_step=checkpoint_step,
                video=wandb.Video(
                    str(video_path),
                    caption=caption,
                    fps=fps,
                    format="mp4",
                ),
                artifact_ref_video=f"{media_artifact_ref}/{video_path.name}",
                metadata_json=json.dumps(record.metadata, sort_keys=True),
            )
        )

    success_rate = float(np.mean([result.success for result in results])) if results else 0.0
    mean_max_reward = float(np.mean([result.max_reward for result in results])) if results else 0.0
    return EvalBundle(
        metrics={
            "primary_score": success_rate,
            "success_rate": success_rate,
            "mean_max_reward": mean_max_reward,
            "num_examples": len(results),
            "checkpoint_step": checkpoint_step,
        },
        results=results,
        video_records=video_records,
        example_rows=example_rows,
    )


def _write_results_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _write_results_csv(path: pathlib.Path, results: list[EpisodeResult]) -> None:
    csv_path = path.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("example_id", "prompt", "seed", "success", "max_reward", "num_steps", "video_path"),
        )
        writer.writeheader()
        for result in results:
            row = result.as_dict()
            writer.writerow({key: row[key] for key in writer.fieldnames})


def _save_video(path: pathlib.Path, frames: list[np.ndarray], *, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(path, [np.asarray(frame) for frame in frames], fps=fps)
