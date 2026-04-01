from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import json
import logging
import pathlib
import re
import sys
from typing import Any
from typing import Literal

import imageio
import numpy as np
import tyro
import wandb

ALOHA_SIM_EXAMPLES_ROOT = pathlib.Path(__file__).resolve().parents[1] / "examples" / "aloha_sim"
if str(ALOHA_SIM_EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(ALOHA_SIM_EXAMPLES_ROOT))

import env as _aloha_env
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from openpi.training import eval_manifest as _eval_manifest
from openpi.training import eval_tracking as _eval_tracking
from openpi.utils.wandb import ArtifactRecord
from openpi.utils.wandb import ExampleRecord
from openpi.utils.wandb import LeaderboardRow
from openpi.utils.wandb import LeaderboardTableLogger
from openpi.utils.wandb import VideoLogger
from openpi.utils.wandb import VideoRecord
from openpi.utils.wandb import WandbArtifactManager
from openpi.utils.wandb import WandbRunContext
from openpi.utils.wandb import WandbTableLogger
from openpi.utils.wandb import build_artifact_ref
from openpi_client import action_chunk_broker

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


@dataclasses.dataclass
class Args:
    config_name: str
    checkpoint_dir: str
    manifest: pathlib.Path
    video_dir: pathlib.Path

    split: Literal["subsample", "full"] | None = None
    run_name: str | None = None
    checkpoint_step: int | None = None
    checkpoint_alias: str | None = None
    results_path: pathlib.Path | None = None
    num_examples: int = 4

    task: str = DEFAULT_TASK
    action_horizon: int = 10
    max_episode_steps: int = 0
    success_reward_threshold: float = 4.0
    fps: int = 50
    render_mode: str | None = None
    visualization_width: int = 640
    visualization_height: int = 336
    visualization_camera_id: str = "angle"
    attach_to_source_train_run: bool = True

    wandb_enabled: bool = True


def _slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value.strip()).strip("_") or "example"


def _load_manifest(path: pathlib.Path) -> _eval_manifest.EvalManifest:
    return _eval_manifest.load_eval_manifest(path)


def _infer_checkpoint_step(checkpoint_dir: str) -> int | None:
    leaf = checkpoint_dir.rstrip("/").rsplit("/", 1)[-1]
    return int(leaf) if leaf.isdigit() else None


def _infer_source_train_run_name(checkpoint_dir: str) -> str | None:
    parts = [part for part in checkpoint_dir.rstrip("/").split("/") if part]
    if len(parts) < 2:
        return None
    return parts[-2]


def _read_source_train_run_id(checkpoint_dir: str) -> str | None:
    if "://" in checkpoint_dir:
        return None

    run_id_path = pathlib.Path(checkpoint_dir).resolve().parent / "wandb_id.txt"
    if not run_id_path.exists():
        return None
    run_id = run_id_path.read_text().strip()
    return run_id or None


def _build_eval_records(
    manifest: _eval_manifest.EvalManifest, num_examples: int
) -> list[_eval_manifest.ManifestRecord]:
    if manifest.records:
        return manifest.records[:num_examples] if num_examples > 0 else list(manifest.records)

    if num_examples <= 0:
        raise ValueError("num_examples must be greater than 0 when the manifest has no explicit records.")

    base_seed = manifest.selection.seed or 0
    return [
        _eval_manifest.ManifestRecord(
            # Explicit records are preferred. This fallback keeps the runner usable
            # before a manifest is materialized.
            example_id=f"{manifest.split_name}_{index:03d}",
            metadata={"seed": base_seed + index},
        )
        for index in range(num_examples)
    ]


def _resolve_seed(manifest: _eval_manifest.EvalManifest, record: _eval_manifest.ManifestRecord, index: int) -> int:
    if isinstance(record.metadata.get("seed"), int):
        return int(record.metadata["seed"])
    if record.hf_episode_index is not None:
        return int(record.hf_episode_index)
    if manifest.selection.seed is not None:
        return int(manifest.selection.seed + index)
    return index


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


def _resolve_eval_run_id_path(
    checkpoint_dir: str,
    results_path: pathlib.Path,
    split_name: str,
    *,
    attach_to_source_train_run: bool,
) -> pathlib.Path:
    if attach_to_source_train_run and "://" not in checkpoint_dir:
        checkpoint_root = pathlib.Path(checkpoint_dir).resolve().parent
        return checkpoint_root / "wandb_id.txt"
    return results_path.parent / _eval_tracking.run_id_filename_for_split(split_name)


def _should_attach_to_source_train_run(checkpoint_dir: str, source_train_run_id: str | None) -> bool:
    return source_train_run_id is not None and "://" not in checkpoint_dir


def _sync_source_train_summary(
    *,
    config: _config.TrainConfig,
    source_train_run_id: str | None,
    metric_prefix: str,
    metrics: dict[str, Any],
    checkpoint_alias: str,
    eval_run_id: str | None,
    eval_run_name: str,
    results_artifact_ref: str,
    media_artifact_ref: str,
) -> None:
    if source_train_run_id is None:
        return

    entity = config.wandb_entity or (wandb.run.entity if wandb.run is not None else None)
    project = (wandb.run.project if wandb.run is not None else None) or config.project_name
    if entity is None or project is None:
        logging.warning("Skipping source train summary sync because entity/project could not be resolved.")
        return

    try:
        api = wandb.Api()
        source_run = api.run(f"{entity}/{project}/{source_train_run_id}")
        for key, value in metrics.items():
            if value is not None:
                source_run.summary[f"{metric_prefix}/{key}"] = value
        source_run.summary[f"{metric_prefix}/checkpoint_alias"] = checkpoint_alias
        source_run.summary[f"{metric_prefix}/latest_eval_run_id"] = eval_run_id
        source_run.summary[f"{metric_prefix}/latest_eval_run_name"] = eval_run_name
        source_run.summary[f"{metric_prefix}/results_artifact"] = results_artifact_ref
        source_run.summary[f"{metric_prefix}/video_artifact"] = media_artifact_ref
        try:
            source_run.summary.update()
        except TypeError:
            pass
        source_run.update()
    except Exception as exc:  # noqa: BLE001
        logging.warning("Failed to sync %s summary back to train run %s: %s", metric_prefix, source_train_run_id, exc)


def _example_table_logger(key: str = "examples") -> WandbTableLogger[ExampleRecord]:
    return WandbTableLogger(key=key, columns=EXAMPLE_COLUMNS)


def _configure_eval_metric_axes(metric_prefix: str) -> None:
    checkpoint_metric = f"{metric_prefix}/checkpoint_step"
    wandb.define_metric(checkpoint_metric)
    for metric_name in ("primary_score", "success_rate", "mean_max_reward", "num_examples"):
        wandb.define_metric(f"{metric_prefix}/{metric_name}", step_metric=checkpoint_metric)


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, force=True)

    manifest = _load_manifest(args.manifest)
    split_name = args.split or manifest.split_name
    metric_prefix = _eval_tracking.metric_namespace_for_split(split_name)

    config = _config.get_config(args.config_name)
    checkpoint_step = args.checkpoint_step if args.checkpoint_step is not None else _infer_checkpoint_step(args.checkpoint_dir)
    checkpoint_alias = args.checkpoint_alias or (f"step-{checkpoint_step}" if checkpoint_step is not None else "manual")
    source_train_run_name = _infer_source_train_run_name(args.checkpoint_dir)
    source_train_run_id = _read_source_train_run_id(args.checkpoint_dir)
    attach_to_source_train_run = args.attach_to_source_train_run and _should_attach_to_source_train_run(
        args.checkpoint_dir, source_train_run_id
    )
    run_name = args.run_name or _eval_tracking.run_name_for_split(
        source_train_run_name,
        split_name,
        checkpoint_step=checkpoint_step,
        checkpoint_alias=checkpoint_alias,
    )

    video_dir = args.video_dir.resolve()
    video_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.results_path.resolve() if args.results_path is not None else video_dir / _eval_tracking.results_filename_for_split(split_name)
    eval_run_id_path = _resolve_eval_run_id_path(
        args.checkpoint_dir,
        results_path,
        split_name,
        attach_to_source_train_run=attach_to_source_train_run,
    )
    artifact_stem = f"{_slugify(args.config_name)}-{_slugify(run_name)}"
    results_artifact_name = f"{artifact_stem}-{metric_prefix}-results"
    media_artifact_name = f"{artifact_stem}-{metric_prefix}-videos"
    artifact_aliases = tuple(alias for alias in (checkpoint_alias, split_name, "latest") if alias)
    results_artifact_ref = build_artifact_ref(results_artifact_name, (checkpoint_alias,))
    media_artifact_ref = build_artifact_ref(media_artifact_name, (checkpoint_alias,))

    if attach_to_source_train_run:
        wandb_run = WandbRunContext.resume_existing_run(
            config,
            backend="aloha_sim_eval",
            run_id_path=eval_run_id_path,
            run_name=source_train_run_name or run_name,
            enabled=args.wandb_enabled,
        )
    else:
        wandb_run = WandbRunContext.init_for_evaluation(
            config,
            backend="aloha_sim_eval",
            run_name=run_name,
            run_group=_eval_tracking.run_group_for_split(split_name),
            job_type=_eval_tracking.job_type_for_split(split_name),
            run_id_path=eval_run_id_path,
            resuming=eval_run_id_path.exists(),
            enabled=args.wandb_enabled,
            extra_config={
                "manifest": str(args.manifest),
                "checkpoint_dir": args.checkpoint_dir,
                "checkpoint_step": checkpoint_step,
                "checkpoint_alias": checkpoint_alias,
                "task": args.task,
                "metric_prefix": metric_prefix,
                "visualization_width": args.visualization_width,
                "visualization_height": args.visualization_height,
                "visualization_camera_id": args.visualization_camera_id,
                "render_mode": args.render_mode,
                "attach_to_source_train_run": attach_to_source_train_run,
            },
        )
    if args.wandb_enabled and attach_to_source_train_run:
        _configure_eval_metric_axes(metric_prefix)
    artifact_manager = WandbArtifactManager()
    video_logger = VideoLogger()
    leaderboard_logger = LeaderboardTableLogger(key=f"{metric_prefix}/leaderboard")
    example_logger = _example_table_logger(f"{metric_prefix}/examples")
    log_step = None if attach_to_source_train_run else checkpoint_step

    policy = _policy_config.create_trained_policy(config, args.checkpoint_dir, default_prompt=manifest.prompt)
    broker = action_chunk_broker.ActionChunkBroker(policy=policy, action_horizon=args.action_horizon)

    eval_records = _build_eval_records(manifest, args.num_examples)
    results: list[EpisodeResult] = []
    video_records: list[VideoRecord] = []
    example_rows: list[ExampleRecord] = []

    for index, record in enumerate(eval_records):
        seed = _resolve_seed(manifest, record, index)
        prompt = record.prompt or manifest.prompt
        logging.info("Starting eval example %s (seed=%s)", record.example_id, seed)
        env = _aloha_env.AlohaSimEnvironment(
            task=args.task,
            seed=seed,
            render_mode=args.render_mode,
            visualization_width=args.visualization_width,
            visualization_height=args.visualization_height,
            visualization_camera_id=args.visualization_camera_id,
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

            if env.is_episode_complete() or (args.max_episode_steps > 0 and num_steps >= args.max_episode_steps):
                break

        success = env.is_success(args.success_reward_threshold)
        video_path = video_dir / f"{index:03d}_{_slugify(record.example_id)}_{'success' if success else 'failure'}.mp4"
        _save_video(video_path, frames, fps=args.fps)

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
        video_records.append(
            VideoRecord(
                path=str(video_path),
                name=record.example_id,
                caption=(
                    f"step={checkpoint_step} | {record.example_id} | "
                    f"success={int(success)} | reward={env.episode_reward:.2f}"
                ),
                fps=args.fps,
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
                    caption=(
                        f"step={checkpoint_step} | {record.example_id} | "
                        f"success={int(success)} | reward={env.episode_reward:.2f}"
                    ),
                    fps=args.fps,
                    format="mp4",
                ),
                artifact_ref_video=f"{media_artifact_ref}/{video_path.name}",
                metadata_json=json.dumps(record.metadata, sort_keys=True),
            )
        )

    success_rate = float(np.mean([result.success for result in results])) if results else 0.0
    mean_max_reward = float(np.mean([result.max_reward for result in results])) if results else 0.0
    metrics = {
        "primary_score": success_rate,
        "success_rate": success_rate,
        "mean_max_reward": mean_max_reward,
        "num_examples": len(results),
        "checkpoint_step": checkpoint_step,
    }
    if args.wandb_enabled:
        wandb_run.log_metrics(metrics, step=log_step, prefix=metric_prefix)
        wandb_run.log_summary({f"{metric_prefix}/{key}": value for key, value in metrics.items() if value is not None})
        wandb_run.log_summary(
            {
                f"{metric_prefix}/split_name": split_name,
                f"{metric_prefix}/checkpoint_alias": checkpoint_alias,
                f"{metric_prefix}/source_train_run_id": source_train_run_id,
                f"{metric_prefix}/source_train_run_name": source_train_run_name,
            }
        )

        video_logger.log_video_files(f"{metric_prefix}/videos", video_records, step=log_step)
        example_logger.log_immutable(example_rows, step=log_step)

    results_payload = {
        "config_name": args.config_name,
        "checkpoint_dir": args.checkpoint_dir,
        "checkpoint_step": checkpoint_step,
        "checkpoint_alias": checkpoint_alias,
        "manifest": str(args.manifest),
        "split_name": split_name,
        "metrics": metrics,
        "results": [result.as_dict() for result in results],
    }
    _write_results_json(results_path, results_payload)
    _write_results_csv(results_path, results)

    artifact_metadata = {
        "config_name": args.config_name,
        "task_name": manifest.task_name,
        "dataset_name": manifest.dataset_name,
        "split_name": split_name,
        "checkpoint_step": checkpoint_step,
        "checkpoint_alias": checkpoint_alias,
        "source_train_run_id": source_train_run_id,
        "source_train_run_name": source_train_run_name,
    }
    if args.wandb_enabled:
        artifact_manager.log_artifact(
            ArtifactRecord(
                name=results_artifact_name,
                type="eval-results",
                path=str(results_path),
                aliases=artifact_aliases,
                description=f"ALOHA Sim {split_name} evaluation results",
                metadata=artifact_metadata,
            )
        )
        artifact_manager.log_artifact(
            ArtifactRecord(
                name=media_artifact_name,
                type="eval-video-bundle",
                path=str(video_dir),
                aliases=artifact_aliases,
                description=f"ALOHA Sim {split_name} evaluation videos",
                metadata=artifact_metadata,
            )
        )
        wandb_run.log_summary(
            {
                f"{metric_prefix}/results_artifact": results_artifact_ref,
                f"{metric_prefix}/video_artifact": media_artifact_ref,
            }
        )

        leaderboard_logger.log_row(
            LeaderboardRow(
                eval_run_id=wandb.run.id if wandb.run is not None else run_name,
                source_train_run_id=source_train_run_id,
                source_train_run_name=source_train_run_name,
                eval_name=manifest.name,
                eval_split=split_name,
                model_family=getattr(config.model, "model_type", type(config.model).__name__),
                config_name=args.config_name,
                task_name=manifest.task_name,
                dataset_name=manifest.dataset_name,
                checkpoint_alias=checkpoint_alias,
                checkpoint_step=checkpoint_step,
                primary_score=success_rate,
                success_rate=success_rate,
                mean_max_reward=mean_max_reward,
                num_examples=len(results),
                artifact_ref_results=results_artifact_ref,
                artifact_ref_media=media_artifact_ref,
                created_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                notes=f"success_reward_threshold={args.success_reward_threshold}",
            ),
            step=log_step,
        )

    if args.wandb_enabled and not attach_to_source_train_run:
        _sync_source_train_summary(
            config=config,
            source_train_run_id=source_train_run_id,
            metric_prefix=metric_prefix,
            metrics=metrics,
            checkpoint_alias=checkpoint_alias,
            eval_run_id=wandb.run.id if wandb.run is not None else None,
            eval_run_name=run_name,
            results_artifact_ref=results_artifact_ref,
            media_artifact_ref=media_artifact_ref,
        )

    wandb_run.finish()


if __name__ == "__main__":
    main(tyro.cli(Args))
