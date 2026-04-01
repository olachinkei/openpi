from __future__ import annotations

import dataclasses
import json
import logging
import os
import pathlib
from typing import Any

import wandb

import openpi.training.config as _config
import openpi.training.eval_manifest as _eval_manifest
from openpi.utils.wandb import ArtifactRecord
from openpi.utils.wandb import WandbArtifactManager
import openpi.utils.wandb.run_context as _run_context
from wandb.errors import CommError


logger = logging.getLogger(__name__)


def _use_wandb_artifact(artifact_ref: str) -> "wandb.sdk.artifacts.artifact.Artifact | None":
    if wandb.run is None or not artifact_ref:
        return None
    try:
        return wandb.run.use_artifact(artifact_ref)
    except (CommError, ValueError) as exc:
        logger.warning("Unable to resolve WandB artifact %s: %s", artifact_ref, exc)
        return None


def _dataset_url(repo_id: str | None) -> str | None:
    if not repo_id or "/" not in repo_id or "://" in repo_id:
        return None
    return f"https://huggingface.co/datasets/{repo_id}"


def _artifact_root(config: _config.TrainConfig) -> pathlib.Path:
    root = config.checkpoint_dir / "_wandb_inputs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def configured_dataset_artifact_refs(config: _config.TrainConfig) -> dict[str, str]:
    refs: dict[str, str] = {}
    if config.train_dataset_artifact_ref:
        refs["train"] = config.train_dataset_artifact_ref
    if config.eval_dataset_artifact_ref:
        refs["eval"] = config.eval_dataset_artifact_ref
    if config.eval_final_dataset_artifact_ref:
        refs["eval_final"] = config.eval_final_dataset_artifact_ref
    return refs


def _download_artifact_payload(artifact_ref: str, target_dir: pathlib.Path) -> pathlib.Path | None:
    if wandb.run is None:
        return None

    artifact = _use_wandb_artifact(artifact_ref)
    if artifact is None:
        return None
    download_root = pathlib.Path(artifact.download(root=str(target_dir)))
    json_paths = sorted(download_root.rglob("*.json"))
    if not json_paths:
        return None
    return json_paths[0]


def bind_dataset_artifact_payloads(
    config: _config.TrainConfig,
    refs: dict[str, str],
) -> tuple[_config.TrainConfig, dict[str, str]]:
    if not refs:
        return config, refs

    artifact_root = _artifact_root(config) / "resolved"
    artifact_root.mkdir(parents=True, exist_ok=True)

    updated_config = config

    if train_ref := refs.get("train"):
        _use_wandb_artifact(train_ref)

    if eval_ref := refs.get("eval"):
        resolved_eval_manifest = _download_artifact_payload(eval_ref, artifact_root / "eval")
        if resolved_eval_manifest is not None:
            updated_config = dataclasses.replace(updated_config, eval_manifest_path=str(resolved_eval_manifest))

    resolved_final_manifest: pathlib.Path | None = None
    if eval_final_ref := refs.get("eval_final"):
        resolved_final_manifest = _download_artifact_payload(eval_final_ref, artifact_root / "eval_final")
        if resolved_final_manifest is not None:
            updated_config = dataclasses.replace(updated_config, final_eval_manifest_path=str(resolved_final_manifest))

    if resolved_final_manifest is not None and isinstance(updated_config.data, _config.LeRobotAlohaDataConfig):
        updated_config = dataclasses.replace(
            updated_config,
            data=dataclasses.replace(
                updated_config.data,
                exclude_episodes_manifest_path=str(resolved_final_manifest),
            ),
        )

    return updated_config, refs


def _load_manifest_payload(path_str: str | None) -> tuple[pathlib.Path | None, _eval_manifest.EvalManifest | None, dict[str, Any] | None]:
    path = _eval_manifest.resolve_repo_path(path_str)
    if path is None or not path.exists():
        return None, None, None
    payload = json.loads(path.read_text())
    payload["resolved_path"] = str(path)
    return path, _eval_manifest.load_eval_manifest(path), payload


def _write_json(path: pathlib.Path, payload: dict[str, Any]) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def _build_train_dataset_payload(
    config: _config.TrainConfig,
    *,
    eval_manifest: _eval_manifest.EvalManifest | None,
    final_manifest: _eval_manifest.EvalManifest | None,
) -> dict[str, Any]:
    data_factory_payload = dataclasses.asdict(config.data)
    held_out_records = final_manifest.records if final_manifest is not None else ()
    held_out_episode_indices = [record.hf_episode_index for record in held_out_records if record.hf_episode_index is not None]
    return {
        "config_name": config.name,
        "exp_name": config.exp_name,
        "task_name": config.task_name,
        "dataset_name": config.dataset_name,
        "source_dataset_repo_id": getattr(config.data, "repo_id", None),
        "source_dataset_url": _dataset_url(getattr(config.data, "repo_id", None)),
        "data_factory": data_factory_payload,
        "train_split_policy": "dataset_minus_final_eval_manifest",
        "periodic_eval_manifest_path": config.eval_manifest_path,
        "final_eval_manifest_path": config.final_eval_manifest_path,
        "held_out_example_count": len(held_out_records),
        "held_out_example_ids": [record.example_id for record in held_out_records],
        "held_out_episode_indices": held_out_episode_indices,
        "periodic_eval_example_count": 0 if eval_manifest is None else len(eval_manifest.records),
    }


def register_dataset_artifacts(
    config: _config.TrainConfig,
    artifact_manager: WandbArtifactManager,
) -> dict[str, str]:
    artifact_root = _artifact_root(config)
    refs: dict[str, str] = {}

    eval_manifest_path, eval_manifest, eval_payload = _load_manifest_payload(config.eval_manifest_path)
    final_manifest_path, final_manifest, final_payload = _load_manifest_payload(config.final_eval_manifest_path)

    train_payload = _build_train_dataset_payload(config, eval_manifest=eval_manifest, final_manifest=final_manifest)
    train_payload_path = _write_json(artifact_root / "train_dataset_spec.json", train_payload)
    train_artifact_name = f"{config.name}-{config.exp_name}-train-dataset"
    train_ref = artifact_manager.log_artifact(
        ArtifactRecord(
            name=train_artifact_name,
            type="dataset",
            path=str(train_payload_path),
            aliases=("train", "latest"),
            description="Training dataset specification and held-out split metadata",
            metadata={
                "config_name": config.name,
                "task_name": config.task_name,
                "dataset_name": config.dataset_name,
                "source_dataset_repo_id": getattr(config.data, "repo_id", None),
                "held_out_example_count": train_payload["held_out_example_count"],
            },
        )
    )
    if train_ref is not None:
        refs["train"] = train_ref

    if eval_manifest_path is not None and eval_manifest is not None and eval_payload is not None:
        eval_artifact_name = f"{config.name}-{config.exp_name}-eval-dataset"
        eval_ref = artifact_manager.log_artifact(
            ArtifactRecord(
                name=eval_artifact_name,
                type="dataset",
                path=str(eval_manifest_path),
                aliases=("eval", "latest"),
                description="Periodic evaluation manifest",
                metadata={
                    "config_name": config.name,
                    "task_name": eval_manifest.task_name,
                    "dataset_name": eval_manifest.dataset_name,
                    "split_name": eval_manifest.split_name,
                    "num_examples": len(eval_manifest.records),
                    "selection_policy": eval_payload["selection"]["policy"],
                },
            )
        )
        if eval_ref is not None:
            refs["eval"] = eval_ref

    if final_manifest_path is not None and final_manifest is not None and final_payload is not None:
        final_artifact_name = f"{config.name}-{config.exp_name}-eval-final-dataset"
        final_ref = artifact_manager.log_artifact(
            ArtifactRecord(
                name=final_artifact_name,
                type="dataset",
                path=str(final_manifest_path),
                aliases=("eval_final", "latest"),
                description="Final held-out evaluation manifest",
                metadata={
                    "config_name": config.name,
                    "task_name": final_manifest.task_name,
                    "dataset_name": final_manifest.dataset_name,
                    "split_name": final_manifest.split_name,
                    "num_examples": len(final_manifest.records),
                    "selection_policy": final_payload["selection"]["policy"],
                },
            )
        )
        if final_ref is not None:
            refs["eval_final"] = final_ref

    return refs


def publish_dataset_artifacts(
    config: _config.TrainConfig,
    *,
    backend: str,
    enabled: bool,
) -> dict[str, str]:
    if not enabled:
        return {}

    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT") or config.project_name,
        entity=config.wandb_entity or os.environ.get("WANDB_ENTITY"),
        name=f"{config.exp_name}-inputs",
        group="inputs",
        job_type="dataset_registry",
        tags=list(_run_context.build_run_tags(config, backend)),
        config=_run_context.build_run_config(config, backend),
    )
    if run is None:
        return {}

    try:
        run.summary["run/backend"] = backend
        run.summary["run/config_name"] = config.name
        run.summary["run/task_name"] = config.task_name
        run.summary["run/dataset_name"] = config.dataset_name
        run.summary["run/group"] = "inputs"
        run.summary["run/job_type"] = "dataset_registry"
        refs = register_dataset_artifacts(config, WandbArtifactManager())
        run.summary["data/train_dataset_artifact"] = refs.get("train")
        run.summary["data/eval_dataset_artifact"] = refs.get("eval")
        run.summary["data/eval_final_dataset_artifact"] = refs.get("eval_final")
        return refs
    finally:
        wandb.finish()
