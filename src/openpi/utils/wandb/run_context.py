from __future__ import annotations

import dataclasses
import os
import pathlib
import platform
import re
import socket
import subprocess
from typing import Any

import wandb

import openpi.training.config as _config

RUN_ID_FILENAME = "wandb_id.txt"
REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]


def _git_output(*args: str) -> str | None:
    try:
        output = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), *args],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return output or None


def _dedupe(items: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return tuple(result)


def _parse_env_tags(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return _dedupe([tag.strip() for tag in re.split(r"[\n,]", value) if tag.strip()])


def build_run_tags(config: _config.TrainConfig, backend: str) -> tuple[str, ...]:
    del backend
    return _dedupe([*config.wandb_tags, *_parse_env_tags(os.environ.get("WANDB_TAGS"))])


def build_run_metadata(config: _config.TrainConfig, backend: str) -> dict[str, Any]:
    return {
        "backend": backend,
        "config_name": config.name,
        "task_name": config.task_name,
        "dataset_name": config.dataset_name,
        "eval_manifest_path": config.eval_manifest_path,
        "final_eval_manifest_path": config.final_eval_manifest_path,
        "baseline_checkpoint_ref": config.baseline_checkpoint_ref,
        "git_sha": _git_output("rev-parse", "HEAD"),
        "git_branch": _git_output("branch", "--show-current"),
        "hostname": socket.gethostname(),
        "platform_node": platform.node(),
        "cluster_name": os.environ.get("SLURM_CLUSTER_NAME") or os.environ.get("CLUSTER_NAME"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
        "slurm_node_name": os.environ.get("SLURMD_NODENAME"),
        "gpu_type": os.environ.get("GPU_TYPE") or os.environ.get("NVIDIA_PRODUCT_NAME"),
    }


def build_run_config(config: _config.TrainConfig, backend: str) -> dict[str, Any]:
    payload = dataclasses.asdict(config)
    payload["training_backend"] = backend
    payload["wandb_tracking"] = build_run_metadata(config, backend)
    return payload


def _resolve_project_name(config: _config.TrainConfig) -> str:
    return os.environ.get("WANDB_PROJECT") or config.project_name


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


@dataclasses.dataclass
class WandbRunContext:
    config_name: str
    backend: str
    enabled: bool
    run_id_path: pathlib.Path

    @classmethod
    def init_for_training(
        cls,
        config: _config.TrainConfig,
        *,
        backend: str,
        resuming: bool,
        resume_mode: str = "must",
        enabled: bool = True,
        log_code_root: pathlib.Path | None = None,
    ) -> WandbRunContext:
        run_id_path = config.checkpoint_dir / RUN_ID_FILENAME

        init_kwargs = {
            "project": _resolve_project_name(config),
            "entity": config.wandb_entity or os.environ.get("WANDB_ENTITY"),
            "name": config.exp_name,
            "group": config.wandb_group or os.environ.get("WANDB_GROUP"),
            "job_type": config.wandb_job_type or os.environ.get("WANDB_JOB_TYPE") or "train",
            "tags": list(build_run_tags(config, backend)),
        }

        return cls._init(
            config_name=config.name,
            backend=backend,
            enabled=enabled,
            run_id_path=run_id_path,
            resuming=resuming,
            resume_mode=resume_mode,
            init_kwargs=init_kwargs,
            run_config=build_run_config(config, backend),
            log_code_root=log_code_root,
            summary={
                "run/backend": backend,
                "run/config_name": config.name,
                "run/task_name": config.task_name,
                "run/dataset_name": config.dataset_name,
                "run/job_type": init_kwargs["job_type"],
                "run/group": init_kwargs["group"],
            },
        )

    @classmethod
    def init_for_evaluation(
        cls,
        config: _config.TrainConfig,
        *,
        backend: str,
        run_name: str,
        run_group: str,
        job_type: str,
        run_id_path: pathlib.Path,
        resuming: bool = False,
        resume_mode: str = "allow",
        enabled: bool = True,
        extra_config: dict[str, Any] | None = None,
    ) -> WandbRunContext:
        run_config = build_run_config(config, backend)
        if extra_config:
            run_config["evaluation"] = extra_config

        return cls._init(
            config_name=config.name,
            backend=backend,
            enabled=enabled,
            run_id_path=run_id_path,
            resuming=resuming,
            resume_mode=resume_mode,
            init_kwargs={
                "project": _resolve_project_name(config),
                "entity": config.wandb_entity or os.environ.get("WANDB_ENTITY"),
                "name": run_name,
                "group": run_group,
                "job_type": job_type,
                "tags": list(build_run_tags(config, backend)),
            },
            run_config=run_config,
            summary={
                "run/backend": backend,
                "run/config_name": config.name,
                "run/task_name": config.task_name,
                "run/dataset_name": config.dataset_name,
                "run/job_type": job_type,
                "run/group": run_group,
            },
        )

    @classmethod
    def resume_existing_run(
        cls,
        config: _config.TrainConfig,
        *,
        backend: str,
        run_id_path: pathlib.Path | None = None,
        run_name: str | None = None,
        run_group: str | None = None,
        job_type: str | None = None,
        enabled: bool = True,
    ) -> WandbRunContext:
        resolved_run_id_path = run_id_path or (config.checkpoint_dir / RUN_ID_FILENAME)
        return cls._init(
            config_name=config.name,
            backend=backend,
            enabled=enabled,
            run_id_path=resolved_run_id_path,
            resuming=True,
            resume_mode="allow",
            init_kwargs={
                "project": _resolve_project_name(config),
                "entity": config.wandb_entity or os.environ.get("WANDB_ENTITY"),
                "name": run_name or _optional_string(getattr(config, "exp_name", None)),
                "group": run_group or _optional_string(config.wandb_group) or os.environ.get("WANDB_GROUP"),
                "job_type": job_type or _optional_string(config.wandb_job_type) or os.environ.get("WANDB_JOB_TYPE") or "train",
                "tags": list(build_run_tags(config, backend)),
            },
            summary={},
        )

    @classmethod
    def _init(
        cls,
        *,
        config_name: str,
        backend: str,
        enabled: bool,
        run_id_path: pathlib.Path,
        resuming: bool,
        resume_mode: str,
        init_kwargs: dict[str, Any],
        run_config: dict[str, Any] | None = None,
        log_code_root: pathlib.Path | None = None,
        summary: dict[str, Any] | None = None,
    ) -> WandbRunContext:
        run_id_path.parent.mkdir(parents=True, exist_ok=True)

        if not enabled:
            wandb.init(mode="disabled")
            return cls(config_name=config_name, backend=backend, enabled=False, run_id_path=run_id_path)

        if resuming:
            if not run_id_path.exists():
                raise FileNotFoundError(f"W&B run id file not found at {run_id_path}")
            init_kwargs["id"] = run_id_path.read_text().strip()
            init_kwargs["resume"] = resume_mode
        elif run_config is not None:
            init_kwargs["config"] = run_config

        wandb.init(**{key: value for key, value in init_kwargs.items() if value not in (None, [], ())})
        if wandb.run is None:
            raise RuntimeError("wandb.init() did not create an active run")

        if not resuming:
            run_id_path.write_text(wandb.run.id)

        if log_code_root is not None:
            wandb.run.log_code(str(log_code_root))

        context = cls(config_name=config_name, backend=backend, enabled=True, run_id_path=run_id_path)
        context.log_summary(summary or {})
        return context

    def log(self, payload: dict[str, Any], *, step: int | None = None) -> None:
        if payload:
            wandb.log(payload, step=step)

    def log_metrics(self, metrics: dict[str, Any], *, step: int | None = None, prefix: str | None = None) -> None:
        payload = {}
        for key, value in metrics.items():
            if value is None:
                continue
            metric_key = f"{prefix}/{key}" if prefix else key
            payload[metric_key] = value
        self.log(payload, step=step)

    def log_summary(self, metrics: dict[str, Any]) -> None:
        if wandb.run is None:
            return
        for key, value in metrics.items():
            if value is not None:
                wandb.run.summary[key] = value

    def finish(self) -> None:
        if self.enabled and wandb.run is not None:
            wandb.finish()
