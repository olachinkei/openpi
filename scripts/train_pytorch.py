"""
PyTorch training entrypoint for PI0/PI05 with multi-GPU and multi-node (DDP) support.
This script mirrors the behavior of the JAX trainer (`scripts/train.py`) but runs
entirely in PyTorch using the `PI0Pytorch` model and your existing config/data
pipeline from `src/openpi/training/config.py` and `src/openpi/training/data_loader.py`.

Usage
Single GPU:
  python scripts/train_pytorch.py <config_name> --exp_name <run_name> --save_interval <interval>
  Example:
  python scripts/train_pytorch.py debug --exp_name pytorch_ddp_test
  python scripts/train_pytorch.py debug --exp_name pytorch_ddp_test --resume  # Resume from latest checkpoint
Multi-GPU (single node):
  torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> scripts/train_pytorch.py <config_name> --exp_name <run_name>
  Example:
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test --resume
Multi-Node Training:
	torchrun \
    --nnodes=<num_nodes> --nproc_per_node=<gpus_per_node> --node_rank=<rank_of_node> \
    --master_addr=<master_ip> --master_port=<port> \
    scripts/train_pytorch.py <config_name> --exp_name=<run_name> --save_interval <interval>

"""

import dataclasses
import datetime
import gc
import json
import logging
import os
import pathlib
import platform
import shutil
import time
import traceback

import jax
import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
import torch.nn.parallel
import tqdm
import wandb

import openpi.models.pi0_config
import openpi.models_pytorch.pi0_pytorch
import openpi.policies.policy_config as _policy_config
import openpi.shared.normalize as _normalize
import openpi.training.aloha_eval as _aloha_eval
import openpi.training.config as _config
import openpi.training.data_loader as _data
import openpi.training.dataset_artifacts as _dataset_artifacts
import openpi.training.eval_manifest as _eval_manifest
import openpi.training.eval_submission as _eval_submission
import openpi.training.eval_tracking as _eval_tracking
import openpi.utils.wandb.artifacts as _wandb_artifacts
import openpi.utils.wandb.leaderboard as _wandb_leaderboard
import openpi.utils.wandb.run_context as _wandb_run_context
import openpi.utils.wandb.tables as _wandb_tables
import openpi.utils.wandb.types as _wandb_types
import openpi.utils.wandb.videos as _wandb_videos


@dataclasses.dataclass(frozen=True)
class PendingEvalResult:
    split_name: str
    results_path: pathlib.Path
    history_step: int | None = None


def init_logging():
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    """Initialize wandb logging."""
    log_code_root = pathlib.Path(__file__).resolve().parent.parent if log_code else None
    return _wandb_run_context.WandbRunContext.init_for_training(
        config,
        backend="pytorch",
        resuming=resuming,
        enabled=enabled,
        log_code_root=log_code_root,
    )


def _configure_eval_metric_axes(config: _config.TrainConfig) -> None:
    for split_name in {config.eval_split_name, config.final_eval_split_name}:
        metric_prefix = _eval_tracking.metric_namespace_for_split(split_name)
        checkpoint_metric = f"{metric_prefix}/checkpoint_step"
        wandb.define_metric(checkpoint_metric)
        for metric_name in ("primary_score", "success_rate", "mean_max_reward", "num_examples"):
            wandb.define_metric(f"{metric_prefix}/{metric_name}", step_metric=checkpoint_metric)


def _results_import_marker(results_path: pathlib.Path) -> pathlib.Path:
    return results_path.with_name(f".{results_path.stem}.imported_to_train")


def _import_eval_results_to_train_run(
    wandb_run: _wandb_run_context.WandbRunContext,
    *,
    split_name: str,
    results_path: pathlib.Path,
    history_step: int | None = None,
) -> bool:
    marker_path = _results_import_marker(results_path)
    if marker_path.exists() or not results_path.exists():
        return False

    payload = json.loads(results_path.read_text())
    metric_prefix = _eval_tracking.metric_namespace_for_split(split_name)
    metrics = payload.get("metrics", {})
    checkpoint_step = metrics.get("checkpoint_step")

    if metrics:
        wandb_run.log_metrics(metrics, step=history_step, prefix=metric_prefix)
        wandb_run.log_summary({f"{metric_prefix}/{key}": value for key, value in metrics.items() if value is not None})

    video_records = []
    for result in payload.get("results", []):
        video_path = result.get("video_path")
        if not video_path:
            continue
        example_id = result.get("example_id", pathlib.Path(video_path).stem)
        success = int(bool(result.get("success", False)))
        reward = float(result.get("max_reward", 0.0))
        video_records.append(
            _wandb_types.VideoRecord(
                path=video_path,
                name=example_id,
                caption=f"step={checkpoint_step} | {example_id} | success={success} | reward={reward:.2f}",
            )
        )
    if video_records:
        _wandb_videos.VideoLogger().log_video_files(
            f"{metric_prefix}/videos",
            video_records,
            step=history_step,
        )

    marker_path.write_text(json.dumps({"imported_at": int(time.time()), "split_name": split_name}))
    logging.info("Imported %s results into train run from %s", split_name, results_path)
    return True


def _flush_available_eval_results(
    wandb_run: _wandb_run_context.WandbRunContext,
    pending_eval_results: list[PendingEvalResult],
) -> list[PendingEvalResult]:
    remaining = []
    for pending_result in pending_eval_results:
        if not _import_eval_results_to_train_run(
            wandb_run,
            split_name=pending_result.split_name,
            results_path=pending_result.results_path,
            history_step=pending_result.history_step,
        ):
            remaining.append(pending_result)
    return remaining


def _wait_for_eval_results(
    wandb_run: _wandb_run_context.WandbRunContext,
    pending_eval_results: list[PendingEvalResult],
    *,
    poll_interval_secs: int,
) -> list[PendingEvalResult]:
    remaining = pending_eval_results
    while remaining:
        remaining = _flush_available_eval_results(wandb_run, remaining)
        if remaining:
            time.sleep(poll_interval_secs)
    return remaining


def setup_ddp():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    if use_ddp and not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        timeout_secs = int(os.environ.get("TORCH_DISTRIBUTED_TIMEOUT_SECS", "7200"))
        init_kwargs = {
            "backend": backend,
            "init_method": "env://",
            "timeout": datetime.timedelta(seconds=timeout_secs),
        }
        if backend == "nccl" and device.type == "cuda":
            init_kwargs["device_id"] = device
        torch.distributed.init_process_group(**init_kwargs)

        # Set up debugging environment variables for DDP issues.
        if os.environ.get("TORCH_DISTRIBUTED_DEBUG") is None:
            os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"

    return use_ddp, local_rank, device


def cleanup_ddp(device: torch.device | None = None):
    if torch.distributed.is_initialized():
        if device is not None and device.type == "cuda":
            torch.distributed.barrier(device_ids=[device.index])
        else:
            torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def set_seed(seed: int, local_rank: int):
    torch.manual_seed(seed + local_rank)
    np.random.seed(seed + local_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + local_rank)


def build_datasets(config: _config.TrainConfig):
    # Use the unified data loader with PyTorch framework
    data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=True)
    return data_loader, data_loader.data_config()


def get_model_state_dict(model):
    """Get state dict from model, handling DDP wrapper."""
    return (
        model.module.state_dict()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.state_dict()
    )


def get_model_parameters(model):
    """Get parameters from model, handling DDP wrapper."""
    return (
        model.module.parameters()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.parameters()
    )


def get_unwrapped_model(model):
    """Get the underlying model, removing any DDP wrapper."""
    return model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model


def _ddp_barrier(use_ddp: bool, device: torch.device | None = None) -> None:
    if use_ddp and dist.is_initialized():
        if device is not None and device.type == "cuda":
            dist.barrier(device_ids=[device.index])
        else:
            dist.barrier()


def _main_rank_sync_marker(config: _config.TrainConfig, *, step: int, phase: str) -> pathlib.Path:
    return pathlib.Path(config.checkpoint_dir) / f".sync-{phase}-{step}.json"


def _write_main_rank_sync_marker(
    marker_path: pathlib.Path,
    *,
    ok: bool,
    phase: str,
    step: int,
    error: str | None = None,
    traceback_text: str | None = None,
) -> None:
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ok": ok,
        "phase": phase,
        "step": step,
        "timestamp": time.time(),
    }
    if error is not None:
        payload["error"] = error
    if traceback_text is not None:
        payload["traceback"] = traceback_text
    marker_path.write_text(json.dumps(payload))


def _wait_for_main_rank_sync_marker(
    marker_path: pathlib.Path,
    *,
    phase: str,
    step: int,
    poll_interval_secs: int,
    timeout_secs: int = 4 * 60 * 60,
) -> None:
    deadline = time.time() + timeout_secs
    while not marker_path.exists():
        if time.time() > deadline:
            raise TimeoutError(f"Timed out waiting for main-rank {phase} completion at step {step}: {marker_path}")
        time.sleep(poll_interval_secs)

    payload = json.loads(marker_path.read_text())
    if not payload.get("ok", False):
        error_message = payload.get("error", f"Main-rank {phase} failed at step {step}")
        traceback_text = payload.get("traceback")
        if traceback_text:
            raise RuntimeError(f"{error_message}\n{traceback_text}")
        raise RuntimeError(error_message)


def _run_main_rank_only_phase(
    *,
    config: _config.TrainConfig,
    use_ddp: bool,
    is_main: bool,
    step: int,
    phase: str,
    action,
) -> None:
    marker_path = _main_rank_sync_marker(config, step=step, phase=phase)
    if is_main and marker_path.exists():
        marker_path.unlink()

    if not use_ddp:
        action()
        return

    if is_main:
        try:
            action()
        except Exception as exc:
            _write_main_rank_sync_marker(
                marker_path,
                ok=False,
                phase=phase,
                step=step,
                error=f"{type(exc).__name__}: {exc}",
                traceback_text=traceback.format_exc(),
            )
            raise
        else:
            _write_main_rank_sync_marker(marker_path, ok=True, phase=phase, step=step)
    else:
        _wait_for_main_rank_sync_marker(
            marker_path,
            phase=phase,
            step=step,
            poll_interval_secs=config.eval_results_poll_interval_secs,
        )


def save_checkpoint(
    model,
    optimizer,
    global_step,
    config,
    is_main,
    data_config,
    artifact_manager=None,
    wandb_run=None,
    dataset_artifact_refs=None,
):
    """Save a checkpoint with model state, optimizer state, and metadata."""
    if not is_main:
        return False

    # Only save if it's time to save or if it's the final step
    if (global_step % config.save_interval == 0 and global_step > 0) or global_step == config.num_train_steps:
        # Create temporary directory for atomic checkpoint saving
        final_ckpt_dir = config.checkpoint_dir / f"{global_step}"
        tmp_ckpt_dir = config.checkpoint_dir / f"tmp_{global_step}"

        # Remove any existing temp directory and create new one
        if tmp_ckpt_dir.exists():
            shutil.rmtree(tmp_ckpt_dir)
        tmp_ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save model state using safetensors (handle shared tensors)
        model_to_save = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        safetensors.torch.save_model(model_to_save, tmp_ckpt_dir / "model.safetensors")

        # Save optimizer state using PyTorch format
        torch.save(optimizer.state_dict(), tmp_ckpt_dir / "optimizer.pt")

        # Save training metadata (avoid saving full config to prevent JAX/Flax compatibility issues)
        metadata = {
            "global_step": global_step,
            "config": dataclasses.asdict(config),
            "timestamp": time.time(),
        }
        torch.save(metadata, tmp_ckpt_dir / "metadata.pt")

        # save norm stats
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(tmp_ckpt_dir / "assets" / data_config.asset_id, norm_stats)

        # Atomically move temp directory to final location
        if final_ckpt_dir.exists():
            shutil.rmtree(final_ckpt_dir)
        tmp_ckpt_dir.rename(final_ckpt_dir)

        logging.info(f"Saved checkpoint at step {global_step} -> {final_ckpt_dir}")

        if wandb_run is not None:
            wandb_run.log_metrics({"checkpoint_step": global_step}, step=global_step, prefix="train")
            wandb_run.log_summary({"train/checkpoint_step": global_step})

        if config.wandb_enabled and wandb.run is not None:
            if artifact_manager is not None and _wandb_artifacts.should_publish_checkpoint_artifact(
                global_step,
                final_step=config.num_train_steps,
                interval=config.wandb_checkpoint_artifact_interval,
            ):
                artifact_manager.log_checkpoint_directory(
                    final_ckpt_dir,
                    artifact_name=f"{config.name}-{config.exp_name}",
                    aliases=_wandb_artifacts.build_checkpoint_aliases(
                        global_step,
                        is_final=global_step == config.num_train_steps,
                    ),
                    metadata={
                        "backend": "pytorch",
                        "config_name": config.name,
                        "task_name": config.task_name,
                        "dataset_name": config.dataset_name,
                        "checkpoint_step": global_step,
                        "train_dataset_artifact": None if dataset_artifact_refs is None else dataset_artifact_refs.get("train"),
                        "eval_dataset_artifact": None if dataset_artifact_refs is None else dataset_artifact_refs.get("eval"),
                        "eval_final_dataset_artifact": None
                        if dataset_artifact_refs is None
                        else dataset_artifact_refs.get("eval_final"),
                    },
                )
        return True

    return False


def _run_local_eval_and_resume_training(
    config: _config.TrainConfig,
    wandb_run: _wandb_run_context.WandbRunContext,
    artifact_manager: _wandb_artifacts.WandbArtifactManager,
    *,
    model,
    checkpoint_step: int,
    split_name: str,
    num_examples: int | None,
    data_config,
    device: torch.device,
) -> None:
    logging.info("Running inline local %s eval at step %s on the current allocation.", split_name, checkpoint_step)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    _run_inline_aloha_eval(
        config,
        wandb_run,
        artifact_manager,
        model=model,
        checkpoint_step=checkpoint_step,
        split_name=split_name,
        num_examples=num_examples,
        history_step=checkpoint_step,
        data_config=data_config,
        device=device,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def _run_local_final_eval(
    config: _config.TrainConfig,
    wandb_run: _wandb_run_context.WandbRunContext,
    artifact_manager: _wandb_artifacts.WandbArtifactManager,
    *,
    model,
    checkpoint_step: int,
    data_config,
    device: torch.device,
) -> None:
    logging.info("Running inline local final eval at step %s on the current allocation.", checkpoint_step)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    _run_inline_aloha_eval(
        config,
        wandb_run,
        artifact_manager,
        model=model,
        checkpoint_step=checkpoint_step,
        split_name=config.final_eval_split_name,
        num_examples=config.final_eval_num_examples,
        history_step=checkpoint_step,
        data_config=data_config,
        device=device,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def _run_inline_aloha_eval(
    config: _config.TrainConfig,
    wandb_run: _wandb_run_context.WandbRunContext,
    artifact_manager: _wandb_artifacts.WandbArtifactManager,
    *,
    model,
    checkpoint_step: int,
    split_name: str,
    num_examples: int | None,
    history_step: int,
    data_config,
    device: torch.device,
) -> None:
    manifest_path_str = config.eval_manifest_path if split_name == config.eval_split_name else config.final_eval_manifest_path
    manifest_path = _eval_manifest.resolve_repo_path(manifest_path_str)
    if manifest_path is None or not manifest_path.exists():
        raise FileNotFoundError(f"Manifest path for {split_name} eval does not exist: {manifest_path_str}")

    metric_prefix = _eval_tracking.metric_namespace_for_split(split_name)
    checkpoint_alias = f"step-{checkpoint_step}"
    checkpoint_dir = pathlib.Path(config.checkpoint_dir) / str(checkpoint_step)
    video_root, results_path = _eval_submission.build_eval_output_paths(
        config,
        checkpoint_step=checkpoint_step,
        split_name=split_name,
    )
    video_root.mkdir(parents=True, exist_ok=True)

    artifact_stem = f"{_aloha_eval.slugify(config.name)}-{_aloha_eval.slugify(config.exp_name)}"
    results_artifact_name = f"{artifact_stem}-{metric_prefix}-results"
    media_artifact_name = f"{artifact_stem}-{metric_prefix}-videos"
    artifact_aliases = tuple(alias for alias in (checkpoint_alias, split_name, "latest") if alias)
    results_artifact_ref = _wandb_artifacts.build_artifact_ref(results_artifact_name, (checkpoint_alias,))
    media_artifact_ref = _wandb_artifacts.build_artifact_ref(media_artifact_name, (checkpoint_alias,))

    manifest = _aloha_eval.load_manifest(manifest_path)
    model_to_eval = get_unwrapped_model(model)
    was_training = model_to_eval.training
    model_to_eval.eval()
    try:
        policy = _policy_config.create_policy_from_model(
            config,
            model_to_eval,
            default_prompt=manifest.prompt,
            norm_stats=data_config.norm_stats,
            pytorch_device=str(device),
            is_pytorch=True,
        )
        bundle = _aloha_eval.evaluate_policy(
            policy=policy,
            manifest=manifest,
            split_name=split_name,
            checkpoint_step=checkpoint_step,
            num_examples=0 if num_examples is None else num_examples,
            video_dir=video_root,
            media_artifact_ref=media_artifact_ref,
            task=os.environ.get("EVAL_TASK", _aloha_eval.DEFAULT_TASK),
            action_horizon=int(os.environ.get("ACTION_HORIZON", "10")),
            max_episode_steps=int(os.environ.get("MAX_EPISODE_STEPS", "0")),
            success_reward_threshold=float(os.environ.get("SUCCESS_REWARD_THRESHOLD", "4.0")),
            fps=int(os.environ.get("EVAL_FPS", "50")),
            render_mode=os.environ.get("RENDER_MODE"),
            visualization_width=int(os.environ.get("VISUALIZATION_WIDTH", "640")),
            visualization_height=int(os.environ.get("VISUALIZATION_HEIGHT", "336")),
            visualization_camera_id=os.environ.get("VISUALIZATION_CAMERA_ID", "angle"),
        )
    finally:
        if was_training:
            model_to_eval.train()

    _aloha_eval.write_results(
        results_path,
        config_name=config.name,
        checkpoint_dir=str(checkpoint_dir),
        checkpoint_step=checkpoint_step,
        checkpoint_alias=checkpoint_alias,
        manifest_path=manifest_path,
        split_name=split_name,
        metrics=bundle.metrics,
        results=bundle.results,
    )

    wandb_run.log_metrics(bundle.metrics, step=history_step, prefix=metric_prefix)
    wandb_run.log_summary({f"{metric_prefix}/{key}": value for key, value in bundle.metrics.items() if value is not None})
    wandb_run.log_summary(
        {
            f"{metric_prefix}/split_name": split_name,
            f"{metric_prefix}/checkpoint_alias": checkpoint_alias,
            f"{metric_prefix}/source_train_run_id": wandb.run.id if wandb.run is not None else None,
            f"{metric_prefix}/source_train_run_name": config.exp_name,
        }
    )
    _wandb_videos.VideoLogger().log_video_files(f"{metric_prefix}/videos", bundle.video_records, step=history_step)
    _wandb_tables.WandbTableLogger(key=f"{metric_prefix}/examples", columns=_aloha_eval.EXAMPLE_COLUMNS).log_immutable(
        bundle.example_rows,
        step=history_step,
    )
    _wandb_leaderboard.LeaderboardTableLogger(key=f"{metric_prefix}/leaderboard").log_row(
        _wandb_types.LeaderboardRow(
            eval_run_id=wandb.run.id if wandb.run is not None else config.exp_name,
            source_train_run_id=wandb.run.id if wandb.run is not None else None,
            source_train_run_name=config.exp_name,
            eval_name=manifest.name,
            eval_split=split_name,
            model_family=getattr(config.model, "model_type", type(config.model).__name__),
            config_name=config.name,
            task_name=manifest.task_name,
            dataset_name=manifest.dataset_name,
            checkpoint_alias=checkpoint_alias,
            checkpoint_step=checkpoint_step,
            primary_score=bundle.metrics["primary_score"],
            success_rate=bundle.metrics["success_rate"],
            mean_max_reward=bundle.metrics["mean_max_reward"],
            num_examples=bundle.metrics["num_examples"],
            artifact_ref_results=results_artifact_ref,
            artifact_ref_media=media_artifact_ref,
            notes=f"success_reward_threshold={os.environ.get('SUCCESS_REWARD_THRESHOLD', '4.0')}",
        ),
        step=history_step,
    )
    artifact_metadata = {
        "config_name": config.name,
        "task_name": manifest.task_name,
        "dataset_name": manifest.dataset_name,
        "split_name": split_name,
        "checkpoint_step": checkpoint_step,
        "checkpoint_alias": checkpoint_alias,
        "source_train_run_id": wandb.run.id if wandb.run is not None else None,
        "source_train_run_name": config.exp_name,
    }
    artifact_manager.log_artifact(
        _wandb_types.ArtifactRecord(
            name=results_artifact_name,
            type="eval-results",
            path=str(results_path),
            aliases=artifact_aliases,
            description=f"ALOHA Sim {split_name} evaluation results",
            metadata=artifact_metadata,
        )
    )
    artifact_manager.log_artifact(
        _wandb_types.ArtifactRecord(
            name=media_artifact_name,
            type="eval-video-bundle",
            path=str(video_root),
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
    logging.info("Finished inline %s eval at checkpoint step %s.", split_name, checkpoint_step)


def load_checkpoint(model, optimizer, checkpoint_dir, device):
    """Load the latest checkpoint and return the global step."""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]

    if not checkpoint_steps:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    latest_step = max(checkpoint_steps)
    ckpt_dir = checkpoint_dir / f"{latest_step}"

    # Clear memory before loading checkpoints
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "before_loading_checkpoint")

    try:
        # Load model state with error handling
        logging.info("Loading model state...")
        safetensors_path = ckpt_dir / "model.safetensors"

        if safetensors_path.exists():
            model_to_load = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
            safetensors.torch.load_model(model_to_load, safetensors_path, device=str(device))
            logging.info("Loaded model state from safetensors format")
        else:
            raise FileNotFoundError(f"No model checkpoint found at {ckpt_dir}")

        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_model")

        # Load optimizer state with error handling
        logging.info("Loading optimizer state...")
        optimizer_path = ckpt_dir / "optimizer.pt"

        if optimizer_path.exists():
            optimizer_state_dict = torch.load(optimizer_path, map_location=device, weights_only=False)
            logging.info("Loaded optimizer state from pt format")
        else:
            raise FileNotFoundError(f"No optimizer checkpoint found at {ckpt_dir}")

        optimizer.load_state_dict(optimizer_state_dict)
        del optimizer_state_dict
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_optimizer")

        # Load metadata
        logging.info("Loading metadata...")
        metadata = torch.load(ckpt_dir / "metadata.pt", map_location=device, weights_only=False)
        global_step = metadata.get("global_step", latest_step)
        del metadata
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_metadata")

        logging.info(f"Successfully loaded all checkpoint components from step {latest_step}")
        return global_step

    except RuntimeError as e:
        if "out of memory" in str(e):
            # Clear memory and provide detailed error message
            torch.cuda.empty_cache()
            gc.collect()
            logging.error(f"Out of memory error while loading checkpoint: {e!s}")
            log_memory_usage(device, latest_step, "after_oom_error")
            raise RuntimeError(
                "Out of memory while loading checkpoint. Try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
            ) from e
        raise


def get_latest_checkpoint_step(checkpoint_dir):
    """Get the latest checkpoint step number from a checkpoint directory."""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]
    return max(checkpoint_steps) if checkpoint_steps else None


def log_memory_usage(device, step, phase="unknown"):
    """Log detailed memory usage information."""
    if not torch.cuda.is_available():
        return

    memory_allocated = torch.cuda.memory_allocated(device) / 1e9
    memory_reserved = torch.cuda.memory_reserved(device) / 1e9
    memory_free = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    memory_free = memory_free / 1e9

    # Get more detailed memory info
    memory_stats = torch.cuda.memory_stats(device)
    max_memory_allocated = memory_stats.get("allocated_bytes.all.peak", 0) / 1e9
    max_memory_reserved = memory_stats.get("reserved_bytes.all.peak", 0) / 1e9

    # Get DDP info if available
    ddp_info = ""
    if dist.is_initialized():
        ddp_info = f" | DDP: rank={dist.get_rank()}, world_size={dist.get_world_size()}"

    logging.info(
        f"Step {step} ({phase}): GPU memory - allocated: {memory_allocated:.2f}GB, reserved: {memory_reserved:.2f}GB, free: {memory_free:.2f}GB, peak_allocated: {max_memory_allocated:.2f}GB, peak_reserved: {max_memory_reserved:.2f}GB{ddp_info}"
    )


def train_loop(config: _config.TrainConfig):
    use_ddp, local_rank, device = setup_ddp()
    is_main = (not use_ddp) or (dist.get_rank() == 0)
    set_seed(config.seed, local_rank)

    # Initialize checkpoint directory and wandb
    resuming = False
    exp_checkpoint_dir = config.checkpoint_dir
    if config.resume:
        # Find checkpoint directory based on experiment name
        if exp_checkpoint_dir.exists():
            # Use validation to find the latest working checkpoint
            latest_step = get_latest_checkpoint_step(exp_checkpoint_dir)
            if latest_step is not None:
                resuming = True
                logging.info(
                    f"Resuming from experiment checkpoint directory: {exp_checkpoint_dir} at step {latest_step}"
                )
            else:
                raise FileNotFoundError(f"No valid checkpoints found in {exp_checkpoint_dir} for resume")
        else:
            raise FileNotFoundError(f"Experiment checkpoint directory {exp_checkpoint_dir} does not exist for resume")
    elif config.overwrite and config.checkpoint_dir.exists():
        if is_main:
            shutil.rmtree(config.checkpoint_dir)
            logging.info(f"Overwriting checkpoint directory: {config.checkpoint_dir}")
        _ddp_barrier(use_ddp, device)

    # Create checkpoint directory with experiment name
    if not resuming:
        # For new runs, create experiment-specific checkpoint directory
        if is_main:
            exp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Created experiment checkpoint directory: {exp_checkpoint_dir}")
        _ddp_barrier(use_ddp, device)
    else:
        # For resume, checkpoint_dir is already set to the experiment directory
        logging.info(f"Using existing experiment checkpoint directory: {config.checkpoint_dir}")

    # Initialize wandb (only on main process)
    wandb_run = None
    artifact_manager = None
    dataset_artifact_refs = {}
    if is_main:
        wandb_run = init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)
        if config.wandb_enabled:
            _configure_eval_metric_axes(config)
        artifact_manager = _wandb_artifacts.WandbArtifactManager()
        dataset_artifact_refs = _dataset_artifacts.configured_dataset_artifact_refs(config)
        if dataset_artifact_refs:
            config, dataset_artifact_refs = _dataset_artifacts.bind_dataset_artifact_payloads(config, dataset_artifact_refs)
        elif config.publish_dataset_artifacts:
            dataset_artifact_refs = _dataset_artifacts.register_dataset_artifacts(config, artifact_manager)
        if wandb_run is not None:
            wandb_run.log_summary(
                {
                    "data/train_dataset_artifact": dataset_artifact_refs.get("train"),
                    "data/eval_dataset_artifact": dataset_artifact_refs.get("eval"),
                    "data/eval_final_dataset_artifact": dataset_artifact_refs.get("eval_final"),
                }
            )
    if use_ddp:
        config_holder = [config]
        refs_holder = [dataset_artifact_refs]
        dist.broadcast_object_list(config_holder, src=0)
        dist.broadcast_object_list(refs_holder, src=0)
        config = config_holder[0]
        dataset_artifact_refs = refs_holder[0]

    # Build data loader using the unified data loader
    # Calculate effective batch size per GPU for DDP
    # For N GPUs, each GPU should get batch_size/N samples, so total across all GPUs is batch_size
    world_size = torch.distributed.get_world_size() if use_ddp else 1
    effective_batch_size = config.batch_size // world_size
    logging.info(
        f"Using batch size per GPU: {effective_batch_size} (total batch size across {world_size} GPUs: {config.batch_size})"
    )

    # Pass the original batch size to data loader - it will handle DDP splitting internally
    loader, data_config = build_datasets(config)

    # Build model
    if not isinstance(config.model, openpi.models.pi0_config.Pi0Config):
        # Convert dataclass to Pi0Config if needed
        model_cfg = openpi.models.pi0_config.Pi0Config(
            dtype=config.pytorch_training_precision,
            action_dim=config.model.action_dim,
            action_horizon=config.model.action_horizon,
            max_token_len=config.model.max_token_len,
            paligemma_variant=getattr(config.model, "paligemma_variant", "gemma_2b"),
            action_expert_variant=getattr(config.model, "action_expert_variant", "gemma_300m"),
            pi05=getattr(config.model, "pi05", False),
        )
    else:
        model_cfg = config.model
        # Update dtype to match pytorch_training_precision
        object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)

    model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_cfg).to(device)

    if hasattr(model, "gradient_checkpointing_enable"):
        enable_gradient_checkpointing = True
        model.gradient_checkpointing_enable()
        logging.info("Enabled gradient checkpointing for memory optimization")
    else:
        enable_gradient_checkpointing = False
        logging.info("Gradient checkpointing is not supported for this model")

    # Log initial memory usage after model creation
    if is_main and torch.cuda.is_available():
        log_memory_usage(device, 0, "after_model_creation")

    # Enable memory optimizations for large-scale training
    if world_size >= 8:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Set memory allocation configuration
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
        logging.info("Enabled memory optimizations for 8+ GPU training")

    # Load fine-tuning weights before DDP wrapping so every rank starts from the same state.
    ddp_init_sync = True
    if config.pytorch_weight_path is not None:
        logging.info(f"Loading weights from: {config.pytorch_weight_path}")

        model_path = os.path.join(config.pytorch_weight_path, "model.safetensors")
        safetensors.torch.load_model(model, model_path)
        logging.info(f"Loaded PyTorch weights from {config.pytorch_weight_path}")
        ddp_init_sync = False

    if use_ddp:
        logging.info("Wrapping model with DDP on device %s", device)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            init_sync=ddp_init_sync,
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
            static_graph=False,
            broadcast_buffers=False,
        )
        logging.info("DDP model wrapping complete on device %s", device)

    # Optimizer + learning rate schedule from config
    warmup_steps = config.lr_schedule.warmup_steps
    peak_lr = config.lr_schedule.peak_lr
    decay_steps = config.lr_schedule.decay_steps
    end_lr = config.lr_schedule.decay_lr

    # Create optimizer with config parameters
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )

    # Load checkpoint if resuming
    global_step = 0
    if resuming:
        global_step = load_checkpoint(model, optim, config.checkpoint_dir, device)
        logging.info(f"Resumed training from step {global_step}")

    def lr_schedule(step: int):
        if step < warmup_steps:
            # Match JAX behavior: start from peak_lr / (warmup_steps + 1)
            init_lr = peak_lr / (warmup_steps + 1)
            return init_lr + (peak_lr - init_lr) * step / warmup_steps
        # cosine decay
        progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        return end_lr + (peak_lr - end_lr) * cos

    model.train()
    start_time = time.time()
    infos = []  # Collect stats over log interval
    if is_main:
        logging.info(
            f"Running on: {platform.node()} | world_size={torch.distributed.get_world_size() if use_ddp else 1}"
        )
        logging.info(
            f"Training config: batch_size={config.batch_size}, effective_batch_size={effective_batch_size}, num_train_steps={config.num_train_steps}"
        )
        logging.info(f"Memory optimizations: gradient_checkpointing={enable_gradient_checkpointing}")
        logging.info(
            f"LR schedule: warmup={warmup_steps}, peak_lr={peak_lr:.2e}, decay_steps={decay_steps}, end_lr={end_lr:.2e}"
        )
        logging.info(
            f"Optimizer: {type(config.optimizer).__name__}, weight_decay={config.optimizer.weight_decay}, clip_norm={config.optimizer.clip_gradient_norm}"
        )
        logging.info("EMA is not supported for PyTorch training")
        logging.info(f"Training precision: {model_cfg.dtype}")

    # Training loop - iterate until we reach num_train_steps
    pbar = (
        tqdm.tqdm(total=config.num_train_steps, initial=global_step, desc="Training", disable=not is_main)
        if is_main
        else None
    )

    last_eval_job_id = None
    pending_eval_results: list[PendingEvalResult] = []
    while global_step < config.num_train_steps:
        # Set epoch for distributed training
        if use_ddp and hasattr(loader, "set_epoch"):
            loader.set_epoch(global_step // len(loader))

        for observation, actions in loader:
            # Check if we've reached the target number of steps
            if global_step >= config.num_train_steps:
                break

            # The unified data loader returns (observation, actions) tuple
            observation = jax.tree.map(lambda x: x.to(device), observation)  # noqa: PLW2901
            actions = actions.to(torch.float32)  # noqa: PLW2901
            actions = actions.to(device)  # noqa: PLW2901

            # Update LR
            for pg in optim.param_groups:
                pg["lr"] = lr_schedule(global_step)

            # Forward pass
            losses = model(observation, actions)
            # Ensure losses is a tensor and handle different return types
            if isinstance(losses, list | tuple):
                losses = torch.stack(losses)
            elif not isinstance(losses, torch.Tensor):
                losses = torch.tensor(losses, device=device, dtype=torch.float32)

            loss = losses.mean()

            # Backward pass
            loss.backward()

            # Log memory usage after backward pass
            if global_step < 5 and is_main and torch.cuda.is_available():
                log_memory_usage(device, global_step, "after_backward")

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.optimizer.clip_gradient_norm)

            # Optimizer step
            optim.step()
            optim.zero_grad(set_to_none=True)

            # Clear gradients more aggressively
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.detach_()
                    param.grad = None

            # Collect stats
            if is_main:
                infos.append(
                    {
                        "loss": loss.item(),
                        "learning_rate": optim.param_groups[0]["lr"],
                        "grad_norm": float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    }
                )

            if is_main and (global_step % config.log_interval == 0):
                elapsed = time.time() - start_time

                # Average stats over log interval
                avg_loss = sum(info["loss"] for info in infos) / len(infos)
                avg_lr = sum(info["learning_rate"] for info in infos) / len(infos)

                avg_grad_norm = None
                if any("grad_norm" in info for info in infos):
                    vals = [
                        info["grad_norm"] for info in infos if "grad_norm" in info and info["grad_norm"] is not None
                    ]
                    if len(vals) > 0:
                        avg_grad_norm = sum(vals) / len(vals)
                logging.info(
                    f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} grad_norm={avg_grad_norm:.2f} time={elapsed:.1f}s"
                    if avg_grad_norm is not None
                    else f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} time={elapsed:.1f}s"
                )

                # Log to wandb
                if config.wandb_enabled and len(infos) > 0:
                    log_payload = {
                        "loss": avg_loss,
                        "learning_rate": avg_lr,
                        "checkpoint_step": global_step,
                        "step_time_sec": elapsed / config.log_interval,
                    }
                    if avg_grad_norm is not None:
                        log_payload["grad_norm"] = avg_grad_norm
                    wandb_run.log_metrics(log_payload, step=global_step, prefix="train")

                start_time = time.time()
                infos = []  # Reset stats collection

            if is_main and wandb_run is not None:
                pending_eval_results = _flush_available_eval_results(wandb_run, pending_eval_results)

            global_step += 1
            should_save_checkpoint = (
                (global_step % config.save_interval == 0 and global_step > 0)
                or global_step == config.num_train_steps
            )
            should_run_periodic_eval = global_step % config.save_interval == 0 and global_step < config.num_train_steps

            if should_save_checkpoint:
                def _checkpoint_phase() -> None:
                    nonlocal last_eval_job_id, pending_eval_results

                    checkpoint_saved = save_checkpoint(
                        model,
                        optim,
                        global_step,
                        config,
                        True,
                        data_config,
                        artifact_manager,
                        wandb_run,
                        dataset_artifact_refs,
                    )
                    if not checkpoint_saved:
                        return

                    if config.eval_manifest_path is None or (
                        config.eval_execution_mode != "local" and config.eval_job_script_path is None
                    ):
                        return

                    if config.eval_execution_mode == "local" and should_run_periodic_eval:
                        if wandb_run is not None and artifact_manager is not None:
                            _run_local_eval_and_resume_training(
                                config,
                                wandb_run,
                                artifact_manager,
                                model=model,
                                checkpoint_step=global_step,
                                split_name=config.eval_split_name,
                                num_examples=config.periodic_eval_num_examples,
                                data_config=data_config,
                                device=device,
                            )
                    elif should_run_periodic_eval and wandb_run is not None:
                        submitted_job_id = _eval_submission.maybe_submit_periodic_eval(
                            config,
                            checkpoint_step=global_step,
                            dependency_job_id=last_eval_job_id,
                        )
                        if submitted_job_id is not None:
                            eval_prefix = _eval_tracking.metric_namespace_for_split(config.eval_split_name)
                            wandb_run.log_summary(
                                {
                                    f"{eval_prefix}/last_submitted_job_id": submitted_job_id,
                                    f"{eval_prefix}/last_submitted_checkpoint_step": global_step,
                                }
                            )
                            _, results_path = _eval_submission.build_eval_output_paths(
                                config,
                                checkpoint_step=global_step,
                                split_name=config.eval_split_name,
                            )
                            pending_eval_results.append(
                                PendingEvalResult(
                                    split_name=config.eval_split_name,
                                    results_path=results_path,
                                    history_step=global_step if config.block_on_periodic_eval else None,
                                )
                            )
                            if config.block_on_periodic_eval:
                                pending_eval_results = _wait_for_eval_results(
                                    wandb_run,
                                    pending_eval_results,
                                    poll_interval_secs=config.eval_results_poll_interval_secs,
                                )
                            last_eval_job_id = submitted_job_id

                _run_main_rank_only_phase(
                    config=config,
                    use_ddp=use_ddp,
                    is_main=is_main,
                    step=global_step,
                    phase="checkpoint",
                    action=_checkpoint_phase,
                )

            # Update progress bar
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(
                    {"loss": f"{loss.item():.4f}", "lr": f"{optim.param_groups[0]['lr']:.2e}", "step": global_step}
                )

    # Close progress bar
    if pbar is not None:
        pbar.close()

    def _finalize_phase() -> None:
        nonlocal last_eval_job_id, pending_eval_results

        if config.eval_execution_mode == "local" and config.final_eval_manifest_path is not None:
            if wandb_run is not None and artifact_manager is not None:
                _run_local_final_eval(
                    config,
                    wandb_run,
                    artifact_manager,
                    model=model,
                    checkpoint_step=global_step,
                    data_config=data_config,
                    device=device,
                )
        elif config.final_eval_manifest_path is not None and wandb_run is not None:
            job_id = _eval_submission.maybe_submit_final_eval(
                config,
                checkpoint_step=global_step,
                dependency_job_id=last_eval_job_id,
            )
            if job_id is not None:
                eval_prefix = _eval_tracking.metric_namespace_for_split(config.final_eval_split_name)
                wandb_run.log_summary(
                    {
                        f"{eval_prefix}/last_submitted_job_id": job_id,
                        f"{eval_prefix}/last_submitted_checkpoint_step": global_step,
                    }
                )
                _, results_path = _eval_submission.build_eval_output_paths(
                    config,
                    checkpoint_step=global_step,
                    split_name=config.final_eval_split_name,
                )
                pending_eval_results.append(
                    PendingEvalResult(
                        split_name=config.final_eval_split_name,
                        results_path=results_path,
                        history_step=global_step,
                    )
                )
            pending_eval_results = _wait_for_eval_results(
                wandb_run,
                pending_eval_results,
                poll_interval_secs=config.eval_results_poll_interval_secs,
            )

        if wandb_run is not None:
            wandb_run.finish()

    _run_main_rank_only_phase(
        config=config,
        use_ddp=use_ddp,
        is_main=is_main,
        step=global_step,
        phase="finalize",
        action=_finalize_phase,
    )

    _ddp_barrier(use_ddp, device)
    cleanup_ddp(device)


def main():
    init_logging()
    config = _config.cli()
    train_loop(config)


if __name__ == "__main__":
    main()
