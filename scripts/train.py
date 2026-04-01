import dataclasses
import functools
import gc
import json
import logging
import os
import pathlib
import platform
import time
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util

import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.policies.policy_config as _policy_config
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.aloha_eval as _aloha_eval
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.dataset_artifacts as _dataset_artifacts
import openpi.training.eval_manifest as _eval_manifest
import openpi.training.eval_submission as _eval_submission
import openpi.training.eval_tracking as _eval_tracking
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders
import openpi.utils.wandb.artifacts as _wandb_artifacts
import openpi.utils.wandb.leaderboard as _wandb_leaderboard
import openpi.utils.wandb.run_context as _wandb_run_context
import openpi.utils.wandb.tables as _wandb_tables
import openpi.utils.wandb.types as _wandb_types
import openpi.utils.wandb.videos as _wandb_videos
from openpi.utils.wandb.types import VideoRecord


@dataclasses.dataclass(frozen=True)
class PendingEvalResult:
    split_name: str
    results_path: pathlib.Path
    history_step: int | None = None


def init_logging():
    """Custom logging format for better readability."""
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
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    log_code_root = pathlib.Path(__file__).resolve().parent.parent if log_code else None
    return _wandb_run_context.WandbRunContext.init_for_training(
        config,
        backend="jax",
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
            VideoRecord(
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


def _run_local_eval_and_resume_training(
    config: _config.TrainConfig,
    wandb_run: _wandb_run_context.WandbRunContext,
    artifact_manager: _wandb_artifacts.WandbArtifactManager,
    *,
    train_state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
    checkpoint_step: int,
    split_name: str,
    num_examples: int | None,
) -> tuple[training_utils.TrainState, tuple[_model.Observation, _model.Actions]]:
    logging.info("Running inline local %s eval at step %s on the current allocation.", split_name, checkpoint_step)
    gc.collect()
    _run_inline_aloha_eval(
        config,
        wandb_run,
        artifact_manager,
        train_state=train_state,
        checkpoint_step=checkpoint_step,
        split_name=split_name,
        num_examples=num_examples,
        history_step=checkpoint_step,
    )
    gc.collect()
    return train_state, batch


def _run_local_final_eval(
    config: _config.TrainConfig,
    wandb_run: _wandb_run_context.WandbRunContext,
    artifact_manager: _wandb_artifacts.WandbArtifactManager,
    *,
    train_state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
    checkpoint_step: int,
) -> None:
    del batch
    logging.info("Running inline local final eval at step %s on the current allocation.", checkpoint_step)
    gc.collect()
    _run_inline_aloha_eval(
        config,
        wandb_run,
        artifact_manager,
        train_state=train_state,
        checkpoint_step=checkpoint_step,
        split_name=config.final_eval_split_name,
        num_examples=config.final_eval_num_examples,
        history_step=checkpoint_step,
    )
    gc.collect()


def _run_inline_aloha_eval(
    config: _config.TrainConfig,
    wandb_run: _wandb_run_context.WandbRunContext,
    artifact_manager: _wandb_artifacts.WandbArtifactManager,
    *,
    train_state: training_utils.TrainState,
    checkpoint_step: int,
    split_name: str,
    num_examples: int | None,
    history_step: int,
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
    try:
        eval_device = jax.devices("gpu")[0]
    except RuntimeError:
        eval_device = jax.devices()[0]
    eval_mesh = jax.sharding.Mesh(np.array([eval_device]), ("x",))
    eval_sharding = jax.sharding.NamedSharding(eval_mesh, jax.sharding.PartitionSpec())
    eval_params = _model.restore_params(
        checkpoint_dir / "params",
        dtype=jnp.bfloat16,
        sharding=eval_sharding,
    )
    model = config.model.load(eval_params)
    policy = _policy_config.create_policy_from_model(
        config,
        model,
        default_prompt=manifest.prompt,
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


def _maybe_publish_checkpoint_artifact(
    config: _config.TrainConfig,
    artifact_manager: _wandb_artifacts.WandbArtifactManager,
    checkpoint_manager,
    step: int,
    *,
    dataset_artifact_refs: dict[str, str],
) -> None:
    final_step = config.num_train_steps - 1
    if not _wandb_artifacts.should_publish_checkpoint_artifact(
        step,
        final_step=final_step,
        interval=config.wandb_checkpoint_artifact_interval,
    ):
        return

    checkpoint_manager.wait_until_finished()
    checkpoint_dir = epath.Path(config.checkpoint_dir) / str(step)
    artifact_manager.log_checkpoint_directory(
        checkpoint_dir,
        artifact_name=f"{config.name}-{config.exp_name}",
        aliases=_wandb_artifacts.build_checkpoint_aliases(step, is_final=step == final_step),
        metadata={
            "backend": "jax",
            "config_name": config.name,
            "task_name": config.task_name,
            "dataset_name": config.dataset_name,
            "checkpoint_step": step,
            "train_dataset_artifact": dataset_artifact_refs.get("train"),
            "eval_dataset_artifact": dataset_artifact_refs.get("eval"),
            "eval_final_dataset_artifact": dataset_artifact_refs.get("eval_final"),
        },
    )


def _maybe_submit_periodic_eval(
    config: _config.TrainConfig,
    wandb_run: _wandb_run_context.WandbRunContext,
    *,
    step: int,
    dependency_job_id: str | None = None,
) -> str | None:
    job_id = _eval_submission.maybe_submit_periodic_eval(
        config,
        checkpoint_step=step,
        dependency_job_id=dependency_job_id,
    )
    if job_id is not None:
        eval_prefix = _eval_tracking.metric_namespace_for_split(config.eval_split_name)
        wandb_run.log_summary(
            {
                f"{eval_prefix}/last_submitted_job_id": job_id,
                f"{eval_prefix}/last_submitted_checkpoint_step": step,
            }
        )
    return job_id


def _maybe_submit_final_eval(
    config: _config.TrainConfig,
    wandb_run: _wandb_run_context.WandbRunContext,
    *,
    step: int,
    dependency_job_id: str | None = None,
) -> str | None:
    job_id = _eval_submission.maybe_submit_final_eval(
        config,
        checkpoint_step=step,
        dependency_job_id=dependency_job_id,
    )
    if job_id is not None:
        eval_prefix = _eval_tracking.metric_namespace_for_split(config.final_eval_split_name)
        wandb_run.log_summary(
            {
                f"{eval_prefix}/last_submitted_job_id": job_id,
                f"{eval_prefix}/last_submitted_checkpoint_step": step,
            }
        )
    return job_id


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")
    logging.info("JAX devices (%s): %s", jax.device_count(), [str(device) for device in jax.devices()])
    logging.info("FSDP devices: %s", config.fsdp_devices)

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    compilation_cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR")
    if compilation_cache_dir:
        jax.config.update("jax_compilation_cache_dir", compilation_cache_dir)
    else:
        jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))
    logging.info("JAX compilation cache dir: %s", jax.config.jax_compilation_cache_dir)

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        max_to_keep=config.checkpoint_max_to_keep,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
        save_resume_state=config.save_resume_state,
    )
    wandb_run = init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)
    if config.wandb_enabled:
        _configure_eval_metric_axes(config)
    artifact_manager = _wandb_artifacts.WandbArtifactManager()
    dataset_artifact_refs = _dataset_artifacts.configured_dataset_artifact_refs(config)
    if dataset_artifact_refs:
        config, dataset_artifact_refs = _dataset_artifacts.bind_dataset_artifact_payloads(config, dataset_artifact_refs)
    elif config.publish_dataset_artifacts:
        dataset_artifact_refs = _dataset_artifacts.register_dataset_artifacts(config, artifact_manager)
    wandb_run.log_summary(
        {
            "data/train_dataset_artifact": dataset_artifact_refs.get("train"),
            "data/eval_dataset_artifact": dataset_artifact_refs.get("eval"),
            "data/eval_final_dataset_artifact": dataset_artifact_refs.get("eval_final"),
        }
    )

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(
            checkpoint_manager,
            train_state,
            data_loader,
            save_resume_state=config.save_resume_state,
        )

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    last_eval_job_id: str | None = None
    pending_eval_results: list[PendingEvalResult] = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb_run.log_metrics(reduced_info, step=step, prefix="train")
            infos = []
        batch = next(data_iter)
        pending_eval_results = _flush_available_eval_results(wandb_run, pending_eval_results)

        should_save_checkpoint = (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1
        should_run_periodic_eval = step % config.save_interval == 0 and step > start_step
        if should_save_checkpoint:
            _checkpoints.save_state(
                checkpoint_manager,
                train_state,
                data_loader,
                step,
                save_resume_state=config.save_resume_state,
            )
            if config.wandb_enabled:
                wandb_run.log_summary({"train/checkpoint_step": step})
            if config.eval_manifest_path is not None and (
                config.eval_execution_mode == "local" or config.eval_job_script_path is not None
            ):
                checkpoint_manager.wait_until_finished()
                if config.eval_execution_mode == "local" and should_run_periodic_eval:
                    train_state, batch = _run_local_eval_and_resume_training(
                        config,
                        wandb_run,
                        artifact_manager,
                        train_state=train_state,
                        batch=batch,
                        checkpoint_step=step,
                        split_name=config.eval_split_name,
                        num_examples=config.periodic_eval_num_examples,
                    )
                elif should_run_periodic_eval:
                    submitted_job_id = _maybe_submit_periodic_eval(
                        config,
                        wandb_run,
                        step=step,
                        dependency_job_id=last_eval_job_id,
                    )
                    if submitted_job_id is not None:
                        last_eval_job_id = submitted_job_id
                        _, results_path = _eval_submission.build_eval_output_paths(
                            config,
                            checkpoint_step=step,
                            split_name=config.eval_split_name,
                        )
                        pending_eval_results.append(
                            PendingEvalResult(
                                split_name=config.eval_split_name,
                                results_path=results_path,
                                history_step=step if config.block_on_periodic_eval else None,
                            )
                        )
                        if config.block_on_periodic_eval:
                            pending_eval_results = _wait_for_eval_results(
                                wandb_run,
                                pending_eval_results,
                                poll_interval_secs=config.eval_results_poll_interval_secs,
                            )
            if config.wandb_enabled:
                _maybe_publish_checkpoint_artifact(
                    config,
                    artifact_manager,
                    checkpoint_manager,
                    step,
                    dataset_artifact_refs=dataset_artifact_refs,
                )

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()
    if config.eval_execution_mode == "local" and config.final_eval_manifest_path is not None:
        _run_local_final_eval(
            config,
            wandb_run,
            artifact_manager,
            train_state=train_state,
            batch=batch,
            checkpoint_step=config.num_train_steps - 1,
        )
    elif config.final_eval_manifest_path is not None:
        submitted_final_eval_job_id = _maybe_submit_final_eval(
            config,
            wandb_run,
            step=config.num_train_steps - 1,
            dependency_job_id=last_eval_job_id,
        )
        if submitted_final_eval_job_id is not None:
            _, results_path = _eval_submission.build_eval_output_paths(
                config,
                checkpoint_step=config.num_train_steps - 1,
                split_name=config.final_eval_split_name,
            )
            pending_eval_results.append(
                PendingEvalResult(
                    split_name=config.final_eval_split_name,
                    results_path=results_path,
                    history_step=config.num_train_steps - 1,
                )
            )
        pending_eval_results = _wait_for_eval_results(
            wandb_run,
            pending_eval_results,
            poll_interval_secs=config.eval_results_poll_interval_secs,
        )
    wandb_run.finish()


if __name__ == "__main__":
    main(_config.cli())
