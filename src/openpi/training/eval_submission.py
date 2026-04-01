from __future__ import annotations

import logging
import os
import pathlib
import subprocess
import sys

import openpi.training.config as _config
import openpi.training.eval_tracking as _eval_tracking

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]


def _resolve_repo_path(path: str | None) -> pathlib.Path | None:
    if not path:
        return None
    candidate = pathlib.Path(path)
    if candidate.is_absolute():
        return candidate
    return (REPO_ROOT / candidate).resolve()


def build_eval_output_paths(
    config: _config.TrainConfig,
    *,
    checkpoint_step: int,
    split_name: str,
) -> tuple[pathlib.Path, pathlib.Path]:
    metric_namespace = _eval_tracking.metric_namespace_for_split(split_name)
    openpi_root = pathlib.Path(os.environ.get("OPENPI_ROOT", str(REPO_ROOT))).resolve()
    runtime_root = pathlib.Path(os.environ.get("OPENPI_RUNTIME_ROOT", str(openpi_root / "data"))).resolve()
    video_root = runtime_root / "aloha_eval" / config.exp_name / f"{metric_namespace}-step-{checkpoint_step}"
    results_path = video_root / _eval_tracking.results_filename_for_split(split_name)
    return video_root, results_path


def _runtime_paths() -> dict[str, pathlib.Path]:
    openpi_root = pathlib.Path(os.environ.get("OPENPI_ROOT", str(REPO_ROOT))).resolve()
    runtime_root = pathlib.Path(os.environ.get("OPENPI_RUNTIME_ROOT", str(openpi_root / "data"))).resolve()
    return {
        "openpi_root": openpi_root,
        "runtime_root": runtime_root,
        "data_home": pathlib.Path(os.environ.get("OPENPI_DATA_HOME", str(runtime_root / "cache"))).resolve(),
        "wandb_dir": pathlib.Path(os.environ.get("WANDB_DIR", str(runtime_root / "wandb"))).resolve(),
        "wandb_cache_dir": pathlib.Path(os.environ.get("WANDB_CACHE_DIR", str(runtime_root / "wandb-cache"))).resolve(),
        "wandb_data_dir": pathlib.Path(os.environ.get("WANDB_DATA_DIR", str(runtime_root / "wandb-data"))).resolve(),
        "wandb_artifact_dir": pathlib.Path(
            os.environ.get("WANDB_ARTIFACT_DIR", str(runtime_root / "wandb-artifacts"))
        ).resolve(),
        "tmpdir": pathlib.Path(os.environ.get("TMPDIR", str(runtime_root / "tmp"))).resolve(),
        "xdg_cache_home": pathlib.Path(os.environ.get("XDG_CACHE_HOME", str(runtime_root / "xdg-cache"))).resolve(),
        "uv_cache_dir": pathlib.Path(os.environ.get("UV_CACHE_DIR", str(runtime_root / "uv-cache"))).resolve(),
    }


def _prepare_local_eval_env() -> dict[str, str]:
    runtime_paths = _runtime_paths()
    env = os.environ.copy()

    for directory in runtime_paths.values():
        directory.mkdir(parents=True, exist_ok=True)

    env.update(
        {
            "OPENPI_ROOT": str(runtime_paths["openpi_root"]),
            "OPENPI_RUNTIME_ROOT": str(runtime_paths["runtime_root"]),
            "OPENPI_DATA_HOME": str(runtime_paths["data_home"]),
            "WANDB_DIR": str(runtime_paths["wandb_dir"]),
            "WANDB_CACHE_DIR": str(runtime_paths["wandb_cache_dir"]),
            "WANDB_DATA_DIR": str(runtime_paths["wandb_data_dir"]),
            "WANDB_ARTIFACT_DIR": str(runtime_paths["wandb_artifact_dir"]),
            "TMPDIR": str(runtime_paths["tmpdir"]),
            "XDG_CACHE_HOME": str(runtime_paths["xdg_cache_home"]),
            "UV_CACHE_DIR": str(runtime_paths["uv_cache_dir"]),
        }
    )

    pythonpath_entries = [str(runtime_paths["openpi_root"] / "src")]
    if env.get("PYTHONPATH"):
        pythonpath_entries.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(pythonpath_entries)

    env.setdefault("MUJOCO_GL", "osmesa")
    if env["MUJOCO_GL"] == "osmesa":
        env.setdefault("PYOPENGL_PLATFORM", "osmesa")

    vendor_gl_libdir = pathlib.Path(
        env.get("VENDOR_GL_LIBDIR", str(pathlib.Path.home() / "vendor-gl" / "root" / "usr" / "lib" / "x86_64-linux-gnu"))
    )
    if vendor_gl_libdir.exists():
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{vendor_gl_libdir}:{existing}" if existing else str(vendor_gl_libdir)

    return env


def _build_local_eval_command(
    config: _config.TrainConfig,
    *,
    checkpoint_step: int,
    manifest_path_str: str | None,
    split_name: str,
    num_examples: int | None,
) -> tuple[list[str], dict[str, str], pathlib.Path] | None:
    if manifest_path_str is None:
        return None

    manifest_path = _resolve_repo_path(manifest_path_str)
    if manifest_path is None or not manifest_path.exists():
        logging.warning("Skipping local eval because manifest does not exist: %s", manifest_path)
        return None

    checkpoint_dir = pathlib.Path(config.checkpoint_dir) / str(checkpoint_step)
    if not checkpoint_dir.exists():
        logging.warning("Skipping local eval because checkpoint is missing: %s", checkpoint_dir)
        return None

    env = _prepare_local_eval_env()
    openpi_root = pathlib.Path(env["OPENPI_ROOT"])
    video_root, results_path = build_eval_output_paths(
        config,
        checkpoint_step=checkpoint_step,
        split_name=split_name,
    )
    video_root.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(openpi_root / "scripts" / "eval_aloha_dataset.py"),
        "--config-name",
        config.name,
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--manifest",
        str(manifest_path),
        "--video-dir",
        str(video_root),
        "--split",
        split_name,
        "--results-path",
        str(results_path),
        "--checkpoint-alias",
        f"step-{checkpoint_step}",
        "--checkpoint-step",
        str(checkpoint_step),
        "--no-attach-to-source-train-run",
        "--no-wandb-enabled",
    ]
    if num_examples is not None:
        command.extend(["--num-examples", str(num_examples)])

    render_mode = env.get("RENDER_MODE")
    if render_mode:
        command.extend(["--render-mode", render_mode])
    if env.get("VISUALIZATION_WIDTH"):
        command.extend(["--visualization-width", env["VISUALIZATION_WIDTH"]])
    if env.get("VISUALIZATION_HEIGHT"):
        command.extend(["--visualization-height", env["VISUALIZATION_HEIGHT"]])
    if env.get("VISUALIZATION_CAMERA_ID"):
        command.extend(["--visualization-camera-id", env["VISUALIZATION_CAMERA_ID"]])

    return command, env, results_path


def run_eval_locally(
    config: _config.TrainConfig,
    *,
    checkpoint_step: int,
    split_name: str,
    num_examples: int | None,
) -> pathlib.Path | None:
    manifest_path_str = config.eval_manifest_path if split_name == config.eval_split_name else config.final_eval_manifest_path
    invocation = _build_local_eval_command(
        config,
        checkpoint_step=checkpoint_step,
        manifest_path_str=manifest_path_str,
        split_name=split_name,
        num_examples=num_examples,
    )
    if invocation is None:
        return None

    command, env, results_path = invocation
    runtime_root = pathlib.Path(env["OPENPI_RUNTIME_ROOT"])
    try:
        subprocess.run(command, check=True, env=env, cwd=runtime_root)
    except subprocess.CalledProcessError as exc:
        logging.warning(
            "Local eval failed for checkpoint %s (split=%s): returncode=%s",
            checkpoint_step,
            split_name,
            exc.returncode,
        )
        return None

    logging.info("Finished local %s eval for checkpoint step %s.", split_name, checkpoint_step)
    return results_path


def _submit_eval_job(
    config: _config.TrainConfig,
    *,
    checkpoint_step: int,
    manifest_path_str: str | None,
    job_script_path_str: str | None,
    split_name: str,
    num_examples: int | None,
    dependency_job_id: str | None = None,
) -> str | None:
    if manifest_path_str is None or job_script_path_str is None:
        return None

    job_script = _resolve_repo_path(job_script_path_str)
    manifest_path = _resolve_repo_path(manifest_path_str)
    if job_script is None or manifest_path is None:
        return None

    if not job_script.exists():
        logging.warning("Skipping eval submission because job script does not exist: %s", job_script)
        return None
    if not manifest_path.exists():
        logging.warning("Skipping eval submission because manifest does not exist: %s", manifest_path)
        return None

    checkpoint_dir = pathlib.Path(config.checkpoint_dir) / str(checkpoint_step)
    if not checkpoint_dir.exists():
        logging.warning("Skipping eval submission because checkpoint is missing: %s", checkpoint_dir)
        return None

    openpi_root = pathlib.Path(os.environ.get("OPENPI_ROOT", str(REPO_ROOT))).resolve()
    runtime_root = pathlib.Path(os.environ.get("OPENPI_RUNTIME_ROOT", str(openpi_root / "data"))).resolve()
    video_root, results_path = build_eval_output_paths(
        config,
        checkpoint_step=checkpoint_step,
        split_name=split_name,
    )

    env = os.environ.copy()
    env.update(
        {
            "OPENPI_ROOT": str(openpi_root),
            "OPENPI_RUNTIME_ROOT": str(runtime_root),
            "CONFIG_NAME": config.name,
            "CHECKPOINT_DIR": str(checkpoint_dir),
            "MANIFEST_PATH": str(manifest_path),
            "SPLIT_NAME": split_name,
            "CHECKPOINT_STEP": str(checkpoint_step),
            "CHECKPOINT_ALIAS": f"step-{checkpoint_step}",
            "VIDEO_ROOT": str(video_root),
            "RESULTS_PATH": str(results_path),
            "EXTRA_EVAL_ARGS": "--no-attach-to-source-train-run --no-wandb-enabled",
        }
    )
    if num_examples is not None:
        env["NUM_EXAMPLES"] = str(num_examples)

    export_names = [
        "HOME",
        "PATH",
        "USER",
        "LOGNAME",
        "SHELL",
        "TMPDIR",
        "LANG",
        "LC_ALL",
        "UV_BIN",
        "UV_CACHE_DIR",
        "XDG_CACHE_HOME",
        "OPENPI_ROOT",
        "OPENPI_RUNTIME_ROOT",
        "OPENPI_DATA_HOME",
        "CONFIG_NAME",
        "CHECKPOINT_DIR",
        "MANIFEST_PATH",
        "SPLIT_NAME",
        "CHECKPOINT_STEP",
        "CHECKPOINT_ALIAS",
        "VIDEO_ROOT",
        "RESULTS_PATH",
        "EXTRA_EVAL_ARGS",
        "NUM_EXAMPLES",
        "WANDB_API_KEY",
        "WANDB_ENTITY",
        "WANDB_PROJECT",
        "WANDB_TAGS",
        "WANDB_DIR",
        "WANDB_CACHE_DIR",
        "WANDB_DATA_DIR",
        "WANDB_ARTIFACT_DIR",
        "OPENPI_VIDEO_BACKEND",
        "VENDOR_GL_LIBDIR",
        "MUJOCO_GL",
        "PYOPENGL_PLATFORM",
        "RENDER_MODE",
        "VISUALIZATION_WIDTH",
        "VISUALIZATION_HEIGHT",
        "VISUALIZATION_CAMERA_ID",
    ]
    export_arg = ",".join(name for name in export_names if name in env and env[name] != "")

    try:
        submit_cmd = ["sbatch", "--parsable"]
        if dependency_job_id:
            submit_cmd.append(f"--dependency=afterany:{dependency_job_id}")
        submit_cmd.extend([f"--export={export_arg}", str(job_script)])
        result = subprocess.run(
            submit_cmd,
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
    except FileNotFoundError:
        logging.warning("Skipping eval submission because sbatch was not found in PATH.")
        return None
    except subprocess.CalledProcessError as exc:
        logging.warning(
            "Eval submission failed for checkpoint %s (split=%s): %s",
            checkpoint_step,
            split_name,
            exc.stderr.strip() or exc.stdout.strip(),
        )
        return None

    job_id = result.stdout.strip()
    logging.info("Submitted %s eval job %s for checkpoint step %s.", split_name, job_id, checkpoint_step)
    return job_id or None


def maybe_submit_periodic_eval(
    config: _config.TrainConfig,
    *,
    checkpoint_step: int,
    dependency_job_id: str | None = None,
) -> str | None:
    return _submit_eval_job(
        config,
        checkpoint_step=checkpoint_step,
        manifest_path_str=config.eval_manifest_path,
        job_script_path_str=config.eval_job_script_path,
        split_name=config.eval_split_name,
        num_examples=config.periodic_eval_num_examples,
        dependency_job_id=dependency_job_id,
    )


def maybe_submit_final_eval(
    config: _config.TrainConfig,
    *,
    checkpoint_step: int,
    dependency_job_id: str | None = None,
) -> str | None:
    return _submit_eval_job(
        config,
        checkpoint_step=checkpoint_step,
        manifest_path_str=config.final_eval_manifest_path,
        job_script_path_str=config.final_eval_job_script_path or config.eval_job_script_path,
        split_name=config.final_eval_split_name,
        num_examples=config.final_eval_num_examples,
        dependency_job_id=dependency_job_id,
    )
