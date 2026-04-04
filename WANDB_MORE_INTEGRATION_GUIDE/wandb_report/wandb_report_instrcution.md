# W&B Report Writing Notes

This file is a working memo that organizes the OpenPI W&B integration changes in a format that is easy to reuse in a public report or blog post. Every item should always be explained in the order **"Purpose" -> "Changes."**

## Basic Policy

- `openpi/WANDB_MORE_INTEGRATION_GUIDE/wandb_report/wandb_report_en.md` and `openpi/WANDB_MORE_INTEGRATION_GUIDE/wandb_report/wandb_report_jp.md` should be managed as the markdown source of truth.
- Use `openpi/WANDB_MORE_INTEGRATION_GUIDE/wandb_report/wandb_report.py` to create and update W&B Reports from those markdown files.
- Assume the report will be published to:
  - `WANDB_ENTITY=wandb-smle`
  - `WANDB_PROJECT=openpi-aloha-wandb-integration`
- The notes below are intended to explain the OpenPI-side code changes in the order **"Purpose -> Changes"** without exception.

## Code Change Notes

The items below are ordered in a way that makes them easier to turn into a public-facing report.  
The goal is not to describe every implementation detail, but to clearly communicate what problem we wanted to solve and what we changed to solve it.

### 1. Make training runs easier to compare

**Purpose**

- It is hard to compare runs in the Workspace if task name, dataset name, backend, and git information are not recorded consistently.
- We want to make it easy to filter both JAX and PyTorch runs along the same axes.

**Changes**

- Added `src/openpi/utils/wandb/run_context.py` to unify W&B run initialization.
- Made `wandb.config` record metadata consistently, including `config_name`, `task_name`, `dataset_name`, `backend`, `git sha`, `branch`, `hostname`, `cluster`, and `gpu_type`.
- Allowed the `WANDB_PROJECT` environment variable to override the default project name from config.
- Simplified tag handling so that it mainly relies on `wandb_tags` and `WANDB_TAGS` instead of automatically adding too many tags.
- Aligned JAX and PyTorch so both leave behind comparison-friendly metadata in the same general format.

### 2. Show train / periodic eval / final eval in the same W&B run

**Purpose**

- If train runs and eval runs are separated, the context becomes fragmented in W&B and comparison gets harder.
- If the step alignment is unclear, it becomes difficult to understand checkpoint-by-checkpoint changes.

**Changes**

- Extended `scripts/train.py` so that train, periodic eval, and final eval are aggregated into the same W&B run.
- Standardized metric namespaces as `train/*`, `eval/*`, and `eval_final/*`.
- Logged periodic eval and final eval results into the same run history, summary, and media.
- Shifted the user-facing workflow away from separate visible eval runs so that the train run itself becomes the main place to inspect everything.

### 3. Run evaluation inside the same training allocation instead of a separate node or job

**Purpose**

- Using additional eval nodes separately from training increases operational cost.
- If eval jobs sit in the pending queue, results do not come back during training.

**Changes**

- Added `src/openpi/training/aloha_eval.py` to factor out a reusable evaluation loop for ALOHA Sim.
- Added an inline local eval flow to `scripts/train.py`, so periodic eval and final eval can run in the same Slurm job and on the same node after checkpoint save.
- This made it possible to run training and evaluation without splitting them into separate allocations.

### 4. Make checkpoint steps and eval results easier to match

**Purpose**

- With asynchronous eval, metrics and videos can appear to belong to whatever training step happened to be current at that time, which is confusing.
- We want to inspect results per checkpoint, such as `5000`, `10000`, and `15000`.

**Changes**

- Triggered periodic eval at checkpoint save time and always recorded the corresponding `eval/checkpoint_step`.
- Enabled `block_on_periodic_eval=True` for `pi0_aloha_sim`, so training does not move to the next step until periodic eval finishes.
- As a result, each eval result has a clear meaning: it is the result of evaluating a specific checkpoint.

### 5. Fix the held-out eval split so results are comparable

**Purpose**

- If each run evaluates different episodes, run-to-run comparison becomes unstable.
- We want periodic eval and final eval to serve different roles.

**Changes**

- Added `scripts/prepare_aloha_eval_manifests.py` so the ALOHA Sim Transfer Cube held-out split can be generated as a manifest.
- Created `manifests/aloha_sim_transfer_cube/eval_dataset_subsample.json` and `eval_dataset_full.json`.
- Excluded held-out episodes from training, used a small fixed subset for periodic eval, and used the full held-out split for final eval.
- Added shared manifest loading in `src/openpi/training/eval_manifest.py`.

### 6. Track dataset lineage through W&B artifacts

**Purpose**

- We want to be able to trace which training dataset and eval dataset were used for a run.
- We do not want to re-upload datasets every time and clutter the artifact space.

**Changes**

- Added `src/openpi/training/dataset_artifacts.py` to centralize dataset artifact reference and resolution logic.
- Made it possible to resolve training, periodic eval, and final eval artifact refs from config.
- Used `wandb.run.use_artifact(...)` so manifests are resolved from existing registry artifacts instead of repeatedly uploading the same data.
- Also left a path available for publishing dataset artifacts when needed.

### 7. Publish checkpoints as W&B artifacts while keeping storage lightweight

**Purpose**

- We want to preserve checkpoints as W&B artifacts.
- At the same time, full resume state is heavy, and on CoreWeave it creates storage and I/O pressure.

**Changes**

- Adjusted `src/openpi/training/checkpoints.py` and `scripts/train.py` so `pi0_aloha_sim` uses a lightweight checkpoint strategy centered on `params` and `assets`.
- Standardized on `save_resume_state=False`, so large optimizer and resume state are not forcibly retained.
- Made checkpoint artifact upload intervals configurable, so they can be disabled for smoke runs and enabled for full runs.

### 8. Make videos and metrics easier to inspect in W&B

**Purpose**

- Logging too many images and videos during training makes the W&B UI noisy.
- For periodic eval and final eval, we want rollout videos as well as scalar metrics in the same run.

**Changes**

- Removed sample image and sample video logging from the training phase and kept training logging mostly scalar-focused.
- Logged rollout videos only during evaluation using `wandb.Video`.
- Added `src/openpi/utils/wandb/videos.py` to stabilize video logging key names.
- Adjusted video resolution to `640x336` so it is easier to inspect in the W&B UI.
- Switched ALOHA Sim camera views away from `top` and toward `angle`-based views for more readable side-view videos.

### 9. Use W&B Tables to inspect leaderboards and sample-level details

**Purpose**

- Run summary alone does not make it easy to tell which episode succeeded or which video corresponds to which result.
- We want a table that makes checkpoint-by-checkpoint evaluation results easy to compare.

**Changes**

- Updated `scripts/eval_aloha_dataset.py` so it outputs both episode-level results and run-level summaries as W&B Tables.
- Added `src/openpi/utils/wandb/tables.py` and `src/openpi/utils/wandb/leaderboard.py` to provide a stable-schema table logger.
- Kept the leaderboard table minimal, with columns:
  - `checkpoint_step`
  - `primary_score`
  - `success_rate`
  - `mean_max_reward`
  - `num_examples`
- Added an example table so readers can trace episode-level results together with the corresponding videos.

### 10. Make JAX + MuJoCo + W&B stable on CoreWeave

**Purpose**

- Even if things work locally, they often fail on CoreWeave because of GL, OSMesa, cache paths, or home directory quota issues.
- If the JAX compilation cache or W&B staging directory lands in home, capacity problems show up quickly.

**Changes**

- Prepared `jobs/train_aloha_sim_jax_8gpu_6h.sbatch` around an `OPENPI_RUNTIME_ROOT=/mnt/data/...` workflow.
- Redirected `WANDB_DIR`, `WANDB_CACHE_DIR`, `WANDB_ARTIFACT_DIR`, `TMPDIR`, `XDG_CACHE_HOME`, `UV_CACHE_DIR`, and `JAX_COMPILATION_CACHE_DIR` into `/mnt/data`.
- Added `MUJOCO_GL=osmesa`, `PYOPENGL_PLATFORM=osmesa`, and vendor GL library settings to the job script so headless eval works on GPU nodes.
- Updated `scripts/train.py` so the JAX compilation cache path also respects the environment configuration.

### 11. Make larger JAX training and follow-up sweeps easier to run

**Purpose**

- We want to run not only smoke tests, but also larger JAX training jobs closer to full-scale runs on CoreWeave.
- We want to keep launching follow-up experiments while looking at W&B results.

**Changes**

- Added `jobs/train_aloha_sim_jax_8gpu_6h.sbatch` as a shared 1-node / 8-GPU JAX training job.
- Added `jobs/submit_aloha_sweep_4x8gpu_3h.sh`, `jobs/submit_aloha_sweep_3x8gpu_3h.sh`, `jobs/submit_aloha_followup_2x8gpu_8h.sh`, and `jobs/submit_aloha_followup_4x8gpu_8h.sh` so follow-up experiments can be launched in batches.
- Based on earlier time-limit failures, made `TIME_LIMIT` configurable from outside the submit scripts.

### 12. Make the W&B experience in PyTorch match JAX as closely as possible

**Purpose**

- Even if the backend changes, we want the W&B view and operational flow to stay consistent.
- If only JAX supports same-run eval, comparison with PyTorch remains awkward.

**Changes**

- Added the same metric namespaces to `scripts/train_pytorch.py`: `train/*`, `eval/*`, and `eval_final/*`.
- Added an inline local eval path so PyTorch periodic eval and final eval are also logged into the same W&B run.
- Used `src/openpi/training/aloha_eval.py` and `src/openpi/policies/policy_config.py` so an ALOHA Sim eval policy can be built directly from the PyTorch model.
- Added `PendingEvalResult`, result import markers, and flush / wait logic so result JSON can be imported back into the parent train run when needed.
- Ensured that in DDP, only the main rank handles eval and W&B logging, while other ranks synchronize via marker files and barriers.
- Limited video, table, and summary handling to the main rank so same-run logging is less likely to break.
- Aligned dataset artifact handling with JAX by resolving manifests from existing registry artifacts.

### 13. Add job scripts for running PyTorch on CoreWeave

**Purpose**

- We want stable smoke runs and full runs on the PyTorch side as well.
- We want a runtime, cache, and W&B directory layout that is comparable to the JAX side.

**Changes**

- Added `jobs/train_aloha_sim_pytorch_smoke_1gpu.sbatch` for short smoke runs on a single GPU.
- Added `jobs/train_aloha_sim_pytorch_8gpu_6h.sbatch` for full 1-node / 8-GPU PyTorch training.
- Explicitly set `OPENPI_RUNTIME_ROOT`, `WANDB_DIR`, `WANDB_CACHE_DIR`, `WANDB_ARTIFACT_DIR`, `TMPDIR`, `XDG_CACHE_HOME`, and `UV_CACHE_DIR` in both job scripts so runtime and cache paths stay organized.
- Added `MUJOCO_GL=osmesa` and vendor GL settings so headless eval also works on the PyTorch side.
- Added `jobs/submit_aloha_pytorch_sweep_4x8gpu_2h.sh` and `jobs/submit_aloha_pytorch_sweep_4x8gpu_8h.sh` so PyTorch sweeps can also be launched in batches.

### 14. Make it possible to create and update W&B Reports from markdown

**Purpose**

- We want the report body to be managed in markdown as the source of truth.
- After editing the report text, we want to update the W&B report quickly.

**Changes**

- Added `WANDB_MORE_INTEGRATION_GUIDE/wandb_report/wandb_report.py`.
- Made it read `wandb_report_jp.md` and `wandb_report_en.md`, then split them into H1 / H2 / H3 headings and body blocks for W&B Report generation.
- Represented body sections as `MarkdownBlock` and kept headings as heading blocks in the report itself.
- Added `--dry-run` so title and block counts can be checked before upload.
- Added `--report-url` so an existing report can be overwritten by URL.
- Set the default destination to `wandb-smle/openpi-aloha-wandb-integration`.

## How To Write The Public Report

- Always write `Purpose` first in each section.
- Then write `Changes`.
- Keep implementation details to the minimum needed, and also state how the change improved the user or researcher experience.
- The following four points are especially important and should be emphasized repeatedly:
  - Train, periodic eval, and final eval are aggregated into the same W&B run.
  - Inline eval makes it possible to run evaluation without increasing the number of extra nodes.
  - Dataset, checkpoint, video, and metrics can all be tracked consistently in W&B.
  - JAX and PyTorch are aligned so the W&B experience is as similar as possible across both backends.

## What To Prioritize In The Public Report

- In the public-facing report body, focus mainly on items 1 through 13 first.
- Item 14, the report generation script, can be mentioned separately as an operational note if needed.
- The main thing readers care about is how the research workflow improved, so report automation should not become the main storyline.
