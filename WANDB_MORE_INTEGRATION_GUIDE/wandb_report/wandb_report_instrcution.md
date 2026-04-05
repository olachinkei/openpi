# OpenPI W&B Integration Outline

This document organizes the main OpenPI W&B integration changes for the report. Each section starts with the workflow issue we wanted to solve, then summarizes the corresponding changes.

Overview reports:
English: https://wandb.ai/wandb-smle/openpi-aloha-wandb-integration/reports/OpenPI-x-W-B--VmlldzoxNjQyMzc4Mg
Japanese: https://wandb.ai/wandb-smle/openpi-aloha-wandb-integration/reports/OpenPI-x-W-B--VmlldzoxNjQxNjczNQ



## Code Change Notes

The sections below are organized around the workflow improvements that matter most in the report.  
The goal is to keep implementation details concise and make the change in workflow easy to understand.

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

- Evaluation can be run either inside the training allocation or on separate nodes or jobs.
- In this integration, we chose the inline path so training and evaluation are easier to inspect together and the overall code path stays simpler.

**Changes**

- Added `src/openpi/training/aloha_eval.py` to factor out a reusable evaluation loop for ALOHA Sim.
- Added an inline local eval flow to `scripts/train.py`, so periodic eval and final eval can run in the same Slurm job and on the same node after checkpoint save.
- This keeps training and evaluation easier to inspect together in one place and simplifies the overall control flow. Running evaluation on separate nodes or separate jobs is still a valid approach when that fits the workflow better.

### 4. Make checkpoint steps and eval results easier to match

**Purpose**

- With asynchronous eval, metrics and videos can appear to belong to whatever training step happened to be current at that time, which is confusing.
- We want to inspect results per checkpoint, such as `5000`, `10000`, and `15000`.

**Changes**

- Triggered periodic eval at checkpoint save time and always recorded the corresponding `eval/checkpoint_step`.
- Enabled `block_on_periodic_eval=True` for `pi0_aloha_sim`, so training does not move to the next step until periodic eval finishes. If evaluation is run on a separate node or as a separate job, this is not required.
- As a result, each eval result has a clear meaning: it is the result of evaluating a specific checkpoint.

### 5. Fix the held-out eval split so results are comparable

**Purpose**

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
- At the same time, full resume state is heavy and can create storage and I/O pressure in GPU environments.

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

### 10. Make JAX + MuJoCo + W&B stable in GPU environments

**Purpose**

- We want JAX training and evaluation to run stably in a remote GPU environment.

**Changes**

- Added job-side runtime and cache configuration so training and evaluation run more reliably in remote environments.
- Added headless evaluation settings needed for MuJoCo-based evaluation on GPU nodes.
- Updated `scripts/train.py` so the JAX compilation cache path also respects the environment configuration.

### 11. Make larger JAX training and follow-up sweeps easier to run

**Purpose**

- We want to run not only smoke tests, but also larger JAX training jobs closer to full-scale runs in GPU environments.
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

### 13. Add job scripts for running PyTorch in GPU environments

**Purpose**

- We want stable smoke runs and full runs on the PyTorch side as well.
- We want the runtime setup to stay aligned with the JAX side.

**Changes**

- Added `jobs/train_aloha_sim_pytorch_smoke_1gpu.sbatch` for short smoke runs on a single GPU.
- Added `jobs/train_aloha_sim_pytorch_8gpu_6h.sbatch` for full 1-node / 8-GPU PyTorch training.
- Added runtime, cache, and headless evaluation settings needed to run PyTorch jobs more reliably in remote environments.
- Added `jobs/submit_aloha_pytorch_sweep_4x8gpu_2h.sh` and `jobs/submit_aloha_pytorch_sweep_4x8gpu_8h.sh` so PyTorch sweeps can also be launched in batches.
