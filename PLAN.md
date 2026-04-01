# openpi ALOHA Sim + W&B Integration Plan

- Please write actual steps for comamnd in DEMO_STEPS.md　(including how to use CW console). Please make the training part undependent on GPU vendors.
- Please write what the integration is like in WANDB_INTEGRATION.md for those who will use wandb integration. This will be a doc for users. 

## Status

- This document is a planning and runbook document only.
- No code changes for W&B, evaluation, or automation are included yet.
- The primary execution environment is CoreWeave SUNK.
- Cluster-specific access rules come from `CW_console_doc.md`.

## CoreWeave source hierarchy

Use these sources in this order when operating the cluster:

1. `CW_console_doc.md`
2. CoreWeave SUNK official docs
3. Generic Slurm assumptions

This matters because:

- the training cluster has a cluster-specific login path and org suffix
- official SUNK docs describe the generic access model and Slurm best practices
- generic Slurm habits are useful, but they should not override cluster-specific instructions

## Official SUNK behaviors that affect this plan

From the official SUNK docs, the assumptions baked into this plan are:

- login nodes are for access, job submission, data preparation, and status checks
- compute work belongs on Slurm compute nodes, not on login nodes
- compute access should happen through Slurm commands such as `srun`, `sbatch`, and `salloc`
- direct SSH to compute nodes should be treated as debug-only
- shared storage and object storage are both valid building blocks for multi-job workflows

## Working decision

- Use ALOHA Sim as the primary demo and training target.
- Use one specific task first, not a multi-task mixture.
- The chosen first task is `Transfer cube`.
- The starting config is `pi0_aloha_sim`.

Why this choice:

- the repo already ships an ALOHA Sim config and default checkpoint path
- ALOHA Sim is easier to evaluate repeatedly than real ALOHA
- `Transfer cube` is the simplest path to a stable end-to-end demo
- it preserves the option to move to real ALOHA later without throwing away the W&B integration layer

## Goal

Build a reproducible workflow that supports:

1. Baseline evaluation of the public ALOHA Sim checkpoint.
2. Fine-tuning for one ALOHA Sim task, `Transfer cube`.
3. W&B-based tracking for train metrics, checkpoints, evaluation results, tables, and videos.
4. Periodic evaluation during training on a fixed eval subsample.
5. Final evaluation on a full evaluation dataset.
6. Side-by-side comparison across baseline and custom runs.

## Non-goals for the first pass

- multi-task ALOHA training
- real-robot ALOHA evaluation
- external benchmark integration
- multi-node training
- changing the core model architecture

## Current repo state

Existing building blocks in the repo:

- JAX training loop: `scripts/train.py`
- PyTorch training loop: `scripts/train_pytorch.py`
- Policy serving: `scripts/serve_policy.py`
- ALOHA Sim run example: `examples/aloha_sim/main.py`
- ALOHA Sim video saver: `examples/aloha_sim/saver.py`
- ALOHA real runtime example: `examples/aloha_real/main.py`
- ALOHA input/output transforms: `src/openpi/policies/aloha_policy.py`
- ALOHA training configs: `src/openpi/training/config.py`

Relevant configs already in the repo:

- `pi0_aloha_sim`
- `pi0_aloha_pen_uncap`
- `pi05_aloha_pen_uncap`

Current logging behavior:

- `scripts/train.py` already logs train loss, grad norm, param norm, and first-batch camera views to W&B
- `scripts/train_pytorch.py` already logs train loss, learning rate, grad norm, and checkpoint step to W&B
- neither path currently runs automated eval during training
- neither path currently publishes model checkpoints as W&B artifacts

Current ALOHA Sim behavior:

- `examples/aloha_sim/main.py` runs the policy in simulation
- `examples/aloha_sim/saver.py` already saves episode videos
- there is not yet a dedicated eval-dataset runner for ALOHA Sim

## Specific task definition

The first training and evaluation target is:

- environment family: `ALOHA_SIM`
- task: `Transfer cube`
- training config: `pi0_aloha_sim`
- default prompt: `Transfer cube`

This task-specific choice is intentional.

Reasons:

- it reduces ambiguity in model selection and evaluation
- it makes the leaderboard easier to interpret
- it keeps the first W&B schema simple

The W&B utility layer should still stay generic.
Only the experiment configuration should be task-specific.

## Evaluation model

Use two evaluation layers.

### 1. Periodic eval during training

Purpose:

- run repeatedly during training
- be cheap enough to execute every few thousand steps
- provide a stable signal for model selection

Data source:

- a fixed eval subsample
- this subsample must come from the final evaluation dataset, not from random train minibatches

Outputs:

- eval loss on the subsample
- any additional action-level metrics we define later
- a curated set of ALOHA Sim videos

### 2. Final evaluation

Purpose:

- serve as the main comparison surface across models and checkpoints
- be the result recorded in the final leaderboard

Data source:

- the full evaluation dataset

Outputs:

- full-dataset evaluation metrics
- selected ALOHA Sim videos for qualitative review
- a one-row leaderboard result per evaluated run

## Data split policy

Preferred:

- `train_dataset`
- `eval_dataset_full`
- `eval_dataset_subsample`

Rules:

- `eval_dataset_subsample` must be a fixed subset of `eval_dataset_full`
- `eval_dataset_full` must not overlap with the training split
- all manifests must be versioned and tracked

Fallback if no official split exists:

- create a fixed held-out eval split from the available ALOHA Sim dataset
- materialize it as a manifest file
- store it in git and as a W&B artifact

Important:

- mid-training eval always uses the same subsample
- final eval always uses the full eval dataset
- do not use random train batches as a proxy for evaluation

## ALOHA experiment ladder

### Phase 0: pipeline shakedown

Use the existing `pi0_aloha_sim` path first.

Why:

- the dataset and config already exist
- the default checkpoint already exists
- the sim runtime already exists

### Phase 1: baseline evaluation

Evaluate the public ALOHA Sim checkpoint first:

- config: `pi0_aloha_sim`
- checkpoint: `gs://openpi-assets/checkpoints/pi0_aloha_sim`

This gives:

- an immediate baseline
- a first target for custom runs
- a fixed comparison row in W&B

### Phase 2: fine-tune from the base model

Use the existing `pi0_aloha_sim` config path as the starting point.

Keep the first run simple:

- one task
- one prompt
- one eval split
- one visualization style

### Phase 3: iterate

Once the workflow is stable:

- compare multiple checkpoints from the same run
- compare multiple hyperparameter variants
- optionally test `pi05` or a real ALOHA dataset path later

## Primary metrics philosophy

Training selection should use:

- train loss for health and debugging
- eval-dataset metrics for actual checkpoint selection

The checkpoint chosen as `best` should come from periodic eval, not from train loss alone.

## W&B integration plan

## Project structure

Recommended W&B project layout:

- project: `openpi-aloha`
- entity: your team entity

Use run groups:

- `baseline_eval`
- `train`
- `periodic_eval`
- `final_eval`

Use tags:

- `env:aloha_sim`
- `task:transfer_cube`
- `backend:jax` or `backend:pytorch`
- `config:<config_name>`
- `dataset:<dataset_name>`
- `split:subsample` or `split:full`
- `baseline` / `custom`

Recommended run name pattern:

- `aloha-train-<config>-<exp_name>`
- `aloha-eval-subsample-<config>-step<step>`
- `aloha-eval-full-<config>-<checkpoint_alias>`

## Evaluation visibility requirements

The W&B design must support two views at the same time:

1. Run-centric inspection
2. Cross-run leaderboard inspection

To support both:

- every eval-capable run should write scalar metrics into `run.summary`
- every eval-capable run should also emit a stable evaluation table row
- the evaluation table contract should be `1 evaluated run = 1 row`
- episode-level and media-level details should be stored separately from the leaderboard row

This is important because:

- summary metrics are best for normal multi-run charts and filters
- a one-row evaluation table is best for leaderboard-style comparison
- episode/media detail should not pollute the top-level comparison table

## W&B table model

Use three table layers.

### 1. Leaderboard table

Purpose:

- cross-run comparison
- one row per evaluated run or checkpoint

Cardinality:

- exactly one row per evaluation run

Suggested key:

- `eval/leaderboard`

Suggested schema:

- `eval_run_id`
- `source_train_run_id`
- `source_train_run_name`
- `eval_name`
- `eval_split`
- `model_family`
- `config_name`
- `task_name`
- `dataset_name`
- `checkpoint_alias`
- `checkpoint_step`
- `primary_score`
- `eval_loss`
- `num_examples`
- `artifact_ref_results`
- `artifact_ref_media`
- `created_at`
- `notes`

Usage notes:

- the schema must be stable across environments and tasks
- task-specific details belong in the details table, not in the shared leaderboard row
- this table should stay small and queryable

### 2. Example details table

Purpose:

- per-example drill-down
- error analysis and qualitative review

Cardinality:

- one row per evaluated example or episode

Suggested key:

- `eval/examples`

Suggested schema:

- `example_id`
- `prompt`
- `task_name`
- `split`
- `metric_primary`
- `metric_aux_json`
- `checkpoint_step`
- `video`
- `artifact_ref_video`
- `metadata_json`

### 3. Media gallery table

Purpose:

- curated visual review
- compact dashboard inspection

Cardinality:

- selected rows only

Suggested key:

- `eval/gallery`

Suggested contents:

- representative successes
- representative failures
- visually informative edge cases

## Table logging modes

Recommended default choices:

- leaderboard table: `IMMUTABLE`
- final example details table: `IMMUTABLE`
- optional streaming example table during long eval: `INCREMENTAL`

Use `INCREMENTAL` only if live progress during evaluation becomes useful.
At the end of the run, materialize an `IMMUTABLE` final table for stable analysis.

## Leaderboard strategy

To make the dashboard usable later:

- each evaluation run writes one stable leaderboard row table
- the same run mirrors headline metrics into `run.summary`
- a later aggregation step can stitch those rows into a project-level leaderboard artifact if needed
- treat logged tables as artifact-backed data that can be reloaded and merged later

The first implementation should not depend on a global mutable leaderboard.
Instead, it should guarantee:

- stable row schema
- stable keys
- stable summary metric names

## Config to record in every run

Store at least:

- git SHA
- branch name if useful
- config name
- exp name
- dataset identifier
- eval manifest path
- checkpoint source
- backend (`jax` or `pytorch`)
- model family
- task name
- cluster / node / GPU type
- save interval
- eval interval

## Metrics to log

### Training metrics

Keep existing metrics and standardize them:

- `train/loss`
- `train/grad_norm`
- `train/param_norm`
- `train/learning_rate`
- `train/step_time_sec` if added later
- `train/data_time_sec` if added later

### Periodic eval metrics

- `eval_subsample/primary_score`
- `eval_subsample/loss`
- `eval_subsample/num_examples`
- `eval_subsample/checkpoint_step`

### Final eval metrics

- `eval_full/primary_score`
- `eval_full/loss`
- `eval_full/num_examples`
- `eval_full/checkpoint_step`

### Summary mirroring rule

Any metric needed for cross-run comparison should exist in two forms:

- as a scalar summary metric
- as a column in the one-row leaderboard table

Examples:

- `eval_full/primary_score`
- `eval_full/loss`
- `eval_full/num_examples`
- `eval_full/checkpoint_step`

## Artifact strategy

Track artifacts explicitly.

Artifact types:

- `dataset-manifest`
- `model-checkpoint`
- `eval-video-bundle`
- `eval-results`

Checkpoint artifact policy:

- save checkpoints locally at the repo-configured interval
- upload only milestone checkpoints to W&B, not every single save
- always upload:
  - the best eval checkpoint
  - the final checkpoint
  - the baseline checkpoint reference used in comparisons

Recommended aliases:

- `latest`
- `best-eval`
- `final`
- `step-<n>`
- `baseline-public`

Video artifact policy:

- upload a curated sample per eval run
- do not upload every video if storage becomes noisy
- prefer logging finalized video files rather than raw frame tensors
- keep:
  - a fixed gallery of representative cases
  - failures
  - best-case examples

## Media logging strategy

Use two storage paths for media.

### 1. Run-visible media

Use this for:

- dashboard inspection
- a small curated set of ALOHA Sim videos or images

Recommended objects:

- `wandb.Video` for videos
- `wandb.Image` for stills and overlays
- table media columns for example drill-down

Recommended default for videos:

- save finalized local `.mp4` files first
- then log those files to W&B

### 2. Artifact-backed media bundles

Use this for:

- larger eval result bundles
- raw outputs that should be retained but not forced into the main run page

Examples:

- `eval-video-bundle`
- `eval-gallery`
- `failure-cases`

## ALOHA visualization requirements

Visualization must be ALOHA-specific.

For the first version, log:

- `cam_high` video for selected examples
- any additional camera views only if they are available and useful

The canonical visualization unit should be:

- one finalized ALOHA Sim rollout video per selected example

This keeps the visuals aligned with the target environment and avoids non-ALOHA assumptions.

## Object-oriented utility plan

Create a generic W&B utility package and keep task-specific logic out of it.

Proposed location:

- `src/openpi/utils/wandb/`

Design rule:

- environment-specific code should prepare normalized records
- `utils/wandb` should only know how to log generic records, tables, media, and artifacts

### Proposed module layout

- `src/openpi/utils/wandb/__init__.py`
- `src/openpi/utils/wandb/types.py`
- `src/openpi/utils/wandb/run_context.py`
- `src/openpi/utils/wandb/artifacts.py`
- `src/openpi/utils/wandb/tables.py`
- `src/openpi/utils/wandb/media.py`
- `src/openpi/utils/wandb/videos.py`
- `src/openpi/utils/wandb/leaderboard.py`

### Proposed generic data objects

- `LeaderboardRow`
- `ExampleRecord`
- `MediaRecord`
- `VideoRecord`
- `ArtifactRecord`

These should be generic dataclasses or typed records.
They should not contain ALOHA-only field names.

### Proposed generic logger objects

- `WandbRunContext`
- `WandbArtifactManager`
- `WandbTableLogger[T]`
- `LeaderboardTableLogger`
- `ExampleTableLogger`
- `MediaLogger`
- `VideoLogger`

### Design responsibilities

`WandbRunContext`

- owns the active run
- standardizes config, tags, job type, and summary metrics

`WandbArtifactManager`

- publishes and aliases model, dataset, result, and media artifacts
- keeps artifact naming consistent

`WandbTableLogger[T]`

- owns schema validation
- converts typed records into table rows
- hides table mode details from callers

`LeaderboardTableLogger`

- accepts exactly one logical evaluation result and writes one leaderboard row
- also mirrors the same headline fields into `run.summary`

`ExampleTableLogger`

- logs per-example detail rows
- supports a final immutable materialization step

`MediaLogger`

- handles generic images, html, and rich media attachments

`VideoLogger`

- handles local video discovery, captioning, and `wandb.Video` conversion
- keeps file-based upload policy out of task code

## Separation of concerns rule

Task code should do only this:

- run the environment or dataset evaluation
- compute raw scores
- produce normalized generic records

Task code should not:

- construct W&B tables by hand
- know artifact naming rules
- decide media upload policy
- embed W&B-specific branching logic everywhere

This keeps the W&B integration reusable for:

- ALOHA Sim periodic eval
- ALOHA Sim final eval
- future ALOHA real eval
- future non-ALOHA eval paths

## Dashboard and report contract

The dashboard should be able to answer these questions without custom code:

- what is the current best run?
- what is the latest full-eval score per run?
- which checkpoint produced the best periodic eval?
- what do representative failures look like?

Minimum dashboard surfaces:

- line charts for scalar training metrics
- line charts for periodic eval metrics over checkpoint step
- leaderboard table panel
- example details table panel
- media gallery panel

## W&B reports and workspace assembly

Once the table contracts above are fixed, create one standard workspace/report that includes:

- training loss over time
- periodic eval score over checkpoint step
- final eval leaderboard table
- example drill-down table
- linked media panels

The report should rely on stable scalar keys and stable table keys, not on task-specific custom panels.

## Proposed implementation phases

### Phase A: generic W&B utility foundation

Scope:

- create `src/openpi/utils/wandb/`
- define generic typed records and logger classes
- standardize W&B naming, tags, config, summary metrics, and artifact naming
- keep task-specific code out of the utility layer

Outcome:

- one reusable W&B foundation for training, periodic eval, and final eval jobs

### Phase B: normalize existing training logging

Scope:

- keep current train logging
- route training metrics and checkpoints through the generic W&B helpers
- add checkpoint artifact publishing hooks

Outcome:

- all train runs are queryable and comparable
- checkpoint lineage is standardized

### Phase C: ALOHA Sim periodic eval runner

Scope:

- add an automated ALOHA Sim evaluation script
- make it run against `eval_dataset_subsample`
- make it emit:
  - summary metrics
  - one-row leaderboard table
  - example details table
  - curated media gallery
  - CSV/JSON results and artifacts

Outcome:

- repeatable training-time evaluation

### Phase D: checkpoint-to-eval orchestration

Scope:

- whenever a training checkpoint hits a milestone step, run periodic eval
- log results to a dedicated W&B eval run
- link eval run to checkpoint artifact

Outcome:

- train and eval are connected by checkpoint lineage

### Phase E: final eval dataset runner

Scope:

- run final evaluation on `eval_dataset_full`
- compare:
  - baseline public checkpoint
  - best custom checkpoint
  - final custom checkpoint
- log outputs in the shared leaderboard shape

Outcome:

- final scorecard across models and checkpoints

## CoreWeave SUNK runbook

Important operating rule:

- perform cluster access and Slurm job submission from the Slurm login node
- do not SSH into compute nodes to run workloads directly
- do not use the login node for long-running GPU work

## Cluster profile for this project

Current training cluster assumptions from `CW_console_doc.md`:

- training org suffix: `cwb607`
- login host: `sunk.cwb607-training.coreweave.app`
- user pattern: `<coreweave_user>+cwb607`

If any of these change, update this document before running jobs.

## Access prerequisites

Before the first real training session:

1. Get invited to the CoreWeave Training Cluster org and assigned to the correct group.
2. Register and log in to CoreWeave Console using the invited account.
3. Upload your SSH public key in Console settings.
4. Add your private key to your local SSH agent or keychain.

Recommended local setup:

```bash
ssh-add ~/.ssh/id_ed25519
```

If you do not already have an SSH key:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519
```

Operational note:

- the cluster-specific login name may not match your local macOS username
- if `$(whoami)+cwb607` does not match your invited login, replace it with the exact cluster username manually

## Access pattern

From a local machine:

1. Connect to the SUNK login node.
2. Perform repo setup and Slurm submission from the login node.
3. Use Slurm to start compute work.
4. Keep long-running training and evaluation off the login node.

Primary login command for this cluster:

```bash
ssh -o IdentitiesOnly=yes $(whoami)+cwb607@sunk.cwb607-training.coreweave.app
```

If your local username does not match the cluster username:

```bash
ssh -o IdentitiesOnly=yes <your_cluster_user>+cwb607@sunk.cwb607-training.coreweave.app
```

Cluster sanity checks on the login node:

```bash
whoami
hostname
sinfo
squeue -u "$(whoami)"
```

## What runs where

### Local laptop

Use it for:

- SSH access
- editing local notes
- optional tunnel setup

### Slurm login node

Use it for:

- repo checkout
- `uv` setup
- W&B login
- manifest preparation
- Slurm submission
- lightweight inspection and notebook use

Do not use it for:

- long GPU jobs
- real training
- full evaluation jobs

### Slurm compute nodes

Use them only through Slurm for:

- training
- periodic eval
- final eval

## First-session checklist

Run this once before the first training attempt:

1. SSH into the login node.
2. Confirm `sinfo` and `squeue` work.
3. Install `uv` on the login node if it is not already available.
4. Clone the repo and sync dependencies.
5. Log in to W&B.
6. Run a tiny Slurm smoke test.
7. Only then submit the first training job.

Optional `uv` install on the login node:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
exec -l "$SHELL"
which uv
```

## Repo setup on the login node

```bash
cd <shared_workspace_root>
git clone --recurse-submodules <your_openpi_repo_url> openpi
cd openpi

GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

Notes:

- keep the repo on shared storage that both training and evaluation jobs can read
- if `uv` is not found, install it first on the login node

## W&B setup on the login node

Preferred:

```bash
wandb login
```

Or via environment variables:

```bash
export WANDB_ENTITY=<your_entity>
export WANDB_PROJECT=openpi-aloha
export WANDB_API_KEY=<your_api_key>
```

Optional run metadata env vars:

```bash
export OPENPI_EXPERIMENT_OWNER=<your_name>
export OPENPI_CLUSTER=coreweave-sunk
```

## Slurm smoke test before real training

Do this before the first real job submission.

Create `jobs/smoke.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=openpi_smoke
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

set -euo pipefail

mkdir -p logs
hostname
nvidia-smi
which uv || true
```

Submit:

```bash
sbatch jobs/smoke.sbatch
```

Check:

```bash
squeue -u "$(whoami)"
sacct -j <job_id>
tail -f logs/openpi_smoke_<job_id>.out
```

## ALOHA Sim data preparation

### Training dataset

Use the existing dataset implied by `pi0_aloha_sim`:

- repo id: `lerobot/aloha_sim_transfer_cube_human`

### Eval dataset plan

Create and version:

- `eval_dataset_full`
- `eval_dataset_subsample`

Recommended first approach:

1. define a held-out eval split from the available ALOHA Sim data
2. materialize a manifest file for the full eval split
3. derive a fixed smaller subsample manifest from it
4. track both manifests in git and W&B artifacts

## Training command

Use the existing config path first:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi0_aloha_sim --exp-name=<run_name> --overwrite
```

## Slurm training job template

Create `jobs/train_aloha_sim_jax.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=openpi_aloha_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

set -euo pipefail

cd /path/to/openpi
mkdir -p logs

export WANDB_ENTITY=<your_entity>
export WANDB_PROJECT=openpi-aloha
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

uv run scripts/train.py pi0_aloha_sim --exp-name=<run_name> --overwrite
```

Submit from the login node:

```bash
sbatch jobs/train_aloha_sim_jax.sbatch
```

Monitor:

```bash
squeue -u "$(whoami)"
sacct -j <job_id>
tail -f logs/openpi_aloha_train_<job_id>.out
```

Cancel if needed:

```bash
scancel <job_id>
```

## Optional interactive debug path

Use only when you need to debug a problem before switching back to batch jobs:

```bash
srun --pty --nodes=1 --ntasks=1 --gres=gpu:1 --time=01:00:00 bash
```

Inside the interactive shell:

```bash
nvidia-smi
cd /path/to/openpi
uv run scripts/train.py debug
```

## Storage recommendation

For this project, separate storage concerns into:

- repo and run scripts on persistent shared storage
- manifests and outputs on persistent shared storage
- W&B for experiment metadata and curated artifacts

Recommended usage:

- keep eval manifests in a location that both training and eval jobs can access
- keep milestone checkpoints local to the run directory and upload selected checkpoints to W&B artifacts

## Immediate baseline demo path

This is available now, even before new eval code exists.

1. Start a policy server for the baseline public checkpoint.
2. Run ALOHA Sim.
3. Save rollout videos.

Baseline server:

```bash
uv run scripts/serve_policy.py --env ALOHA_SIM
```

Simulation:

```bash
MUJOCO_GL=egl python examples/aloha_sim/main.py
```

This path is useful for:

- early demos
- collecting qualitative examples
- validating the checkpoint behavior before building automated eval

## Future periodic eval execution path

Planned, not implemented yet:

```bash
uv run scripts/eval_aloha_dataset.py \
  --checkpoint_dir <checkpoint_dir> \
  --config_name pi0_aloha_sim \
  --manifest <eval_subsample_manifest.json> \
  --split subsample \
  --video_dir data/aloha_eval/<run_name>/<step>
```

Expected behavior:

- runs the fixed eval subsample
- logs metrics to W&B
- writes videos
- emits a machine-readable results file

## Future final eval execution path

Planned, not implemented yet:

```bash
uv run scripts/eval_aloha_dataset.py \
  --checkpoint_dir <checkpoint_dir> \
  --config_name pi0_aloha_sim \
  --manifest <eval_full_manifest.json> \
  --split full \
  --video_dir data/aloha_eval_final/<run_name>/<step>
```

Expected comparison set:

- baseline public checkpoint
- best custom checkpoint
- final custom checkpoint

## Recommended milestone cadence

Training cadence:

- save checkpoints on the existing repo interval
- run periodic eval every 2k to 5k steps

Artifact cadence:

- upload milestone checkpoints every 5k steps
- always upload best and final

Final eval cadence:

- run on every experiment candidate that is worth comparing
- do not run the full eval dataset at every checkpoint save

## Comparison workflow

For every experiment family, compare:

- public baseline checkpoint
- latest checkpoint
- best periodic-eval checkpoint
- final checkpoint

Comparison dimensions:

- train loss trajectory
- periodic eval score trajectory
- final eval score
- representative ALOHA Sim videos
- representative failure videos

## Implementation backlog

1. Standardize W&B run names, groups, tags, and metadata in the existing train scripts.
2. Create `src/openpi/utils/wandb/` with generic typed records and logger classes.
3. Add checkpoint artifact publishing hooks.
4. Build one ALOHA dataset eval runner for both subsample and full eval.
5. Add W&B logging for eval metrics, one-row leaderboard tables, example detail tables, and videos.
6. Add a checkpoint watcher or milestone-triggered eval launcher.
7. Add a lightweight runbook command set under `jobs/` or `scripts/slurm/`.

## Risks and mitigations

Risk:

- train loss improves while task quality does not

Mitigation:

- use periodic eval subsample as the checkpoint-selection signal

Risk:

- the eval split is not truly independent

Mitigation:

- materialize and review the held-out eval manifest before first real training

Risk:

- W&B artifact storage becomes noisy or expensive

Mitigation:

- upload milestone checkpoints only, plus best and final

Risk:

- cluster-specific login or namespace details differ from the generic plan

Mitigation:

- update `CW_console_doc.md` first, then update this plan

## First concrete next actions

1. Freeze the task as `Transfer cube`.
2. Freeze `pi0_aloha_sim` as the initial config and baseline path.
3. Define `eval_dataset_full` and `eval_dataset_subsample`.
4. Freeze the leaderboard row schema and example/media table schemas.
5. Standardize W&B project/entity/run naming.
6. Prepare one CoreWeave `sbatch` script for JAX training.
7. Run the public `pi0_aloha_sim` checkpoint once in ALOHA Sim to validate the end-to-end path.
8. Only then implement the eval dataset runner.
