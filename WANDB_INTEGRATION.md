## First-pass contract

The first tracked environment is:

- environment: `ALOHA_SIM`
- task: `Transfer cube`
- config: `pi0_aloha_sim`
- default project: `openpi-aloha`
- runtime override: `WANDB_PROJECT=<project>`

## Stable run groups

- `baseline_eval`
- `train`
- `periodic_eval`
- `final_eval`

## Stable tag pattern

Tags are explicit and intentionally sparse.

- use `wandb_tags` in config only for human-curated labels
- use `WANDB_TAGS=test,smoke` for ad hoc run labeling
- do not mirror task/config/dataset/backend metadata into tags

## Scalar metric keys

Training:

- `train/loss`
- `train/grad_norm`
- `train/param_norm`
- `train/learning_rate`
- `train/step_time_sec`
- `train/checkpoint_step`
- no train images or videos by default

Periodic eval:

- `eval_subsample/primary_score`
- `eval_subsample/success_rate`
- `eval_subsample/mean_max_reward`
- `eval_subsample/num_examples`
- `eval_subsample/checkpoint_step`
- `eval_subsample/videos/*`

Final eval:

- `eval_full/primary_score`
- `eval_full/success_rate`
- `eval_full/mean_max_reward`
- `eval_full/num_examples`
- `eval_full/checkpoint_step`
- `eval_full/videos/*`

## Table keys

- `eval_subsample/leaderboard`
- `eval_subsample/examples`
- `eval_full/leaderboard`
- `eval_full/examples`

## Artifact types

- `dataset-manifest`
- `model-checkpoint`
- `eval-video-bundle`
- `eval-results`

## Current implementation status

- common W&B run initialization now lives under `src/openpi/utils/wandb/`
- `WANDB_PROJECT` overrides config-default projects
- tags are explicit-only (`wandb_tags` and optional `WANDB_TAGS`)
- training metrics are logged with stable `train/*` keys and no media
- checkpoint artifact upload hooks exist for milestone and final checkpoints
- checkpoint-triggered periodic eval can submit `jobs/eval_aloha_sim.sbatch` automatically when `eval_manifest_path` and `eval_job_script_path` are configured
- ALOHA Sim eval now logs success metrics plus rollout videos under `eval_subsample/*` and `eval_full/*`
- ALOHA Sim manifest templates and a shared schema are tracked in git
