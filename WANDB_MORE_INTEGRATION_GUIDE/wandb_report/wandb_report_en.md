# OpenPI x W&B: A Guide to Connecting Physical AI Workflows

## Introduction

[OpenPI](https://github.com/Physical-Intelligence/openpi) is an open-source robotics framework published by [Physical Intelligence](https://www.physicalintelligence.company/), and it has been drawing rapid attention in the Physical AI space. In this report, we use OpenPI as an example to explain how W&B can be used more effectively for Physical AI development. OpenPI supports both JAX and PyTorch, and while it already includes basic W&B integration such as logging loss and learning rate, we also implemented additional integration features such as simulation result visualization.

- [**GitHub repository with the added integrations**](https://github.com/olachinkei/openpi):
- W&B Project (OpenPI integration example):

Using OpenPI as a concrete example, let us walk through useful ways to use W&B in Physical AI, step by step. For the technical implementation details of the added integration, please refer to the [README in the GitHub repository](https://github.com/olachinkei/openpi).

There is also a white paper explaining the value of W&B in Physical AI, along with a general demo video using IsaacLab as an example, so please take a look at those as well.

- [White Paper: Advancing Physical AI: From learning to embodied intelligence](https://wandb.ai/site/resources/whitepapers/advancing-physical-ai/)
- [W&B demo video (IsaacLab)](https://www.youtube.com/watch?v=45Beo0ZkJJA)
- [Example W&B project (IsaacLab)](https://wandb.ai/wandb-smle/isaaclab-wandb-crwv?nw=nwuseranushravvatsa)

In AI development, a single training run produces a large amount of information. Looking only at train loss is not enough. To judge whether a model is actually good, you need to look at the evaluation episode list, checkpoints, success rate, and maximum reward together. Once these pieces are scattered across different scripts, different jobs, and different storage locations, comparison quickly becomes difficult. In Physical AI in particular, simulation visualization matters just as much as metrics.

That is why we developed additional integrations for the OpenPI codebase from the following three perspectives:
1. Strengthening the experiment comparison foundation
2. Strengthening the evaluation pipeline with visualization
3. Managing assets with Artifacts / Registry
In this report, together with the added integrations, we also walk through useful UI operations that make these workflows easier to use in practice.

---

## Example Task: ALOHA Sim Transfer Cube

In this demo, we use the **Transfer Cube** task, where a two-arm ALOHA robot grasps a cube, passes it from one arm to the other, and holds it stably. The training dataset is `lerobot/aloha_sim_transfer_cube_human`, and the configuration is `pi0_aloha_sim`.

This dataset contains 50 episodes in total. In this setup, 10 episodes are held out for final evaluation, while the remaining 40 episodes are used for training. In addition, periodic evaluation uses a fixed subset of 4 episodes from those 10 held-out episodes. Training runs up to 20,000 steps.

The main goal here is to demonstrate how training, evaluation, and asset management can be connected within W&B, so the dataset size is intentionally small. As a result, the outcomes shown here should be understood as workflow examples rather than as a strict benchmark or a statistically robust performance comparison.

---

## 1. Strengthening the Experiment Comparison Foundation

### Saving detailed experiment conditions in config

In W&B, you can save not only time-series data such as loss, but also the conditions of each run in `config`. If you record information such as `config_name`, `task_name`, `dataset_name`, `backend`, `git sha`, and `hostname`, it becomes much easier later to filter specific conditions in the Workspace or compare runs by individual parameters. As shown later, these config values can also be used to color-code runs.

In the original OpenPI W&B integration, the information needed for comparison was not fully captured in `config`. To address this, we added `src/openpi/utils/wandb/run_context.py` and unified W&B run initialization. This makes it possible to store the metadata that is useful for comparison and organization directly in `wandb.config`. The same approach works whether you are using JAX or PyTorch.

For example, run information in W&B can be structured like this:

```python
import wandb

wandb.init(
    project="openpi-integration",
    config={
        "config_name": "pi0_aloha_sim",
        "task_name": "Transfer cube",
        "dataset_name": "lerobot/aloha_sim_transfer_cube_human",
        "training_backend": "jax",
    },
    tags=["aloha", "transfer-cube"],
)
```

### Comparing many experiments in the Workspace

With the config structure above in place, let us look at several useful Workspace features in W&B for comparing many runs.

### Useful Feature 1: Parallel Coordinates Chart

The [Parallel Coordinates Chart](https://docs.wandb.ai/models/app/features/panels/parallel-coordinates) is a panel that lets you view relationships between multiple hyperparameters and result metrics at once. For example, if you place `learning_rate`, `batch_size`, `training_backend`, `eval/success_rate`, and `eval_final/primary_score` on the axes, you can visually trace which run configurations led to higher success rates.

When you are running many follow-up experiments, it is already useful to simply see where the promising runs cluster together. Even when the final loss alone is hard to interpret, looking at success rate and reward alongside it makes the relationship between settings and outcomes much easier to read.

When you run a large search manually, you need to launch jobs one by one with different commands and settings each time. With **W&B Sweeps**, you can define the search space in YAML, and `wandb agent` can automatically choose candidate settings and launch runs. We did not use it in this demo, but it is easy to use and well worth checking out.

### Useful Feature 2: Pinned Runs and Baseline Comparison

Pinned Runs let you keep important baseline runs at the top of the run list. Even as many new experiments are added, your reference runs stay easy to find.

When combined with Baseline Comparison, the baseline run is highlighted directly on line plots, which makes it much easier to judge whether a new run actually improved over the baseline. This is especially useful when comparing fine-tuning runs against a public checkpoint.

Below, the runs are colored by `lr_schedule.peak_lr`.

![Pinned Runs and Baseline Comparison](https://mintcdn.com/wb-21fd5541/57wwTAGN9Q-FX-xN/images/models/pinned-and-baseline-runs/runs-table-with-pinned-and-baseline-runs.png?w=1100&fit=max&auto=format&n=57wwTAGN9Q-FX-xN&q=85&s=288f18afe190c9e11ce65f9e3b3086e1)

Demo video: https://www.loom.com/share/b8a5352c01594778ac38ff0ad5fa18d8

### Useful Feature 3: Semantic Colors by Config Values

Semantic Colors by Config Values automatically assigns colors to runs based on config values. Simply having consistent colors by learning rate, model family, or backend can make a large difference in how readable the charts are.

For example, if `training_backend=jax` and `training_backend=pytorch` are automatically shown in different colors, you can quickly see backend-related trends in the same Workspace. Because individual runs remain easy to distinguish before aggregation, it also speeds up judgment during follow-up experiments.

![Semantic Colors by Config Values](https://mintcdn.com/wb-21fd5541/_OEDykSS2PIumrEw/images/track/color-code-runs-plot.png?w=1100&fit=max&auto=format&n=_OEDykSS2PIumrEw&q=85&s=93b9f741937503187baa665f41568973)

Demo video: https://www.loom.com/share/640c6d2c04ec4c328c92b530516778bd

### Useful Feature 4: Saved Views

W&B Workspace [Saved Views](https://docs.wandb.ai/models/track/workspaces) let you save panel layouts, filter conditions, color assignments, and display targets exactly as they are.

This is more than just a saved layout. What matters is that you can save a research perspective itself. For example, it is useful to keep separate views such as:

- For training checks: a view centered on `train/loss` and `train/grad_norm`
- For periodic evaluation: a view centered on `eval/success_rate` and `eval/leaderboard`
- For video review: a view centered on `eval_final/videos`

If a team shares the same Saved View, everyone is literally looking at the same screen when discussing results. Even that alone can make reviews much faster.

![Saved Views Menu](https://mintcdn.com/wb-21fd5541/4kbs1cW6PdjDOqU3/images/app_ui/Menu_No_views.jpg?w=1100&fit=max&auto=format&n=4kbs1cW6PdjDOqU3&q=85&s=7ee0771a9d10880e774d04deff43ed01)

### Automatically choosing and launching the next experiment: W&B Skills

W&B also provides [Skills](https://wandb.ai/site/skills/). This is a framework that makes it easier for coding agents to work with W&B experiment management features. Agents can review training results and run comparisons, consider what conditions to try next, and even proceed to code changes when necessary.

For example, once several training runs have completed, an agent can review the runs and suggest what conditions to try next. When used well, this makes it easier to build a workflow that continuously improves experiments while looking at the latest results.

In our experiments as well, we used W&B Skills as a lightweight aid when deciding later follow-up conditions. W&B Skills is still a fairly new tool, so we would love for you to try it and share lots of feedback.

```bash
# Local (current project):
npx skills add wandb/skills --skill '*' --yes

# Global (all projects):
npx skills add wandb/skills --skill '*' --yes --global

# To link skills to a specific agent (for example, Claude Code):
npx skills add wandb/skills --agent claude-code --skill '*' --yes --global
```

---

## 2. Strengthening the Evaluation Pipeline

In Physical AI, checking the simulation itself is extremely important. Quantitative metrics such as success rate or reward do not tell you how the robot actually moved, where it became unstable, or how it failed. That is why visualizing the simulation and checking behavior directly is a necessary part of evaluation.

In other words, both of the following are essential in Physical AI evaluation:

- Quantitative evaluation: indicators such as success rate, maximum reward, and number of steps, which are easy to compare across runs
- Qualitative evaluation: reviewing rollout videos to understand how the robot moves and how it fails

### Quantitative evaluation: building a leaderboard with W&B Tables

The `scripts/eval_aloha_dataset.py` script we added does more than save evaluation results as JSON or CSV. It also logs them as W&B Tables. W&B Tables let you store structured data in the UI directly.

There are two main ideas here. The first is that we prepared a leaderboard that gives an overview of evaluation results for each run and checkpoint. This makes it easy to compare which checkpoints performed well when multiple runs are displayed side by side in the Workspace.

The second is that we do not stop at the aggregated results. We also preserve sample-level evaluation details. That makes it possible to inspect questions such as "the overall scores are similar, but which samples differ?" or "which episodes failed?" In the W&B Report, you can choose which run to display with the run filter below the panel.

The details of what information goes into the Tables are implemented in `scripts/eval_aloha_dataset.py` and `src/openpi/utils/wandb/leaderboard.py`. The main evaluation values are also written into `run.summary`, so they can be used for filtering and sorting in the Workspace even without opening the Table itself.

<!-- Insert a screenshot of the OpenPI leaderboard table here -->

### Qualitative evaluation: viewing rollout videos in W&B Media Panels

In this integration, ALOHA Sim rollouts are logged as `wandb.Video` and can be viewed step by step in W&B Media Panels. Because rollout videos are kept for each evaluation sample, you can later trace how the robot behaved at each checkpoint.

This makes it possible to visually confirm whether the robot is actually learning to solve the task as training progresses, or whether instability remains along the way. The key point is that improvements and failure modes that are hard to see in scalar metrics alone can now be tracked as simulation videos at each step.

For implementation details, see `scripts/eval_aloha_dataset.py`, `src/openpi/training/aloha_eval.py`, and `src/openpi/utils/wandb/videos.py`.

<!-- Insert a screenshot of OpenPI rollout videos or a W&B Media Panel capture here -->

### Especially useful Media Panel features

As described in the [Media Panels documentation](https://docs.wandb.ai/models/app/features/panels/media#media-panels), W&B Media Panels continued to improve in 2025. As interest in Physical AI grew, the tools for handling simulations and rollout videos also became more capable.

Here, we highlight a few Media Panel features that are especially useful in the Physical AI context.

#### Synchronized Video Playback

This feature lets you play multiple videos in sync. It is useful when comparing successful and failed examples from the same checkpoint, or the same prompt across a JAX run and a PyTorch run. Once you scrub one video, all the others align to the same timestamp, so it becomes easy to compare grasp timing or handoff behavior frame by frame.

Demo video: https://www.loom.com/share/244cb3be1de04ad8a4ee22654d730b0f

#### Synced Media Sliders

This feature synchronizes media sliders across multiple panels. It is useful not only for videos, but also when you want to compare related images or derived media at the same step.

Demo video: https://app.getbeamer.com/pictures?id=519192-Ge-_vXnesu-_vTUj77-977-977-977-9Pu-_vVtqbXDWhu-_vX0X77-91r7vv70uRzRd77-9&v=4

#### Bulk control of Media Panels

This feature lets you manage panel settings together across an entire Workspace or a specific Section. As the number of video panels grows, manual configuration becomes costly. Bulk control helps you align display styles and grid layouts all at once, so you can spend more time analyzing and less time configuring.

Demo video: https://app.getbeamer.com/pictures?id=504954-Y17vv73vv73vv73vv71_Ie-_vUI1zZg4PO-_ve-_vWHvv73vv70oC--_ve-_ve-_ve-_vXta3ZZz&v=4

---

## 3. Asset Management

### Why asset management matters

In Physical AI, it is important to be able to trace, later on, which data was used for training, which checkpoints were evaluated, and what results were obtained. As the number of checkpoints grows, managing everything through local filenames alone quickly becomes difficult.

That is why this integration places strong emphasis on versioning checkpoints and preserving lineage from data to models to evaluation results.

### Artifacts: strengthening checkpoint version management

W&B Artifacts are designed to manage the many checkpoints and result files that appear during experiments as versioned assets. In this integration, we especially strengthened how model checkpoints are handled during training.

For `pi0_aloha_sim`, milestone checkpoints every 5,000 steps and the final checkpoint are published, with aliases such as `step-5000`, `step-10000`, `step-15000`, and `final`. This keeps storage costs under control while still making it easy to compare training progress checkpoint by checkpoint. These settings are also easy to change.

In addition, by using lineage features, we made it easier to trace which datasets and parameters were used to produce each checkpoint, so that model versioning becomes something useful for later comparison and validation rather than simple file storage.

### Registry: preserving dataset and model lineage

W&B Registry is designed to make shared datasets and models easy for a team to reference. In this integration, we first register the training dataset and evaluation datasets in the Registry, and then start training and evaluation by referencing them. While Artifacts are designed to manage the large number of assets generated during experiments, Registry is designed to keep the important assets that a team wants to share easy to find and reuse.

The training dataset and evaluation datasets are registered in the Registry, and training and evaluation are launched by referring to those assets. The artifact refs used here are:

- train: `wandb32/wandb-registry-Physical AI - openpi/training dataset:v0`
- eval: `wandb32/wandb-registry-Physical AI - openpi/evaluation dataset for openpi:v0`
- holdout (`eval_final`): `wandb32/wandb-registry-Physical AI - openpi/evaluation dataset for openpi:v1`

In practice, the implementation references the train / eval / holdout (`eval_final`) artifacts using `use_artifact` as shown below. The train artifact is referenced so the dataset version used for training is recorded, while the eval and holdout artifacts are used to resolve manifests and drive actual evaluation.

```python
dataset_artifact_refs = _dataset_artifacts.configured_dataset_artifact_refs(config)
if train_ref := dataset_artifact_refs.get("train"):
    wandb.run.use_artifact(train_ref)

if eval_ref := dataset_artifact_refs.get("eval"):
    wandb.run.use_artifact(eval_ref)
    resolved_eval_manifest = _download_artifact_payload(eval_ref, artifact_root / "eval")
    if resolved_eval_manifest is not None:
        config = dataclasses.replace(config, eval_manifest_path=str(resolved_eval_manifest))

if holdout_ref := dataset_artifact_refs.get("eval_final"):
    wandb.run.use_artifact(holdout_ref)
    resolved_holdout_manifest = _download_artifact_payload(holdout_ref, artifact_root / "eval_final")
    if resolved_holdout_manifest is not None:
        config = dataclasses.replace(config, final_eval_manifest_path=str(resolved_holdout_manifest))
```

Because the same dataset does not need to be uploaded again for every run, operations stay easier to manage, and it also becomes much easier to understand the assumptions behind a model when revisiting it later.

![Registry page](https://mintcdn.com/wb-21fd5541/AXlwJe6YUBax3n2I/images/registry/registry_landing_page.png?w=2500&fit=max&auto=format&n=AXlwJe6YUBax3n2I&q=85&s=88562e36bd19c3d5a7e492a6cabb604c)

---

## Conclusion

With this OpenPI x W&B integration, our goal was to make it possible to follow, from a single run, the configuration that was used, evaluation results at each checkpoint, rollout videos, and the connections between training and evaluation data.

In Physical AI, loss and success rate alone are not enough. By making simulation results easy to visualize, we made it possible to understand performance in a deeper and more practical way.

We hope the integrations introduced here, and the workflow shown in this report, will be useful as a reference when building an experiment foundation for Physical AI.
