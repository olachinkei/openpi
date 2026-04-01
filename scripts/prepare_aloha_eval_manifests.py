from __future__ import annotations

import dataclasses
import json
import pathlib
import random
import sys

import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import tyro

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclasses.dataclass
class Args:
    repo_id: str = "lerobot/aloha_sim_transfer_cube_human"
    config_name: str = "pi0_aloha_sim"
    task_name: str = "Transfer cube"
    prompt: str = "Transfer cube"
    split_seed: int = 42
    eval_episode_count: int = 10
    subsample_episode_count: int = 4
    full_output: pathlib.Path = pathlib.Path("manifests/aloha_sim_transfer_cube/eval_dataset_full.json")
    subsample_output: pathlib.Path = pathlib.Path("manifests/aloha_sim_transfer_cube/eval_dataset_subsample.json")


def _record_for_episode(episode_index: int, prompt: str) -> dict[str, object]:
    return {
        "example_id": f"episode_{episode_index:03d}",
        "episode_id": f"episode_{episode_index:06d}",
        "prompt": prompt,
        "hf_episode_index": episode_index,
        "metadata": {
            "seed": episode_index,
            "source_episode_index": episode_index,
        },
    }


def _write_json(path: pathlib.Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def main(args: Args) -> None:
    metadata = lerobot_dataset.LeRobotDatasetMetadata(args.repo_id)
    available_episodes = sorted(int(episode_index) for episode_index in metadata.episodes)
    if args.eval_episode_count <= 0:
        raise ValueError("eval_episode_count must be greater than 0.")
    if args.subsample_episode_count <= 0:
        raise ValueError("subsample_episode_count must be greater than 0.")
    if args.eval_episode_count >= len(available_episodes):
        raise ValueError(
            f"eval_episode_count={args.eval_episode_count} must be smaller than total_episodes={len(available_episodes)}."
        )
    if args.subsample_episode_count > args.eval_episode_count:
        raise ValueError("subsample_episode_count must be less than or equal to eval_episode_count.")

    rng = random.Random(args.split_seed)
    shuffled_episodes = list(available_episodes)
    rng.shuffle(shuffled_episodes)
    eval_episodes = shuffled_episodes[: args.eval_episode_count]
    subsample_episodes = eval_episodes[: args.subsample_episode_count]

    full_output = args.full_output if args.full_output.is_absolute() else (REPO_ROOT / args.full_output).resolve()
    subsample_output = (
        args.subsample_output if args.subsample_output.is_absolute() else (REPO_ROOT / args.subsample_output).resolve()
    )
    full_source = full_output.relative_to(REPO_ROOT)

    full_payload = {
        "$schema": "../../schemas/eval_manifest.schema.json",
        "manifest_version": "1.0",
        "name": "aloha_sim_transfer_cube_eval_full",
        "environment_family": "ALOHA_SIM",
        "task_name": args.task_name,
        "config_name": args.config_name,
        "dataset_name": args.repo_id,
        "split_name": "full",
        "prompt": args.prompt,
        "selection": {
            "policy": "seeded_holdout_from_train_only_dataset",
            "source": args.repo_id,
            "seed": args.split_seed,
            "notes": (
                "Dataset exposes only a train split, so this manifest defines a deterministic held-out eval subset "
                f"of {args.eval_episode_count} episodes from total_episodes={len(available_episodes)}."
            ),
        },
        "records": [_record_for_episode(episode_index, args.prompt) for episode_index in eval_episodes],
    }
    subsample_payload = {
        "$schema": "../../schemas/eval_manifest.schema.json",
        "manifest_version": "1.0",
        "name": "aloha_sim_transfer_cube_eval_subsample",
        "environment_family": "ALOHA_SIM",
        "task_name": args.task_name,
        "config_name": args.config_name,
        "dataset_name": args.repo_id,
        "split_name": "subsample",
        "prompt": args.prompt,
        "selection": {
            "policy": "fixed_prefix_of_full_eval",
            "source": str(full_source),
            "seed": args.split_seed,
            "notes": f"Fixed {args.subsample_episode_count}-episode prefix of the seeded full eval manifest.",
        },
        "records": [_record_for_episode(episode_index, args.prompt) for episode_index in subsample_episodes],
    }

    _write_json(full_output, full_payload)
    _write_json(subsample_output, subsample_payload)
    print(
        json.dumps(
            {
                "repo_id": args.repo_id,
                "total_episodes": len(available_episodes),
                "full_eval_episodes": eval_episodes,
                "subsample_eval_episodes": subsample_episodes,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
