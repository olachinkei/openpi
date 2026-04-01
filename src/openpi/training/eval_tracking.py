from __future__ import annotations

import re


def _slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value.strip()).strip("_") or "eval"


def metric_namespace_for_split(split_name: str) -> str:
    normalized = split_name.strip().lower()
    if normalized in {"subsample", "eval", "periodic", "periodic_eval"}:
        return "eval"
    if normalized in {"full", "final", "final_eval", "eval_final"}:
        return "eval_final"
    return f"eval_{_slugify(normalized)}"


def run_group_for_split(split_name: str) -> str:
    return metric_namespace_for_split(split_name)


def job_type_for_split(split_name: str) -> str:
    return "final_eval" if metric_namespace_for_split(split_name) == "eval_final" else "periodic_eval"


def run_id_filename_for_split(split_name: str) -> str:
    namespace = metric_namespace_for_split(split_name)
    if namespace == "eval":
        return "wandb_eval_id.txt"
    if namespace == "eval_final":
        return "wandb_eval_final_id.txt"
    return f"wandb_{namespace}_id.txt"


def results_filename_for_split(split_name: str) -> str:
    return f"{metric_namespace_for_split(split_name)}_results.json"


def run_name_for_split(
    source_train_run_name: str | None,
    split_name: str,
    *,
    checkpoint_step: int | None = None,
    checkpoint_alias: str | None = None,
) -> str:
    stem = source_train_run_name or "aloha-eval"
    suffix = "eval-final" if metric_namespace_for_split(split_name) == "eval_final" else "eval"
    return f"{stem}-{suffix}"
