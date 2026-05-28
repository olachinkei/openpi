from __future__ import annotations

import pathlib
from typing import Any

import wandb

from openpi.utils.wandb.types import ArtifactRecord


def should_publish_checkpoint_artifact(step: int, *, final_step: int, interval: int | None) -> bool:
    """Publish final and milestone checkpoints unless uploads are explicitly disabled."""
    if interval is not None and interval < 0:
        return False
    if step == final_step:
        return True
    return interval is not None and step > 0 and step % interval == 0


def build_artifact_ref(name: str, aliases: tuple[str, ...] = ()) -> str:
    alias = aliases[0] if aliases else "latest"
    return f"{name}:{alias}"


def build_checkpoint_aliases(step: int, *, is_final: bool) -> tuple[str, ...]:
    aliases = [f"step-{step}", "latest"]
    if is_final:
        aliases.append("final")
    return tuple(aliases)


class WandbArtifactManager:
    """Publishes model, dataset, results, and media artifacts with stable naming."""

    def log_artifact(self, record: ArtifactRecord) -> str | None:
        if wandb.run is None:
            return None

        artifact = wandb.Artifact(
            record.name,
            type=record.type,
            description=record.description,
            metadata=record.metadata or None,
        )

        path = pathlib.Path(record.path)
        if path.is_dir():
            artifact.add_dir(str(path))
        else:
            artifact.add_file(str(path), name=path.name)

        wandb.run.log_artifact(artifact, aliases=list(record.aliases))
        return build_artifact_ref(record.name, record.aliases)

    def log_checkpoint_directory(
        self,
        checkpoint_dir: pathlib.Path,
        *,
        artifact_name: str,
        aliases: tuple[str, ...],
        metadata: dict[str, Any],
    ) -> str | None:
        return self.log_artifact(
            ArtifactRecord(
                name=artifact_name,
                type="model-checkpoint",
                path=str(checkpoint_dir),
                description="Training checkpoint directory",
                aliases=aliases,
                metadata=metadata,
            )
        )
