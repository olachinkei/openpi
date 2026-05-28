from __future__ import annotations

from collections.abc import Sequence

import wandb


class MediaLogger:
    """Small helpers for curated run-visible media."""

    def log_images(self, key: str, images: Sequence[object], *, step: int | None = None) -> None:
        if not images:
            return
        wandb.log({key: list(images)}, step=step)

    def log_html(self, key: str, html: str, *, step: int | None = None) -> None:
        wandb.log({key: wandb.Html(html)}, step=step)
