from __future__ import annotations

from collections.abc import Sequence
import pathlib
import re

import wandb

from openpi.utils.wandb.types import VideoRecord


class VideoLogger:
    """Logs finalized local video files with stable key naming."""

    @staticmethod
    def _key_suffix(record: VideoRecord, fallback_index: int) -> str:
        raw_name = record.name or pathlib.Path(record.path).stem or str(fallback_index)
        return re.sub(r"[^a-zA-Z0-9_.-]+", "_", raw_name).strip("_") or str(fallback_index)

    def log_video_files(self, key: str, videos: Sequence[VideoRecord], *, step: int | None = None) -> None:
        payload = {}
        for index, record in enumerate(videos):
            path = pathlib.Path(record.path)
            payload[f"{key}/{self._key_suffix(record, index)}"] = wandb.Video(
                str(path),
                caption=record.caption or path.name,
                fps=record.fps,
                format=record.format,
            )
        if payload:
            wandb.log(payload, step=step)
