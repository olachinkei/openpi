from __future__ import annotations

import wandb

from openpi.utils.wandb.tables import WandbTableLogger
from openpi.utils.wandb.types import LeaderboardRow

LEADERBOARD_COLUMNS = (
    "checkpoint_step",
    "primary_score",
    "success_rate",
    "mean_max_reward",
    "num_examples",
)


class LeaderboardTableLogger(WandbTableLogger[LeaderboardRow]):
    """Logs exactly one logical evaluation result into a stable leaderboard row."""

    def __init__(self, key: str = "eval/leaderboard"):
        super().__init__(key=key, columns=LEADERBOARD_COLUMNS)

    def log_row(self, row: LeaderboardRow, *, step: int | None = None) -> None:
        filtered_row = {column: getattr(row, column) for column in self.columns}
        self.log_immutable([filtered_row], step=step)
        if wandb.run is None:
            return
        for summary_key in (
            "primary_score",
            "success_rate",
            "mean_max_reward",
            "num_examples",
            "checkpoint_step",
        ):
            value = getattr(row, summary_key)
            if value is not None:
                wandb.run.summary[f"leaderboard/{summary_key}"] = value
