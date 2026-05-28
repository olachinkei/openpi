from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
from typing import Any, Generic, TypeVar

import wandb

T = TypeVar("T")


def _record_to_dict(record: T | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(record, Mapping):
        return dict(record)
    if dataclasses.is_dataclass(record) and not isinstance(record, type):
        return dataclasses.asdict(record)
    raise TypeError(f"Unsupported record type for W&B table logging: {type(record)!r}")


class WandbTableLogger(Generic[T]):
    """Materializes typed records into a stable wandb.Table schema."""

    def __init__(self, key: str, columns: Sequence[str]):
        self._key = key
        self._columns = tuple(columns)

    @property
    def columns(self) -> tuple[str, ...]:
        return self._columns

    def _validate_row(self, row: Mapping[str, Any]) -> None:
        missing = [column for column in self._columns if column not in row]
        unexpected = sorted(set(row) - set(self._columns))
        if missing or unexpected:
            raise ValueError(
                f"Row for {self._key!r} does not match the stable schema. "
                f"Missing={missing}, unexpected={unexpected}"
            )

    def materialize(self, records: Sequence[T | Mapping[str, Any]]) -> wandb.Table:
        table = wandb.Table(columns=list(self._columns))
        for record in records:
            row = _record_to_dict(record)
            self._validate_row(row)
            table.add_data(*(row[column] for column in self._columns))
        return table

    def log_immutable(self, records: Sequence[T | Mapping[str, Any]], *, step: int | None = None) -> wandb.Table:
        table = self.materialize(records)
        wandb.log({self._key: table}, step=step)
        return table
