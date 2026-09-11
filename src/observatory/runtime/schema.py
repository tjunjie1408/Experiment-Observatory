"""Pydantic models for the V0 run record: manifest.json, events.jsonl, snapshots.json.

Implements the static file contract from BEHAVIOR_SPECIFICATION_V0.2.md section 3.3.
These are the Python producer types; the TS consumer schema in
web/src/lib/schema.ts must be kept in sync by hand (a small amount of
hand-written duplication across the language boundary is acceptable for
V0 instead of generating one side from the other).

Wire format uses camelCase (aliases) to match the TS consumer; Python code
uses snake_case field names via `populate_by_name`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

SCHEMA_VERSION = 1

RunStatus = Literal["created", "running", "completed", "failed"]
StopReason = Literal["max_steps", "numerical_error", "io_error", "user_cancelled", "runtime_error"]


def _camel(name: str) -> str:
    head, *tail = name.split("_")
    return head + "".join(part.capitalize() for part in tail)


class DataConfig(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True, alias_generator=_camel)

    generator: str
    n_samples: int
    true_bias: float
    true_weight: float
    noise_std: float
    seed: int


class ModelConfig(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True, alias_generator=_camel)

    algorithm: Literal["linear_regression_gradient_descent"]
    initial_bias: float
    initial_weight: float
    learning_rate: float
    n_updates: int


class DatasetSummary(BaseModel):
    """Real generated data embedded in the manifest so the run is self-contained."""

    model_config = ConfigDict(frozen=True, populate_by_name=True, alias_generator=_camel)

    generator_id: str
    sample_ids: list[str]
    x: list[float]
    y: list[float]


class CodeProvenance(BaseModel):
    """Source code identity for reproducibility. Empty fields must state why."""

    model_config = ConfigDict(frozen=True, populate_by_name=True, alias_generator=_camel)

    git_commit: str | None = None
    git_dirty: bool | None = None
    unavailable_reason: str | None = None


class RunManifest(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True, alias_generator=_camel)

    schema_version: int = SCHEMA_VERSION
    run_id: str
    experiment_id: str
    created_at: str
    status: RunStatus
    stop_reason: StopReason | None = None
    last_valid_step: int | None = None
    error_message: str | None = None
    data_config: DataConfig
    dataset: DatasetSummary
    training_config: ModelConfig
    code_provenance: CodeProvenance
    observed_sample_ids: list[str]
    n_snapshots_written: int = 0


class Snapshot(BaseModel):
    """One fully-consistent training state: all values belong to the same theta_k."""

    model_config = ConfigDict(frozen=True, populate_by_name=True, alias_generator=_camel)

    step: int
    b: float
    w: float
    gradient_b: float
    gradient_w: float
    train_mse: float
    observed_predictions: dict[str, float]


class Event(BaseModel):
    """Minimal V0 event: run lifecycle markers. seq is strictly increasing within a run."""

    model_config = ConfigDict(frozen=True, populate_by_name=True, alias_generator=_camel)

    schema_version: int = SCHEMA_VERSION
    run_id: str
    seq: int
    kind: Literal["run.created", "run.started", "step.recorded", "run.completed", "run.failed"]
    step: int | None = None
    message: str | None = None
