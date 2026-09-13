"""HTTP request models for the local M4 run service."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from observatory.runtime.schema import ModelConfig, _camel


class SyntheticDataRequest(BaseModel):
    model_config = ConfigDict(
        alias_generator=_camel, populate_by_name=True, extra="forbid", allow_inf_nan=False
    )

    n_samples: int = Field(ge=2, strict=True)
    true_bias: float
    true_weight: float
    noise_std: float = Field(ge=0)
    seed: int = Field(ge=0, strict=True)

    @field_validator("n_samples", "seed", mode="before")
    @classmethod
    def reject_boolean_integers(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("boolean values are not valid integers")
        return value


class ModelRequest(BaseModel):
    model_config = ConfigDict(
        alias_generator=_camel, populate_by_name=True, extra="forbid", allow_inf_nan=False
    )

    algorithm: Literal["linear_regression_gradient_descent"]
    initial_bias: float
    initial_weight: float
    learning_rate: float = Field(gt=0)
    n_updates: int = Field(ge=0, strict=True)

    @field_validator("n_updates", mode="before")
    @classmethod
    def reject_boolean_updates(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("boolean values are not valid integers")
        return value

    def to_runtime_config(self) -> ModelConfig:
        return ModelConfig.model_validate(self.model_dump())


class RunRequest(BaseModel):
    model_config = ConfigDict(alias_generator=_camel, populate_by_name=True, extra="forbid")

    experiment_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    data: SyntheticDataRequest
    model: ModelRequest
