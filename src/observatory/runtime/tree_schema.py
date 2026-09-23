"""Schema-v4 WDBC tree artifact models; older schemas remain separate."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from observatory.runtime.schema import CodeProvenance, RunStatus, _camel


class _TreeModel(BaseModel):
    model_config = ConfigDict(
        frozen=True, populate_by_name=True, alias_generator=_camel, extra="forbid"
    )


class TreeDataConfig(_TreeModel):
    source: Literal["external_dataset"] = "external_dataset"
    dataset_id: Literal["uci-wdbc"] = "uci-wdbc"
    dataset_version: Literal["1.0.0"] = "1.0.0"
    version_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    processed_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    split_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    split_strategy: Literal["stratified-sha256-2026-v1"] = "stratified-sha256-2026-v1"
    feature_names: list[str] = Field(min_length=30, max_length=30)
    target_mapping: dict[str, int]
    preprocessing: Literal["none"] = "none"
    train_count: Literal[397] = 397
    validation_count: Literal[172] = 172


class TreeTrainingConfig(_TreeModel):
    algorithm: Literal["cart_gini"] = "cart_gini"
    max_depth: int = Field(ge=1, le=5, strict=True)
    min_samples_split: Literal[2] = 2


class TreeRosterRow(_TreeModel):
    sample_id: str
    target: int = Field(ge=0, le=1, strict=True)
    split: Literal["train", "validation"]


class TreeObservedRow(TreeRosterRow):
    features: list[float] = Field(min_length=30, max_length=30)


class TreeDatasetSummary(_TreeModel):
    roster: list[TreeRosterRow] = Field(min_length=569, max_length=569)
    observed_rows: list[TreeObservedRow] = Field(min_length=20, max_length=20)


class TreeEvaluation(_TreeModel):
    correct: int = Field(ge=0)
    total: int = Field(gt=0)
    accuracy: float = Field(ge=0, le=1, allow_inf_nan=False)


class TreePrediction(_TreeModel):
    sample_id: str
    leaf_id: str
    predicted_class: int = Field(ge=0, le=1, strict=True)


class TreeRunManifest(_TreeModel):
    schema_version: Literal[4] = 4
    run_id: str
    experiment_id: str
    created_at: str
    status: RunStatus
    stop_reason: Literal["tree_complete", "user_cancelled", "runtime_error", "io_error"] | None = (
        None
    )
    last_valid_step: int | None = None
    error_message: str | None = None
    data_config: TreeDataConfig
    dataset: TreeDatasetSummary
    training_config: TreeTrainingConfig
    code_provenance: CodeProvenance
    observed_sample_ids: list[str]
    n_snapshots_written: int = 0
    train_evaluation: TreeEvaluation | None = None
    validation_evaluation: TreeEvaluation | None = None
    predictions: list[TreePrediction] = Field(default_factory=list)


class TreeSnapshot(_TreeModel):
    step: int = Field(ge=0)
    node_id: str
    parent_id: str | None
    side: Literal["left", "right"] | None
    depth: int = Field(ge=0)
    sample_ids: list[str]
    n_samples: int = Field(gt=0)
    class_counts: tuple[int, int]
    gini_parent: float = Field(ge=0, le=0.5, allow_inf_nan=False)
    is_leaf: bool
    split_feature_index: int | None = None
    split_feature_name: str | None = None
    split_threshold: float | None = Field(default=None, allow_inf_nan=False)
    left_child_id: str | None = None
    right_child_id: str | None = None
    left_count: int | None = None
    right_count: int | None = None
    gini_left: float | None = Field(default=None, allow_inf_nan=False)
    gini_right: float | None = Field(default=None, allow_inf_nan=False)
    weighted_gini_decrease: float | None = Field(default=None, allow_inf_nan=False)
    predicted_class: Literal[0, 1] | None = None
    predicted_label: Literal["Benign", "Malignant"] | None = None
    leaf_reason: Literal["pure", "max_depth", "insufficient_samples", "no_positive_gain"] | None = (
        None
    )

    @model_validator(mode="after")
    def check_variant(self) -> TreeSnapshot:
        split_fields = (
            self.split_feature_index,
            self.split_feature_name,
            self.split_threshold,
            self.left_child_id,
            self.right_child_id,
            self.left_count,
            self.right_count,
            self.gini_left,
            self.gini_right,
            self.weighted_gini_decrease,
        )
        if self.is_leaf:
            if any(value is not None for value in split_fields):
                raise ValueError("leaf must not contain split fields")
            if (
                self.predicted_class is None
                or self.predicted_label is None
                or self.leaf_reason is None
            ):
                raise ValueError("leaf requires predicted class and reason")
        elif any(value is None for value in split_fields) or (
            self.predicted_class is not None
            or self.predicted_label is not None
            or self.leaf_reason is not None
        ):
            raise ValueError("split must contain split fields only")
        return self


class TreeEvent(_TreeModel):
    schema_version: Literal[4] = 4
    run_id: str
    seq: int = Field(gt=0)
    kind: Literal[
        "run.created",
        "run.started",
        "tree.node_split",
        "tree.node_leaf",
        "run.completed",
        "run.cancelled",
        "run.failed",
    ]
    step: int | None = None
    node_id: str | None = None
    message: str | None = None
