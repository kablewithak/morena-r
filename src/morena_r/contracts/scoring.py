from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


NonEmptyStr = Annotated[str, Field(min_length=1)]


class FailureLabel(StrEnum):
    EXTRINSIC_CLAIM = "extrinsic_claim"
    IGNORED_EVIDENCE = "ignored_evidence"
    WRONG_SPAN = "wrong_span"
    PARTIAL_SUPPORT = "partial_support"
    WRONG_ANSWERABILITY = "wrong_answerability"
    CONFLICT_ERROR = "conflict_error"
    CITATION_MISMATCH = "citation_mismatch"
    MEMORY_OVERRIDE = "memory_override"

    MISSED_CALL = "missed_call"
    UNNECESSARY_CALL = "unnecessary_call"
    WRONG_TOOL = "wrong_tool"
    WRONG_ARGUMENT = "wrong_argument"
    INVALID_SCHEMA = "invalid_schema"
    MISSING_CLARIFICATION = "missing_clarification"
    LOOP = "loop"
    IGNORED_RESULT = "ignored_result"

    FALSE_REFUSAL = "false_refusal"
    UNSAFE_COMPLIANCE = "unsafe_compliance"
    INCONSISTENT_HANDLING = "inconsistent_handling"
    EXCESSIVE_DISCLAIMER = "excessive_disclaimer"
    BENIGN_MISCLASSIFICATION = "benign_misclassification"

    DECISION_FLIP = "decision_flip"
    IMPLAUSIBLE_CODE_SWITCHING = "implausible_code_switching"
    ENTITY_CORRUPTION = "entity_corruption"
    REGISTER_ERROR = "register_error"
    SEMANTIC_TRANSLATION_LOSS = "semantic_translation_loss"
    LANGUAGE_DRIFT = "language_drift"

    OVERFLOW = "overflow"
    TIMEOUT = "timeout"
    OOM = "oom"
    NUMERICAL_ERROR = "numerical_error"
    CORRUPTED_ARTIFACT = "corrupted_artifact"
    PARSER_FAILURE = "parser_failure"
    INCOMPLETE_RUN = "incomplete_run"

    CONVERSION_MISMATCH = "conversion_mismatch"
    QUANTIZATION_POLICY_SHIFT = "quantization_policy_shift"
    LANGUAGE_REGRESSION = "language_regression"
    MEMORY_OVERRUN = "memory_overrun"
    UNSTABLE_LATENCY = "unstable_latency"


class ScoreRecord(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal["1.0"] = "1.0"

    run_id: NonEmptyStr
    case_id: NonEmptyStr
    family_id: NonEmptyStr

    model_identity_hash: NonEmptyStr
    dataset_hash: NonEmptyStr
    scorer_version: NonEmptyStr
    metric_id: NonEmptyStr

    eligible: bool

    value: float | None = Field(
        default=None,
        allow_inf_nan=False,
    )

    failure_labels: tuple[FailureLabel, ...] = ()
    evidence_refs: tuple[NonEmptyStr, ...] = ()

    review_status: NonEmptyStr
    attempt_status: NonEmptyStr

    missing_reason: NonEmptyStr | None = None

    @model_validator(mode="after")
    def validate_missing_value_reason(self) -> "ScoreRecord":
        if self.value is None and self.missing_reason is None:
            raise ValueError(
                "A missing score value requires missing_reason."
            )

        if self.value is not None and self.missing_reason is not None:
            raise ValueError(
                "missing_reason is only valid when value is missing."
            )

        return self
