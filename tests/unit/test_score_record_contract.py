from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from morena_r.contracts.scoring import FailureLabel, ScoreRecord


def valid_payload() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "run_id": "b0-pilot-001",
        "case_id": "case-001",
        "family_id": "family-001",
        "model_identity_hash": "model-hash",
        "dataset_hash": "dataset-hash",
        "scorer_version": "scorer-v1",
        "metric_id": "fully_faithful_grounded_success",
        "eligible": True,
        "value": 1.0,
        "failure_labels": [],
        "evidence_refs": ["output:case-001"],
        "review_status": "not_required",
        "attempt_status": "completed",
    }


def test_valid_score_record() -> None:
    record = ScoreRecord.model_validate(valid_payload())

    assert record.value == 1.0
    assert record.missing_reason is None


def test_zero_score_is_preserved() -> None:
    payload = valid_payload()
    payload["value"] = 0.0

    record = ScoreRecord.model_validate(payload)

    assert record.value == 0.0


def test_missing_value_requires_reason() -> None:
    payload = valid_payload()
    payload["value"] = None

    with pytest.raises(
        ValidationError,
        match="missing score value requires missing_reason",
    ):
        ScoreRecord.model_validate(payload)


def test_missing_value_with_reason_is_valid() -> None:
    payload = valid_payload()
    payload["value"] = None
    payload["missing_reason"] = "runtime_timeout"

    record = ScoreRecord.model_validate(payload)

    assert record.value is None
    assert record.missing_reason == "runtime_timeout"


def test_missing_reason_for_present_value_is_rejected() -> None:
    payload = valid_payload()
    payload["missing_reason"] = "should-not-exist"

    with pytest.raises(
        ValidationError,
        match="missing_reason is only valid",
    ):
        ScoreRecord.model_validate(payload)


@pytest.mark.parametrize(
    "invalid_value",
    [
        math.nan,
        math.inf,
        -math.inf,
    ],
)
def test_non_finite_values_are_rejected(
    invalid_value: float,
) -> None:
    payload = valid_payload()
    payload["value"] = invalid_value

    with pytest.raises(ValidationError):
        ScoreRecord.model_validate(payload)


def test_unknown_fields_are_rejected() -> None:
    payload = valid_payload()
    payload["silent_extra_field"] = "bad"

    with pytest.raises(ValidationError):
        ScoreRecord.model_validate(payload)


def test_failure_taxonomy_is_closed() -> None:
    payload = valid_payload()
    payload["value"] = 0.0
    payload["failure_labels"] = [
        FailureLabel.MEMORY_OVERRIDE,
        FailureLabel.CITATION_MISMATCH,
    ]

    record = ScoreRecord.model_validate(payload)

    assert record.failure_labels == (
        FailureLabel.MEMORY_OVERRIDE,
        FailureLabel.CITATION_MISMATCH,
    )


def test_unknown_failure_label_is_rejected() -> None:
    payload = valid_payload()
    payload["failure_labels"] = ["made_up_failure"]

    with pytest.raises(ValidationError):
        ScoreRecord.model_validate(payload)
