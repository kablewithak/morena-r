from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from morena_r.contracts.actions import (
    Answerability,
    Decision,
    GetRecordArgs,
    GetRecordCall,
    ToolName,
)
from morena_r.contracts.evaluation import (
    AttemptObservation,
    AttemptTransportStatus,
    EvalCase,
    EvalGold,
    EvalInput,
    EvalMessage,
)


def valid_input() -> EvalInput:
    return EvalInput(
        case_id="case-001",
        family_id="family-001",
        languages=("en",),
        messages=(
            EvalMessage(
                role="user",
                content="Synthetic request.",
            ),
        ),
        evidence=(),
        available_tools=(
            ToolName.GET_RECORD,
        ),
        permitted_tools=(
            ToolName.GET_RECORD,
        ),
        evaluation_time_utc=datetime(
            2026,
            1,
            1,
            tzinfo=timezone.utc,
        ),
    )


def test_naive_evaluation_time_is_rejected() -> None:
    with pytest.raises(
        ValidationError,
        match="timezone-aware",
    ):
        EvalInput(
            case_id="case-001",
            family_id="family-001",
            languages=("en",),
            messages=(
                EvalMessage(
                    role="user",
                    content="Synthetic request.",
                ),
            ),
            evaluation_time_utc=datetime(
                2026,
                1,
                1,
            ),
        )


def test_permitted_tool_must_be_available() -> None:
    with pytest.raises(
        ValidationError,
        match="Permitted tools must be available",
    ):
        EvalInput(
            case_id="case-001",
            family_id="family-001",
            languages=("en",),
            messages=(
                EvalMessage(
                    role="user",
                    content="Synthetic request.",
                ),
            ),
            available_tools=(),
            permitted_tools=(
                ToolName.GET_RECORD,
            ),
            evaluation_time_utc=datetime(
                2026,
                1,
                1,
                tzinfo=timezone.utc,
            ),
        )


def test_expected_tool_must_be_permitted() -> None:
    with pytest.raises(
        ValidationError,
        match="Expected tool call must be permitted",
    ):
        EvalGold(
            case_id="case-001",
            expected_decision=Decision.CALL_TOOL,
            expected_answerability=(
                Answerability.NOT_APPLICABLE
            ),
            permitted_tools=(),
            expected_tool_call=GetRecordCall(
                tool="get_record",
                arguments=GetRecordArgs(
                    record_id="record-001",
                ),
            ),
            rubric="Synthetic rubric.",
        )


def test_input_and_gold_permission_state_must_match() -> None:
    with pytest.raises(
        ValidationError,
        match="permission state must match",
    ):
        EvalCase(
            input=valid_input(),
            gold=EvalGold(
                case_id="case-001",
                expected_decision=Decision.RESPOND,
                expected_answerability=(
                    Answerability.SUPPORTED
                ),
                permitted_tools=(),
                rubric="Synthetic rubric.",
            ),
            source="synthetic-g2",
            provenance="unit-test",
            license="internal-test",
            review_status="not_required",
        )


def test_attempt_observation_v1_historical_success_is_readable() -> None:
    observation = AttemptObservation(
        schema_version="1.0",
        case_id="historical-case",
        transport_status=(
            AttemptTransportStatus.COMPLETED
        ),
        started_at_utc=datetime(
            2026,
            1,
            1,
            tzinfo=timezone.utc,
        ),
        completed_at_utc=datetime(
            2026,
            1,
            1,
            0,
            0,
            1,
            tzinfo=timezone.utc,
        ),
        elapsed_seconds=1.0,
        worker_returncode=0,
        model_inference_seconds=0.5,
        input_token_count=10,
        output_token_count=1,
        stop_reason="eos",
        prompt_sha256="abc123",
    )

    assert observation.schema_version == "1.0"
    assert observation.runtime_device is None
    assert observation.generated_ids is None


def test_attempt_observation_v11_requires_runtime_evidence() -> None:
    with pytest.raises(
        ValidationError,
        match="requires runtime and token evidence",
    ):
        AttemptObservation(
            schema_version="1.1",
            case_id="v11-case",
            transport_status=(
                AttemptTransportStatus.COMPLETED
            ),
            started_at_utc=datetime(
                2026,
                1,
                1,
                tzinfo=timezone.utc,
            ),
            completed_at_utc=datetime(
                2026,
                1,
                1,
                0,
                0,
                1,
                tzinfo=timezone.utc,
            ),
            elapsed_seconds=1.0,
            worker_returncode=0,
            model_inference_seconds=0.5,
            input_token_count=10,
            output_token_count=1,
            stop_reason="eos",
            prompt_sha256="abc123",
        )
