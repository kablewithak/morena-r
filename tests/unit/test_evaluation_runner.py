from __future__ import annotations

from datetime import datetime, timezone

import pytest

from morena_r.contracts.actions import (
    Answerability,
    Decision,
    GetRecordArgs,
    GetRecordCall,
    ToolName,
)
from morena_r.contracts.evaluation import (
    AttemptErrorCode,
    AttemptStatus,
    EvalCase,
    EvalGold,
    EvalInput,
    EvalMessage,
)
from morena_r.contracts.scoring import (
    FailureLabel,
)
from morena_r.evaluation.runner import (
    EvaluationRunConfig,
    EvaluationRunner,
    ModelContextOverflowError,
    ModelTimeoutError,
)
from morena_r.reporting.run_report import (
    build_run_report,
    canonical_report_json,
)


class FakeAdapter:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate(
        self,
        eval_input: EvalInput,
    ) -> str:
        self.calls.append(
            eval_input.case_id
        )

        outputs = {
            "case-valid": (
                '{"decision":"RESPOND",'
                '"answerability":"SUPPORTED",'
                '"answer":"Harare"}'
            ),
            "case-invalid": '{"decision":',
            "case-wrong-action": (
                '{"decision":"CALL_TOOL",'
                '"answerability":"NOT_APPLICABLE",'
                '"tool_call":{'
                '"tool":"get_record",'
                '"arguments":{"record_id":"record-001"}}}'
            ),
            "case-second": (
                '{"decision":"RESPOND",'
                '"answerability":"SUPPORTED",'
                '"answer":"Second answer"}'
            ),
        }

        if eval_input.case_id == "case-timeout":
            raise ModelTimeoutError(
                "fixture timeout"
            )

        if eval_input.case_id == "case-overflow":
            raise ModelContextOverflowError(
                "fixture context overflow"
            )

        return outputs[eval_input.case_id]


def make_case(
    *,
    case_id: str,
    family_id: str,
    decision: Decision,
    answerability: Answerability,
    tool: ToolName | None = None,
) -> EvalCase:
    permitted_tools = (
        ToolName.GET_RECORD,
    )

    expected_tool_call = (
        GetRecordCall(
            tool="get_record",
            arguments=GetRecordArgs(
                record_id="record-001",
            ),
        )
        if tool == ToolName.GET_RECORD
        else None
    )

    return EvalCase(
        input=EvalInput(
            case_id=case_id,
            family_id=family_id,
            languages=("en",),
            messages=(
                EvalMessage(
                    role="user",
                    content="Synthetic test request.",
                ),
            ),
            evidence=(),
            available_tools=permitted_tools,
            permitted_tools=permitted_tools,
            evaluation_time_utc=datetime(
                2026,
                1,
                1,
                tzinfo=timezone.utc,
            ),
        ),
        gold=EvalGold(
            case_id=case_id,
            expected_decision=decision,
            expected_answerability=answerability,
            permitted_tools=permitted_tools,
            expected_tool_call=expected_tool_call,
            expected_claims=(),
            rubric=(
                "Score the frozen synthetic decision "
                "and answerability contract."
            ),
        ),
        source="synthetic-g2",
        provenance="unit-test",
        license="internal-test",
        review_status="not_required",
    )


def config(
    run_id: str = "g2-pilot-test",
) -> EvaluationRunConfig:
    return EvaluationRunConfig(
        run_id=run_id,
        model_identity_hash="model-hash",
        dataset_hash="dataset-hash",
    )


def test_runner_accounts_for_every_case() -> None:
    cases = (
        make_case(
            case_id="case-valid",
            family_id="family-1",
            decision=Decision.RESPOND,
            answerability=Answerability.SUPPORTED,
        ),
        make_case(
            case_id="case-invalid",
            family_id="family-2",
            decision=Decision.RESPOND,
            answerability=Answerability.SUPPORTED,
        ),
        make_case(
            case_id="case-wrong-action",
            family_id="family-3",
            decision=Decision.RESPOND,
            answerability=Answerability.SUPPORTED,
        ),
        make_case(
            case_id="case-timeout",
            family_id="family-4",
            decision=Decision.RESPOND,
            answerability=Answerability.SUPPORTED,
        ),
    )

    runner = EvaluationRunner(
        config=config(),
        adapter=FakeAdapter(),
    )

    result = runner.run(cases)

    assert result.summary.expected_cases == 4
    assert result.summary.attempted_cases == 4

    assert result.summary.completed == 2
    assert result.summary.invalid_responses == 1
    assert result.summary.timeouts == 1
    assert result.summary.execution_errors == 0

    assert result.summary.score_rows == 16


def test_raw_invalid_output_is_preserved() -> None:
    case = make_case(
        case_id="case-invalid",
        family_id="family-2",
        decision=Decision.RESPOND,
        answerability=Answerability.SUPPORTED,
    )

    result = EvaluationRunner(
        config=config("raw-preservation-test"),
        adapter=FakeAdapter(),
    ).run((case,))

    attempt = result.attempts[0]

    assert (
        attempt.status
        == AttemptStatus.INVALID_RESPONSE
    )

    assert attempt.raw_output == '{"decision":'


def test_timeout_is_runtime_missing_not_parser_failure() -> None:
    case = make_case(
        case_id="case-timeout",
        family_id="family-4",
        decision=Decision.RESPOND,
        answerability=Answerability.SUPPORTED,
    )

    result = EvaluationRunner(
        config=config("timeout-test"),
        adapter=FakeAdapter(),
    ).run((case,))

    attempt = result.attempts[0]

    assert attempt.status == AttemptStatus.TIMEOUT
    assert (
        attempt.runtime_error_code
        == AttemptErrorCode.TIMEOUT
    )

    schema_score = next(
        score
        for score in result.scores
        if score.metric_id == "schema_valid"
    )

    assert schema_score.eligible is False
    assert schema_score.value is None

    action_score = next(
        score
        for score in result.scores
        if score.metric_id == "action_correct"
    )

    assert action_score.eligible is True
    assert action_score.value == 0.0

    answerability_score = next(
        score
        for score in result.scores
        if score.metric_id == "answerability_correct"
    )

    assert answerability_score.eligible is True
    assert answerability_score.value == 0.0

    action_score = next(
        score
        for score in result.scores
        if score.metric_id == "action_correct"
    )

    assert action_score.eligible is True
    assert action_score.value == 0.0

    answerability_score = next(
        score
        for score in result.scores
        if score.metric_id == "answerability_correct"
    )

    assert answerability_score.eligible is True
    assert answerability_score.value == 0.0
    assert (
        FailureLabel.TIMEOUT
        in schema_score.failure_labels
    )
    assert (
        FailureLabel.PARSER_FAILURE
        not in schema_score.failure_labels
    )


def test_context_overflow_is_runtime_failure() -> None:
    case = make_case(
        case_id="case-overflow",
        family_id="family-overflow",
        decision=Decision.RESPOND,
        answerability=Answerability.SUPPORTED,
    )

    result = EvaluationRunner(
        config=config("overflow-test"),
        adapter=FakeAdapter(),
    ).run((case,))

    attempt = result.attempts[0]

    assert (
        attempt.status
        == AttemptStatus.EXECUTION_ERROR
    )

    assert (
        attempt.runtime_error_code
        == AttemptErrorCode.CONTEXT_OVERFLOW
    )

    schema_score = next(
        score
        for score in result.scores
        if score.metric_id == "schema_valid"
    )

    assert schema_score.eligible is False
    assert schema_score.value is None
    assert (
        FailureLabel.OVERFLOW
        in schema_score.failure_labels
    )


def test_resume_does_not_rerun_existing_case() -> None:
    first_case = make_case(
        case_id="case-valid",
        family_id="family-1",
        decision=Decision.RESPOND,
        answerability=Answerability.SUPPORTED,
    )

    second_case = make_case(
        case_id="case-second",
        family_id="family-2",
        decision=Decision.RESPOND,
        answerability=Answerability.SUPPORTED,
    )

    adapter = FakeAdapter()

    first = EvaluationRunner(
        config=config("resume-test"),
        adapter=adapter,
    ).run((first_case,))

    resumed = EvaluationRunner(
        config=config("resume-test"),
        adapter=adapter,
    ).run(
        (
            first_case,
            second_case,
        ),
        existing_attempts=first.attempts,
    )

    assert adapter.calls == [
        "case-valid",
        "case-second",
    ]

    assert resumed.summary.attempted_cases == 2


def test_duplicate_resume_attempt_is_rejected() -> None:
    case = make_case(
        case_id="case-valid",
        family_id="family-1",
        decision=Decision.RESPOND,
        answerability=Answerability.SUPPORTED,
    )

    adapter = FakeAdapter()

    first = EvaluationRunner(
        config=config("duplicate-test"),
        adapter=adapter,
    ).run((case,))

    with pytest.raises(
        ValueError,
        match="Duplicate existing attempt",
    ):
        EvaluationRunner(
            config=config("duplicate-test"),
            adapter=adapter,
        ).run(
            (case,),
            existing_attempts=(
                first.attempts[0],
                first.attempts[0],
            ),
        )


def test_report_regeneration_is_byte_deterministic() -> None:
    case = make_case(
        case_id="case-valid",
        family_id="family-1",
        decision=Decision.RESPOND,
        answerability=Answerability.SUPPORTED,
    )

    result = EvaluationRunner(
        config=config("report-test"),
        adapter=FakeAdapter(),
    ).run((case,))

    first = canonical_report_json(
        build_run_report(result)
    )

    second = canonical_report_json(
        build_run_report(result)
    )

    assert first == second
