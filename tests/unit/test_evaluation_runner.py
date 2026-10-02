from __future__ import annotations

from morena_r.contracts.actions import (
    Answerability,
    Decision,
    ToolName,
)
from morena_r.contracts.evaluation import (
    AttemptStatus,
    EvalCase,
    EvalGold,
    EvalInput,
)
from morena_r.evaluation.runner import (
    EvaluationRunConfig,
    EvaluationRunner,
    ModelTimeoutError,
)


class FakeAdapter:
    def generate(
        self,
        eval_input: EvalInput,
    ) -> str:
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
        }

        if eval_input.case_id == "case-timeout":
            raise ModelTimeoutError(
                "fixture timeout"
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
    return EvalCase(
        input=EvalInput(
            case_id=case_id,
            family_id=family_id,
            language="en",
            user_message="Synthetic test request.",
            available_tools=(
                ToolName.GET_RECORD,
            ),
        ),
        gold=EvalGold(
            case_id=case_id,
            expected_decision=decision,
            expected_answerability=answerability,
            expected_tool=tool,
        ),
        source="synthetic-g2",
        provenance="unit-test",
        license="internal-test",
        review_status="not_required",
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
        config=EvaluationRunConfig(
            run_id="g2-pilot-test",
            model_identity_hash="model-hash",
            dataset_hash="dataset-hash",
        ),
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

    assert result.summary.case_ids == (
        "case-valid",
        "case-invalid",
        "case-wrong-action",
        "case-timeout",
    )


def test_raw_invalid_output_is_preserved() -> None:
    case = make_case(
        case_id="case-invalid",
        family_id="family-2",
        decision=Decision.RESPOND,
        answerability=Answerability.SUPPORTED,
    )

    runner = EvaluationRunner(
        config=EvaluationRunConfig(
            run_id="raw-preservation-test",
            model_identity_hash="model-hash",
            dataset_hash="dataset-hash",
        ),
        adapter=FakeAdapter(),
    )

    result = runner.run((case,))

    attempt = result.attempts[0]

    assert (
        attempt.status
        == AttemptStatus.INVALID_RESPONSE
    )
    assert attempt.raw_output == '{"decision":'


def test_timeout_is_not_silently_dropped() -> None:
    case = make_case(
        case_id="case-timeout",
        family_id="family-4",
        decision=Decision.RESPOND,
        answerability=Answerability.SUPPORTED,
    )

    runner = EvaluationRunner(
        config=EvaluationRunConfig(
            run_id="timeout-test",
            model_identity_hash="model-hash",
            dataset_hash="dataset-hash",
        ),
        adapter=FakeAdapter(),
    )

    result = runner.run((case,))

    assert result.summary.attempted_cases == 1
    assert result.summary.timeouts == 1
    assert len(result.scores) == 4
