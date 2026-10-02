from __future__ import annotations

from morena_r.contracts.actions import (
    CallToolResponse,
    Decision,
    StrictContract,
    ToolName,
)
from morena_r.contracts.evaluation import (
    AttemptErrorCode,
    AttemptRecord,
    AttemptStatus,
    EvalCase,
)
from morena_r.contracts.scoring import (
    FailureLabel,
    ScoreRecord,
)


SCORER_VERSION = "g2-deterministic-v1"


class ScoreContext(StrictContract):
    model_identity_hash: str
    dataset_hash: str
    scorer_version: str = SCORER_VERSION


def _runtime_missing_reason(
    attempt: AttemptRecord,
) -> str | None:
    if attempt.status == AttemptStatus.TIMEOUT:
        return "runtime_timeout"

    if attempt.status == AttemptStatus.EXECUTION_ERROR:
        code = attempt.runtime_error_code

        assert code is not None

        return f"runtime_{code.value.lower()}"

    return None


def _attempt_failure_labels(
    attempt: AttemptRecord,
) -> tuple[FailureLabel, ...]:
    if attempt.status == AttemptStatus.INVALID_RESPONSE:
        return (
            FailureLabel.PARSER_FAILURE,
            FailureLabel.INVALID_SCHEMA,
        )

    if attempt.status == AttemptStatus.TIMEOUT:
        return (
            FailureLabel.TIMEOUT,
        )

    if attempt.status == AttemptStatus.EXECUTION_ERROR:
        if (
            attempt.runtime_error_code
            == AttemptErrorCode.CONTEXT_OVERFLOW
        ):
            return (
                FailureLabel.OVERFLOW,
            )

        return (
            FailureLabel.INCOMPLETE_RUN,
        )

    return ()


def _action_failure_labels(
    case: EvalCase,
    attempt: AttemptRecord,
) -> tuple[FailureLabel, ...]:
    response = attempt.parsed_response

    if response is None:
        return _attempt_failure_labels(attempt)

    expected = case.gold.expected_decision
    actual = Decision(response.decision)

    labels: list[FailureLabel] = []

    if (
        expected == Decision.CALL_TOOL
        and actual != Decision.CALL_TOOL
    ):
        labels.append(
            FailureLabel.MISSED_CALL
        )

    if (
        expected != Decision.CALL_TOOL
        and actual == Decision.CALL_TOOL
    ):
        labels.append(
            FailureLabel.UNNECESSARY_CALL
        )

    if (
        expected == Decision.CALL_TOOL
        and actual == Decision.CALL_TOOL
        and isinstance(response, CallToolResponse)
        and case.gold.expected_tool_call is not None
        and ToolName(response.tool_call.tool)
        != ToolName(case.gold.expected_tool_call.tool)
    ):
        labels.append(
            FailureLabel.WRONG_TOOL
        )

    return tuple(labels)


def score_attempt(
    *,
    context: ScoreContext,
    case: EvalCase,
    attempt: AttemptRecord,
) -> tuple[ScoreRecord, ...]:
    response = attempt.parsed_response
    runtime_missing = _runtime_missing_reason(
        attempt
    )

    common = {
        "run_id": attempt.run_id,
        "case_id": case.input.case_id,
        "family_id": case.input.family_id,
        "model_identity_hash": context.model_identity_hash,
        "dataset_hash": context.dataset_hash,
        "scorer_version": context.scorer_version,
        "review_status": case.review_status,
        "attempt_status": attempt.status.value,
        "evidence_refs": (
            f"attempt:{attempt.attempt_id}",
        ),
    }

    failure_labels = _attempt_failure_labels(
        attempt
    )

    schema_valid = ScoreRecord(
        **common,
        metric_id="schema_valid",
        eligible=runtime_missing is None,
        value=(
            None
            if runtime_missing is not None
            else (
                1.0
                if response is not None
                else 0.0
            )
        ),
        missing_reason=runtime_missing,
        failure_labels=failure_labels,
    )

    action_correct = ScoreRecord(
        **common,
        metric_id="action_correct",
        eligible=runtime_missing is None,
        value=(
            None
            if runtime_missing is not None
            else (
                1.0
                if (
                    response is not None
                    and Decision(response.decision)
                    == case.gold.expected_decision
                )
                else 0.0
            )
        ),
        missing_reason=runtime_missing,
        failure_labels=_action_failure_labels(
            case,
            attempt,
        ),
    )

    answerability_correct = ScoreRecord(
        **common,
        metric_id="answerability_correct",
        eligible=runtime_missing is None,
        value=(
            None
            if runtime_missing is not None
            else (
                1.0
                if (
                    response is not None
                    and response.answerability
                    == case.gold.expected_answerability
                )
                else 0.0
            )
        ),
        missing_reason=runtime_missing,
        failure_labels=(
            failure_labels
            if response is None
            else (
                ()
                if response.answerability
                == case.gold.expected_answerability
                else (
                    FailureLabel.WRONG_ANSWERABILITY,
                )
            )
        ),
    )

    if (
        case.gold.expected_decision
        != Decision.CALL_TOOL
    ):
        tool_identity = ScoreRecord(
            **common,
            metric_id="tool_identity_correct",
            eligible=False,
            value=None,
            missing_reason="not_tool_required",
        )

    elif runtime_missing is not None:
        tool_identity = ScoreRecord(
            **common,
            metric_id="tool_identity_correct",
            eligible=False,
            value=None,
            missing_reason=runtime_missing,
            failure_labels=failure_labels,
        )

    else:
        expected_call = case.gold.expected_tool_call

        assert expected_call is not None

        tool_correct = (
            response is not None
            and isinstance(
                response,
                CallToolResponse,
            )
            and ToolName(response.tool_call.tool)
            == ToolName(expected_call.tool)
        )

        tool_identity = ScoreRecord(
            **common,
            metric_id="tool_identity_correct",
            eligible=True,
            value=1.0 if tool_correct else 0.0,
            failure_labels=_action_failure_labels(
                case,
                attempt,
            ),
        )

    return (
        schema_valid,
        action_correct,
        answerability_correct,
        tool_identity,
    )
