from __future__ import annotations

from morena_r.contracts.actions import (
    CallToolResponse,
    Decision,
    StrictContract,
)
from morena_r.contracts.evaluation import (
    AttemptRecord,
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


def _schema_failure_labels(
    attempt: AttemptRecord,
) -> tuple[FailureLabel, ...]:
    if attempt.parsed_response is None:
        return (
            FailureLabel.PARSER_FAILURE,
            FailureLabel.INVALID_SCHEMA,
        )

    return ()


def _action_failure_labels(
    case: EvalCase,
    attempt: AttemptRecord,
) -> tuple[FailureLabel, ...]:
    response = attempt.parsed_response

    if response is None:
        return _schema_failure_labels(attempt)

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
        and response.tool_call.tool
        != case.gold.expected_tool
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

    schema_valid = ScoreRecord(
        **common,
        metric_id="schema_valid",
        eligible=True,
        value=1.0 if response is not None else 0.0,
        failure_labels=_schema_failure_labels(
            attempt
        ),
    )

    action_correct = ScoreRecord(
        **common,
        metric_id="action_correct",
        eligible=True,
        value=(
            1.0
            if (
                response is not None
                and Decision(response.decision)
                == case.gold.expected_decision
            )
            else 0.0
        ),
        failure_labels=_action_failure_labels(
            case,
            attempt,
        ),
    )

    answerability_correct = ScoreRecord(
        **common,
        metric_id="answerability_correct",
        eligible=True,
        value=(
            1.0
            if (
                response is not None
                and response.answerability
                == case.gold.expected_answerability
            )
            else 0.0
        ),
        failure_labels=(
            _schema_failure_labels(attempt)
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
        == Decision.CALL_TOOL
    ):
        tool_correct = (
            response is not None
            and isinstance(
                response,
                CallToolResponse,
            )
            and response.tool_call.tool
            == case.gold.expected_tool
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
    else:
        tool_identity = ScoreRecord(
            **common,
            metric_id="tool_identity_correct",
            eligible=False,
            value=None,
            missing_reason="not_tool_required",
        )

    return (
        schema_valid,
        action_correct,
        answerability_correct,
        tool_identity,
    )
