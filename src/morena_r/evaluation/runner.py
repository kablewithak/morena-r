from __future__ import annotations

from collections import Counter
from typing import Protocol

from morena_r.contracts.actions import StrictContract
from morena_r.contracts.evaluation import (
    AttemptErrorCode,
    AttemptRecord,
    AttemptStatus,
    EvalCase,
    EvalInput,
    RunSummary,
)
from morena_r.contracts.scoring import ScoreRecord
from morena_r.evaluation.parser import (
    parse_model_response,
)
from morena_r.scoring.deterministic import (
    ScoreContext,
    score_attempt,
)


class ModelTimeoutError(TimeoutError):
    pass


class ModelExecutionError(RuntimeError):
    pass


class ModelAdapter(Protocol):
    def generate(
        self,
        eval_input: EvalInput,
    ) -> str:
        ...


class EvaluationRunConfig(StrictContract):
    run_id: str
    model_identity_hash: str
    dataset_hash: str
    scorer_version: str = "g2-deterministic-v1"


class EvaluationRunResult(StrictContract):
    config: EvaluationRunConfig
    attempts: tuple[AttemptRecord, ...]
    scores: tuple[ScoreRecord, ...]
    summary: RunSummary


class EvaluationRunner:
    def __init__(
        self,
        *,
        config: EvaluationRunConfig,
        adapter: ModelAdapter,
    ) -> None:
        self._config = config
        self._adapter = adapter

    def run(
        self,
        cases: tuple[EvalCase, ...],
    ) -> EvaluationRunResult:
        case_ids = tuple(
            case.input.case_id
            for case in cases
        )

        if len(set(case_ids)) != len(case_ids):
            raise ValueError(
                "Evaluation run contains duplicate case IDs."
            )

        attempts: list[AttemptRecord] = []
        scores: list[ScoreRecord] = []

        score_context = ScoreContext(
            model_identity_hash=(
                self._config.model_identity_hash
            ),
            dataset_hash=self._config.dataset_hash,
            scorer_version=self._config.scorer_version,
        )

        for case in cases:
            attempt = self._run_case(case)

            attempts.append(attempt)

            scores.extend(
                score_attempt(
                    context=score_context,
                    case=case,
                    attempt=attempt,
                )
            )

        counts = Counter(
            attempt.status
            for attempt in attempts
        )

        summary = RunSummary(
            run_id=self._config.run_id,
            expected_cases=len(cases),
            attempted_cases=len(attempts),
            completed=counts[
                AttemptStatus.COMPLETED
            ],
            invalid_responses=counts[
                AttemptStatus.INVALID_RESPONSE
            ],
            timeouts=counts[
                AttemptStatus.TIMEOUT
            ],
            execution_errors=counts[
                AttemptStatus.EXECUTION_ERROR
            ],
            score_rows=len(scores),
            case_ids=tuple(
                attempt.case_id
                for attempt in attempts
            ),
        )

        return EvaluationRunResult(
            config=self._config,
            attempts=tuple(attempts),
            scores=tuple(scores),
            summary=summary,
        )

    def _run_case(
        self,
        case: EvalCase,
    ) -> AttemptRecord:
        attempt_id = (
            f"{self._config.run_id}:"
            f"{case.input.case_id}:1"
        )

        try:
            raw_output = self._adapter.generate(
                case.input
            )
        except ModelTimeoutError:
            return AttemptRecord(
                attempt_id=attempt_id,
                run_id=self._config.run_id,
                case_id=case.input.case_id,
                family_id=case.input.family_id,
                status=AttemptStatus.TIMEOUT,
                runtime_error_code=(
                    AttemptErrorCode.TIMEOUT
                ),
            )
        except ModelExecutionError:
            return AttemptRecord(
                attempt_id=attempt_id,
                run_id=self._config.run_id,
                case_id=case.input.case_id,
                family_id=case.input.family_id,
                status=(
                    AttemptStatus.EXECUTION_ERROR
                ),
                runtime_error_code=(
                    AttemptErrorCode.EXECUTION_ERROR
                ),
            )

        if not isinstance(raw_output, str):
            return AttemptRecord(
                attempt_id=attempt_id,
                run_id=self._config.run_id,
                case_id=case.input.case_id,
                family_id=case.input.family_id,
                status=(
                    AttemptStatus.EXECUTION_ERROR
                ),
                runtime_error_code=(
                    AttemptErrorCode.EXECUTION_ERROR
                ),
            )

        parsed = parse_model_response(
            raw_output
        )

        if parsed.response is not None:
            return AttemptRecord(
                attempt_id=attempt_id,
                run_id=self._config.run_id,
                case_id=case.input.case_id,
                family_id=case.input.family_id,
                status=AttemptStatus.COMPLETED,
                raw_output=raw_output,
                parsed_response=parsed.response,
            )

        return AttemptRecord(
            attempt_id=attempt_id,
            run_id=self._config.run_id,
            case_id=case.input.case_id,
            family_id=case.input.family_id,
            status=AttemptStatus.INVALID_RESPONSE,
            raw_output=raw_output,
            parse_failure=parsed.failure,
        )
