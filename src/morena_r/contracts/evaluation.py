from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, model_validator

from morena_r.contracts.actions import (
    Answerability,
    Decision,
    ModelResponse,
    NonEmptyStr,
    StrictContract,
    ToolName,
)


class AttemptStatus(StrEnum):
    COMPLETED = "completed"
    INVALID_RESPONSE = "invalid_response"
    TIMEOUT = "timeout"
    EXECUTION_ERROR = "execution_error"


class AttemptErrorCode(StrEnum):
    PARSE_ERROR = "PARSE_ERROR"
    UNKNOWN_FIELD = "UNKNOWN_FIELD"
    INVALID_TOOL = "INVALID_TOOL"
    ARGUMENT_INVALID = "ARGUMENT_INVALID"
    CONTEXT_OVERFLOW = "CONTEXT_OVERFLOW"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    TIMEOUT = "TIMEOUT"
    EXECUTION_ERROR = "EXECUTION_ERROR"


class EvalInput(StrictContract):
    schema_version: Literal["1.0"] = "1.0"

    case_id: NonEmptyStr
    family_id: NonEmptyStr

    language: NonEmptyStr

    user_message: Annotated[
        str,
        Field(
            min_length=1,
            max_length=20_000,
        ),
    ]

    available_tools: tuple[ToolName, ...] = ()


class EvalGold(StrictContract):
    schema_version: Literal["1.0"] = "1.0"

    case_id: NonEmptyStr

    expected_decision: Decision
    expected_answerability: Answerability

    expected_tool: ToolName | None = None

    @model_validator(mode="after")
    def validate_tool_expectation(self) -> "EvalGold":
        if (
            self.expected_decision == Decision.CALL_TOOL
            and self.expected_tool is None
        ):
            raise ValueError(
                "CALL_TOOL gold requires expected_tool."
            )

        if (
            self.expected_decision != Decision.CALL_TOOL
            and self.expected_tool is not None
        ):
            raise ValueError(
                "expected_tool is only valid for CALL_TOOL gold."
            )

        return self


class EvalCase(StrictContract):
    schema_version: Literal["1.0"] = "1.0"

    input: EvalInput
    gold: EvalGold

    split: Literal["development_pilot"] = "development_pilot"

    source: NonEmptyStr
    provenance: NonEmptyStr
    license: NonEmptyStr
    review_status: NonEmptyStr

    @model_validator(mode="after")
    def validate_case_identity(self) -> "EvalCase":
        if self.input.case_id != self.gold.case_id:
            raise ValueError(
                "EvalInput and EvalGold case IDs must match."
            )

        return self


class ParseFailure(StrictContract):
    code: AttemptErrorCode
    message: NonEmptyStr


class ParseOutcome(StrictContract):
    raw_output: str

    response: ModelResponse | None = None
    failure: ParseFailure | None = None

    @model_validator(mode="after")
    def validate_parse_state(self) -> "ParseOutcome":
        success = self.response is not None
        failed = self.failure is not None

        if success == failed:
            raise ValueError(
                "ParseOutcome requires exactly one of response or failure."
            )

        return self


class AttemptRecord(StrictContract):
    schema_version: Literal["1.0"] = "1.0"

    attempt_id: NonEmptyStr
    run_id: NonEmptyStr
    case_id: NonEmptyStr
    family_id: NonEmptyStr

    status: AttemptStatus

    raw_output: str | None = None
    parsed_response: ModelResponse | None = None

    parse_failure: ParseFailure | None = None
    runtime_error_code: AttemptErrorCode | None = None

    @model_validator(mode="after")
    def validate_attempt_state(self) -> "AttemptRecord":
        if self.status == AttemptStatus.COMPLETED:
            if self.raw_output is None:
                raise ValueError(
                    "Completed attempt requires raw_output."
                )

            if self.parsed_response is None:
                raise ValueError(
                    "Completed attempt requires parsed_response."
                )

            if self.parse_failure is not None:
                raise ValueError(
                    "Completed attempt cannot contain parse_failure."
                )

            if self.runtime_error_code is not None:
                raise ValueError(
                    "Completed attempt cannot contain runtime_error_code."
                )

        if self.status == AttemptStatus.INVALID_RESPONSE:
            if self.raw_output is None:
                raise ValueError(
                    "Invalid response attempt requires raw_output."
                )

            if self.parsed_response is not None:
                raise ValueError(
                    "Invalid response cannot contain parsed_response."
                )

            if self.parse_failure is None:
                raise ValueError(
                    "Invalid response requires parse_failure."
                )

            if self.runtime_error_code is not None:
                raise ValueError(
                    "Invalid response cannot contain runtime_error_code."
                )

        if self.status in {
            AttemptStatus.TIMEOUT,
            AttemptStatus.EXECUTION_ERROR,
        }:
            if self.raw_output is not None:
                raise ValueError(
                    "Runtime failure cannot contain raw_output."
                )

            if self.parsed_response is not None:
                raise ValueError(
                    "Runtime failure cannot contain parsed_response."
                )

            if self.parse_failure is not None:
                raise ValueError(
                    "Runtime failure cannot contain parse_failure."
                )

            if self.runtime_error_code is None:
                raise ValueError(
                    "Runtime failure requires runtime_error_code."
                )

        return self


class RunSummary(StrictContract):
    schema_version: Literal["1.0"] = "1.0"

    run_id: NonEmptyStr

    expected_cases: int = Field(ge=0)
    attempted_cases: int = Field(ge=0)

    completed: int = Field(ge=0)
    invalid_responses: int = Field(ge=0)
    timeouts: int = Field(ge=0)
    execution_errors: int = Field(ge=0)

    score_rows: int = Field(ge=0)

    case_ids: tuple[NonEmptyStr, ...]

    @model_validator(mode="after")
    def validate_accounting(self) -> "RunSummary":
        accounted = (
            self.completed
            + self.invalid_responses
            + self.timeouts
            + self.execution_errors
        )

        if self.expected_cases != self.attempted_cases:
            raise ValueError(
                "Final run summary must account for every expected case."
            )

        if accounted != self.attempted_cases:
            raise ValueError(
                "Attempt status counts do not reconcile."
            )

        if len(self.case_ids) != self.attempted_cases:
            raise ValueError(
                "case_ids length does not match attempted_cases."
            )

        if len(set(self.case_ids)) != len(self.case_ids):
            raise ValueError(
                "Duplicate case IDs are not permitted."
            )

        return self
