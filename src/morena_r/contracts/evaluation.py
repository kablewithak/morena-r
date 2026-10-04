from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, model_validator

from morena_r.contracts.actions import (
    Answerability,
    Decision,
    ModelResponse,
    NonEmptyStr,
    StrictContract,
    ToolCall,
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


class AttemptTransportStatus(StrEnum):
    COMPLETED = "completed"
    TIMEOUT = "timeout"
    EXECUTION_ERROR = "execution_error"


class AttemptObservation(StrictContract):
    schema_version: Literal["1.0"] = "1.0"

    case_id: NonEmptyStr

    transport_status: AttemptTransportStatus

    started_at_utc: datetime
    completed_at_utc: datetime

    elapsed_seconds: float = Field(
        ge=0,
        allow_inf_nan=False,
    )

    worker_returncode: int | None = None

    model_inference_seconds: float | None = Field(
        default=None,
        ge=0,
        allow_inf_nan=False,
    )

    input_token_count: int | None = Field(
        default=None,
        ge=0,
    )

    output_token_count: int | None = Field(
        default=None,
        ge=0,
    )

    stop_reason: NonEmptyStr | None = None
    prompt_sha256: NonEmptyStr | None = None

    @model_validator(mode="after")
    def validate_observation(self) -> "AttemptObservation":
        for value in (
            self.started_at_utc,
            self.completed_at_utc,
        ):
            if (
                value.tzinfo is None
                or value.utcoffset() is None
            ):
                raise ValueError(
                    "Attempt observation timestamps must be timezone-aware."
                )

        if (
            self.completed_at_utc
            < self.started_at_utc
        ):
            raise ValueError(
                "Attempt completion cannot precede start."
            )

        if (
            self.transport_status
            == AttemptTransportStatus.COMPLETED
        ):
            if self.worker_returncode != 0:
                raise ValueError(
                    "Completed transport requires worker_returncode=0."
                )

            required = (
                self.model_inference_seconds,
                self.input_token_count,
                self.output_token_count,
                self.stop_reason,
                self.prompt_sha256,
            )

            if any(
                value is None
                for value in required
            ):
                raise ValueError(
                    "Completed transport requires generation metadata."
                )

        if (
            self.transport_status
            == AttemptTransportStatus.TIMEOUT
            and self.worker_returncode is not None
        ):
            raise ValueError(
                "Timeout transport must not claim a worker return code."
            )

        return self


class EvalMessage(StrictContract):
    role: Literal[
        "system",
        "user",
        "assistant",
        "tool",
    ]

    content: Annotated[
        str,
        Field(
            min_length=1,
            max_length=20_000,
        ),
    ]


class EvidenceDocument(StrictContract):
    document_id: NonEmptyStr
    version: NonEmptyStr

    content: Annotated[
        str,
        Field(
            min_length=1,
            max_length=20_000,
        ),
    ]

    observed_at_utc: datetime
    authoritative: bool = True

    @model_validator(mode="after")
    def validate_timezone(self) -> "EvidenceDocument":
        if (
            self.observed_at_utc.tzinfo is None
            or self.observed_at_utc.utcoffset() is None
        ):
            raise ValueError(
                "Evidence timestamps must be timezone-aware."
            )

        return self


class EvalInput(StrictContract):
    schema_version: Literal["1.0"] = "1.0"

    case_id: NonEmptyStr
    family_id: NonEmptyStr

    languages: tuple[NonEmptyStr, ...] = Field(
        min_length=1,
    )

    messages: tuple[EvalMessage, ...] = Field(
        min_length=1,
    )

    evidence: tuple[EvidenceDocument, ...] = ()

    available_tools: tuple[ToolName, ...] = ()
    permitted_tools: tuple[ToolName, ...] = ()

    evaluation_time_utc: datetime

    @model_validator(mode="after")
    def validate_input_state(self) -> "EvalInput":
        if (
            self.evaluation_time_utc.tzinfo is None
            or self.evaluation_time_utc.utcoffset() is None
        ):
            raise ValueError(
                "evaluation_time_utc must be timezone-aware."
            )

        if len(set(self.languages)) != len(self.languages):
            raise ValueError(
                "languages must not contain duplicates."
            )

        if not any(
            message.role == "user"
            for message in self.messages
        ):
            raise ValueError(
                "EvalInput requires at least one user message."
            )

        if (
            len(set(self.available_tools))
            != len(self.available_tools)
        ):
            raise ValueError(
                "available_tools must not contain duplicates."
            )

        if (
            len(set(self.permitted_tools))
            != len(self.permitted_tools)
        ):
            raise ValueError(
                "permitted_tools must not contain duplicates."
            )

        if not set(self.permitted_tools).issubset(
            set(self.available_tools)
        ):
            raise ValueError(
                "Permitted tools must be available tools."
            )

        evidence_ids = tuple(
            document.document_id
            for document in self.evidence
        )

        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError(
                "Evidence document IDs must be unique."
            )

        return self


class GoldClaim(StrictContract):
    claim: NonEmptyStr
    evidence_refs: tuple[NonEmptyStr, ...] = ()


class EvalGold(StrictContract):
    schema_version: Literal["1.0"] = "1.0"

    case_id: NonEmptyStr

    expected_decision: Decision
    expected_answerability: Answerability

    permitted_tools: tuple[ToolName, ...] = ()

    expected_tool_call: ToolCall | None = None

    expected_claims: tuple[GoldClaim, ...] = ()
    rubric: NonEmptyStr

    @model_validator(mode="after")
    def validate_tool_expectation(self) -> "EvalGold":
        if (
            self.expected_decision == Decision.CALL_TOOL
            and self.expected_tool_call is None
        ):
            raise ValueError(
                "CALL_TOOL gold requires expected_tool_call."
            )

        if (
            self.expected_decision != Decision.CALL_TOOL
            and self.expected_tool_call is not None
        ):
            raise ValueError(
                "expected_tool_call is only valid for CALL_TOOL gold."
            )

        if (
            self.expected_tool_call is not None
            and ToolName(self.expected_tool_call.tool)
            not in self.permitted_tools
        ):
            raise ValueError(
                "Expected tool call must be permitted."
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
    def validate_case_state(self) -> "EvalCase":
        if self.input.case_id != self.gold.case_id:
            raise ValueError(
                "EvalInput and EvalGold case IDs must match."
            )

        if set(self.input.permitted_tools) != set(
            self.gold.permitted_tools
        ):
            raise ValueError(
                "Input and gold permission state must match."
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

        if self.status == AttemptStatus.TIMEOUT:
            if self.raw_output is not None:
                raise ValueError(
                    "Timeout cannot contain raw_output."
                )

            if self.parsed_response is not None:
                raise ValueError(
                    "Timeout cannot contain parsed_response."
                )

            if self.parse_failure is not None:
                raise ValueError(
                    "Timeout cannot contain parse_failure."
                )

            if (
                self.runtime_error_code
                != AttemptErrorCode.TIMEOUT
            ):
                raise ValueError(
                    "Timeout requires TIMEOUT error code."
                )

        if self.status == AttemptStatus.EXECUTION_ERROR:
            if self.raw_output is not None:
                raise ValueError(
                    "Execution failure cannot contain raw_output."
                )

            if self.parsed_response is not None:
                raise ValueError(
                    "Execution failure cannot contain parsed_response."
                )

            if self.parse_failure is not None:
                raise ValueError(
                    "Execution failure cannot contain parse_failure."
                )

            if self.runtime_error_code is None:
                raise ValueError(
                    "Execution failure requires runtime_error_code."
                )

            if (
                self.runtime_error_code
                == AttemptErrorCode.TIMEOUT
            ):
                raise ValueError(
                    "TIMEOUT must use timeout attempt status."
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
