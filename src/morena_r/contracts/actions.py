from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Annotated, Literal, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    model_validator,
)


NonEmptyStr = Annotated[
    str,
    Field(min_length=1),
]


class StrictContract(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


class Decision(StrEnum):
    RESPOND = "RESPOND"
    CALL_TOOL = "CALL_TOOL"
    ASK_CLARIFICATION = "ASK_CLARIFICATION"
    REFUSE = "REFUSE"


class Answerability(StrEnum):
    SUPPORTED = "SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    CONFLICTING = "CONFLICTING"
    INSUFFICIENT_CONTEXT = "INSUFFICIENT_CONTEXT"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class ToolName(StrEnum):
    SEARCH_RECORDS = "search_records"
    GET_RECORD = "get_record"
    CALCULATE = "calculate"
    LOOKUP_STATUS = "lookup_status"


class CalculationOperation(StrEnum):
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"


class ToolErrorCode(StrEnum):
    PARSE_ERROR = "PARSE_ERROR"
    UNKNOWN_FIELD = "UNKNOWN_FIELD"
    INVALID_TOOL = "INVALID_TOOL"
    ARGUMENT_INVALID = "ARGUMENT_INVALID"
    CONTEXT_OVERFLOW = "CONTEXT_OVERFLOW"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"

    PERMISSION_DENIED = "PERMISSION_DENIED"
    NOT_FOUND = "NOT_FOUND"
    TIMEOUT = "TIMEOUT"
    MALFORMED_RESPONSE = "MALFORMED_RESPONSE"
    STALE_DATA = "STALE_DATA"
    STAGNATION = "STAGNATION"


class SearchRecordsArgs(StrictContract):
    query: Annotated[str, Field(min_length=1, max_length=500)]
    as_of_date: date | None = None
    limit: Annotated[int, Field(ge=1, le=10)] = 5


class GetRecordArgs(StrictContract):
    record_id: NonEmptyStr


class CalculateArgs(StrictContract):
    operation: CalculationOperation

    left: float = Field(
        allow_inf_nan=False,
    )

    right: float = Field(
        allow_inf_nan=False,
    )


class LookupStatusArgs(StrictContract):
    subject_id: NonEmptyStr


class SearchRecordsCall(StrictContract):
    tool: Literal["search_records"]
    arguments: SearchRecordsArgs


class GetRecordCall(StrictContract):
    tool: Literal["get_record"]
    arguments: GetRecordArgs


class CalculateCall(StrictContract):
    tool: Literal["calculate"]
    arguments: CalculateArgs


class LookupStatusCall(StrictContract):
    tool: Literal["lookup_status"]
    arguments: LookupStatusArgs


ToolCall = Annotated[
    Union[
        SearchRecordsCall,
        GetRecordCall,
        CalculateCall,
        LookupStatusCall,
    ],
    Field(discriminator="tool"),
]


class EvidenceClaim(StrictContract):
    claim: NonEmptyStr
    evidence_refs: tuple[NonEmptyStr, ...] = ()


class RespondResponse(StrictContract):
    schema_version: Literal["1.0"] = "1.0"
    status: Literal["OK"] = "OK"

    decision: Literal["RESPOND"]
    answerability: Answerability

    answer: NonEmptyStr

    claims: tuple[EvidenceClaim, ...] = ()
    citations: tuple[NonEmptyStr, ...] = ()


class CallToolResponse(StrictContract):
    schema_version: Literal["1.0"] = "1.0"
    status: Literal["OK"] = "OK"

    decision: Literal["CALL_TOOL"]
    answerability: Answerability

    tool_call: ToolCall


class AskClarificationResponse(StrictContract):
    schema_version: Literal["1.0"] = "1.0"
    status: Literal["OK"] = "OK"

    decision: Literal["ASK_CLARIFICATION"]
    answerability: Answerability

    question: NonEmptyStr


class RefuseResponse(StrictContract):
    schema_version: Literal["1.0"] = "1.0"
    status: Literal["OK"] = "OK"

    decision: Literal["REFUSE"]
    answerability: Answerability

    reason: NonEmptyStr


ModelResponse = Annotated[
    Union[
        RespondResponse,
        CallToolResponse,
        AskClarificationResponse,
        RefuseResponse,
    ],
    Field(discriminator="decision"),
]


MODEL_RESPONSE_ADAPTER = TypeAdapter(ModelResponse)
TOOL_CALL_ADAPTER = TypeAdapter(ToolCall)


class RecordFixture(StrictContract):
    record_id: NonEmptyStr
    text: Annotated[str, Field(min_length=1, max_length=20_000)]
    observed_at: date
    authoritative: bool = True


class StatusFixture(StrictContract):
    subject_id: NonEmptyStr
    status: NonEmptyStr
    observed_at: date


class SearchRecordsData(StrictContract):
    kind: Literal["search_records"] = "search_records"
    records: tuple[RecordFixture, ...]


class GetRecordData(StrictContract):
    kind: Literal["get_record"] = "get_record"
    record: RecordFixture


class CalculateData(StrictContract):
    kind: Literal["calculate"] = "calculate"

    result: float = Field(
        allow_inf_nan=False,
    )


class LookupStatusData(StrictContract):
    kind: Literal["lookup_status"] = "lookup_status"
    status: StatusFixture


ToolPayload = Annotated[
    Union[
        SearchRecordsData,
        GetRecordData,
        CalculateData,
        LookupStatusData,
    ],
    Field(discriminator="kind"),
]


class ToolResult(StrictContract):
    schema_version: Literal["1.0"] = "1.0"

    call_id: NonEmptyStr
    tool: ToolName

    success: bool
    error_code: ToolErrorCode | None = None

    payload_hash: NonEmptyStr | None = None
    data: ToolPayload | None = None

    @model_validator(mode="after")
    def validate_result_state(self) -> "ToolResult":
        if self.success:
            if self.error_code is not None:
                raise ValueError(
                    "Successful tool result cannot contain error_code."
                )

            if self.data is None:
                raise ValueError(
                    "Successful tool result requires data."
                )

            if self.payload_hash is None:
                raise ValueError(
                    "Successful tool result requires payload_hash."
                )

        if not self.success:
            if self.error_code is None:
                raise ValueError(
                    "Failed tool result requires error_code."
                )

            if self.data is not None:
                raise ValueError(
                    "Failed tool result cannot contain data."
                )

        return self
