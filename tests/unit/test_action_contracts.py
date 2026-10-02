from __future__ import annotations

import pytest
from pydantic import ValidationError

from morena_r.contracts.actions import (
    Answerability,
    CallToolResponse,
    Decision,
    MODEL_RESPONSE_ADAPTER,
)


def test_respond_contract_is_valid() -> None:
    response = MODEL_RESPONSE_ADAPTER.validate_python(
        {
            "decision": "RESPOND",
            "answerability": "SUPPORTED",
            "answer": "The supplied evidence supports this answer.",
            "claims": [],
            "citations": [],
        }
    )

    assert response.decision == Decision.RESPOND
    assert response.answerability == Answerability.SUPPORTED


def test_call_tool_requires_typed_tool_call() -> None:
    response = MODEL_RESPONSE_ADAPTER.validate_python(
        {
            "decision": "CALL_TOOL",
            "answerability": "NOT_APPLICABLE",
            "tool_call": {
                "tool": "get_record",
                "arguments": {
                    "record_id": "record-001",
                },
            },
        }
    )

    assert isinstance(response, CallToolResponse)
    assert response.tool_call.tool == "get_record"


def test_unknown_tool_is_rejected() -> None:
    with pytest.raises(ValidationError):
        MODEL_RESPONSE_ADAPTER.validate_python(
            {
                "decision": "CALL_TOOL",
                "answerability": "NOT_APPLICABLE",
                "tool_call": {
                    "tool": "delete_account",
                    "arguments": {},
                },
            }
        )


def test_unknown_tool_argument_is_rejected() -> None:
    with pytest.raises(ValidationError):
        MODEL_RESPONSE_ADAPTER.validate_python(
            {
                "decision": "CALL_TOOL",
                "answerability": "NOT_APPLICABLE",
                "tool_call": {
                    "tool": "get_record",
                    "arguments": {
                        "record_id": "record-001",
                        "hidden_extra": True,
                    },
                },
            }
        )


def test_refuse_cannot_carry_executable_call() -> None:
    with pytest.raises(ValidationError):
        MODEL_RESPONSE_ADAPTER.validate_python(
            {
                "decision": "REFUSE",
                "answerability": "NOT_APPLICABLE",
                "reason": "The request is disallowed.",
                "tool_call": {
                    "tool": "get_record",
                    "arguments": {
                        "record_id": "record-001",
                    },
                },
            }
        )


def test_clarification_requires_question() -> None:
    with pytest.raises(ValidationError):
        MODEL_RESPONSE_ADAPTER.validate_python(
            {
                "decision": "ASK_CLARIFICATION",
                "answerability": "INSUFFICIENT_CONTEXT",
            }
        )
