from __future__ import annotations

from morena_r.contracts.evaluation import (
    AttemptErrorCode,
)
from morena_r.evaluation.parser import (
    parse_model_response,
)


def test_valid_response_preserves_raw_output() -> None:
    raw = (
        '{"decision":"RESPOND",'
        '"answerability":"SUPPORTED",'
        '"answer":"Harare"}'
    )

    result = parse_model_response(raw)

    assert result.raw_output == raw
    assert result.response is not None
    assert result.failure is None


def test_invalid_json_is_parse_error() -> None:
    raw = '{"decision":'

    result = parse_model_response(raw)

    assert result.raw_output == raw
    assert result.response is None
    assert result.failure is not None
    assert (
        result.failure.code
        == AttemptErrorCode.PARSE_ERROR
    )


def test_unknown_field_is_typed() -> None:
    raw = (
        '{"decision":"RESPOND",'
        '"answerability":"SUPPORTED",'
        '"answer":"Harare",'
        '"secret_extra":"bad"}'
    )

    result = parse_model_response(raw)

    assert result.response is None
    assert result.failure is not None
    assert (
        result.failure.code
        == AttemptErrorCode.UNKNOWN_FIELD
    )


def test_unknown_tool_is_typed() -> None:
    raw = (
        '{"decision":"CALL_TOOL",'
        '"answerability":"NOT_APPLICABLE",'
        '"tool_call":{'
        '"tool":"delete_account",'
        '"arguments":{}}}'
    )

    result = parse_model_response(raw)

    assert result.response is None
    assert result.failure is not None
    assert (
        result.failure.code
        == AttemptErrorCode.INVALID_TOOL
    )


def test_invalid_tool_arguments_are_typed() -> None:
    raw = (
        '{"decision":"CALL_TOOL",'
        '"answerability":"NOT_APPLICABLE",'
        '"tool_call":{'
        '"tool":"get_record",'
        '"arguments":{"record_id":""}}}'
    )

    result = parse_model_response(raw)

    assert result.response is None
    assert result.failure is not None
    assert (
        result.failure.code
        == AttemptErrorCode.ARGUMENT_INVALID
    )
