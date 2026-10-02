from __future__ import annotations

import json

from pydantic import ValidationError

from morena_r.contracts.actions import MODEL_RESPONSE_ADAPTER
from morena_r.contracts.evaluation import (
    AttemptErrorCode,
    ParseFailure,
    ParseOutcome,
)


def _classify_validation_error(
    error: ValidationError,
) -> AttemptErrorCode:
    errors = error.errors()

    for item in errors:
        if item.get("type") == "extra_forbidden":
            return AttemptErrorCode.UNKNOWN_FIELD

    for item in errors:
        location = tuple(
            str(part)
            for part in item.get("loc", ())
        )

        if (
            item.get("type") == "union_tag_invalid"
            and "tool_call" in location
        ):
            return AttemptErrorCode.INVALID_TOOL

    for item in errors:
        location = tuple(
            str(part)
            for part in item.get("loc", ())
        )

        if "arguments" in location:
            return AttemptErrorCode.ARGUMENT_INVALID

    return AttemptErrorCode.PARSE_ERROR


def parse_model_response(
    raw_output: str,
) -> ParseOutcome:
    if not raw_output.strip():
        return ParseOutcome(
            raw_output=raw_output,
            failure=ParseFailure(
                code=AttemptErrorCode.PARSE_ERROR,
                message="Model output was empty.",
            ),
        )

    try:
        payload = json.loads(raw_output)
    except json.JSONDecodeError as error:
        return ParseOutcome(
            raw_output=raw_output,
            failure=ParseFailure(
                code=AttemptErrorCode.PARSE_ERROR,
                message=(
                    "Model output is not valid JSON: "
                    f"{error.msg}"
                ),
            ),
        )

    if not isinstance(payload, dict):
        return ParseOutcome(
            raw_output=raw_output,
            failure=ParseFailure(
                code=AttemptErrorCode.PARSE_ERROR,
                message=(
                    "Model output JSON must be an object."
                ),
            ),
        )

    try:
        response = MODEL_RESPONSE_ADAPTER.validate_python(
            payload
        )
    except ValidationError as error:
        return ParseOutcome(
            raw_output=raw_output,
            failure=ParseFailure(
                code=_classify_validation_error(error),
                message="Model response failed schema validation.",
            ),
        )

    return ParseOutcome(
        raw_output=raw_output,
        response=response,
    )
