from __future__ import annotations

import json

from morena_r.contracts.actions import ToolResult
from morena_r.contracts.evaluation import EvalInput


B0_PROMPT_VERSION = "b0-minimal-json-v1"

USER_MARKER = "<reserved_0>"
ASSISTANT_MARKER = "<reserved_1>"


def render_b0_prompt(
    eval_input: EvalInput,
    *,
    tool_results: tuple[ToolResult, ...] = (),
) -> str:
    message_lines = [
        f"{message.role}: {message.content}"
        for message in eval_input.messages
    ]

    evidence_lines = [
        (
            f"{document.document_id}: "
            f"{document.content}"
        )
        for document in eval_input.evidence
    ]

    available_tools = [
        tool.value
        for tool in eval_input.available_tools
    ]

    permitted_tools = [
        tool.value
        for tool in eval_input.permitted_tools
    ]

    tool_result_lines = [
        json.dumps(
            result.model_dump(
                mode="json",
                exclude_none=True,
            ),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        for result in tool_results
    ]

    body = "\n".join(
        [
            "Return exactly one JSON object and no other text.",
            (
                "decision must be RESPOND, CALL_TOOL, "
                "ASK_CLARIFICATION, or REFUSE."
            ),
            (
                "answerability must be SUPPORTED, UNSUPPORTED, "
                "CONFLICTING, INSUFFICIENT_CONTEXT, or NOT_APPLICABLE."
            ),
            (
                "RESPOND requires answer. "
                "CALL_TOOL requires tool_call with tool and arguments. "
                "ASK_CLARIFICATION requires question. "
                "REFUSE requires reason."
            ),
            "",
            "MESSAGES",
            *message_lines,
            "",
            "EVIDENCE",
            *(evidence_lines or ["none"]),
            "",
            (
                "AVAILABLE_TOOLS="
                + json.dumps(
                    available_tools,
                    ensure_ascii=False,
                )
            ),
            (
                "PERMITTED_TOOLS="
                + json.dumps(
                    permitted_tools,
                    ensure_ascii=False,
                )
            ),
            "",
            "TOOL_RESULTS",
            *(tool_result_lines or ["none"]),
        ]
    )

    return (
        f"{USER_MARKER}\n"
        f"{body}\n"
        f"{ASSISTANT_MARKER}\n"
    )
