from __future__ import annotations

from datetime import datetime, timezone

from morena_r.contracts.actions import (
    ToolName,
)
from morena_r.contracts.evaluation import (
    EvalInput,
    EvalMessage,
    EvidenceDocument,
)
from morena_r.models.native_morena import (
    GenerationConfig,
)
from morena_r.prompting.b0 import (
    B0_PROMPT_VERSION,
    render_b0_prompt,
)


def make_input() -> EvalInput:
    return EvalInput(
        case_id="prompt-case",
        family_id="prompt-family",
        languages=("sn",),
        messages=(
            EvalMessage(
                role="user",
                content=(
                    "Ndeipi guta guru reZimbabwe?"
                ),
            ),
        ),
        evidence=(
            EvidenceDocument(
                document_id=(
                    "doc-zim-capital"
                ),
                version="1",
                content="Capital: Harare.",
                observed_at_utc=datetime(
                    2026,
                    1,
                    1,
                    tzinfo=timezone.utc,
                ),
            ),
        ),
        available_tools=(
            ToolName.GET_RECORD,
        ),
        permitted_tools=(),
        evaluation_time_utc=datetime(
            2026,
            1,
            15,
            tzinfo=timezone.utc,
        ),
    )


def test_b0_prompt_uses_native_chat_markers() -> None:
    prompt = render_b0_prompt(
        make_input()
    )

    assert prompt.startswith(
        "<reserved_0>\n"
    )

    assert prompt.endswith(
        "<reserved_1>\n"
    )


def test_b0_prompt_contains_case_not_gold() -> None:
    prompt = render_b0_prompt(
        make_input()
    )

    assert (
        "Ndeipi guta guru reZimbabwe?"
        in prompt
    )

    assert "Capital: Harare." in prompt

    assert (
        'AVAILABLE_TOOLS=["get_record"]'
        in prompt
    )

    assert (
        "expected_decision"
        not in prompt
    )

    assert (
        "rubric"
        not in prompt.lower()
    )


def test_b0_prompt_version_is_frozen() -> None:
    assert (
        B0_PROMPT_VERSION
        == "b0-minimal-json-v1"
    )


def test_generation_config_is_frozen_for_probe() -> None:
    config = GenerationConfig()

    assert config.seed == 1234
    assert config.temperature == 0.7
    assert config.top_p == 0.9
    assert config.max_new_tokens == 48
