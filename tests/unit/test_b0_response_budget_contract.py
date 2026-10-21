"""Read-only, pinned-tokenizer checks for the frozen P01 decision boundary.

No model weights are loaded. The historical P01 artifacts are never modified.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from tokenizers import Tokenizer

from morena_r.evaluation.dataset import load_eval_cases
from morena_r.evaluation.parser import parse_model_response
from morena_r.models.native_morena import REVISION, TOKENIZER_SHA256
from morena_r.prompting.b0 import render_b0_prompt

ROOT = Path(__file__).resolve().parents[2]
TOKENIZER_PATH = (
    ROOT / ".cache" / "upstream" / "morena-1.5b-instruct" / REVISION / "tokenizer.json"
)
DATASET_PATH = ROOT / "data" / "pilot" / "g2_development_pilot_v1.json"
EVIDENCE_DIR = ROOT / "reports" / "b0" / "initial_pilot_v1" / "final"
MAX_NEW_TOKENS = 48  # Frozen P01 protocol; not a proposed new B1 budget.


@pytest.fixture(scope="module")
def pinned_tokenizer() -> Tokenizer:
    actual_sha256 = hashlib.sha256(TOKENIZER_PATH.read_bytes()).hexdigest()
    assert actual_sha256 == TOKENIZER_SHA256, "Pinned tokenizer identity mismatch"
    return Tokenizer.from_file(str(TOKENIZER_PATH))


def _serialized_response(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


MINIMAL_RESPONSES = (
    pytest.param(
        "RESPOND",
        {"decision": "RESPOND", "answerability": "SUPPORTED", "answer": "Harare"},
        id="respond",
    ),
    pytest.param(
        "ASK_CLARIFICATION",
        {
            "decision": "ASK_CLARIFICATION",
            "answerability": "INSUFFICIENT_CONTEXT",
            "question": "Which subject ID?",
        },
        id="ask-clarification",
    ),
    pytest.param(
        "REFUSE",
        {
            "decision": "REFUSE",
            "answerability": "NOT_APPLICABLE",
            "reason": "This action is not permitted.",
        },
        id="refuse",
    ),
    pytest.param(
        "CALL_TOOL",
        {
            "decision": "CALL_TOOL",
            "answerability": "NOT_APPLICABLE",
            "tool_call": {"tool": "get_record", "arguments": {"record_id": "rec-001"}},
        },
        id="call-tool",
    ),
)


@pytest.mark.parametrize("decision,payload", MINIMAL_RESPONSES)
def test_minimal_valid_response_budget_is_measured(
    pinned_tokenizer: Tokenizer, decision: str, payload: dict[str, Any]
) -> None:
    serialized = _serialized_response(payload)
    result = parse_model_response(serialized)
    assert result.response is not None, f"Invalid contract for {decision}: {result.failure}"
    assert result.failure is None

    token_count = len(pinned_tokenizer.encode(serialized, add_special_tokens=False).ids)
    print(
        "S1_G3_MINIMAL_RESPONSE="
        + json.dumps(
            {
                "decision": decision,
                "serialized_response": serialized,
                "pinned_token_count": token_count,
                "frozen_max_new_tokens": MAX_NEW_TOKENS,
                "fits_frozen_budget": token_count <= MAX_NEW_TOKENS,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    # Fit is an observed diagnostic, not an assertion that the frozen P01 budget is adequate.


def test_grounded_response_length_is_reported_without_quality_claim(
    pinned_tokenizer: Tokenizer,
) -> None:
    # Richer serialization is diagnostic, not a promise of 48-token multilingual sufficiency.
    payload = {
        "decision": "RESPOND",
        "answerability": "SUPPORTED",
        "answer": "Harare iguta guru reZimbabwe.",
        "claims": [
            {
                "claim": "Harare is the capital of Zimbabwe.",
                "evidence_refs": ["doc-zim-capital"],
            }
        ],
        "citations": ["doc-zim-capital"],
    }
    serialized = _serialized_response(payload)
    result = parse_model_response(serialized)
    assert result.response is not None
    assert result.failure is None

    token_count = len(pinned_tokenizer.encode(serialized, add_special_tokens=False).ids)
    print(
        "S1_G3_GROUNDED_RESPONSE="
        + json.dumps(
            {
                "serialized_response": serialized,
                "pinned_token_count": token_count,
                "frozen_max_new_tokens": MAX_NEW_TOKENS,
                "fits_frozen_budget": token_count <= MAX_NEW_TOKENS,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def test_historical_prompt_input_token_ids_match_frozen_receipts(
    pinned_tokenizer: Tokenizer,
) -> None:
    manifest = json.loads((EVIDENCE_DIR / "run_manifest.json").read_text(encoding="utf-8"))
    receipts = [
        json.loads(line)
        for line in (EVIDENCE_DIR / "case_receipts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    by_case = {case.input.case_id: case for case in load_eval_cases(DATASET_PATH)}
    by_receipt = {receipt["attempt"]["case_id"]: receipt for receipt in receipts}
    expected_ids = {case["case_id"] for case in manifest["cases"]}

    assert len(expected_ids) == 4
    assert set(by_receipt) == expected_ids

    for identity in manifest["cases"]:
        case_id = identity["case_id"]
        prompt = render_b0_prompt(by_case[case_id].input)
        actual_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        assert actual_sha256 == identity["prompt_sha256"]

        # Upstream and native MORENA runtime explicitly prefix the chat prompt with EOS.
        input_ids = (2, *pinned_tokenizer.encode(prompt, add_special_tokens=False).ids)
        recorded_ids = tuple(by_receipt[case_id]["observation"]["input_ids"])
        assert input_ids == recorded_ids, f"Historical token boundary differs: {case_id}"
        print(f"S1_G3_HISTORICAL_INPUT={case_id} tokens={len(input_ids)} match=true")
