from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import TypeAdapter

from morena_r.contracts.evaluation import EvalCase


EVAL_CASES_ADAPTER = TypeAdapter(
    tuple[EvalCase, ...]
)


def load_eval_cases(
    path: Path,
) -> tuple[EvalCase, ...]:
    payload = json.loads(
        path.read_text(
            encoding="utf-8-sig",
        )
    )

    cases = EVAL_CASES_ADAPTER.validate_python(
        payload
    )

    case_ids = tuple(
        case.input.case_id
        for case in cases
    )

    if len(set(case_ids)) != len(case_ids):
        raise ValueError(
            "Dataset contains duplicate case IDs."
        )

    return cases


def canonical_dataset_bytes(
    cases: tuple[EvalCase, ...],
) -> bytes:
    payload = [
        case.model_dump(
            mode="json",
            exclude_none=True,
        )
        for case in cases
    ]

    serialized = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    )

    return serialized.encode("utf-8")


def dataset_sha256(
    cases: tuple[EvalCase, ...],
) -> str:
    return hashlib.sha256(
        canonical_dataset_bytes(cases)
    ).hexdigest()


def case_sha256(
    case: EvalCase,
) -> str:
    serialized = (
        json.dumps(
            case.model_dump(
                mode="json",
                exclude_none=True,
            ),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    )

    return hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()
