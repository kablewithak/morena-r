from __future__ import annotations

import csv
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

from morena_r.contracts.evaluation import EvalCase
from morena_r.evaluation.dataset import (
    case_sha256,
    dataset_sha256,
)
from morena_r.evaluation.runner import (
    EvaluationRunResult,
)
from morena_r.reporting.run_report import (
    build_run_report,
    canonical_report_json,
)


REQUIRED_BUNDLE_FILES = (
    "manifest.json",
    "resolved_config.json",
    "environment.json",
    "inputs_manifest.json",
    "outputs.jsonl",
    "scores.jsonl",
    "events.jsonl",
    "failures.jsonl",
    "metrics.json",
    "timing.csv",
    "checksums.sha256",
)


def _canonical_json(
    value: Any,
) -> str:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    )


def _write_json(
    path: Path,
    value: Any,
) -> None:
    path.write_text(
        _canonical_json(value),
        encoding="utf-8",
        newline="\n",
    )


def _write_jsonl(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    content = "".join(
        _canonical_json(row)
        for row in rows
    )

    path.write_text(
        content,
        encoding="utf-8",
        newline="\n",
    )


def _sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def write_run_bundle(
    *,
    output_dir: Path,
    cases: tuple[EvalCase, ...],
    result: EvaluationRunResult,
    implementation_commit: str,
    git_dirty: bool,
    environment_lock_sha256: str,
    bundle_kind: str,
    timing_mode: str,
) -> str:
    if output_dir.exists() and any(
        output_dir.iterdir()
    ):
        raise FileExistsError(
            f"Run bundle directory is not empty: {output_dir}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    observed_dataset_hash = dataset_sha256(
        cases
    )

    if (
        observed_dataset_hash
        != result.config.dataset_hash
    ):
        raise ValueError(
            "Run config dataset hash does not match cases."
        )

    manifest = {
        "schema_version": "1.0",
        "bundle_kind": bundle_kind,
        "run_id": result.config.run_id,
        "model_identity_hash": (
            result.config.model_identity_hash
        ),
        "dataset_hash": observed_dataset_hash,
        "scorer_version": (
            result.config.scorer_version
        ),
        "implementation_commit": (
            implementation_commit
        ),
        "git_dirty": git_dirty,
        "environment_lock_sha256": (
            environment_lock_sha256
        ),
        "required_files": list(
            REQUIRED_BUNDLE_FILES
        ),
    }

    resolved_config = {
        **result.config.model_dump(
            mode="json",
        ),
        "bundle_kind": bundle_kind,
        "timing_mode": timing_mode,
    }

    environment = {
        "python_version": (
            platform.python_version()
        ),
        "platform": platform.platform(),
        "implementation_commit": (
            implementation_commit
        ),
        "git_dirty": git_dirty,
        "environment_lock_sha256": (
            environment_lock_sha256
        ),
    }

    inputs_manifest = {
        "schema_version": "1.0",
        "dataset_hash": (
            observed_dataset_hash
        ),
        "case_count": len(cases),
        "cases": [
            {
                "case_id": case.input.case_id,
                "family_id": (
                    case.input.family_id
                ),
                "sha256": case_sha256(case),
            }
            for case in cases
        ],
    }

    _write_json(
        output_dir / "manifest.json",
        manifest,
    )

    _write_json(
        output_dir / "resolved_config.json",
        resolved_config,
    )

    _write_json(
        output_dir / "environment.json",
        environment,
    )

    _write_json(
        output_dir / "inputs_manifest.json",
        inputs_manifest,
    )

    _write_jsonl(
        output_dir / "outputs.jsonl",
        [
            attempt.model_dump(
                mode="json",
                exclude_none=True,
            )
            for attempt in result.attempts
        ],
    )

    scores = sorted(
        result.scores,
        key=lambda row: (
            row.case_id,
            row.metric_id,
        ),
    )

    _write_jsonl(
        output_dir / "scores.jsonl",
        [
            score.model_dump(
                mode="json",
                exclude_none=True,
            )
            for score in scores
        ],
    )

    case_by_id = {
        case.input.case_id: case
        for case in cases
    }

    events: list[dict[str, Any]] = []

    for sequence, attempt in enumerate(
        result.attempts,
        start=1,
    ):
        case = case_by_id[
            attempt.case_id
        ]

        events.append(
            {
                "sequence": sequence,
                "case_id": attempt.case_id,
                "actor": (
                    "scripted_reference_adapter"
                ),
                "action": "model_attempt",
                "status": attempt.status.value,
                "elapsed_seconds": 0.0,
                "artifact_ref": (
                    f"attempt:{attempt.attempt_id}"
                ),
                "event_time_utc": (
                    case.input
                    .evaluation_time_utc
                    .isoformat()
                ),
                "time_source": (
                    "frozen_fixture_evaluation_time"
                ),
            }
        )

    _write_jsonl(
        output_dir / "events.jsonl",
        events,
    )

    scores_by_case: dict[
        str,
        list,
    ] = {}

    for score in result.scores:
        scores_by_case.setdefault(
            score.case_id,
            [],
        ).append(score)

    failures: list[dict[str, Any]] = []

    for attempt in result.attempts:
        labels = sorted(
            {
                label.value
                for score in scores_by_case.get(
                    attempt.case_id,
                    [],
                )
                for label
                in score.failure_labels
            }
        )

        if (
            attempt.status.value
            != "completed"
            or labels
        ):
            failures.append(
                {
                    "attempt_id": (
                        attempt.attempt_id
                    ),
                    "case_id": (
                        attempt.case_id
                    ),
                    "status": (
                        attempt.status.value
                    ),
                    "parse_error": (
                        attempt.parse_failure.code.value
                        if attempt.parse_failure
                        is not None
                        else None
                    ),
                    "runtime_error": (
                        attempt.runtime_error_code.value
                        if attempt.runtime_error_code
                        is not None
                        else None
                    ),
                    "failure_labels": labels,
                }
            )

    _write_jsonl(
        output_dir / "failures.jsonl",
        failures,
    )

    report = build_run_report(result)

    (output_dir / "metrics.json").write_text(
        canonical_report_json(report),
        encoding="utf-8",
        newline="\n",
    )

    with (
        output_dir / "timing.csv"
    ).open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.writer(
            handle,
            lineterminator="\n",
        )

        writer.writerow(
            [
                "case_id",
                "attempt_status",
                "elapsed_seconds",
                "timing_mode",
            ]
        )

        for attempt in result.attempts:
            writer.writerow(
                [
                    attempt.case_id,
                    attempt.status.value,
                    "0.000000",
                    timing_mode,
                ]
            )

    checksum_targets = sorted(
        path
        for path in output_dir.iterdir()
        if (
            path.is_file()
            and path.name
            != "checksums.sha256"
        )
    )

    checksum_lines = [
        (
            f"{_sha256_file(path)}"
            f"  {path.name}\n"
        )
        for path in checksum_targets
    ]

    checksum_path = (
        output_dir
        / "checksums.sha256"
    )

    checksum_path.write_text(
        "".join(checksum_lines),
        encoding="utf-8",
        newline="\n",
    )

    actual_files = {
        path.name
        for path in output_dir.iterdir()
        if path.is_file()
    }

    if actual_files != set(
        REQUIRED_BUNDLE_FILES
    ):
        raise RuntimeError(
            "Run bundle file set does not match contract."
        )

    return _sha256_file(
        checksum_path
    )
