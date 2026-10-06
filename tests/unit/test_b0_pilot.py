from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from morena_r.contracts.evaluation import (
    AttemptObservation,
    AttemptTransportStatus,
    EvalInput,
)
from morena_r.evaluation.b0_pilot import (
    B0AttemptIntent,
    B0PilotJournal,
    B0UnresolvedAttemptError,
    rebuild_result_from_receipts,
    resolve_pilot_inputs,
    run_or_resume_case,
    run_or_resume_cases,
    write_pilot_bundle,
)
from morena_r.evaluation.b0_subprocess import (
    B0TransportDiagnostics,
)


ROOT = Path(__file__).resolve().parents[2]


class FakePilotAdapter:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self._observations: dict[
            str,
            AttemptObservation,
        ] = {}
        self._diagnostics: dict[
            str,
            B0TransportDiagnostics,
        ] = {}

    def generate(
        self,
        eval_input: EvalInput,
    ) -> str:
        self.calls.append(
            eval_input.case_id
        )

        self._observations[
            eval_input.case_id
        ] = AttemptObservation(
            case_id=eval_input.case_id,
            transport_status=(
                AttemptTransportStatus.COMPLETED
            ),
            started_at_utc=datetime(
                2026,
                1,
                1,
                tzinfo=timezone.utc,
            ),
            completed_at_utc=datetime(
                2026,
                1,
                1,
                0,
                0,
                1,
                tzinfo=timezone.utc,
            ),
            elapsed_seconds=1.0,
            worker_returncode=0,
            model_inference_seconds=0.5,
            input_token_count=10,
            output_token_count=1,
            stop_reason="eos",
            prompt_sha256=self._prompt_hash(
                eval_input.case_id
            ),
            input_ids=(2, 3),
            generated_ids=(49,),
            runtime_device="cuda:0",
            parameter_dtype="torch.bfloat16",
            attention_mode="sdpa",
            model_load_seconds=0.25,
        )

        self._diagnostics[
            eval_input.case_id
        ] = B0TransportDiagnostics(
            case_id=eval_input.case_id,
        )

        return (
            '{"decision":"RESPOND",'
            '"answerability":"SUPPORTED",'
            '"answer":"fixture"}'
        )

    def _prompt_hash(
        self,
        case_id: str,
    ) -> str:
        manifest, _ = resolve_pilot_inputs(
            root=ROOT,
            implementation_commit=(
                "test-commit"
            ),
        )

        return next(
            item.prompt_sha256
            for item in manifest.cases
            if item.case_id == case_id
        )

    def get_observation(
        self,
        case_id: str,
    ) -> AttemptObservation:
        return self._observations[
            case_id
        ]

    def get_diagnostics(
        self,
        case_id: str,
    ) -> B0TransportDiagnostics:
        return self._diagnostics[
            case_id
        ]


def resolved():
    return resolve_pilot_inputs(
        root=ROOT,
        implementation_commit=(
            "test-commit"
        ),
    )


def test_resolver_selects_exact_frozen_four_cases() -> None:
    manifest, cases = resolved()

    expected = (
        "ref-respond-supported",
        "ref-ask-clarification",
        "ref-refuse",
        "ref-tool-correct",
    )

    assert tuple(
        case.input.case_id
        for case in cases
    ) == expected

    assert tuple(
        item.case_id
        for item in manifest.cases
    ) == expected

    assert (
        manifest.runtime.profile_id
        == "kaggle-t4-bf16-sdpa-v1"
    )

    assert (
        manifest.runtime.device
        == "cuda:0"
    )


def test_manifest_identity_mismatch_rejects_resume(
    tmp_path: Path,
) -> None:
    first, _ = resolved()

    second, _ = resolve_pilot_inputs(
        root=ROOT,
        implementation_commit=(
            "different-commit"
        ),
    )

    journal = B0PilotJournal(
        tmp_path
    )

    journal.ensure_manifest(
        first
    )

    with pytest.raises(
        RuntimeError,
        match="does not match",
    ):
        journal.ensure_manifest(
            second
        )


def test_resume_reuses_terminal_receipts(
    tmp_path: Path,
) -> None:
    manifest, cases = resolved()

    journal = B0PilotJournal(
        tmp_path
    )

    journal.ensure_manifest(
        manifest
    )

    first_adapter = (
        FakePilotAdapter()
    )

    first_receipts = []

    for case, identity in zip(
        cases[:2],
        manifest.cases[:2],
        strict=True,
    ):
        first_receipts.append(
            run_or_resume_case(
                manifest=manifest,
                case=case,
                case_identity=identity,
                journal=journal,
                adapter=first_adapter,
            )
        )

    assert first_adapter.calls == [
        "ref-respond-supported",
        "ref-ask-clarification",
    ]

    second_adapter = (
        FakePilotAdapter()
    )

    receipts = run_or_resume_cases(
        manifest=manifest,
        cases=cases,
        journal=journal,
        adapter=second_adapter,
    )

    assert len(receipts) == 4

    assert second_adapter.calls == [
        "ref-refuse",
        "ref-tool-correct",
    ]

    assert (
        receipts[0]
        .attempt
        .attempt_id
        == first_receipts[0]
        .attempt
        .attempt_id
    )


def test_unresolved_start_intent_blocks_silent_rerun(
    tmp_path: Path,
) -> None:
    manifest, cases = resolved()

    journal = B0PilotJournal(
        tmp_path
    )

    journal.ensure_manifest(
        manifest
    )

    identity = manifest.cases[0]

    journal.write_intent(
        B0AttemptIntent(
            run_id=manifest.run_id,
            identity_sha256=(
                manifest.identity_sha256
            ),
            implementation_commit=(
                manifest.execution_commit
            ),
            protocol_sha256=(
                manifest.protocol_sha256
            ),
            case_id=identity.case_id,
            family_id=identity.family_id,
            case_sha256=(
                identity.case_sha256
            ),
            started_at_utc=datetime(
                2026,
                1,
                1,
                tzinfo=timezone.utc,
            ),
        )
    )

    adapter = FakePilotAdapter()

    with pytest.raises(
        B0UnresolvedAttemptError,
        match="Do not silently rerun",
    ):
        run_or_resume_case(
            manifest=manifest,
            case=cases[0],
            case_identity=identity,
            journal=journal,
            adapter=adapter,
        )

    assert adapter.calls == []


def test_final_bundle_rebuilds_without_model_calls(
    tmp_path: Path,
) -> None:
    manifest, cases = resolved()

    journal = B0PilotJournal(
        tmp_path
        / "journal-root"
    )

    journal.ensure_manifest(
        manifest
    )

    adapter = FakePilotAdapter()

    receipts = run_or_resume_cases(
        manifest=manifest,
        cases=cases,
        journal=journal,
        adapter=adapter,
    )

    result = rebuild_result_from_receipts(
        manifest=manifest,
        cases=cases,
        receipts=receipts,
    )

    assert result.summary.expected_cases == 4
    assert result.summary.attempted_cases == 4

    output = (
        tmp_path
        / "final"
    )

    checksum = write_pilot_bundle(
        output_dir=output,
        manifest=manifest,
        cases=cases,
        receipts=receipts,
        result=result,
    )

    assert len(checksum) == 64

    assert {
        path.name
        for path in output.iterdir()
    } == {
        "BASELINE_REPORT.md",
        "case_receipts.jsonl",
        "checksums.sha256",
        "failures.jsonl",
        "metrics.json",
        "run_manifest.json",
        "scores.jsonl",
        "timing.csv",
    }

    report = (
        output
        / "BASELINE_REPORT.md"
    ).read_text(
        encoding="utf-8",
    )

    assert (
        "not final G5 baseline"
        in report
    )

    assert (
        "does not claim model improvement"
        in report
    )

    with pytest.raises(
        FileExistsError,
        match="Refusing to overwrite",
    ):
        write_pilot_bundle(
            output_dir=output,
            manifest=manifest,
            cases=cases,
            receipts=receipts,
            result=result,
        )
