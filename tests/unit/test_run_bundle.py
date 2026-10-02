from __future__ import annotations

from pathlib import Path

from morena_r.contracts.evaluation import EvalInput
from morena_r.evaluation.dataset import (
    dataset_sha256,
    load_eval_cases,
)
from morena_r.evaluation.runner import (
    EvaluationRunConfig,
    EvaluationRunner,
)
from morena_r.reporting.run_bundle import (
    REQUIRED_BUNDLE_FILES,
    write_run_bundle,
)


ROOT = Path(__file__).resolve().parents[2]

PILOT_PATH = (
    ROOT
    / "data"
    / "pilot"
    / "g2_development_pilot_v1.json"
)


class DeterministicAdapter:
    def generate(
        self,
        eval_input: EvalInput,
    ) -> str:
        return (
            '{"decision":"RESPOND",'
            '"answerability":"SUPPORTED",'
            '"answer":"fixture"}'
        )


def test_reference_pilot_is_valid_and_unique() -> None:
    cases = load_eval_cases(
        PILOT_PATH
    )

    assert len(cases) == 8

    case_ids = [
        case.input.case_id
        for case in cases
    ]

    assert len(set(case_ids)) == 8


def test_dataset_hash_is_stable() -> None:
    first = load_eval_cases(
        PILOT_PATH
    )

    second = load_eval_cases(
        PILOT_PATH
    )

    assert (
        dataset_sha256(first)
        == dataset_sha256(second)
    )


def test_run_bundle_is_byte_deterministic(
    tmp_path: Path,
) -> None:
    cases = (
        load_eval_cases(
            PILOT_PATH
        )[:1]
    )

    dataset_hash = dataset_sha256(
        cases
    )

    result = EvaluationRunner(
        config=EvaluationRunConfig(
            run_id="deterministic-bundle-test",
            model_identity_hash="model-hash",
            dataset_hash=dataset_hash,
        ),
        adapter=DeterministicAdapter(),
    ).run(cases)

    first = tmp_path / "first"
    second = tmp_path / "second"

    first_receipt = write_run_bundle(
        output_dir=first,
        cases=cases,
        result=result,
        implementation_commit="commit-fixture",
        git_dirty=False,
        environment_lock_sha256="lock-fixture",
        bundle_kind="unit_test",
        timing_mode="synthetic_reference_zero",
    )

    second_receipt = write_run_bundle(
        output_dir=second,
        cases=cases,
        result=result,
        implementation_commit="commit-fixture",
        git_dirty=False,
        environment_lock_sha256="lock-fixture",
        bundle_kind="unit_test",
        timing_mode="synthetic_reference_zero",
    )

    assert first_receipt == second_receipt

    assert {
        path.name
        for path in first.iterdir()
        if path.is_file()
    } == set(REQUIRED_BUNDLE_FILES)

    for filename in REQUIRED_BUNDLE_FILES:
        assert (
            (first / filename).read_bytes()
            == (second / filename).read_bytes()
        )
