from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

sys.path.insert(
    0,
    str(SRC),
)


from morena_r.evaluation.b0_pilot import (  # noqa: E402
    B0PilotJournal,
    make_subprocess_adapter,
    rebuild_result_from_receipts,
    resolve_pilot_inputs,
    run_or_resume_cases,
    validate_live_runtime,
    write_pilot_bundle,
)


WORKER_PATH = (
    ROOT
    / "scripts"
    / "b0_case_worker.py"
)

RUN_ROOT = (
    ROOT
    / "reports"
    / "b0"
    / "initial_pilot_v1"
)

FINAL_BUNDLE = (
    RUN_ROOT
    / "final"
)


def git_output(
    *args: str,
) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=ROOT,
        text=True,
    ).strip()


git_status = git_output(
    "status",
    "--porcelain",
)

if git_status:
    raise RuntimeError(
        "B0 scientific execution requires a clean Git worktree."
    )


implementation_commit = git_output(
    "rev-parse",
    "HEAD",
)


manifest, cases = resolve_pilot_inputs(
    root=ROOT,
    implementation_commit=(
        implementation_commit
    ),
)


validate_live_runtime(
    manifest
)


journal = B0PilotJournal(
    RUN_ROOT
)

journal.ensure_manifest(
    manifest
)


adapter = make_subprocess_adapter(
    root=ROOT,
    worker_path=WORKER_PATH,
    manifest=manifest,
    python_executable=Path(
        sys.executable
    ),
)


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


if result.summary.expected_cases != 4:
    raise RuntimeError(
        "B0 pilot expected exactly four cases."
    )

if result.summary.attempted_cases != 4:
    raise RuntimeError(
        "B0 pilot did not account for all four cases."
    )


bundle_checksum = write_pilot_bundle(
    output_dir=FINAL_BUNDLE,
    manifest=manifest,
    cases=cases,
    receipts=receipts,
    result=result,
)


print("B0_INITIAL_PILOT=COMPLETE")
print(
    f"RUN_ID={manifest.run_id}"
)
print(
    "EXECUTION_COMMIT="
    f"{manifest.execution_commit}"
)
print(
    "RUNTIME_PROFILE="
    f"{manifest.runtime.profile_id}"
)

for receipt in receipts:
    attempt = receipt.attempt

    observed_decision = (
        attempt
        .parsed_response
        .decision
        .value
        if (
            attempt.parsed_response
            is not None
        )
        else "NONE"
    )

    print(
        "CASE="
        f"{attempt.case_id} "
        "STATUS="
        f"{attempt.status.value} "
        "DECISION="
        f"{observed_decision}"
    )

print(
    "COMPLETED="
    f"{result.summary.completed}"
)
print(
    "INVALID_RESPONSES="
    f"{result.summary.invalid_responses}"
)
print(
    "TIMEOUTS="
    f"{result.summary.timeouts}"
)
print(
    "EXECUTION_ERRORS="
    f"{result.summary.execution_errors}"
)
print(
    "FINAL_BUNDLE_CHECKSUM_SHA256="
    f"{bundle_checksum}"
)
print(
    "FINAL_BUNDLE="
    f"{FINAL_BUNDLE.relative_to(ROOT)}"
)
print("BASELINE_REPRODUCED=FALSE")
print("IMPROVEMENT_CLAIMED=FALSE")
