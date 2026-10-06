from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol

import torch
from pydantic import Field

from morena_r.contracts.actions import (
    NonEmptyStr,
    StrictContract,
)
from morena_r.contracts.evaluation import (
    EvalCase,
)
from morena_r.evaluation.b0_subprocess import (
    B0CaseReceipt,
    B0SubprocessAdapter,
    B0TransportDiagnostics,
)
from morena_r.evaluation.dataset import (
    case_sha256,
    dataset_sha256,
    load_eval_cases,
)
from morena_r.evaluation.runner import (
    EvaluationRunConfig,
    EvaluationRunResult,
    EvaluationRunner,
)
from morena_r.models.native_morena import (
    ATTENTION_MODE,
    REPOSITORY,
    REVISION,
    GenerationConfig,
)
from morena_r.prompting.b0 import (
    B0_PROMPT_VERSION,
    render_b0_prompt,
)
from morena_r.reporting.run_report import (
    build_run_report,
    canonical_report_json,
)


class B0PilotGeneration(StrictContract):
    seed: int
    temperature: float = Field(gt=0)
    top_p: float = Field(gt=0, le=1)
    max_new_tokens: int = Field(gt=0)
    wall_clock_timeout_seconds: float = Field(gt=0)


class B0PilotRuntimeIdentity(StrictContract):
    profile_id: NonEmptyStr
    parent_runtime: NonEmptyStr

    model_repository: NonEmptyStr
    model_revision: NonEmptyStr
    model_identity_sha256: NonEmptyStr
    weight_sha256: NonEmptyStr

    kq0_decision_sha256: NonEmptyStr
    kq0_checksums_v2_sha256: NonEmptyStr
    environment_lock_sha256: NonEmptyStr

    python_version: NonEmptyStr
    torch_version: NonEmptyStr
    device: NonEmptyStr
    accelerator: NonEmptyStr
    cuda_visible_devices: NonEmptyStr
    parameter_dtype: NonEmptyStr
    attention_mode: NonEmptyStr

    autocast_used: bool
    precision_conversion_used: bool
    multi_gpu_execution_used: bool
    bf16_without_emulation: bool
    native_hardware_bf16_claimed: bool


class B0PilotCaseIdentity(StrictContract):
    case_id: NonEmptyStr
    family_id: NonEmptyStr
    case_sha256: NonEmptyStr
    prompt_sha256: NonEmptyStr


class B0PilotRunManifest(StrictContract):
    schema_version: Literal["1.0"] = "1.0"

    run_id: NonEmptyStr
    identity_sha256: NonEmptyStr

    protocol_id: NonEmptyStr
    protocol_path: NonEmptyStr
    protocol_sha256: NonEmptyStr

    source_dataset_path: NonEmptyStr
    source_dataset_sha256: NonEmptyStr
    selected_dataset_sha256: NonEmptyStr

    execution_commit: NonEmptyStr

    prompt_version: NonEmptyStr
    generation: B0PilotGeneration
    runtime: B0PilotRuntimeIdentity

    cases: tuple[B0PilotCaseIdentity, ...] = Field(
        min_length=1,
    )


class B0AttemptIntent(StrictContract):
    schema_version: Literal["1.0"] = "1.0"

    run_id: NonEmptyStr
    identity_sha256: NonEmptyStr
    implementation_commit: NonEmptyStr
    protocol_sha256: NonEmptyStr

    case_id: NonEmptyStr
    family_id: NonEmptyStr
    case_sha256: NonEmptyStr

    started_at_utc: datetime


class B0UnresolvedAttemptError(RuntimeError):
    pass


class B0PilotAdapter(Protocol):
    def generate(
        self,
        eval_input: Any,
    ) -> str:
        ...

    def get_observation(
        self,
        case_id: str,
    ) -> Any:
        ...

    def get_diagnostics(
        self,
        case_id: str,
    ) -> B0TransportDiagnostics:
        ...


class _ForbiddenAdapter:
    def generate(
        self,
        eval_input: Any,
    ) -> str:
        raise AssertionError(
            "Receipt reconstruction must not execute the model."
        )


def _canonical_json_bytes(
    value: Any,
) -> bytes:
    if hasattr(value, "model_dump"):
        value = value.model_dump(
            mode="json",
            exclude_none=True,
        )

    serialized = (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    )

    return serialized.encode("utf-8")


def _sha256_bytes(
    value: bytes,
) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(
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


def _load_json(
    path: Path,
) -> dict[str, Any]:
    payload = json.loads(
        path.read_text(
            encoding="utf-8-sig",
        )
    )

    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected JSON object: {path}"
        )

    return payload


def _atomic_write_json(
    path: Path,
    value: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing artifact: {path}"
        )

    temporary = path.with_name(
        f".{path.name}.tmp"
    )

    if temporary.exists():
        raise FileExistsError(
            f"Unresolved temporary artifact exists: {temporary}"
        )

    payload = _canonical_json_bytes(
        value
    )

    temporary.write_bytes(
        payload
    )

    json.loads(
        temporary.read_text(
            encoding="utf-8",
        )
    )

    os.replace(
        temporary,
        path,
    )


def _verify_checksum_manifest(
    *,
    root: Path,
    manifest_path: Path,
) -> None:
    for line in manifest_path.read_text(
        encoding="utf-8",
    ).splitlines():
        if not line.strip():
            continue

        expected, relative_path = line.split(
            maxsplit=1,
        )

        artifact = (
            root
            / relative_path.strip()
        )

        if not artifact.is_file():
            raise FileNotFoundError(
                f"KQ0 checksum target missing: {artifact}"
            )

        actual = sha256_file(
            artifact
        )

        if actual != expected:
            raise RuntimeError(
                "KQ0 evidence checksum mismatch: "
                f"{relative_path.strip()}"
            )


def _require_equal(
    *,
    label: str,
    actual: Any,
    expected: Any,
) -> None:
    if actual != expected:
        raise RuntimeError(
            f"{label} mismatch: expected={expected!r}, actual={actual!r}"
        )


def _select_cases(
    *,
    cases: tuple[EvalCase, ...],
    case_ids: tuple[str, ...],
) -> tuple[EvalCase, ...]:
    if len(set(case_ids)) != len(case_ids):
        raise ValueError(
            "Frozen pilot case_ids contain duplicates."
        )

    by_id = {
        case.input.case_id: case
        for case in cases
    }

    missing = [
        case_id
        for case_id in case_ids
        if case_id not in by_id
    ]

    if missing:
        raise ValueError(
            "Frozen pilot references missing cases: "
            + ", ".join(missing)
        )

    return tuple(
        by_id[case_id]
        for case_id in case_ids
    )


def resolve_pilot_inputs(
    *,
    root: Path,
    implementation_commit: str,
) -> tuple[
    B0PilotRunManifest,
    tuple[EvalCase, ...],
]:
    protocol_path = (
        root
        / "configs"
        / "b0_initial_decision_pilot_v1.json"
    )

    dataset_path = (
        root
        / "data"
        / "pilot"
        / "g2_development_pilot_v1.json"
    )

    runtime_profile_path = (
        root
        / "reports"
        / "kaggle"
        / "kq0_runtime_profile_v1.json"
    )

    decision_path = (
        root
        / "reports"
        / "kaggle"
        / "kq0_decision_v1.json"
    )

    checksums_v2_path = (
        root
        / "reports"
        / "kaggle"
        / "checksums_v2.sha256"
    )

    model_identity_path = (
        root
        / "reports"
        / "kaggle"
        / "kq0_model_identity_v1.json"
    )

    environment_lock_path = (
        root
        / "requirements"
        / "kaggle-kq0-environment.lock.txt"
    )

    required_paths = (
        protocol_path,
        dataset_path,
        runtime_profile_path,
        decision_path,
        checksums_v2_path,
        model_identity_path,
        environment_lock_path,
    )

    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(
                f"Required B0 pilot input is missing: {path}"
            )

    _verify_checksum_manifest(
        root=root,
        manifest_path=checksums_v2_path,
    )

    protocol = _load_json(
        protocol_path
    )

    runtime_profile = _load_json(
        runtime_profile_path
    )

    decision = _load_json(
        decision_path
    )

    protocol_sha256 = sha256_file(
        protocol_path
    )

    _require_equal(
        label="KQ0 parent protocol hash",
        actual=runtime_profile[
            "parent_protocol_sha256"
        ],
        expected=protocol_sha256,
    )

    _require_equal(
        label="KQ0 parent protocol id",
        actual=runtime_profile[
            "parent_protocol_id"
        ],
        expected=protocol[
            "protocol_id"
        ],
    )

    _require_equal(
        label="KQ0 decision",
        actual=decision["decision"],
        expected="PASS",
    )

    _require_equal(
        label="KQ0 pass flag",
        actual=decision["kq0_pass"],
        expected=True,
    )

    _require_equal(
        label="KQ0 outcome",
        actual=decision["outcome"],
        expected="A_EXACT_PROFILE_QUALIFIES",
    )

    _require_equal(
        label="KQ0 runtime profile",
        actual=decision[
            "runtime_profile_id"
        ],
        expected=runtime_profile[
            "profile_id"
        ],
    )

    _require_equal(
        label="Protocol model repository",
        actual=protocol["model"][
            "repository"
        ],
        expected=REPOSITORY,
    )

    _require_equal(
        label="Protocol model revision",
        actual=protocol["model"][
            "revision"
        ],
        expected=REVISION,
    )

    _require_equal(
        label="KQ0 model repository",
        actual=decision[
            "model_identity"
        ]["repository"],
        expected=REPOSITORY,
    )

    _require_equal(
        label="KQ0 model revision",
        actual=decision[
            "model_identity"
        ]["revision"],
        expected=REVISION,
    )

    _require_equal(
        label="Protocol prompt version",
        actual=protocol[
            "prompt"
        ]["version"],
        expected=B0_PROMPT_VERSION,
    )

    _require_equal(
        label="Protocol attention mode",
        actual=protocol[
            "model"
        ]["attention_mode"],
        expected=ATTENTION_MODE,
    )

    _require_equal(
        label="KQ0 attention mode",
        actual=decision[
            "runtime"
        ]["attention_mode"],
        expected=ATTENTION_MODE,
    )

    frozen_generation = (
        runtime_profile[
            "frozen_contract_preserved"
        ]["generation"]
    )

    _require_equal(
        label="KQ0 generation contract",
        actual=frozen_generation,
        expected=protocol["generation"],
    )

    runtime_override = (
        runtime_profile[
            "runtime_override"
        ]
    )

    _require_equal(
        label="KQ0 runtime device",
        actual=runtime_override[
            "device"
        ],
        expected=decision[
            "runtime"
        ]["device"],
    )

    _require_equal(
        label="KQ0 runtime dtype",
        actual=runtime_override[
            "checkpoint_parameter_dtype"
        ],
        expected=decision[
            "runtime"
        ][
            "checkpoint_parameter_dtype"
        ],
    )

    _require_equal(
        label="KQ0 runtime torch version",
        actual=runtime_override[
            "torch_version"
        ],
        expected=decision[
            "runtime"
        ]["torch_version"],
    )

    all_cases = load_eval_cases(
        dataset_path
    )

    frozen_case_ids = tuple(
        str(case_id)
        for case_id in protocol[
            "case_ids"
        ]
    )

    if len(frozen_case_ids) != 4:
        raise ValueError(
            "Initial B0 pilot must contain exactly four frozen cases."
        )

    selected_cases = _select_cases(
        cases=all_cases,
        case_ids=frozen_case_ids,
    )

    case_identities = tuple(
        B0PilotCaseIdentity(
            case_id=case.input.case_id,
            family_id=case.input.family_id,
            case_sha256=case_sha256(
                case
            ),
            prompt_sha256=_sha256_bytes(
                render_b0_prompt(
                    case.input
                ).encode("utf-8")
            ),
        )
        for case in selected_cases
    )

    environment_lock_sha256 = (
        sha256_file(
            environment_lock_path
        )
    )

    _require_equal(
        label="KQ0 environment lock",
        actual=runtime_profile[
            "provenance"
        ][
            "environment_lock_sha256"
        ],
        expected=environment_lock_sha256,
    )

    generation = B0PilotGeneration(
        **protocol["generation"]
    )

    runtime = B0PilotRuntimeIdentity(
        profile_id=decision[
            "runtime_profile_id"
        ],
        parent_runtime=runtime_profile[
            "parent_runtime"
        ],
        model_repository=REPOSITORY,
        model_revision=REVISION,
        model_identity_sha256=sha256_file(
            model_identity_path
        ),
        weight_sha256=decision[
            "model_identity"
        ]["weight_sha256"],
        kq0_decision_sha256=sha256_file(
            decision_path
        ),
        kq0_checksums_v2_sha256=sha256_file(
            checksums_v2_path
        ),
        environment_lock_sha256=(
            environment_lock_sha256
        ),
        python_version=decision[
            "runtime"
        ]["python_version"],
        torch_version=decision[
            "runtime"
        ]["torch_version"],
        device=decision[
            "runtime"
        ]["device"],
        accelerator=decision[
            "runtime"
        ]["accelerator"],
        cuda_visible_devices=decision[
            "runtime"
        ]["cuda_visible_devices"],
        parameter_dtype=decision[
            "runtime"
        ][
            "checkpoint_parameter_dtype"
        ],
        attention_mode=decision[
            "runtime"
        ]["attention_mode"],
        autocast_used=decision[
            "runtime"
        ]["autocast_used"],
        precision_conversion_used=decision[
            "runtime"
        ][
            "precision_conversion_used"
        ],
        multi_gpu_execution_used=decision[
            "runtime"
        ][
            "multi_gpu_execution_used"
        ],
        bf16_without_emulation=decision[
            "runtime"
        ]["bf16_without_emulation"],
        native_hardware_bf16_claimed=decision[
            "runtime"
        ][
            "native_hardware_bf16_claimed"
        ],
    )

    identity_payload = {
        "protocol_id": protocol[
            "protocol_id"
        ],
        "protocol_sha256": (
            protocol_sha256
        ),
        "selected_dataset_sha256": (
            dataset_sha256(
                selected_cases
            )
        ),
        "execution_commit": (
            implementation_commit
        ),
        "prompt_version": (
            B0_PROMPT_VERSION
        ),
        "generation": generation.model_dump(
            mode="json"
        ),
        "runtime": runtime.model_dump(
            mode="json"
        ),
        "cases": [
            item.model_dump(
                mode="json"
            )
            for item in case_identities
        ],
    }

    identity_sha256 = _sha256_bytes(
        _canonical_json_bytes(
            identity_payload
        )
    )

    run_id = (
        f"{protocol['protocol_id']}-"
        f"{identity_sha256[:16]}"
    )

    manifest = B0PilotRunManifest(
        run_id=run_id,
        identity_sha256=(
            identity_sha256
        ),
        protocol_id=protocol[
            "protocol_id"
        ],
        protocol_path=(
            protocol_path
            .relative_to(root)
            .as_posix()
        ),
        protocol_sha256=(
            protocol_sha256
        ),
        source_dataset_path=(
            dataset_path
            .relative_to(root)
            .as_posix()
        ),
        source_dataset_sha256=(
            sha256_file(
                dataset_path
            )
        ),
        selected_dataset_sha256=(
            dataset_sha256(
                selected_cases
            )
        ),
        execution_commit=(
            implementation_commit
        ),
        prompt_version=(
            B0_PROMPT_VERSION
        ),
        generation=generation,
        runtime=runtime,
        cases=case_identities,
    )

    return (
        manifest,
        selected_cases,
    )


def validate_live_runtime(
    manifest: B0PilotRunManifest,
) -> None:
    runtime = manifest.runtime

    _require_equal(
        label="Live Python version",
        actual=platform.python_version(),
        expected=runtime.python_version,
    )

    _require_equal(
        label="Live Torch version",
        actual=str(torch.__version__),
        expected=runtime.torch_version,
    )

    _require_equal(
        label="CUDA_VISIBLE_DEVICES",
        actual=os.environ.get(
            "CUDA_VISIBLE_DEVICES"
        ),
        expected=runtime.cuda_visible_devices,
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Qualified B0 pilot requires CUDA."
        )

    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "Qualified B0 pilot requires exactly one visible logical GPU."
        )

    device = torch.device(
        runtime.device
    )

    actual_accelerator = (
        torch.cuda.get_device_name(
            device
        )
    )

    if runtime.accelerator not in actual_accelerator:
        raise RuntimeError(
            "GPU identity differs from qualified KQ0 profile: "
            f"expected={runtime.accelerator!r}, "
            f"actual={actual_accelerator!r}"
        )


def _case_filename(
    case_id: str,
) -> str:
    allowed = set(
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789-_."
    )

    if not case_id:
        raise ValueError(
            "case_id must not be empty."
        )

    if any(
        character not in allowed
        for character in case_id
    ):
        raise ValueError(
            f"Unsafe case_id for journal path: {case_id!r}"
        )

    return f"{case_id}.json"


class B0PilotJournal:
    def __init__(
        self,
        root: Path,
    ) -> None:
        self.root = root
        self.manifest_path = (
            root
            / "run_manifest.json"
        )
        self.intent_dir = (
            root
            / "journal"
            / "intents"
        )
        self.receipt_dir = (
            root
            / "journal"
            / "receipts"
        )

    def ensure_manifest(
        self,
        manifest: B0PilotRunManifest,
    ) -> None:
        expected = _canonical_json_bytes(
            manifest
        )

        if self.manifest_path.exists():
            actual = (
                self.manifest_path
                .read_bytes()
            )

            if actual != expected:
                raise RuntimeError(
                    "Stored B0 run manifest does not match "
                    "the current resolved execution identity."
                )

            return

        _atomic_write_json(
            self.manifest_path,
            manifest,
        )

    def intent_path(
        self,
        case_id: str,
    ) -> Path:
        return (
            self.intent_dir
            / _case_filename(
                case_id
            )
        )

    def receipt_path(
        self,
        case_id: str,
    ) -> Path:
        return (
            self.receipt_dir
            / _case_filename(
                case_id
            )
        )

    def write_intent(
        self,
        intent: B0AttemptIntent,
    ) -> None:
        _atomic_write_json(
            self.intent_path(
                intent.case_id
            ),
            intent,
        )

    def write_receipt(
        self,
        receipt: B0CaseReceipt,
    ) -> None:
        if not self.intent_path(
            receipt.attempt.case_id
        ).is_file():
            raise RuntimeError(
                "Cannot store terminal receipt without "
                "a preserved start intent."
            )

        _atomic_write_json(
            self.receipt_path(
                receipt.attempt.case_id
            ),
            receipt,
        )

    def _load_intent(
        self,
        case_id: str,
    ) -> B0AttemptIntent:
        return B0AttemptIntent.model_validate_json(
            self.intent_path(
                case_id
            ).read_text(
                encoding="utf-8",
            )
        )

    def _load_receipt(
        self,
        case_id: str,
    ) -> B0CaseReceipt:
        return B0CaseReceipt.model_validate_json(
            self.receipt_path(
                case_id
            ).read_text(
                encoding="utf-8",
            )
        )

    def load_terminal_receipt(
        self,
        *,
        manifest: B0PilotRunManifest,
        case_identity: B0PilotCaseIdentity,
    ) -> B0CaseReceipt | None:
        intent_path = self.intent_path(
            case_identity.case_id
        )

        receipt_path = self.receipt_path(
            case_identity.case_id
        )

        if receipt_path.exists() and not intent_path.exists():
            raise RuntimeError(
                "Terminal receipt exists without its start intent: "
                f"{case_identity.case_id}"
            )

        if intent_path.exists() and not receipt_path.exists():
            intent = self._load_intent(
                case_identity.case_id
            )

            _validate_intent(
                manifest=manifest,
                case_identity=case_identity,
                intent=intent,
            )

            raise B0UnresolvedAttemptError(
                "A started B0 attempt has no terminal receipt: "
                f"{case_identity.case_id}. "
                "Do not silently rerun it. Classify the interruption first."
            )

        if not receipt_path.exists():
            return None

        intent = self._load_intent(
            case_identity.case_id
        )

        _validate_intent(
            manifest=manifest,
            case_identity=case_identity,
            intent=intent,
        )

        receipt = self._load_receipt(
            case_identity.case_id
        )

        _validate_receipt(
            manifest=manifest,
            case_identity=case_identity,
            receipt=receipt,
        )

        return receipt


def _validate_intent(
    *,
    manifest: B0PilotRunManifest,
    case_identity: B0PilotCaseIdentity,
    intent: B0AttemptIntent,
) -> None:
    expected = {
        "run_id": manifest.run_id,
        "identity_sha256": (
            manifest.identity_sha256
        ),
        "implementation_commit": (
            manifest.execution_commit
        ),
        "protocol_sha256": (
            manifest.protocol_sha256
        ),
        "case_id": (
            case_identity.case_id
        ),
        "family_id": (
            case_identity.family_id
        ),
        "case_sha256": (
            case_identity.case_sha256
        ),
    }

    actual = {
        "run_id": intent.run_id,
        "identity_sha256": (
            intent.identity_sha256
        ),
        "implementation_commit": (
            intent.implementation_commit
        ),
        "protocol_sha256": (
            intent.protocol_sha256
        ),
        "case_id": intent.case_id,
        "family_id": intent.family_id,
        "case_sha256": (
            intent.case_sha256
        ),
    }

    if actual != expected:
        raise RuntimeError(
            "Stored B0 start intent does not match "
            "the current execution identity."
        )


def _validate_receipt(
    *,
    manifest: B0PilotRunManifest,
    case_identity: B0PilotCaseIdentity,
    receipt: B0CaseReceipt,
) -> None:
    _require_equal(
        label="Receipt run id",
        actual=receipt.run_id,
        expected=manifest.run_id,
    )

    _require_equal(
        label="Receipt implementation commit",
        actual=receipt.implementation_commit,
        expected=manifest.execution_commit,
    )

    _require_equal(
        label="Receipt protocol hash",
        actual=receipt.protocol_sha256,
        expected=manifest.protocol_sha256,
    )

    _require_equal(
        label="Receipt case hash",
        actual=receipt.case_sha256,
        expected=case_identity.case_sha256,
    )

    _require_equal(
        label="Receipt case id",
        actual=receipt.attempt.case_id,
        expected=case_identity.case_id,
    )

    _require_equal(
        label="Receipt family id",
        actual=receipt.attempt.family_id,
        expected=case_identity.family_id,
    )

    observation = receipt.observation

    if observation.prompt_sha256 is not None:
        _require_equal(
            label="Observed prompt hash",
            actual=observation.prompt_sha256,
            expected=case_identity.prompt_sha256,
        )

    if observation.runtime_device is not None:
        _require_equal(
            label="Observed runtime device",
            actual=observation.runtime_device,
            expected=manifest.runtime.device,
        )

    if observation.parameter_dtype is not None:
        _require_equal(
            label="Observed parameter dtype",
            actual=observation.parameter_dtype,
            expected=manifest.runtime.parameter_dtype,
        )

    if observation.attention_mode is not None:
        _require_equal(
            label="Observed attention mode",
            actual=observation.attention_mode,
            expected=manifest.runtime.attention_mode,
        )


def _runner_config(
    manifest: B0PilotRunManifest,
) -> EvaluationRunConfig:
    return EvaluationRunConfig(
        run_id=manifest.run_id,
        model_identity_hash=(
            manifest.runtime
            .model_identity_sha256
        ),
        dataset_hash=(
            manifest.selected_dataset_sha256
        ),
    )


def run_or_resume_case(
    *,
    manifest: B0PilotRunManifest,
    case: EvalCase,
    case_identity: B0PilotCaseIdentity,
    journal: B0PilotJournal,
    adapter: B0PilotAdapter,
) -> B0CaseReceipt:
    _require_equal(
        label="Case order identity",
        actual=case.input.case_id,
        expected=case_identity.case_id,
    )

    existing = (
        journal.load_terminal_receipt(
            manifest=manifest,
            case_identity=case_identity,
        )
    )

    if existing is not None:
        return existing

    intent = B0AttemptIntent(
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
        case_id=(
            case_identity.case_id
        ),
        family_id=(
            case_identity.family_id
        ),
        case_sha256=(
            case_identity.case_sha256
        ),
        started_at_utc=datetime.now(
            timezone.utc
        ),
    )

    journal.write_intent(
        intent
    )

    result = EvaluationRunner(
        config=_runner_config(
            manifest
        ),
        adapter=adapter,
    ).run(
        (case,)
    )

    attempt = result.attempts[0]

    observation = (
        adapter.get_observation(
            case.input.case_id
        )
    )

    diagnostics = (
        adapter.get_diagnostics(
            case.input.case_id
        )
    )

    receipt = B0CaseReceipt(
        run_id=manifest.run_id,
        implementation_commit=(
            manifest.execution_commit
        ),
        protocol_sha256=(
            manifest.protocol_sha256
        ),
        case_sha256=(
            case_identity.case_sha256
        ),
        attempt=attempt,
        observation=observation,
        diagnostics=diagnostics,
    )

    _validate_receipt(
        manifest=manifest,
        case_identity=case_identity,
        receipt=receipt,
    )

    journal.write_receipt(
        receipt
    )

    return receipt


def run_or_resume_cases(
    *,
    manifest: B0PilotRunManifest,
    cases: tuple[EvalCase, ...],
    journal: B0PilotJournal,
    adapter: B0PilotAdapter,
) -> tuple[B0CaseReceipt, ...]:
    case_ids = tuple(
        case.input.case_id
        for case in cases
    )

    expected_ids = tuple(
        item.case_id
        for item in manifest.cases
    )

    _require_equal(
        label="Frozen pilot case order",
        actual=case_ids,
        expected=expected_ids,
    )

    receipts: list[
        B0CaseReceipt
    ] = []

    for case, case_identity in zip(
        cases,
        manifest.cases,
        strict=True,
    ):
        receipts.append(
            run_or_resume_case(
                manifest=manifest,
                case=case,
                case_identity=(
                    case_identity
                ),
                journal=journal,
                adapter=adapter,
            )
        )

    return tuple(
        receipts
    )


def rebuild_result_from_receipts(
    *,
    manifest: B0PilotRunManifest,
    cases: tuple[EvalCase, ...],
    receipts: tuple[B0CaseReceipt, ...],
) -> EvaluationRunResult:
    if len(receipts) != len(cases):
        raise ValueError(
            "Receipt count does not match frozen case count."
        )

    receipt_by_case = {
        receipt.attempt.case_id: receipt
        for receipt in receipts
    }

    if len(receipt_by_case) != len(receipts):
        raise ValueError(
            "Duplicate receipt case IDs are not permitted."
        )

    ordered_receipts: list[
        B0CaseReceipt
    ] = []

    for case, case_identity in zip(
        cases,
        manifest.cases,
        strict=True,
    ):
        receipt = receipt_by_case.get(
            case.input.case_id
        )

        if receipt is None:
            raise ValueError(
                "Missing terminal receipt for case: "
                f"{case.input.case_id}"
            )

        _validate_receipt(
            manifest=manifest,
            case_identity=case_identity,
            receipt=receipt,
        )

        ordered_receipts.append(
            receipt
        )

    return EvaluationRunner(
        config=_runner_config(
            manifest
        ),
        adapter=_ForbiddenAdapter(),
    ).run(
        cases,
        existing_attempts=tuple(
            receipt.attempt
            for receipt
            in ordered_receipts
        ),
    )


def generation_config(
    manifest: B0PilotRunManifest,
) -> GenerationConfig:
    return GenerationConfig(
        seed=manifest.generation.seed,
        temperature=(
            manifest.generation.temperature
        ),
        top_p=manifest.generation.top_p,
        max_new_tokens=(
            manifest.generation.max_new_tokens
        ),
    )


def _write_json(
    path: Path,
    value: Any,
) -> None:
    path.write_bytes(
        _canonical_json_bytes(
            value
        )
    )


def _write_jsonl(
    path: Path,
    rows: list[Any],
) -> None:
    path.write_bytes(
        b"".join(
            _canonical_json_bytes(
                row
            )
            for row in rows
        )
    )


def _build_failure_rows(
    *,
    cases: tuple[EvalCase, ...],
    result: EvaluationRunResult,
) -> list[dict[str, Any]]:
    scores_by_case: dict[
        str,
        list[Any],
    ] = {}

    for score in result.scores:
        scores_by_case.setdefault(
            score.case_id,
            [],
        ).append(score)

    case_by_id = {
        case.input.case_id: case
        for case in cases
    }

    rows: list[
        dict[str, Any]
    ] = []

    for attempt in result.attempts:
        labels = sorted(
            {
                label.value
                for score
                in scores_by_case.get(
                    attempt.case_id,
                    [],
                )
                for label
                in score.failure_labels
            }
        )

        if (
            attempt.status.value
            == "completed"
            and not labels
        ):
            continue

        case = case_by_id[
            attempt.case_id
        ]

        rows.append(
            {
                "case_id": attempt.case_id,
                "expected_decision": (
                    case.gold
                    .expected_decision
                    .value
                ),
                "observed_decision": (
                    attempt
                    .parsed_response
                    .decision
                    if (
                        attempt.parsed_response
                        is not None
                    )
                    else None
                ),
                "status": (
                    attempt.status.value
                ),
                "parse_error": (
                    attempt
                    .parse_failure
                    .code
                    .value
                    if (
                        attempt.parse_failure
                        is not None
                    )
                    else None
                ),
                "runtime_error": (
                    attempt
                    .runtime_error_code
                    .value
                    if (
                        attempt.runtime_error_code
                        is not None
                    )
                    else None
                ),
                "failure_labels": labels,
            }
        )

    return rows


def _baseline_report(
    *,
    manifest: B0PilotRunManifest,
    cases: tuple[EvalCase, ...],
    result: EvaluationRunResult,
) -> str:
    scores_by_case: dict[
        str,
        list[Any],
    ] = {}

    for score in result.scores:
        scores_by_case.setdefault(
            score.case_id,
            [],
        ).append(score)

    attempt_by_case = {
        attempt.case_id: attempt
        for attempt in result.attempts
    }

    report = build_run_report(
        result
    )

    lines = [
        "# MORENA-R Initial B0 Pilot Baseline Report",
        "",
        f"- Run ID: `{manifest.run_id}`",
        f"- Execution commit: `{manifest.execution_commit}`",
        f"- Runtime profile: `{manifest.runtime.profile_id}`",
        f"- Model revision: `{manifest.runtime.model_revision}`",
        f"- Prompt version: `{manifest.prompt_version}`",
        "- Scope: four-case development diagnostic; not final G5 baseline",
        "- Review status: harness-only fixtures",
        "",
        "## Case outcomes",
        "",
        (
            "| Case | Expected decision | Observed decision | "
            "Attempt status | Failure labels |"
        ),
        "|---|---|---|---|---|",
    ]

    for case in cases:
        attempt = attempt_by_case[
            case.input.case_id
        ]

        observed = (
            attempt
            .parsed_response
            .decision
            if (
                attempt.parsed_response
                is not None
            )
            else "—"
        )

        labels = sorted(
            {
                label.value
                for score
                in scores_by_case.get(
                    case.input.case_id,
                    [],
                )
                for label
                in score.failure_labels
            }
        )

        failure_text = (
            ", ".join(labels)
            if labels
            else "none"
        )

        lines.append(
            "| "
            f"`{case.input.case_id}` | "
            f"{case.gold.expected_decision.value} | "
            f"{observed} | "
            f"{attempt.status.value} | "
            f"{failure_text} |"
        )

    lines.extend(
        [
            "",
            "## Deterministic metrics",
            "",
            (
                "| Metric | Eligible | Missing | Estimate |"
            ),
            "|---|---:|---:|---:|",
        ]
    )

    for metric in report.metrics:
        estimate = (
            f"{metric.estimate:.6f}"
            if metric.estimate is not None
            else "—"
        )

        lines.append(
            "| "
            f"`{metric.metric_id}` | "
            f"{metric.eligible_instances} | "
            f"{metric.missing_instances} | "
            f"{estimate} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            (
                "This four-case run is diagnostic evidence for the "
                "initial B0 execution path. It is not a statistically "
                "sufficient estimate of final multilingual reliability, "
                "does not complete G5, and is not protected evaluation."
            ),
            "",
            (
                "KQ0 qualified the compute profile. This report does not "
                "claim model improvement, CPU/GPU token equivalence, or "
                "native no-emulation T4 BF16 support."
            ),
            "",
        ]
    )

    return "\n".join(
        lines
    )


def write_pilot_bundle(
    *,
    output_dir: Path,
    manifest: B0PilotRunManifest,
    cases: tuple[EvalCase, ...],
    receipts: tuple[B0CaseReceipt, ...],
    result: EvaluationRunResult,
) -> str:
    if output_dir.exists():
        raise FileExistsError(
            "Refusing to overwrite existing B0 final bundle: "
            f"{output_dir}"
        )

    temporary = output_dir.with_name(
        f".{output_dir.name}.tmp"
    )

    if temporary.exists():
        raise FileExistsError(
            "Unresolved B0 finalization directory exists: "
            f"{temporary}"
        )

    temporary.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary.mkdir()

    _write_json(
        temporary
        / "run_manifest.json",
        manifest,
    )

    _write_jsonl(
        temporary
        / "case_receipts.jsonl",
        list(
            receipts
        ),
    )

    scores = sorted(
        result.scores,
        key=lambda row: (
            row.case_id,
            row.metric_id,
        ),
    )

    _write_jsonl(
        temporary
        / "scores.jsonl",
        list(
            scores
        ),
    )

    (
        temporary
        / "metrics.json"
    ).write_text(
        canonical_report_json(
            build_run_report(
                result
            )
        ),
        encoding="utf-8",
        newline="\n",
    )

    _write_jsonl(
        temporary
        / "failures.jsonl",
        _build_failure_rows(
            cases=cases,
            result=result,
        ),
    )

    with (
        temporary
        / "timing.csv"
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
                "transport_status",
                "elapsed_seconds",
                "model_load_seconds",
                "model_inference_seconds",
                "input_token_count",
                "output_token_count",
                "stop_reason",
            ]
        )

        for receipt in receipts:
            observation = (
                receipt.observation
            )

            writer.writerow(
                [
                    receipt.attempt.case_id,
                    receipt.attempt.status.value,
                    (
                        observation
                        .transport_status
                        .value
                    ),
                    (
                        f"{observation.elapsed_seconds:.6f}"
                    ),
                    (
                        ""
                        if (
                            observation
                            .model_load_seconds
                            is None
                        )
                        else (
                            f"{observation.model_load_seconds:.6f}"
                        )
                    ),
                    (
                        ""
                        if (
                            observation
                            .model_inference_seconds
                            is None
                        )
                        else (
                            f"{observation.model_inference_seconds:.6f}"
                        )
                    ),
                    (
                        ""
                        if (
                            observation
                            .input_token_count
                            is None
                        )
                        else (
                            observation
                            .input_token_count
                        )
                    ),
                    (
                        ""
                        if (
                            observation
                            .output_token_count
                            is None
                        )
                        else (
                            observation
                            .output_token_count
                        )
                    ),
                    (
                        observation.stop_reason
                        or ""
                    ),
                ]
            )

    (
        temporary
        / "BASELINE_REPORT.md"
    ).write_text(
        _baseline_report(
            manifest=manifest,
            cases=cases,
            result=result,
        ),
        encoding="utf-8",
        newline="\n",
    )

    checksum_targets = sorted(
        path
        for path in temporary.iterdir()
        if (
            path.is_file()
            and path.name
            != "checksums.sha256"
        )
    )

    (
        temporary
        / "checksums.sha256"
    ).write_text(
        "".join(
            (
                f"{sha256_file(path)}"
                f"  {path.name}\n"
            )
            for path
            in checksum_targets
        ),
        encoding="utf-8",
        newline="\n",
    )

    os.replace(
        temporary,
        output_dir,
    )

    return sha256_file(
        output_dir
        / "checksums.sha256"
    )


def make_subprocess_adapter(
    *,
    root: Path,
    worker_path: Path,
    manifest: B0PilotRunManifest,
    python_executable: Path,
) -> B0SubprocessAdapter:
    from morena_r.models.native_morena import RuntimeConfig

    return B0SubprocessAdapter(
        root=root,
        worker_path=worker_path,
        generation_config=(
            generation_config(
                manifest
            )
        ),
        timeout_seconds=(
            manifest.generation
            .wall_clock_timeout_seconds
        ),
        runtime_config=RuntimeConfig(
            device=manifest.runtime.device,
        ),
        python_executable=(
            python_executable
        ),
    )
