from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Literal

from pydantic import Field, ValidationError, model_validator

from morena_r.contracts.actions import (
    NonEmptyStr,
    StrictContract,
)
from morena_r.contracts.evaluation import (
    AttemptObservation,
    AttemptRecord,
    AttemptTransportStatus,
    EvalInput,
)
from morena_r.evaluation.runner import (
    ModelExecutionError,
    ModelTimeoutError,
)
from morena_r.models.native_morena import (
    GenerationConfig,
    RuntimeConfig,
)


class B0WorkerResult(StrictContract):
    schema_version: Literal["1.1"] = "1.1"

    case_id: NonEmptyStr
    raw_output: str

    prompt_sha256: NonEmptyStr

    input_ids: tuple[int, ...] = Field(
        min_length=1,
    )
    generated_ids: tuple[int, ...]

    runtime_device: NonEmptyStr
    parameter_dtype: NonEmptyStr
    attention_mode: NonEmptyStr

    model_load_seconds: float = Field(
        ge=0,
        allow_inf_nan=False,
    )

    input_token_count: int = Field(
        ge=0,
    )

    output_token_count: int = Field(
        ge=0,
    )

    stop_reason: NonEmptyStr

    model_inference_seconds: float = Field(
        ge=0,
        allow_inf_nan=False,
    )


class B0TransportDiagnostics(StrictContract):
    schema_version: Literal["1.0"] = "1.0"

    case_id: NonEmptyStr

    worker_stdout: str | None = None
    worker_stderr: str | None = None


class B0CaseReceipt(StrictContract):
    schema_version: Literal["1.1"] = "1.1"

    run_id: NonEmptyStr
    implementation_commit: NonEmptyStr
    protocol_sha256: NonEmptyStr
    case_sha256: NonEmptyStr

    attempt: AttemptRecord
    observation: AttemptObservation
    diagnostics: B0TransportDiagnostics

    @model_validator(mode="after")
    def validate_receipt(self) -> "B0CaseReceipt":
        if self.attempt.run_id != self.run_id:
            raise ValueError(
                "Receipt run ID does not match attempt."
            )

        if (
            self.attempt.case_id
            != self.observation.case_id
        ):
            raise ValueError(
                "Attempt and observation case IDs differ."
            )

        if (
            self.attempt.case_id
            != self.diagnostics.case_id
        ):
            raise ValueError(
                "Attempt and diagnostics case IDs differ."
            )

        return self


def _captured_text(
    value: str | bytes | None,
) -> str | None:
    if value is None:
        return None

    if isinstance(value, bytes):
        return value.decode(
            "utf-8",
            errors="replace",
        )

    return value


class B0SubprocessAdapter:
    def __init__(
        self,
        *,
        root: Path,
        worker_path: Path,
        generation_config: GenerationConfig,
        timeout_seconds: float,
        runtime_config: RuntimeConfig | None = None,
        python_executable: Path | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError(
                "timeout_seconds must be positive."
            )

        if not worker_path.is_file():
            raise FileNotFoundError(
                f"B0 worker is missing: {worker_path}"
            )

        selected_python = (
            python_executable
            if python_executable is not None
            else Path(sys.executable)
        )

        if not selected_python.is_file():
            raise FileNotFoundError(
                "B0 Python executable is missing: "
                f"{selected_python}"
            )

        self._root = root
        self._worker_path = worker_path
        self._generation_config = (
            generation_config
        )
        self._runtime_config = (
            runtime_config
            if runtime_config is not None
            else RuntimeConfig()
        )
        self._timeout_seconds = (
            timeout_seconds
        )
        self._python_executable = (
            selected_python
        )

        self._observations: dict[
            str,
            AttemptObservation,
        ] = {}

        self._diagnostics: dict[
            str,
            B0TransportDiagnostics,
        ] = {}

    def get_observation(
        self,
        case_id: str,
    ) -> AttemptObservation:
        try:
            return self._observations[
                case_id
            ]
        except KeyError as error:
            raise KeyError(
                f"No observation exists for case {case_id!r}."
            ) from error

    def get_diagnostics(
        self,
        case_id: str,
    ) -> B0TransportDiagnostics:
        try:
            return self._diagnostics[
                case_id
            ]
        except KeyError as error:
            raise KeyError(
                f"No diagnostics exist for case {case_id!r}."
            ) from error

    def _record_transport(
        self,
        *,
        case_id: str,
        observation: AttemptObservation,
        worker_stdout: str | None,
        worker_stderr: str | None,
    ) -> None:
        self._observations[
            case_id
        ] = observation

        self._diagnostics[
            case_id
        ] = B0TransportDiagnostics(
            case_id=case_id,
            worker_stdout=worker_stdout,
            worker_stderr=worker_stderr,
        )

    def generate(
        self,
        eval_input: EvalInput,
    ) -> str:
        case_id = eval_input.case_id

        if case_id in self._observations:
            raise RuntimeError(
                f"Case already executed in this adapter: {case_id}"
            )

        payload = {
            "eval_input": eval_input.model_dump(
                mode="json",
                exclude_none=True,
            ),
            "generation": asdict(
                self._generation_config
            ),
            "runtime": asdict(
                self._runtime_config
            ),
        }

        serialized = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

        started_at = datetime.now(
            timezone.utc
        )

        started = monotonic()

        try:
            completed = subprocess.run(
                [
                    str(self._python_executable),
                    str(self._worker_path),
                ],
                cwd=self._root,
                input=serialized,
                text=True,
                capture_output=True,
                timeout=self._timeout_seconds,
                check=False,
            )

        except subprocess.TimeoutExpired as error:
            elapsed = monotonic() - started

            self._record_transport(
                case_id=case_id,
                observation=AttemptObservation(
                    case_id=case_id,
                    transport_status=(
                        AttemptTransportStatus.TIMEOUT
                    ),
                    started_at_utc=started_at,
                    completed_at_utc=datetime.now(
                        timezone.utc
                    ),
                    elapsed_seconds=elapsed,
                ),
                worker_stdout=_captured_text(
                    error.stdout
                ),
                worker_stderr=_captured_text(
                    error.stderr
                ),
            )

            raise ModelTimeoutError(
                f"B0 worker exceeded {self._timeout_seconds} seconds."
            ) from error

        except OSError as error:
            elapsed = monotonic() - started

            self._record_transport(
                case_id=case_id,
                observation=AttemptObservation(
                    case_id=case_id,
                    transport_status=(
                        AttemptTransportStatus.EXECUTION_ERROR
                    ),
                    started_at_utc=started_at,
                    completed_at_utc=datetime.now(
                        timezone.utc
                    ),
                    elapsed_seconds=elapsed,
                ),
                worker_stdout=None,
                worker_stderr=str(error),
            )

            raise ModelExecutionError(
                "B0 worker could not be launched."
            ) from error

        elapsed = monotonic() - started

        completed_at = datetime.now(
            timezone.utc
        )

        if completed.returncode != 0:
            self._record_transport(
                case_id=case_id,
                observation=AttemptObservation(
                    case_id=case_id,
                    transport_status=(
                        AttemptTransportStatus.EXECUTION_ERROR
                    ),
                    started_at_utc=started_at,
                    completed_at_utc=completed_at,
                    elapsed_seconds=elapsed,
                    worker_returncode=(
                        completed.returncode
                    ),
                ),
                worker_stdout=completed.stdout,
                worker_stderr=completed.stderr,
            )

            raise ModelExecutionError(
                "B0 worker exited with return code "
                f"{completed.returncode}."
            )

        try:
            worker_payload = json.loads(
                completed.stdout
            )

            worker_result = (
                B0WorkerResult.model_validate(
                    worker_payload
                )
            )

        except (
            json.JSONDecodeError,
            ValidationError,
        ) as error:
            self._record_transport(
                case_id=case_id,
                observation=AttemptObservation(
                    case_id=case_id,
                    transport_status=(
                        AttemptTransportStatus.EXECUTION_ERROR
                    ),
                    started_at_utc=started_at,
                    completed_at_utc=completed_at,
                    elapsed_seconds=elapsed,
                    worker_returncode=0,
                ),
                worker_stdout=completed.stdout,
                worker_stderr=completed.stderr,
            )

            raise ModelExecutionError(
                "B0 worker returned an invalid result envelope."
            ) from error

        if worker_result.case_id != case_id:
            self._record_transport(
                case_id=case_id,
                observation=AttemptObservation(
                    case_id=case_id,
                    transport_status=(
                        AttemptTransportStatus.EXECUTION_ERROR
                    ),
                    started_at_utc=started_at,
                    completed_at_utc=completed_at,
                    elapsed_seconds=elapsed,
                    worker_returncode=0,
                ),
                worker_stdout=completed.stdout,
                worker_stderr=completed.stderr,
            )

            raise ModelExecutionError(
                "B0 worker result case ID does not match request."
            )

        self._record_transport(
            case_id=case_id,
            observation=AttemptObservation(
                case_id=case_id,
                transport_status=(
                    AttemptTransportStatus.COMPLETED
                ),
                started_at_utc=started_at,
                completed_at_utc=completed_at,
                elapsed_seconds=elapsed,
                worker_returncode=0,
                model_inference_seconds=(
                    worker_result.model_inference_seconds
                ),
                input_token_count=(
                    worker_result.input_token_count
                ),
                output_token_count=(
                    worker_result.output_token_count
                ),
                stop_reason=(
                    worker_result.stop_reason
                ),
                prompt_sha256=(
                    worker_result.prompt_sha256
                ),
                input_ids=(
                    worker_result.input_ids
                ),
                generated_ids=(
                    worker_result.generated_ids
                ),
                runtime_device=(
                    worker_result.runtime_device
                ),
                parameter_dtype=(
                    worker_result.parameter_dtype
                ),
                attention_mode=(
                    worker_result.attention_mode
                ),
                model_load_seconds=(
                    worker_result.model_load_seconds
                ),
            ),
            worker_stdout=None,
            worker_stderr=(
                completed.stderr
                if completed.stderr
                else None
            ),
        )

        return worker_result.raw_output
