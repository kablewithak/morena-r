from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from morena_r.contracts.evaluation import (
    AttemptTransportStatus,
    EvalInput,
    EvalMessage,
)
from morena_r.evaluation.b0_subprocess import (
    B0SubprocessAdapter,
)
from morena_r.evaluation.runner import (
    ModelExecutionError,
    ModelTimeoutError,
)
from morena_r.models.native_morena import (
    GenerationConfig,
    RuntimeConfig,
)


def make_input(
    case_id: str = "subprocess-case",
) -> EvalInput:
    return EvalInput(
        case_id=case_id,
        family_id="subprocess-family",
        languages=("en",),
        messages=(
            EvalMessage(
                role="user",
                content="Synthetic subprocess test.",
            ),
        ),
        evidence=(),
        available_tools=(),
        permitted_tools=(),
        evaluation_time_utc=datetime(
            2026,
            1,
            1,
            tzinfo=timezone.utc,
        ),
    )


def write_worker(
    path: Path,
    source: str,
) -> Path:
    path.write_text(
        source,
        encoding="utf-8",
        newline="\n",
    )

    return path


def test_subprocess_adapter_returns_raw_output(
    tmp_path: Path,
) -> None:
    worker = write_worker(
        tmp_path / "success_worker.py",
        """import json
import sys

payload = json.loads(sys.stdin.read())
case_id = payload["eval_input"]["case_id"]

if payload["runtime"]["device"] != "cpu":
    raise SystemExit(7)

result = {
    "schema_version": "1.1",
    "case_id": case_id,
    "raw_output": '{"decision":"RESPOND","answerability":"SUPPORTED","answer":"fixture"}',
    "prompt_sha256": "abc123",
    "input_ids": [2, 10],
    "generated_ids": [49],
    "runtime_device": "cpu",
    "parameter_dtype": "torch.bfloat16",
    "attention_mode": "sdpa",
    "model_load_seconds": 0.02,
    "input_token_count": 10,
    "output_token_count": 5,
    "stop_reason": "eos",
    "model_inference_seconds": 0.01,
}

sys.stdout.write(json.dumps(result))
""",
    )

    adapter = B0SubprocessAdapter(
        root=tmp_path,
        worker_path=worker,
        generation_config=GenerationConfig(
            max_new_tokens=1,
        ),
        timeout_seconds=5,
    )

    raw = adapter.generate(
        make_input()
    )

    assert '"decision":"RESPOND"' in raw

    observation = adapter.get_observation(
        "subprocess-case"
    )

    assert (
        observation.transport_status
        == AttemptTransportStatus.COMPLETED
    )
    assert observation.worker_returncode == 0
    assert observation.input_token_count == 10
    assert observation.output_token_count == 5
    assert observation.input_ids == (2, 10)
    assert observation.generated_ids == (49,)
    assert observation.runtime_device == "cpu"
    assert observation.parameter_dtype == "torch.bfloat16"
    assert observation.attention_mode == "sdpa"
    assert observation.model_load_seconds == 0.02
    assert observation.elapsed_seconds >= 0


def test_subprocess_adapter_enforces_timeout(
    tmp_path: Path,
) -> None:
    worker = write_worker(
        tmp_path / "slow_worker.py",
        """import time
time.sleep(2)
""",
    )

    adapter = B0SubprocessAdapter(
        root=tmp_path,
        worker_path=worker,
        generation_config=GenerationConfig(
            max_new_tokens=1,
        ),
        timeout_seconds=0.05,
    )

    with pytest.raises(
        ModelTimeoutError,
    ):
        adapter.generate(
            make_input(
                "timeout-case"
            )
        )

    observation = adapter.get_observation(
        "timeout-case"
    )

    assert (
        observation.transport_status
        == AttemptTransportStatus.TIMEOUT
    )
    assert observation.worker_returncode is None
    assert observation.elapsed_seconds < 2


def test_subprocess_adapter_records_worker_failure(
    tmp_path: Path,
) -> None:
    worker = write_worker(
        tmp_path / "failure_worker.py",
        """import sys
sys.exit(3)
""",
    )

    adapter = B0SubprocessAdapter(
        root=tmp_path,
        worker_path=worker,
        generation_config=GenerationConfig(
            max_new_tokens=1,
        ),
        timeout_seconds=5,
    )

    with pytest.raises(
        ModelExecutionError,
    ):
        adapter.generate(
            make_input(
                "failure-case"
            )
        )

    observation = adapter.get_observation(
        "failure-case"
    )

    assert (
        observation.transport_status
        == AttemptTransportStatus.EXECUTION_ERROR
    )
    assert observation.worker_returncode == 3


def test_subprocess_adapter_rejects_malformed_worker_result(
    tmp_path: Path,
) -> None:
    worker = write_worker(
        tmp_path / "malformed_worker.py",
        """print("not-json")
""",
    )

    adapter = B0SubprocessAdapter(
        root=tmp_path,
        worker_path=worker,
        generation_config=GenerationConfig(
            max_new_tokens=1,
        ),
        timeout_seconds=5,
    )

    with pytest.raises(
        ModelExecutionError,
    ):
        adapter.generate(
            make_input(
                "malformed-case"
            )
        )

    observation = adapter.get_observation(
        "malformed-case"
    )

    assert (
        observation.transport_status
        == AttemptTransportStatus.EXECUTION_ERROR
    )
    assert observation.worker_returncode == 0


def test_subprocess_adapter_propagates_explicit_runtime_config(
    tmp_path: Path,
) -> None:
    worker = write_worker(
        tmp_path / "runtime_worker.py",
        """import json
import sys

payload = json.loads(sys.stdin.read())
case_id = payload["eval_input"]["case_id"]
device = payload["runtime"]["device"]

result = {
    "schema_version": "1.1",
    "case_id": case_id,
    "raw_output": device,
    "prompt_sha256": "runtime123",
    "input_ids": [2],
    "generated_ids": [49],
    "runtime_device": device,
    "parameter_dtype": "torch.bfloat16",
    "attention_mode": "sdpa",
    "model_load_seconds": 0.02,
    "input_token_count": 1,
    "output_token_count": 1,
    "stop_reason": "eos",
    "model_inference_seconds": 0.01,
}

sys.stdout.write(json.dumps(result))
""",
    )

    adapter = B0SubprocessAdapter(
        root=tmp_path,
        worker_path=worker,
        generation_config=GenerationConfig(
            max_new_tokens=1,
        ),
        timeout_seconds=5,
        runtime_config=RuntimeConfig(
            device="cuda:1",
        ),
    )

    raw = adapter.generate(
        make_input(
            "runtime-case"
        )
    )

    assert raw == "cuda:1"

    observation = adapter.get_observation(
        "runtime-case"
    )

    assert (
        observation.transport_status
        == AttemptTransportStatus.COMPLETED
    )


def test_explicit_runtime_observation_preserves_device(
    tmp_path: Path,
) -> None:
    worker = write_worker(
        tmp_path / "runtime_evidence_worker.py",
        """import json
import sys

payload = json.loads(sys.stdin.read())
case_id = payload["eval_input"]["case_id"]
device = payload["runtime"]["device"]

result = {
    "schema_version": "1.1",
    "case_id": case_id,
    "raw_output": "fixture",
    "prompt_sha256": "runtime456",
    "input_ids": [2, 3],
    "generated_ids": [49],
    "runtime_device": device,
    "parameter_dtype": "torch.bfloat16",
    "attention_mode": "sdpa",
    "model_load_seconds": 0.02,
    "input_token_count": 2,
    "output_token_count": 1,
    "stop_reason": "eos",
    "model_inference_seconds": 0.01,
}

sys.stdout.write(json.dumps(result))
""",
    )

    adapter = B0SubprocessAdapter(
        root=tmp_path,
        worker_path=worker,
        generation_config=GenerationConfig(
            max_new_tokens=1,
        ),
        timeout_seconds=5,
        runtime_config=RuntimeConfig(
            device="cuda:0",
        ),
    )

    adapter.generate(
        make_input(
            "runtime-evidence-case"
        )
    )

    observation = adapter.get_observation(
        "runtime-evidence-case"
    )

    assert observation.runtime_device == "cuda:0"
    assert observation.parameter_dtype == "torch.bfloat16"
    assert observation.attention_mode == "sdpa"
    assert observation.input_ids == (2, 3)
    assert observation.generated_ids == (49,)
    assert observation.model_load_seconds == 0.02


def test_subprocess_adapter_rejects_missing_python_executable(
    tmp_path: Path,
) -> None:
    worker = write_worker(
        tmp_path / "worker.py",
        "print('unused')\n",
    )

    missing_python = (
        tmp_path / "missing-python"
    )

    with pytest.raises(
        FileNotFoundError,
        match="Python executable is missing",
    ):
        B0SubprocessAdapter(
            root=tmp_path,
            worker_path=worker,
            generation_config=GenerationConfig(
                max_new_tokens=1,
            ),
            timeout_seconds=5,
            python_executable=missing_python,
        )


def test_subprocess_adapter_uses_explicit_python_executable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = write_worker(
        tmp_path / "unused_worker.py",
        "print('unused')\n",
    )

    selected_python = write_worker(
        tmp_path / "qualified-python",
        "synthetic executable marker\n",
    )

    worker_result = {
        "schema_version": "1.1",
        "case_id": "python-executable-case",
        "raw_output": "fixture",
        "prompt_sha256": "python123",
        "input_ids": [2, 3],
        "generated_ids": [49],
        "runtime_device": "cpu",
        "parameter_dtype": "torch.bfloat16",
        "attention_mode": "sdpa",
        "model_load_seconds": 0.02,
        "input_token_count": 2,
        "output_token_count": 1,
        "stop_reason": "eos",
        "model_inference_seconds": 0.01,
    }

    captured_command: list[str] = []

    class Completed:
        returncode = 0
        stdout = json.dumps(
            worker_result
        )

    def fake_run(
        command: list[str],
        **kwargs: object,
    ) -> Completed:
        captured_command.extend(
            command
        )
        return Completed()

    monkeypatch.setattr(
        "morena_r.evaluation.b0_subprocess.subprocess.run",
        fake_run,
    )

    adapter = B0SubprocessAdapter(
        root=tmp_path,
        worker_path=worker,
        generation_config=GenerationConfig(
            max_new_tokens=1,
        ),
        timeout_seconds=5,
        python_executable=selected_python,
    )

    raw = adapter.generate(
        make_input(
            "python-executable-case"
        )
    )

    assert raw == "fixture"
    assert captured_command == [
        str(selected_python),
        str(worker),
    ]
