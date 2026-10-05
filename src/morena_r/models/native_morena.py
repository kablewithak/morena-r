from __future__ import annotations

import gc
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer


REPOSITORY = "vamboai/morena-1.5b-instruct"
REVISION = "b4be1225c9b593ffa79f2bf46d8a84a83d385c67"

CONFIG_SHA256 = (
    "e46cbea0588ff434cc5ffec0f06d444579b1f55a47e93f11ce43658a04926391"
)
TOKENIZER_SHA256 = (
    "97e5dc822407e4ecb6bac02b0fb8a883e465344953e2fac3becf7b5aacae4de3"
)
MODEL_SOURCE_SHA256 = (
    "1b2693d823e709a4614f091025448febc19ca0e32aa6a431c28f090e3b486de9"
)
REFERENCE_LOADER_SHA256 = (
    "b0730bd0f6afea78ccb9151c340f0216eacf6335245d24957edae91316bae285"
)

ATTENTION_MODE = "sdpa"
MAX_SEQUENCE_TOKENS = 4096


@dataclass(frozen=True)
class RuntimeConfig:
    device: str = "cpu"

    def resolve_device(self) -> torch.device:
        if self.device == "cpu":
            return torch.device("cpu")

        if not self.device.startswith("cuda:"):
            raise ValueError(
                "device must be 'cpu' or explicit 'cuda:<index>'."
            )

        index_text = self.device.removeprefix(
            "cuda:"
        )

        if not index_text.isdigit():
            raise ValueError(
                "CUDA device must use a non-negative integer index."
            )

        index = int(index_text)

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA device requested but CUDA is unavailable."
            )

        device_count = torch.cuda.device_count()

        if index >= device_count:
            raise RuntimeError(
                "CUDA device index is unavailable: "
                f"requested={index}, available={device_count}."
            )

        return torch.device(
            "cuda",
            index,
        )


@dataclass(frozen=True)
class GenerationConfig:
    seed: int = 1234
    temperature: float = 0.7
    top_p: float = 0.9
    max_new_tokens: int = 48

    def __post_init__(self) -> None:
        if self.max_new_tokens < 1:
            raise ValueError(
                "max_new_tokens must be positive."
            )

        if self.temperature <= 0:
            raise ValueError(
                "temperature must be positive."
            )

        if not 0 < self.top_p <= 1:
            raise ValueError(
                "top_p must be in (0, 1]."
            )


@dataclass(frozen=True)
class GenerationResult:
    raw_output: str

    input_ids: tuple[int, ...]
    generated_ids: tuple[int, ...]

    input_token_count: int
    output_token_count: int

    stop_reason: str
    inference_seconds: float

    @property
    def output_tokens_per_second(self) -> float:
        if self.output_token_count == 0:
            return 0.0

        return (
            self.output_token_count
            / self.inference_seconds
        )


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(
                16 * 1024 * 1024
            ),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


class NativeMorenaRuntime:
    def __init__(
        self,
        *,
        root: Path,
        runtime_config: RuntimeConfig | None = None,
    ) -> None:
        self._root = root

        self._runtime_config = (
            runtime_config
            if runtime_config is not None
            else RuntimeConfig()
        )

        self._device = (
            self._runtime_config.resolve_device()
        )

        self._upstream = (
            root
            / ".cache"
            / "upstream"
            / "morena-1.5b-instruct"
            / REVISION
        )

        self._config_path = (
            self._upstream / "config.json"
        )

        self._tokenizer_path = (
            self._upstream / "tokenizer.json"
        )

        self._weights_path = (
            self._upstream / "model.safetensors"
        )

        self._model_source = (
            self._upstream / "modeling_morena.py"
        )

        self._reference_loader = (
            self._upstream / "load_example.py"
        )

        self._weight_receipt_path = (
            root
            / "reports"
            / "g1"
            / "weights_integrity.json"
        )

        self._verify_identity()

        self.tokenizer = Tokenizer.from_file(
            str(self._tokenizer_path)
        )

        self.eos_id = self.tokenizer.token_to_id(
            "<eos>"
        )

        assert self.eos_id == 2
        assert (
            self.tokenizer.token_to_id(
                "<reserved_0>"
            )
            == 3
        )
        assert (
            self.tokenizer.token_to_id(
                "<reserved_1>"
            )
            == 4
        )

        sys.path.insert(
            0,
            str(self._upstream),
        )

        import modeling_morena as M

        config = json.loads(
            self._config_path.read_text(
                encoding="utf-8-sig"
            )
        )

        self._synchronize_device()

        model_load_started = (
            time.perf_counter()
        )

        with torch.device("meta"):
            self.model = M.Transformer(
                M.ModelConfig(
                    **config["model"]
                ),
                ATTENTION_MODE,
            )

        state_dict = load_file(
            self._weights_path,
            device=str(self._device),
        )

        load_result = self.model.load_state_dict(
            state_dict,
            strict=True,
            assign=True,
        )

        assert not load_result.missing_keys
        assert not load_result.unexpected_keys

        del state_dict
        gc.collect()

        self.model.eval()

        parameter_dtypes = {
            str(parameter.dtype)
            for parameter
            in self.model.parameters()
        }

        parameter_devices = {
            str(parameter.device)
            for parameter
            in self.model.parameters()
        }

        if parameter_dtypes != {
            "torch.bfloat16"
        }:
            raise RuntimeError(
                "Loaded parameter dtype differs from qualified BF16 identity: "
                f"{sorted(parameter_dtypes)}"
            )

        expected_device = str(
            self._device
        )

        if parameter_devices != {
            expected_device
        }:
            raise RuntimeError(
                "Loaded parameter device differs from requested runtime device: "
                f"expected={expected_device}, "
                f"actual={sorted(parameter_devices)}"
            )

        self._parameter_dtype = next(
            iter(parameter_dtypes)
        )

        self._synchronize_device()

        self._model_load_seconds = (
            time.perf_counter()
            - model_load_started
        )

    @property
    def runtime_device(self) -> str:
        return str(self._device)

    @property
    def parameter_dtype(self) -> str:
        return self._parameter_dtype

    @property
    def attention_mode(self) -> str:
        return ATTENTION_MODE

    @property
    def model_load_seconds(self) -> float:
        return self._model_load_seconds

    def _synchronize_device(self) -> None:
        if self._device.type == "cuda":
            torch.cuda.synchronize(
                self._device
            )

    def _verify_identity(self) -> None:
        expected = {
            self._config_path: (
                CONFIG_SHA256
            ),
            self._tokenizer_path: (
                TOKENIZER_SHA256
            ),
            self._model_source: (
                MODEL_SOURCE_SHA256
            ),
            self._reference_loader: (
                REFERENCE_LOADER_SHA256
            ),
        }

        for path, digest in expected.items():
            if not path.is_file():
                raise FileNotFoundError(
                    f"Required model artifact missing: {path}"
                )

            if sha256_file(path) != digest:
                raise RuntimeError(
                    f"Model artifact identity mismatch: {path.name}"
                )

        receipt = json.loads(
            self._weight_receipt_path.read_text(
                encoding="utf-8-sig"
            )
        )

        if receipt["checksum_match"] is not True:
            raise RuntimeError(
                "G1 weight checksum receipt is not qualified."
            )

        if (
            sha256_file(
                self._weights_path
            )
            != receipt["actual_sha256"]
        ):
            raise RuntimeError(
                "Model weights differ from G1 qualified identity."
            )

    def generate(
        self,
        *,
        prompt: str,
        config: GenerationConfig,
    ) -> GenerationResult:
        body_ids = self.tokenizer.encode(
            prompt,
            add_special_tokens=False,
        ).ids

        input_ids = [
            self.eos_id,
            *body_ids,
        ]

        if (
            len(input_ids)
            + config.max_new_tokens
            > MAX_SEQUENCE_TOKENS
        ):
            raise ValueError(
                "Requested generation exceeds the 4096-token context limit."
            )

        x = torch.tensor(
            [input_ids],
            dtype=torch.long,
            device=self._device,
        )

        torch.manual_seed(
            config.seed
        )

        generated_ids: list[int] = []

        self._synchronize_device()

        started = time.perf_counter()

        with torch.inference_mode():
            for _ in range(
                config.max_new_tokens
            ):
                positions = torch.arange(
                    x.shape[1],
                    device=x.device,
                ).unsqueeze(0)

                logits = self.model(
                    x,
                    positions,
                    None,
                    x.shape[1],
                    None,
                )[:, -1, :].float()

                probabilities = torch.softmax(
                    logits
                    / max(
                        config.temperature,
                        1e-5,
                    ),
                    dim=-1,
                )

                (
                    sorted_probabilities,
                    sorted_indices,
                ) = torch.sort(
                    probabilities,
                    descending=True,
                    dim=-1,
                )

                sorted_probabilities[
                    sorted_probabilities.cumsum(-1)
                    - sorted_probabilities
                    > config.top_p
                ] = 0.0

                next_token = (
                    sorted_indices.gather(
                        -1,
                        torch.multinomial(
                            sorted_probabilities
                            / sorted_probabilities.sum(
                                -1,
                                keepdim=True,
                            ),
                            1,
                        ),
                    )
                )

                token_id = int(
                    next_token[0, 0]
                )

                if token_id == self.eos_id:
                    stop_reason = "eos"
                    break

                generated_ids.append(
                    token_id
                )

                x = torch.cat(
                    [
                        x,
                        next_token,
                    ],
                    dim=1,
                )
            else:
                stop_reason = (
                    "max_new_tokens"
                )

        self._synchronize_device()

        inference_seconds = (
            time.perf_counter()
            - started
        )

        raw_output = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )

        return GenerationResult(
            raw_output=raw_output,
            input_ids=tuple(input_ids),
            generated_ids=tuple(
                generated_ids
            ),
            input_token_count=len(
                input_ids
            ),
            output_token_count=len(
                generated_ids
            ),
            stop_reason=stop_reason,
            inference_seconds=(
                inference_seconds
            ),
        )
