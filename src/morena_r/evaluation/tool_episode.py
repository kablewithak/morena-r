from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from time import monotonic
from typing import Literal, Protocol

from pydantic import Field, model_validator

from morena_r.contracts.actions import (
    CallToolResponse,
    ModelResponse,
    NonEmptyStr,
    StrictContract,
    ToolErrorCode,
    ToolName,
    ToolResult,
)
from morena_r.contracts.evaluation import (
    EvalInput,
    ParseFailure,
)
from morena_r.evaluation.episode import (
    EpisodePolicy,
    EpisodeTracker,
)
from morena_r.evaluation.parser import (
    parse_model_response,
)
from morena_r.tools.simulator import (
    FixtureState,
    FixtureToolSimulator,
    ToolPermissions,
)


class EpisodeTerminalStatus(StrEnum):
    COMPLETED = "completed"
    INVALID_RESPONSE = "invalid_response"
    BUDGET_EXHAUSTED = "budget_exhausted"
    STAGNATION = "stagnation"


class EpisodeEvent(StrictContract):
    sequence: int = Field(ge=1)
    case_id: NonEmptyStr

    actor: Literal[
        "model",
        "tool",
        "runtime",
    ]

    action: NonEmptyStr
    status: NonEmptyStr

    event_time_utc: datetime

    elapsed_seconds: float = Field(
        ge=0,
        allow_inf_nan=False,
    )

    artifact_ref: NonEmptyStr

    @model_validator(mode="after")
    def validate_event_time(self) -> "EpisodeEvent":
        if (
            self.event_time_utc.tzinfo is None
            or self.event_time_utc.utcoffset() is None
        ):
            raise ValueError(
                "Episode event timestamp must be timezone-aware."
            )

        return self


class EpisodeTurn(StrictContract):
    turn_index: int = Field(ge=1)
    kind: Literal["generation", "format_repair"]

    raw_output: str

    response: ModelResponse | None = None
    parse_failure: ParseFailure | None = None

    @model_validator(mode="after")
    def validate_turn(self) -> "EpisodeTurn":
        success = self.response is not None
        failed = self.parse_failure is not None

        if success == failed:
            raise ValueError(
                "EpisodeTurn requires exactly one response or parse_failure."
            )

        return self


class ToolEpisodeResult(StrictContract):
    schema_version: Literal["1.0"] = "1.0"

    case_id: NonEmptyStr
    status: EpisodeTerminalStatus

    final_response: ModelResponse | None = None
    parse_failure: ParseFailure | None = None
    runtime_error_code: ToolErrorCode | None = None

    turns: tuple[EpisodeTurn, ...]
    tool_results: tuple[ToolResult, ...]
    events: tuple[EpisodeEvent, ...]

    @model_validator(mode="after")
    def validate_terminal_state(self) -> "ToolEpisodeResult":
        if self.status == EpisodeTerminalStatus.COMPLETED:
            if self.final_response is None:
                raise ValueError(
                    "Completed episode requires final_response."
                )

            if self.runtime_error_code is not None:
                raise ValueError(
                    "Completed episode cannot contain runtime_error_code."
                )

        if self.status == EpisodeTerminalStatus.INVALID_RESPONSE:
            if self.parse_failure is None:
                raise ValueError(
                    "Invalid response requires parse_failure."
                )

            if self.final_response is not None:
                raise ValueError(
                    "Invalid response cannot contain final_response."
                )

        if self.status in {
            EpisodeTerminalStatus.BUDGET_EXHAUSTED,
            EpisodeTerminalStatus.STAGNATION,
        }:
            if self.runtime_error_code is None:
                raise ValueError(
                    "Runtime terminal state requires runtime_error_code."
                )

            if self.final_response is not None:
                raise ValueError(
                    "Runtime terminal state cannot contain final_response."
                )

        return self


class ToolEpisodeAdapter(Protocol):
    def generate(
        self,
        eval_input: EvalInput,
        tool_results: tuple[ToolResult, ...],
    ) -> str:
        ...

    def repair(
        self,
        eval_input: EvalInput,
        *,
        raw_output: str,
        parse_failure: ParseFailure,
        tool_results: tuple[ToolResult, ...],
    ) -> str:
        ...


def permissions_from_input(
    eval_input: EvalInput,
) -> ToolPermissions:
    allowed = set(
        eval_input.permitted_tools
    )

    return ToolPermissions(
        search_records=(
            ToolName.SEARCH_RECORDS in allowed
        ),
        get_record=(
            ToolName.GET_RECORD in allowed
        ),
        calculate=(
            ToolName.CALCULATE in allowed
        ),
        lookup_status=(
            ToolName.LOOKUP_STATUS in allowed
        ),
    )


class BoundedToolEpisodeRunner:
    def __init__(
        self,
        *,
        adapter: ToolEpisodeAdapter,
        policy: EpisodePolicy | None = None,
    ) -> None:
        self._adapter = adapter
        self._policy = (
            policy
            if policy is not None
            else EpisodePolicy()
        )

    def run(
        self,
        *,
        eval_input: EvalInput,
        fixture_state: FixtureState,
    ) -> ToolEpisodeResult:
        expected_permissions = permissions_from_input(
            eval_input
        )

        if fixture_state.permissions != expected_permissions:
            raise ValueError(
                "Fixture permissions do not match EvalInput permission state."
            )

        tracker = EpisodeTracker(
            policy=self._policy
        )

        simulator = FixtureToolSimulator(
            fixture_state
        )

        started = monotonic()

        turns: list[EpisodeTurn] = []
        tool_results: list[ToolResult] = []
        events: list[EpisodeEvent] = []

        repair_raw_output: str | None = None
        repair_failure: ParseFailure | None = None

        def record_event(
            *,
            actor: Literal[
                "model",
                "tool",
                "runtime",
            ],
            action: str,
            status: str,
            artifact_ref: str,
        ) -> None:
            events.append(
                EpisodeEvent(
                    sequence=len(events) + 1,
                    case_id=eval_input.case_id,
                    actor=actor,
                    action=action,
                    status=status,
                    event_time_utc=datetime.now(
                        timezone.utc
                    ),
                    elapsed_seconds=(
                        monotonic() - started
                    ),
                    artifact_ref=artifact_ref,
                )
            )

        while True:
            turn_budget = tracker.register_model_turn()

            if turn_budget is not None:
                record_event(
                    actor="runtime",
                    action="model_turn_budget",
                    status=turn_budget.value,
                    artifact_ref=(
                        f"case:{eval_input.case_id}"
                    ),
                )

                return ToolEpisodeResult(
                    case_id=eval_input.case_id,
                    status=(
                        EpisodeTerminalStatus.BUDGET_EXHAUSTED
                    ),
                    parse_failure=repair_failure,
                    runtime_error_code=turn_budget,
                    turns=tuple(turns),
                    tool_results=tuple(tool_results),
                    events=tuple(events),
                )

            repairing = (
                repair_raw_output is not None
                and repair_failure is not None
            )

            if repairing:
                raw_output = self._adapter.repair(
                    eval_input,
                    raw_output=repair_raw_output,
                    parse_failure=repair_failure,
                    tool_results=tuple(tool_results),
                )

                turn_kind = "format_repair"

                repair_raw_output = None
                repair_failure = None
            else:
                raw_output = self._adapter.generate(
                    eval_input,
                    tuple(tool_results),
                )

                turn_kind = "generation"

            parsed = parse_model_response(
                raw_output
            )

            turns.append(
                EpisodeTurn(
                    turn_index=tracker.model_turns,
                    kind=turn_kind,
                    raw_output=raw_output,
                    response=parsed.response,
                    parse_failure=parsed.failure,
                )
            )

            record_event(
                actor="model",
                action=turn_kind,
                status=(
                    "valid"
                    if parsed.response is not None
                    else parsed.failure.code.value
                ),
                artifact_ref=(
                    f"turn:{tracker.model_turns}"
                ),
            )

            if parsed.failure is not None:
                repair_budget = (
                    tracker.register_format_repair()
                )

                if repair_budget is not None:
                    record_event(
                        actor="runtime",
                        action="format_repair_budget",
                        status=repair_budget.value,
                        artifact_ref=(
                            f"turn:{tracker.model_turns}"
                        ),
                    )

                    return ToolEpisodeResult(
                        case_id=eval_input.case_id,
                        status=(
                            EpisodeTerminalStatus.BUDGET_EXHAUSTED
                        ),
                        parse_failure=parsed.failure,
                        runtime_error_code=repair_budget,
                        turns=tuple(turns),
                        tool_results=tuple(tool_results),
                        events=tuple(events),
                    )

                repair_raw_output = raw_output
                repair_failure = parsed.failure

                record_event(
                    actor="runtime",
                    action="format_repair",
                    status="scheduled",
                    artifact_ref=(
                        f"turn:{tracker.model_turns}"
                    ),
                )

                continue

            response = parsed.response

            assert response is not None

            if not isinstance(
                response,
                CallToolResponse,
            ):
                return ToolEpisodeResult(
                    case_id=eval_input.case_id,
                    status=(
                        EpisodeTerminalStatus.COMPLETED
                    ),
                    final_response=response,
                    turns=tuple(turns),
                    tool_results=tuple(tool_results),
                    events=tuple(events),
                )

            call = response.tool_call

            call_budget = tracker.register_tool_call(
                call
            )

            if call_budget is not None:
                terminal = (
                    EpisodeTerminalStatus.STAGNATION
                    if (
                        call_budget
                        == ToolErrorCode.STAGNATION
                    )
                    else (
                        EpisodeTerminalStatus.BUDGET_EXHAUSTED
                    )
                )

                record_event(
                    actor="runtime",
                    action="tool_call_budget",
                    status=call_budget.value,
                    artifact_ref=(
                        f"turn:{tracker.model_turns}"
                    ),
                )

                return ToolEpisodeResult(
                    case_id=eval_input.case_id,
                    status=terminal,
                    runtime_error_code=call_budget,
                    turns=tuple(turns),
                    tool_results=tuple(tool_results),
                    events=tuple(events),
                )

            tool = ToolName(
                call.tool
            )

            call_id = (
                f"{eval_input.case_id}:tool:"
                f"{len(tool_results) + 1}"
            )

            if tool not in eval_input.available_tools:
                tool_result = ToolResult(
                    call_id=call_id,
                    tool=tool,
                    success=False,
                    error_code=(
                        ToolErrorCode.INVALID_TOOL
                    ),
                )
            else:
                tool_result = simulator.execute(
                    call_id=call_id,
                    call=call,
                )

            tool_results.append(
                tool_result
            )

            record_event(
                actor="tool",
                action=tool.value,
                status=(
                    "SUCCESS"
                    if tool_result.success
                    else tool_result.error_code.value
                ),
                artifact_ref=(
                    f"tool_result:{call_id}"
                ),
            )

            tracker.advance_state()
