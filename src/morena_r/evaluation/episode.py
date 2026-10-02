from __future__ import annotations

from dataclasses import dataclass, field

from morena_r.contracts.actions import (
    ToolCall,
    ToolErrorCode,
)
from morena_r.tools.simulator import canonical_tool_call


@dataclass(frozen=True)
class EpisodePolicy:
    max_model_turns: int = 4
    max_tool_calls: int = 2
    max_format_repairs: int = 1

    def __post_init__(self) -> None:
        if self.max_model_turns < 1:
            raise ValueError(
                "max_model_turns must be positive."
            )

        if self.max_tool_calls < 0:
            raise ValueError(
                "max_tool_calls cannot be negative."
            )

        if self.max_format_repairs < 0:
            raise ValueError(
                "max_format_repairs cannot be negative."
            )


@dataclass
class EpisodeTracker:
    policy: EpisodePolicy = field(
        default_factory=EpisodePolicy
    )

    model_turns: int = 0
    tool_calls: int = 0
    format_repairs: int = 0

    state_version: int = 0

    _seen_calls: set[tuple[int, str]] = field(
        default_factory=set,
    )

    def register_model_turn(
        self,
    ) -> ToolErrorCode | None:
        if self.model_turns >= self.policy.max_model_turns:
            return ToolErrorCode.BUDGET_EXHAUSTED

        self.model_turns += 1
        return None

    def register_tool_call(
        self,
        call: ToolCall,
    ) -> ToolErrorCode | None:
        if self.tool_calls >= self.policy.max_tool_calls:
            return ToolErrorCode.BUDGET_EXHAUSTED

        key = (
            self.state_version,
            canonical_tool_call(call),
        )

        if key in self._seen_calls:
            return ToolErrorCode.STAGNATION

        self._seen_calls.add(key)
        self.tool_calls += 1

        return None

    def register_format_repair(
        self,
    ) -> ToolErrorCode | None:
        if (
            self.format_repairs
            >= self.policy.max_format_repairs
        ):
            return ToolErrorCode.BUDGET_EXHAUSTED

        self.format_repairs += 1
        return None

    def advance_state(self) -> None:
        self.state_version += 1
