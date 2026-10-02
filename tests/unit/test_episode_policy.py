from __future__ import annotations

from morena_r.contracts.actions import (
    GetRecordArgs,
    GetRecordCall,
    ToolErrorCode,
)
from morena_r.evaluation.episode import (
    EpisodePolicy,
    EpisodeTracker,
)


def tool_call() -> GetRecordCall:
    return GetRecordCall(
        tool="get_record",
        arguments=GetRecordArgs(
            record_id="record-001",
        ),
    )


def test_four_model_turns_then_budget_exhaustion() -> None:
    tracker = EpisodeTracker()

    assert tracker.register_model_turn() is None
    assert tracker.register_model_turn() is None
    assert tracker.register_model_turn() is None
    assert tracker.register_model_turn() is None

    assert (
        tracker.register_model_turn()
        == ToolErrorCode.BUDGET_EXHAUSTED
    )


def test_two_tool_calls_then_budget_exhaustion() -> None:
    tracker = EpisodeTracker()

    first = GetRecordCall(
        tool="get_record",
        arguments=GetRecordArgs(
            record_id="record-001",
        ),
    )

    second = GetRecordCall(
        tool="get_record",
        arguments=GetRecordArgs(
            record_id="record-002",
        ),
    )

    third = GetRecordCall(
        tool="get_record",
        arguments=GetRecordArgs(
            record_id="record-003",
        ),
    )

    assert tracker.register_tool_call(first) is None
    assert tracker.register_tool_call(second) is None

    assert (
        tracker.register_tool_call(third)
        == ToolErrorCode.BUDGET_EXHAUSTED
    )


def test_repeated_call_without_new_state_is_stagnation() -> None:
    tracker = EpisodeTracker()

    call = tool_call()

    assert tracker.register_tool_call(call) is None

    assert (
        tracker.register_tool_call(call)
        == ToolErrorCode.STAGNATION
    )


def test_same_call_after_state_change_is_permitted() -> None:
    tracker = EpisodeTracker()

    call = tool_call()

    assert tracker.register_tool_call(call) is None

    tracker.advance_state()

    assert tracker.register_tool_call(call) is None


def test_one_format_repair_then_budget_exhaustion() -> None:
    tracker = EpisodeTracker(
        policy=EpisodePolicy(
            max_model_turns=4,
            max_tool_calls=2,
            max_format_repairs=1,
        )
    )

    assert tracker.register_format_repair() is None

    assert (
        tracker.register_format_repair()
        == ToolErrorCode.BUDGET_EXHAUSTED
    )
