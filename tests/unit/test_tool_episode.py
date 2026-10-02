from __future__ import annotations

from datetime import date, datetime, timezone

from morena_r.contracts.actions import (
    RecordFixture,
    ToolErrorCode,
    ToolName,
    ToolResult,
)
from morena_r.contracts.evaluation import (
    EvalInput,
    EvalMessage,
    ParseFailure,
)
from morena_r.evaluation.tool_episode import (
    BoundedToolEpisodeRunner,
    EpisodeTerminalStatus,
)
from morena_r.tools.simulator import (
    FixtureState,
    ToolPermissions,
)


def make_input(
    *,
    case_id: str,
    available_tools: tuple[ToolName, ...],
    permitted_tools: tuple[ToolName, ...],
) -> EvalInput:
    return EvalInput(
        case_id=case_id,
        family_id=f"{case_id}-family",
        languages=("en",),
        messages=(
            EvalMessage(
                role="user",
                content="Synthetic bounded episode.",
            ),
        ),
        evidence=(),
        available_tools=available_tools,
        permitted_tools=permitted_tools,
        evaluation_time_utc=datetime(
            2026,
            1,
            1,
            tzinfo=timezone.utc,
        ),
    )


class NoRepairMixin:
    def repair(
        self,
        eval_input: EvalInput,
        *,
        raw_output: str,
        parse_failure: ParseFailure,
        tool_results: tuple[ToolResult, ...],
    ) -> str:
        raise AssertionError(
            "Unexpected format repair."
        )


class SuccessfulToolAdapter(NoRepairMixin):
    def generate(
        self,
        eval_input: EvalInput,
        tool_results: tuple[ToolResult, ...],
    ) -> str:
        if not tool_results:
            return (
                '{"decision":"CALL_TOOL",'
                '"answerability":"NOT_APPLICABLE",'
                '"tool_call":{'
                '"tool":"get_record",'
                '"arguments":{"record_id":"rec-001"}}}'
            )

        assert tool_results[-1].success is True

        return (
            '{"decision":"RESPOND",'
            '"answerability":"SUPPORTED",'
            '"answer":"Record retrieved."}'
        )


class DeniedToolAdapter(NoRepairMixin):
    def generate(
        self,
        eval_input: EvalInput,
        tool_results: tuple[ToolResult, ...],
    ) -> str:
        if not tool_results:
            return (
                '{"decision":"CALL_TOOL",'
                '"answerability":"NOT_APPLICABLE",'
                '"tool_call":{'
                '"tool":"get_record",'
                '"arguments":{"record_id":"rec-001"}}}'
            )

        assert (
            tool_results[-1].error_code
            == ToolErrorCode.PERMISSION_DENIED
        )

        return (
            '{"decision":"ASK_CLARIFICATION",'
            '"answerability":"INSUFFICIENT_CONTEXT",'
            '"question":"A permitted source is required."}'
        )


class ToolBudgetAdapter(NoRepairMixin):
    def generate(
        self,
        eval_input: EvalInput,
        tool_results: tuple[ToolResult, ...],
    ) -> str:
        record_id = (
            f"rec-{len(tool_results) + 1:03d}"
        )

        return (
            '{"decision":"CALL_TOOL",'
            '"answerability":"NOT_APPLICABLE",'
            '"tool_call":{'
            '"tool":"get_record",'
            '"arguments":{'
            f'"record_id":"{record_id}"'
            '}}}'
        )


class RepairAdapter:
    def generate(
        self,
        eval_input: EvalInput,
        tool_results: tuple[ToolResult, ...],
    ) -> str:
        return '{"decision":'

    def repair(
        self,
        eval_input: EvalInput,
        *,
        raw_output: str,
        parse_failure: ParseFailure,
        tool_results: tuple[ToolResult, ...],
    ) -> str:
        return (
            '{"decision":"RESPOND",'
            '"answerability":"SUPPORTED",'
            '"answer":"Repaired."}'
        )


class BrokenRepairAdapter:
    def generate(
        self,
        eval_input: EvalInput,
        tool_results: tuple[ToolResult, ...],
    ) -> str:
        return '{"decision":'

    def repair(
        self,
        eval_input: EvalInput,
        *,
        raw_output: str,
        parse_failure: ParseFailure,
        tool_results: tuple[ToolResult, ...],
    ) -> str:
        return '{"decision":'


def tool_state(
    *,
    permitted: bool,
) -> FixtureState:
    return FixtureState(
        permissions=ToolPermissions(
            get_record=permitted,
        ),
        records=(
            RecordFixture(
                record_id="rec-001",
                text="Synthetic record.",
                observed_at=date(
                    2026,
                    1,
                    1,
                ),
            ),
        ),
    )


def test_tool_result_reenters_model() -> None:
    result = BoundedToolEpisodeRunner(
        adapter=SuccessfulToolAdapter(),
    ).run(
        eval_input=make_input(
            case_id="tool-success",
            available_tools=(
                ToolName.GET_RECORD,
            ),
            permitted_tools=(
                ToolName.GET_RECORD,
            ),
        ),
        fixture_state=tool_state(
            permitted=True,
        ),
    )

    assert (
        result.status
        == EpisodeTerminalStatus.COMPLETED
    )
    assert len(result.turns) == 2
    assert len(result.tool_results) == 1
    assert result.tool_results[0].success is True


def test_permission_denial_is_typed() -> None:
    result = BoundedToolEpisodeRunner(
        adapter=DeniedToolAdapter(),
    ).run(
        eval_input=make_input(
            case_id="tool-denied",
            available_tools=(
                ToolName.GET_RECORD,
            ),
            permitted_tools=(),
        ),
        fixture_state=tool_state(
            permitted=False,
        ),
    )

    assert (
        result.tool_results[0].error_code
        == ToolErrorCode.PERMISSION_DENIED
    )


def test_tool_budget_is_enforced() -> None:
    result = BoundedToolEpisodeRunner(
        adapter=ToolBudgetAdapter(),
    ).run(
        eval_input=make_input(
            case_id="tool-budget",
            available_tools=(
                ToolName.GET_RECORD,
            ),
            permitted_tools=(
                ToolName.GET_RECORD,
            ),
        ),
        fixture_state=tool_state(
            permitted=True,
        ),
    )

    assert (
        result.status
        == EpisodeTerminalStatus.BUDGET_EXHAUSTED
    )
    assert (
        result.runtime_error_code
        == ToolErrorCode.BUDGET_EXHAUSTED
    )
    assert len(result.tool_results) == 2


def test_one_format_repair_is_allowed() -> None:
    result = BoundedToolEpisodeRunner(
        adapter=RepairAdapter(),
    ).run(
        eval_input=make_input(
            case_id="repair-success",
            available_tools=(),
            permitted_tools=(),
        ),
        fixture_state=FixtureState(
            permissions=ToolPermissions(),
        ),
    )

    assert (
        result.status
        == EpisodeTerminalStatus.COMPLETED
    )
    assert len(result.turns) == 2
    assert result.turns[0].kind == "generation"
    assert result.turns[1].kind == "format_repair"


def test_second_format_failure_exhausts_budget() -> None:
    result = BoundedToolEpisodeRunner(
        adapter=BrokenRepairAdapter(),
    ).run(
        eval_input=make_input(
            case_id="repair-failure",
            available_tools=(),
            permitted_tools=(),
        ),
        fixture_state=FixtureState(
            permissions=ToolPermissions(),
        ),
    )

    assert (
        result.status
        == EpisodeTerminalStatus.BUDGET_EXHAUSTED
    )
    assert (
        result.runtime_error_code
        == ToolErrorCode.BUDGET_EXHAUSTED
    )
    assert result.parse_failure is not None
    assert len(result.turns) == 2


def test_episode_events_are_ordered_and_timed() -> None:
    result = BoundedToolEpisodeRunner(
        adapter=SuccessfulToolAdapter(),
    ).run(
        eval_input=make_input(
            case_id="event-audit",
            available_tools=(
                ToolName.GET_RECORD,
            ),
            permitted_tools=(
                ToolName.GET_RECORD,
            ),
        ),
        fixture_state=tool_state(
            permitted=True,
        ),
    )

    assert len(result.events) == 3

    for expected, event in enumerate(
        result.events,
        start=1,
    ):
        assert event.sequence == expected
        assert event.elapsed_seconds >= 0
        assert event.event_time_utc.tzinfo is not None
