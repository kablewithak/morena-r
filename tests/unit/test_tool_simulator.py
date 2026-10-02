from __future__ import annotations

from datetime import date

from morena_r.contracts.actions import (
    CalculateArgs,
    CalculateCall,
    CalculationOperation,
    GetRecordArgs,
    GetRecordCall,
    LookupStatusArgs,
    LookupStatusCall,
    RecordFixture,
    SearchRecordsArgs,
    SearchRecordsCall,
    StatusFixture,
    ToolErrorCode,
    ToolName,
)
from morena_r.tools.simulator import (
    FixtureState,
    FixtureToolSimulator,
    ToolFault,
    ToolPermissions,
    canonical_tool_call,
)


def fixture_state() -> FixtureState:
    return FixtureState(
        permissions=ToolPermissions(
            search_records=True,
            get_record=True,
            calculate=True,
            lookup_status=True,
        ),
        records=(
            RecordFixture(
                record_id="record-001",
                text="Johannesburg office opened in 2024.",
                observed_at=date(2024, 5, 1),
            ),
            RecordFixture(
                record_id="record-002",
                text="Johannesburg office moved in 2025.",
                observed_at=date(2025, 6, 1),
            ),
        ),
        statuses=(
            StatusFixture(
                subject_id="case-001",
                status="OPEN",
                observed_at=date(2026, 1, 15),
            ),
        ),
    )


def test_get_record_success_has_payload_hash() -> None:
    simulator = FixtureToolSimulator(fixture_state())

    call = GetRecordCall(
        tool="get_record",
        arguments=GetRecordArgs(
            record_id="record-001",
        ),
    )

    first = simulator.execute(
        call_id="call-001",
        call=call,
    )

    second = simulator.execute(
        call_id="call-002",
        call=call,
    )

    assert first.success is True
    assert first.payload_hash is not None
    assert first.payload_hash == second.payload_hash


def test_missing_record_is_typed_failure() -> None:
    simulator = FixtureToolSimulator(fixture_state())

    result = simulator.execute(
        call_id="call-001",
        call=GetRecordCall(
            tool="get_record",
            arguments=GetRecordArgs(
                record_id="missing",
            ),
        ),
    )

    assert result.success is False
    assert result.error_code == ToolErrorCode.NOT_FOUND


def test_permission_denial_prevents_execution() -> None:
    state = FixtureState(
        permissions=ToolPermissions(),
    )

    simulator = FixtureToolSimulator(state)

    result = simulator.execute(
        call_id="call-001",
        call=LookupStatusCall(
            tool="lookup_status",
            arguments=LookupStatusArgs(
                subject_id="case-001",
            ),
        ),
    )

    assert result.success is False
    assert result.error_code == ToolErrorCode.PERMISSION_DENIED


def test_search_respects_as_of_date() -> None:
    simulator = FixtureToolSimulator(fixture_state())

    result = simulator.execute(
        call_id="call-001",
        call=SearchRecordsCall(
            tool="search_records",
            arguments=SearchRecordsArgs(
                query="Johannesburg",
                as_of_date=date(2024, 12, 31),
            ),
        ),
    )

    assert result.success is True
    assert result.data is not None
    assert result.data.kind == "search_records"
    assert len(result.data.records) == 1
    assert result.data.records[0].record_id == "record-001"


def test_division_by_zero_is_argument_failure() -> None:
    simulator = FixtureToolSimulator(fixture_state())

    result = simulator.execute(
        call_id="call-001",
        call=CalculateCall(
            tool="calculate",
            arguments=CalculateArgs(
                operation=CalculationOperation.DIVIDE,
                left=10,
                right=0,
            ),
        ),
    )

    assert result.success is False
    assert result.error_code == ToolErrorCode.ARGUMENT_INVALID


def test_deterministic_fault_fixture() -> None:
    state = FixtureState(
        permissions=ToolPermissions(
            lookup_status=True,
        ),
        faults=(
            ToolFault(
                tool=ToolName.LOOKUP_STATUS,
                error_code=ToolErrorCode.TIMEOUT,
            ),
        ),
    )

    simulator = FixtureToolSimulator(state)

    result = simulator.execute(
        call_id="call-001",
        call=LookupStatusCall(
            tool="lookup_status",
            arguments=LookupStatusArgs(
                subject_id="case-001",
            ),
        ),
    )

    assert result.success is False
    assert result.error_code == ToolErrorCode.TIMEOUT


def test_canonical_tool_call_is_stable() -> None:
    first = GetRecordCall(
        tool="get_record",
        arguments=GetRecordArgs(
            record_id="record-001",
        ),
    )

    second = GetRecordCall(
        tool="get_record",
        arguments=GetRecordArgs(
            record_id="record-001",
        ),
    )

    assert canonical_tool_call(first) == canonical_tool_call(second)
