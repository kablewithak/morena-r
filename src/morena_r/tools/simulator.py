from __future__ import annotations

import hashlib
import json
import math

from pydantic import ConfigDict

from morena_r.contracts.actions import (
    CalculateCall,
    CalculateData,
    CalculationOperation,
    GetRecordCall,
    GetRecordData,
    LookupStatusCall,
    LookupStatusData,
    RecordFixture,
    SearchRecordsCall,
    SearchRecordsData,
    StatusFixture,
    StrictContract,
    ToolCall,
    ToolErrorCode,
    ToolName,
    ToolResult,
)


class ToolPermissions(StrictContract):
    search_records: bool = False
    get_record: bool = False
    calculate: bool = False
    lookup_status: bool = False

    def allows(self, tool: ToolName) -> bool:
        permissions = {
            ToolName.SEARCH_RECORDS: self.search_records,
            ToolName.GET_RECORD: self.get_record,
            ToolName.CALCULATE: self.calculate,
            ToolName.LOOKUP_STATUS: self.lookup_status,
        }

        return permissions[tool]


class ToolFault(StrictContract):
    tool: ToolName
    error_code: ToolErrorCode


class FixtureState(StrictContract):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    permissions: ToolPermissions

    records: tuple[RecordFixture, ...] = ()
    statuses: tuple[StatusFixture, ...] = ()
    faults: tuple[ToolFault, ...] = ()


def canonical_tool_call(call: ToolCall) -> str:
    payload = call.model_dump(
        mode="json",
        exclude_none=True,
    )

    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _payload_hash(data: StrictContract) -> str:
    canonical = json.dumps(
        data.model_dump(
            mode="json",
            exclude_none=True,
        ),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )

    return hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()


class FixtureToolSimulator:
    def __init__(self, state: FixtureState) -> None:
        self._state = state

    def execute(
        self,
        *,
        call_id: str,
        call: ToolCall,
    ) -> ToolResult:
        tool = ToolName(call.tool)

        if not self._state.permissions.allows(tool):
            return self._failure(
                call_id=call_id,
                tool=tool,
                error_code=ToolErrorCode.PERMISSION_DENIED,
            )

        fault = next(
            (
                fixture
                for fixture in self._state.faults
                if fixture.tool == tool
            ),
            None,
        )

        if fault is not None:
            return self._failure(
                call_id=call_id,
                tool=tool,
                error_code=fault.error_code,
            )

        if isinstance(call, SearchRecordsCall):
            return self._search_records(
                call_id=call_id,
                call=call,
            )

        if isinstance(call, GetRecordCall):
            return self._get_record(
                call_id=call_id,
                call=call,
            )

        if isinstance(call, CalculateCall):
            return self._calculate(
                call_id=call_id,
                call=call,
            )

        if isinstance(call, LookupStatusCall):
            return self._lookup_status(
                call_id=call_id,
                call=call,
            )

        raise TypeError(
            f"Unsupported typed tool call: {type(call)!r}"
        )

    def _search_records(
        self,
        *,
        call_id: str,
        call: SearchRecordsCall,
    ) -> ToolResult:
        query = call.arguments.query.casefold()

        matches = [
            record
            for record in self._state.records
            if query in record.text.casefold()
            and (
                call.arguments.as_of_date is None
                or record.observed_at
                <= call.arguments.as_of_date
            )
        ]

        matches.sort(
            key=lambda record: (
                record.observed_at,
                record.record_id,
            ),
            reverse=True,
        )

        data = SearchRecordsData(
            records=tuple(
                matches[: call.arguments.limit]
            )
        )

        return self._success(
            call_id=call_id,
            tool=ToolName.SEARCH_RECORDS,
            data=data,
        )

    def _get_record(
        self,
        *,
        call_id: str,
        call: GetRecordCall,
    ) -> ToolResult:
        record = next(
            (
                candidate
                for candidate in self._state.records
                if candidate.record_id
                == call.arguments.record_id
            ),
            None,
        )

        if record is None:
            return self._failure(
                call_id=call_id,
                tool=ToolName.GET_RECORD,
                error_code=ToolErrorCode.NOT_FOUND,
            )

        data = GetRecordData(
            record=record,
        )

        return self._success(
            call_id=call_id,
            tool=ToolName.GET_RECORD,
            data=data,
        )

    def _calculate(
        self,
        *,
        call_id: str,
        call: CalculateCall,
    ) -> ToolResult:
        operation = call.arguments.operation
        left = call.arguments.left
        right = call.arguments.right

        if (
            operation == CalculationOperation.DIVIDE
            and right == 0
        ):
            return self._failure(
                call_id=call_id,
                tool=ToolName.CALCULATE,
                error_code=ToolErrorCode.ARGUMENT_INVALID,
            )

        operations = {
            CalculationOperation.ADD: lambda: left + right,
            CalculationOperation.SUBTRACT: lambda: left - right,
            CalculationOperation.MULTIPLY: lambda: left * right,
            CalculationOperation.DIVIDE: lambda: left / right,
        }

        result = operations[operation]()

        if not math.isfinite(result):
            return self._failure(
                call_id=call_id,
                tool=ToolName.CALCULATE,
                error_code=ToolErrorCode.ARGUMENT_INVALID,
            )

        data = CalculateData(
            result=result,
        )

        return self._success(
            call_id=call_id,
            tool=ToolName.CALCULATE,
            data=data,
        )

    def _lookup_status(
        self,
        *,
        call_id: str,
        call: LookupStatusCall,
    ) -> ToolResult:
        status = next(
            (
                candidate
                for candidate in self._state.statuses
                if candidate.subject_id
                == call.arguments.subject_id
            ),
            None,
        )

        if status is None:
            return self._failure(
                call_id=call_id,
                tool=ToolName.LOOKUP_STATUS,
                error_code=ToolErrorCode.NOT_FOUND,
            )

        data = LookupStatusData(
            status=status,
        )

        return self._success(
            call_id=call_id,
            tool=ToolName.LOOKUP_STATUS,
            data=data,
        )

    @staticmethod
    def _success(
        *,
        call_id: str,
        tool: ToolName,
        data: StrictContract,
    ) -> ToolResult:
        return ToolResult(
            call_id=call_id,
            tool=tool,
            success=True,
            payload_hash=_payload_hash(data),
            data=data,
        )

    @staticmethod
    def _failure(
        *,
        call_id: str,
        tool: ToolName,
        error_code: ToolErrorCode,
    ) -> ToolResult:
        return ToolResult(
            call_id=call_id,
            tool=tool,
            success=False,
            error_code=error_code,
        )
