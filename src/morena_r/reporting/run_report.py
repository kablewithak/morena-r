from __future__ import annotations

import json
from collections import defaultdict

from pydantic import Field

from morena_r.contracts.actions import (
    NonEmptyStr,
    StrictContract,
)
from morena_r.contracts.evaluation import (
    RunSummary,
)
from morena_r.evaluation.runner import (
    EvaluationRunResult,
)


class MetricAggregate(StrictContract):
    metric_id: NonEmptyStr

    total_rows: int = Field(ge=0)
    eligible_instances: int = Field(ge=0)
    missing_instances: int = Field(ge=0)

    estimate: float | None = Field(
        default=None,
        allow_inf_nan=False,
    )


class DeterministicRunReport(StrictContract):
    schema_version: str = "1.0"

    run_id: NonEmptyStr
    summary: RunSummary

    metrics: tuple[MetricAggregate, ...]


def build_run_report(
    result: EvaluationRunResult,
) -> DeterministicRunReport:
    by_metric: dict[str, list] = defaultdict(list)

    for score in result.scores:
        by_metric[score.metric_id].append(score)

    aggregates: list[MetricAggregate] = []

    for metric_id in sorted(by_metric):
        rows = by_metric[metric_id]

        values = [
            row.value
            for row in rows
            if (
                row.eligible
                and row.value is not None
            )
        ]

        estimate = (
            sum(values) / len(values)
            if values
            else None
        )

        aggregates.append(
            MetricAggregate(
                metric_id=metric_id,
                total_rows=len(rows),
                eligible_instances=len(values),
                missing_instances=sum(
                    row.value is None
                    for row in rows
                ),
                estimate=estimate,
            )
        )

    return DeterministicRunReport(
        run_id=result.config.run_id,
        summary=result.summary,
        metrics=tuple(aggregates),
    )


def canonical_report_json(
    report: DeterministicRunReport,
) -> str:
    return (
        json.dumps(
            report.model_dump(
                mode="json",
            ),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    )
