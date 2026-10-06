# MORENA-R Initial B0 Pilot Baseline Report

- Run ID: `b0-initial-decision-pilot-v1-d1b4c6b61cb3a1aa`
- Execution commit: `cf03e18edbe91567efc9252ef242b3dc9d645b1a`
- Runtime profile: `kaggle-t4-bf16-sdpa-v1`
- Model revision: `b4be1225c9b593ffa79f2bf46d8a84a83d385c67`
- Prompt version: `b0-minimal-json-v1`
- Scope: four-case development diagnostic; not final G5 baseline
- Review status: harness-only fixtures

## Case outcomes

| Case | Expected decision | Observed decision | Attempt status | Failure labels |
|---|---|---|---|---|
| `ref-respond-supported` | RESPOND | — | invalid_response | invalid_schema, parser_failure |
| `ref-ask-clarification` | ASK_CLARIFICATION | — | invalid_response | invalid_schema, parser_failure |
| `ref-refuse` | REFUSE | — | invalid_response | invalid_schema, parser_failure |
| `ref-tool-correct` | CALL_TOOL | — | invalid_response | invalid_schema, parser_failure |

## Deterministic metrics

| Metric | Eligible | Missing | Estimate |
|---|---:|---:|---:|
| `action_correct` | 4 | 0 | 0.000000 |
| `answerability_correct` | 4 | 0 | 0.000000 |
| `schema_valid` | 4 | 0 | 0.000000 |
| `tool_identity_correct` | 1 | 3 | 0.000000 |

## Interpretation limits

This four-case run is diagnostic evidence for the initial B0 execution path. It is not a statistically sufficient estimate of final multilingual reliability, does not complete G5, and is not protected evaluation.

KQ0 qualified the compute profile. This report does not claim model improvement, CPU/GPU token equivalence, or native no-emulation T4 BF16 support.
