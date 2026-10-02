# MORENA-R

Reliability post-training and evaluation research for grounded and tool-using African-language models.

## Research question

Can targeted reliability interventions improve grounded generation and autonomous tool decisions in MORENA 1.5B while preserving the multilingual capabilities that make the model useful?

## Current status

Validated research state:

- G0 identity and environment qualification: PASS
- immutable MORENA 1.5B Instruct revision pinned
- immutable base-model comparator revision pinned
- immutable upstream GGUF revision pinned
- static model archaeology: PASS
- custom MORENA model import on Windows CPU: PASS
- tokenizer contract: PASS (7 tests)
- pinned `model.safetensors` SHA-256 verification: PASS
- checkpoint tensor inventory: 254 tensors
- strict checkpoint-to-architecture state-dict load: PASS
- bounded native-contract CPU inference smoke: PASS

Not yet established:

- reproducible B0 baseline
- intervention results
- protected evaluation
- model improvement

No model-improvement claim has been established.

## Core principles

- reported results are not reproduced results
- reproduced results are not improved results
- development improvement is not protected-holdout improvement
- deterministic validation precedes executable model actions
- protected evaluation data must remain isolated from training and model selection
- every result must resolve to model, data, code, configuration, runtime and scoring identities
- negative and inconclusive findings are valid research outcomes

## Primary subject

Primary instruct model:

- repository: `vamboai/morena-1.5b-instruct`
- revision: `b4be1225c9b593ffa79f2bf46d8a84a83d385c67`

Base-model comparator:

- repository: `vamboai/morena-1.5b-base`
- revision: `e98192f5f3ddb97d118e107ece37399f3cbe84d2`

Upstream GGUF reference:

- repository: `vamboai/morena-1.5b-instruct-gguf`
- revision: `ed64de7262b6edbe1af14c340ed4ea368a36349d`

Qualified experiments must resolve to these immutable identities unless a later gate explicitly records a new model identity.

## Environment

Primary local development environment:

- Windows 11 Pro
- PowerShell 5.1
- Python 3.11 virtual environment
- local CPU development
- no detected local NVIDIA CUDA GPU

GPU notebook environments remain unqualified until separately inspected.
