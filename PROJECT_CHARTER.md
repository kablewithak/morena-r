# MORENA-R Project Charter

## Mission

Investigate whether targeted reliability post-training can improve grounded generation and autonomous tool decisions in MORENA 1.5B without materially degrading multilingual language capability.

## Primary research subject

Vambo AI MORENA 1.5B Instruct.

The exact immutable upstream revision remains to be established during G0.

## North star

Produce an inspectable research package in which another engineer can:

1. identify the exact model and data revisions;
2. reproduce principal evaluation workflows;
3. inspect failures and exclusions;
4. compare frozen baselines with targeted interventions;
5. understand multilingual and deployment trade-offs;
6. determine why a release candidate did or did not qualify.

## Primary objectives

1. Grounded generation reliability.
2. Autonomous tool-policy reliability.
3. Multilingual capability preservation.
4. Reproducible evaluation and release evidence.

## Non-goals

MORENA-R v1 is not:

- a generic chatbot;
- a production customer-support service;
- a broad agent platform;
- a reproduction of foundation-model pretraining;
- a requirement to cover every African language;
- a cloud infrastructure project;
- a claim of improvement before measurement.

## Resource rule

Additional infrastructure spending begins at R0.

Paid compute or services require an explicit decision based on an unresolved research question and exhausted reasonable free/local alternatives.

## Research-state vocabulary

Use the following states precisely:

- proposed
- implemented
- observed
- validated
- verified
- failed
- inconclusive
- blocked

Do not promote a claim to a stronger state without evidence.

## Initial local environment

- OS: Windows 11 Pro 64-bit
- Shell: Windows PowerShell 5.1
- CPU: Intel Core i7-10510U
- CPU cores: 4 physical / 8 logical
- RAM: 15.76 GiB
- C: free storage at initial inspection: 58.3 GiB
- GPU: Intel UHD Graphics
- NVIDIA CUDA runtime detected: no
- Git: 2.51.0.windows.1
- Python selected for project: 3.11
- uv available: yes

## Gate status

G0 Identity and environment: IN PROGRESS.

G1 through G11: NOT YET ATTEMPTED.
