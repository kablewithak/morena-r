# MORENA-R Premise Register

## Confirmed

| Premise | Evidence status |
|---|---|
| Local development target is Windows 11 Pro | Observed locally |
| Shell is Windows PowerShell 5.1 | Observed locally |
| Git is installed | Observed locally |
| Python 3.11 is installed | Observed locally |
| Python 3.12 is installed | Observed locally |
| Python 3.13 is installed | Observed locally |
| uv is installed | Observed locally |
| Local RAM is 15.76 GiB | Observed locally |
| Initial C: free storage is 58.3 GiB | Observed locally |
| CPU is Intel Core i7-10510U, 4C/8T | Observed locally |
| Intel UHD Graphics is present | Observed locally |
| nvidia-smi is not available | Observed locally |
| Local Git repository began without an existing remote | Observed locally |

## Project decisions

| Premise | Status |
|---|---|
| Primary research subject is MORENA 1.5B Instruct | Project decision |
| Python 3.11 is the initial repository runtime | Project decision |
| Additional infrastructure spending starts at R0 | Project decision |
| Model improvement is a hypothesis, not an assumption | Project decision |
| Protected evaluation material must remain outside ordinary development access | Project decision |

## Unverified

| Premise | Required verification |
|---|---|
| Exact MORENA 1.5B Instruct revision | Pin immutable upstream commit and required files |
| Exact tokenizer identity | Hash inspected tokenizer files |
| Correct native chat template | Inspect upstream implementation and fixtures |
| Reference PyTorch loading compatibility | G1 smoke test |
| Local CPU inference feasibility | Measure memory, correctness and throughput |
| Local GGUF runtime availability | Inspect before prescribing |
| Kaggle availability and accelerator type | Inspect account/session when required |
| LoRA compatibility | G6 forward/backward/save/reload proof |
| QLoRA compatibility | G6 only if required and supported |
| Reviewer availability | G3 |
| Public benchmark overlap/contamination | Record and qualify |
| Protected-holdout custodian availability | G3/G4 |
