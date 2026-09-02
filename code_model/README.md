# Open-Growth RSNN Code V1

This branch is a focused Python-code generation experiment derived from the Open-Growth RSNN V2 design.

## Goal

Given a short natural-language instruction, generate Python source code with a small recurrent spiking model suitable for later INT8 mobile inference.

## Real training data only

`collect_real_python.py` does not fabricate tasks or source code. It shallow-clones permissively licensed public Python repositories and pairs actual upstream docstrings with the exact corresponding function/class source at a pinned commit. CI writes a `provenance.json` containing repository, commit and license information.

Current training repositories:

- `psf/requests` — Apache-2.0
- `pallets/flask` — BSD-3-Clause
- `encode/httpx` — BSD-3-Clause
- `pytest-dev/pytest` — MIT
- `fastapi/fastapi` — MIT

Validation is repository-held-out on `pydantic/pydantic` — MIT. This prevents function-level leakage from the same repository into validation.

## Model

- UTF-8 byte tokenizer: 259 symbols including BOS/EOS/PAD.
- Embedding -> stacked sparse LIF recurrent layers -> vocabulary head.
- Surrogate-gradient spiking training.
- FP32 master weights + AdamW.
- Fake INT8 quantization of weights, embeddings, recurrent membrane/synaptic state and readout activations during training.
- Initial sparse topology.
- Structural probe of dormant connections.
- Growth phase followed by rolling selection: current best connections protected, current weak unprotected connections pruned, useful dormant connections regrown.
- Exported deployment bundle stores INT8 weights, sparse masks and per-tensor scales. Integer deployment uses INT32 MAC accumulators followed by requantization.

## Scope of the first run

The first CI run is deliberately a proof run, not a claim of production coding quality. It checks that real data collection, recurrent language-model training, structural growth/pruning, validation loss and INT8 export all work end-to-end. If the loss and held-out validation move in the right direction, the next step is a longer GPU run and Android inference integration.
