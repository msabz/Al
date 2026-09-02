#!/usr/bin/env python3
"""Entry point for the integer-only deployment-compatible proof run.

The generic trainer defines LayerNorm as an optional stabilization component. The
mobile target intentionally disables it so the recurrent path contains only
fixed-scale quantizable operations: embedding, INT8 linear MACs (INT32 accumulators),
LIF state update, clamp/requantize, and vocabulary projection.
"""
from torch import nn
import code_model.train_rsnn_code_v1 as trainer

# Replace normalization construction before model creation. This leaves no LayerNorm
# parameters or floating mean/variance operation in the trained/exported graph.
trainer.nn.LayerNorm = lambda _hidden: nn.Identity()

if __name__ == "__main__":
    trainer.main()
