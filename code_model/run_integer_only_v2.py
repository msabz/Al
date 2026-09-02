#!/usr/bin/env python3
"""Faster deployment-compatible Open-Growth RSNN Code V2 training entry point.

The V1 implementation fake-quantized every recurrent weight matrix again at every token.
That is mathematically unnecessary because weights are constant during one forward pass and
becomes prohibitively expensive for multi-million-parameter runs. V2 computes the same
fake-INT8 effective matrices once per forward sequence and reuses them at each recurrent
step. Gradients still accumulate through the shared quantized weights.

LayerNorm is disabled so the target recurrent inference path remains fixed-scale and
integer-friendly: INT8 weights/activations/state with INT32 MAC accumulation on deployment.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

import code_model.train_rsnn_code_v1 as trainer

# Keep the deployment target free of floating normalization statistics.
trainer.nn.LayerNorm = lambda _hidden: nn.Identity()


def optimized_forward(self: trainer.OpenGrowthRsnnCode, ids: torch.Tensor, probe: bool = False) -> torch.Tensor:
    b, t = ids.shape
    emb = trainer.fake_quant_int8(self.embedding(ids), clip=4.0)

    # One fake-INT8 materialization per layer and forward sequence, not per token.
    eff = []
    for layer in self.layers:
        wi = layer.effective(layer.w_in, layer.mask_in, probe)
        wr = layer.effective(layer.w_rec, layer.mask_rec, probe)
        eff.append((wi, wr))

    mem = [torch.zeros(b, self.hidden, device=ids.device, dtype=emb.dtype) for _ in self.layers]
    syn = [torch.zeros_like(mem[0]) for _ in self.layers]
    spk = [torch.zeros_like(mem[0]) for _ in self.layers]
    outputs = []

    for ti in range(t):
        x = emb[:, ti]
        for li, layer in enumerate(self.layers):
            wi, wr = eff[li]
            cur = F.linear(x, wi) + F.linear(spk[li], wr)
            syn[li] = trainer.fake_quant_int8(layer.syn_decay * syn[li] + cur, clip=layer.mem_clip * 2.0)
            pre = layer.mem_decay * mem[li] + syn[li]
            spk[li] = trainer.spike_fn(pre - layer.threshold)
            mem[li] = trainer.fake_quant_int8(pre - spk[li] * layer.threshold, clip=layer.mem_clip)
            x = spk[li]
        y = trainer.fake_quant_int8(self.norm(mem[-1]), clip=4.0)
        outputs.append(self.head(y))

    return torch.stack(outputs, dim=1)


trainer.OpenGrowthRsnnCode.forward = optimized_forward


if __name__ == "__main__":
    trainer.main()
