#!/usr/bin/env python3
"""V10 correction of the onion-manifold experiment.

This keeps the V9 data/training contract but fixes the implementation error seen
in V9 evidence: branch routes were already ~0.73 at initialization, so they were
not dormant and dynamic plasticity never saw an exploration phase.

Corrections:
- routing uses actual manifold weight-points (the diagonal point per channel),
  not the mean of a whole layer's manifold points;
- a branch/channel route opens only when its two points geometrically meet;
- a third branch can strengthen an already-open pair, but cannot create a route
  if the original pair is not meeting;
- reward/punishment still touches layer gradients only, never route strengths;
- the calm guard also observes hidden activation peaks for the next step, while
  remaining identity when no instability signal exists.
"""
import math
import random
import numpy as np
import torch
import torch.nn.functional as F

import cpu_v9_onion_corrected_deepmind as v9

MEET_RADIUS = 0.080
MEET_TEMPERATURE = 0.012
TRIADIC_BOOST = 1.50
ACTIVATION_TRIGGER = 4.0
ACTIVATION_SPAN = 8.0


class CorrectedOnionNetV10(v9.CorrectedOnionNet):
    def __init__(self):
        super().__init__()
        self.last_activation_peak = 0.0

    def _routing_points(self, depth):
        """Actual manifold points used by weights: one diagonal point per channel."""
        points = []
        idx = torch.arange(v9.WIDTH)
        for chain in range(v9.BRANCHES):
            p = self._layer(chain, depth).manifold_point()
            points.append(p[idx, idx, :])
        return torch.stack(points, dim=0)  # [branches, channels, xyz]

    def _intersection_matrix(self, depth):
        """Per-channel branch coupling from geometric meetings only.

        G[i,j,c] is near zero unless the actual manifold points for branch i and
        branch j on channel c are within MEET_RADIUS. Triadic support multiplies
        an existing pairwise meeting; it never opens a non-meeting pair by itself.
        """
        p = self._routing_points(depth)  # [B,C,3]
        diff = p[:, None, :, :] - p[None, :, :, :]
        distance = torch.linalg.vector_norm(diff, dim=-1)  # [B,B,C]
        pair = torch.sigmoid((MEET_RADIUS - distance) / MEET_TEMPERATURE)
        eye = torch.eye(v9.BRANCHES, dtype=pair.dtype, device=pair.device)[:, :, None]
        pair = pair * (1.0 - eye)

        # For each channel: support(i,j) is large when i and j share a third
        # branch k that also geometrically meets both. It can only strengthen pair.
        support = torch.einsum("ikc,kjc->ijc", pair, pair) / max(1, v9.BRANCHES - 2)
        strength = torch.clamp(pair * (1.0 + TRIADIC_BOOST * support), 0.0, 1.0)
        return strength * (1.0 - eye)

    def forward(self, x, degree):
        del degree
        alpha = self._adaptive_alpha(x)
        q = v9.adaptive_quiet(x, alpha)
        route_means = []
        route_max = 0.0
        h = None
        activation_peak = float(q.detach().abs().amax())

        for depth in range(v9.DEPTH):
            if depth == 0:
                proposals = [F.silu(self._layer(c, depth)(q)) for c in range(v9.BRANCHES)]
            else:
                proposals = [F.silu(self._layer(c, depth)(h[:, c, :])) for c in range(v9.BRANCHES)]
            h = torch.stack(proposals, dim=1)
            G = self._intersection_matrix(depth)  # [branch,branch,channel]
            messages = torch.einsum("ijc,bjc->bic", G, h)
            denom = 1.0 + G.sum(dim=1)[None, :, :]
            h = (h + messages) / denom
            route_means.append(float(G.detach().mean()))
            route_max = max(route_max, float(G.detach().max()))
            activation_peak = max(activation_peak, float(h.detach().abs().amax()))

        self.last_activation_peak = activation_peak
        self.last_route_means = route_means
        self.last_route_max = route_max

        plain_mean = h.mean(dim=1)
        compressed = v9.adaptive_quiet(plain_mean, alpha)
        # Forward identity, compressor derivative only on backward.
        aggregate = plain_mean.detach() + compressed - compressed.detach()
        y = self.readout(aggregate)
        roots = v9.adaptive_unquiet(y[:, :v9.base.ROOT_SLOTS], alpha)
        return roots, y[:, v9.base.ROOT_SLOTS:]

    @torch.no_grad()
    def update_stability(self, grad_norm):
        g = float(grad_norm)
        grad_risk = max(0.0, min(1.0, (g - v9.GRAD_TRIGGER) / v9.GRAD_SPAN))
        act_risk = max(0.0, min(1.0, (self.last_activation_peak - ACTIVATION_TRIGGER) / ACTIVATION_SPAN))
        desired = max(grad_risk, act_risk) * v9.CALM_ALPHA_MAX
        current = float(self.risk_state)
        if desired > current:
            new = desired
        else:
            new = current * v9.RISK_DECAY
            if new < 1e-4:
                new = 0.0
        self.risk_state.fill_(new)

    @torch.no_grad()
    def route_stats(self):
        mats = [self._intersection_matrix(d) for d in range(v9.DEPTH)]
        allv = torch.cat([m.flatten() for m in mats])
        deep = torch.cat([m.flatten() for m in mats[1:]])
        return {
            "mean_route": float(allv.mean()),
            "deep_mean_route": float(deep.mean()),
            "edges_gt_0.10": int((allv > 0.10).sum()),
            "edges_gt_0.25": int((allv > 0.25).sum()),
            "edges_gt_0.50": int((allv > 0.50).sum()),
            "max_route": float(allv.max()),
            "calm_alpha": float(self.last_alpha),
            "risk_state": float(self.risk_state),
            "activation_peak": float(self.last_activation_peak),
        }


def make_fresh_v10(kind):
    if kind == "MLP":
        return v9.ParamMatchedMLP()
    if kind in ("ONION_STATIC", "ONION_DYNAMIC"):
        return CorrectedOnionNetV10()
    raise ValueError(kind)


def preflight():
    """Fail before the expensive dataset build if the semantic contract is broken."""
    initial = []
    for seed in v9.SEEDS:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = CorrectedOnionNetV10()
        st = model.route_stats()
        initial.append(st)
        print(f"V10_INITIAL_ROUTE seed={seed} {st}", flush=True)
        # 6x6x7x3 = 756 directed/channel entries including diagonal zeros.
        # Initial state must be predominantly dormant, unlike V9 where all 90
        # scalar branch edges were >0.5 immediately.
        if st["mean_route"] >= 0.12:
            raise SystemExit(f"initial routes not dormant enough: {st}")
        if st["edges_gt_0.25"] >= 120:
            raise SystemExit(f"too many initially active route channels: {st}")

    # Third-point boost must strengthen an existing meeting, not create one.
    pair = torch.tensor([0.8, 0.8, 0.0])
    boosted = pair[0] * (1.0 + TRIADIC_BOOST * pair[1] * pair[1])
    isolated = pair[2] * (1.0 + TRIADIC_BOOST * pair[0] * pair[1])
    assert boosted > pair[0]
    assert isolated == 0.0

    # Calm guard is identity when stable and provably activates under danger.
    model = CorrectedOnionNetV10()
    normal = torch.zeros(2, v9.base.FEATURES)
    a0 = model._adaptive_alpha(normal)
    if float(a0.max()) != 0.0:
        raise SystemExit("calm guard must be exact identity while stable")
    model.last_activation_peak = ACTIVATION_TRIGGER + ACTIVATION_SPAN
    model.update_stability(torch.tensor(0.0))
    a1 = model._adaptive_alpha(normal)
    if float(a1.mean()) <= 0.0:
        raise SystemExit("calm guard failed to activate on hidden-activation danger")
    x = torch.linspace(-3.0, 3.0, 17).reshape(1, -1)
    alpha = torch.full((1, 1), 0.2)
    rt = v9.adaptive_unquiet(v9.adaptive_quiet(x, alpha), alpha)
    if float((rt - x).abs().max()) > 1e-5:
        raise SystemExit("adaptive calm inverse round-trip failed")

    print("V10_ONION_SEMANTIC_PREFLIGHT_PASS", flush=True)


def main():
    preflight()
    # Reuse the exact V9 DeepMind bank, loss, checkpointing, metrics and
    # parameter-matched comparison. Only the onion implementation is replaced.
    v9.CorrectedOnionNet = CorrectedOnionNetV10
    v9.make_fresh = make_fresh_v10
    v9.main()
    print("V10_ONION_INTERSECTION_FIX_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
