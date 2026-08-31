#!/usr/bin/env python3
"""CPU-only controlled DeepMind test for corrected onion-manifold routing.

Contract:
- official google-deepmind/mathematics_dataset polynomial_roots banks only
- six symmetric branches, three manifold-weight layers per branch
- branch coupling is derived ONLY from manifold-point intersections
- multi-point / triadic coincidence strengthens coupling
- reward/punishment changes ONLY middle-layer plasticity, never branch coupling
- reward/punishment strength adapts to correctness and deep-route engagement
- input/output calming is identity while stable and activates only near instability
- aggregation is forward identity; when calming is active its backward derivative
  follows the same adaptive compressor as the input guard
"""
import copy
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cpu_v6_manifold_lattice_deepmind as base

OFFICIAL_DATA_CONTRACT = "google-deepmind/mathematics_dataset polynomial_roots only"
SEEDS = (11, 22, 33)
STEPS = 5000
BATCH = 192
LR = 1.2e-3
WEIGHT_DECAY = 1e-5
CHECKPOINTS = {1, 100, 250, 500, 1000, 2000, 3000, 4000, 5000}
BRANCHES = 6
DEPTH = 3
WIDTH = 7
INTERSECTION_TAU = 0.22
TRIADIC_BOOST = 1.35
INPUT_TRIGGER = 2.5
INPUT_SPAN = 7.5
GRAD_TRIGGER = 2.0
GRAD_SPAN = 3.0
CALM_ALPHA_MAX = 0.35
RISK_DECAY = 0.90
DEEP_ROUTE_TARGET = 0.28
DEVICE = torch.device("cpu")


def adaptive_quiet(x, alpha):
    """Invertible compressor family; alpha=0 is exactly identity."""
    a = alpha.clamp_min(1e-6)
    compressed = torch.asinh(a * x) / a
    return torch.where(alpha > 1e-6, compressed, x)


def adaptive_unquiet(y, alpha):
    """Exact inverse family for adaptive_quiet, with a numerical safety ceiling."""
    a = alpha.clamp_min(1e-6)
    restored = torch.sinh(torch.clamp(a * y, -8.0, 8.0)) / a
    return torch.where(alpha > 1e-6, restored, y)


class ParamMatchedMLP(nn.Module):
    def __init__(self, hidden=109):
        super().__init__()
        self.fc1 = nn.Linear(base.FEATURES, hidden)
        self.fc2 = nn.Linear(hidden, base.ROOT_SLOTS * 2)

    def forward(self, x, degree):
        del degree
        y = self.fc2(F.silu(self.fc1(x)))
        return y[:, :base.ROOT_SLOTS], y[:, base.ROOT_SLOTS:]


class ManifoldLinear(nn.Module):
    """Effective scalar weights are projections of points on infinite-solution manifolds."""
    def __init__(self, dim, family_offset):
        super().__init__()
        self.dim = dim
        self.theta = nn.Parameter(torch.randn(dim, dim) * 0.35)
        self.phi = nn.Parameter(torch.randn(dim, dim) * 0.35)
        self.bias = nn.Parameter(torch.zeros(dim))
        oi = torch.arange(dim)[:, None]
        ii = torch.arange(dim)[None, :]
        self.register_buffer("family", (oi + 2 * ii + family_offset) % 4, persistent=False)

    def manifold_point(self):
        theta = self.theta
        phi = self.phi
        cp = torch.cos(phi)
        sphere = torch.stack((torch.cos(theta) * cp, torch.sin(theta) * cp, torch.sin(phi)), dim=-1)
        cylinder = torch.stack((torch.cos(theta), torch.sin(theta), torch.tanh(phi)), dim=-1)
        u = torch.tanh(theta)
        v = torch.tanh(phi)
        saddle = torch.stack((u, v, u * v), dim=-1)
        wave = torch.stack((u, v, torch.tanh(torch.sin(math.pi * u) + torch.cos(math.pi * v))), dim=-1)
        f = self.family
        p = torch.where((f == 0)[..., None], sphere, cylinder)
        p = torch.where((f == 2)[..., None], saddle, p)
        p = torch.where((f == 3)[..., None], wave, p)
        return p

    def representative_point(self):
        return self.manifold_point().mean(dim=(0, 1))

    def effective_weight(self):
        p = self.manifold_point()
        return (0.55 * p[..., 0] + 0.30 * p[..., 1] + 0.15 * p[..., 2]) / math.sqrt(self.dim)

    def forward(self, x):
        return F.linear(x, self.effective_weight(), self.bias)


class CorrectedOnionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            ManifoldLinear(WIDTH, family_offset=(chain * DEPTH + depth) % 4)
            for chain in range(BRANCHES)
            for depth in range(DEPTH)
        ])
        self.readout = nn.Linear(WIDTH, base.ROOT_SLOTS * 2)
        self.register_buffer("risk_state", torch.tensor(0.0), persistent=False)
        self.last_alpha = 0.0
        self.last_route_means = [0.0] * DEPTH
        self.last_route_max = 0.0

    def _layer(self, chain, depth):
        return self.layers[chain * DEPTH + depth]

    def _adaptive_alpha(self, x):
        peak = x.detach().abs().amax(dim=1, keepdim=True)
        input_risk = ((peak - INPUT_TRIGGER) / INPUT_SPAN).clamp(0.0, 1.0) * CALM_ALPHA_MAX
        state = self.risk_state.detach().reshape(1, 1).expand_as(input_risk)
        alpha = torch.maximum(input_risk, state)
        self.last_alpha = float(alpha.mean())
        return alpha

    @torch.no_grad()
    def update_stability(self, grad_norm):
        g = float(grad_norm)
        desired = max(0.0, min(1.0, (g - GRAD_TRIGGER) / GRAD_SPAN)) * CALM_ALPHA_MAX
        current = float(self.risk_state)
        if desired > current:
            new = desired
        else:
            new = current * RISK_DECAY
            if new < 1e-4:
                new = 0.0
        self.risk_state.fill_(new)

    def _intersection_matrix(self, depth):
        reps = torch.stack([self._layer(c, depth).representative_point() for c in range(BRANCHES)], dim=0)
        diff = reps[:, None, :] - reps[None, :, :]
        d2 = (diff * diff).sum(dim=-1)
        base_gate = torch.exp(-d2 / INTERSECTION_TAU)
        eye = torch.eye(BRANCHES, dtype=base_gate.dtype, device=base_gate.device)
        base_gate = base_gate * (1.0 - eye)
        triadic = (base_gate @ base_gate) / max(1, BRANCHES - 2)
        strength = 1.0 - torch.exp(-(base_gate + TRIADIC_BOOST * triadic))
        return strength * (1.0 - eye)

    def forward(self, x, degree):
        del degree
        alpha = self._adaptive_alpha(x)
        q = adaptive_quiet(x, alpha)
        route_means = []
        route_max = 0.0
        h = None
        for depth in range(DEPTH):
            if depth == 0:
                proposals = [F.silu(self._layer(c, depth)(q)) for c in range(BRANCHES)]
            else:
                proposals = [F.silu(self._layer(c, depth)(h[:, c, :])) for c in range(BRANCHES)]
            h = torch.stack(proposals, dim=1)
            G = self._intersection_matrix(depth)
            numerator = h + torch.einsum("ij,bjd->bid", G, h)
            denom = 1.0 + G.sum(dim=1)[None, :, None]
            h = numerator / denom
            route_means.append(float(G.detach().mean()))
            route_max = max(route_max, float(G.detach().max()))

        self.last_route_means = route_means
        self.last_route_max = route_max
        plain_mean = h.mean(dim=1)
        compressed = adaptive_quiet(plain_mean, alpha)
        aggregate = plain_mean.detach() + compressed - compressed.detach()
        y = self.readout(aggregate)
        roots = adaptive_unquiet(y[:, :base.ROOT_SLOTS], alpha)
        return roots, y[:, base.ROOT_SLOTS:]

    @torch.no_grad()
    def route_stats(self):
        mats = [self._intersection_matrix(d) for d in range(DEPTH)]
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
        }

    @torch.no_grad()
    def manifold_stats(self, initial_points):
        points = torch.stack([layer.manifold_point() for layer in self.layers], dim=0)
        d = torch.linalg.vector_norm(points - initial_points, dim=-1)
        return {"mean_point_displacement": float(d.mean()), "max_point_displacement": float(d.max())}


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def correctness_score(primary_loss):
    return float(torch.sigmoid((0.30 - primary_loss.detach()) * 12.0))


def dynamic_plasticity(model, primary_loss):
    correctness = correctness_score(primary_loss)
    deep_route = float(np.mean(model.last_route_means[1:])) if len(model.last_route_means) > 1 else 0.0
    exploration_need = max(0.0, min(1.0, (DEEP_ROUTE_TARGET - deep_route) / DEEP_ROUTE_TARGET))
    reward_lambda = 0.05 + 0.35 * (1.0 - exploration_need) * correctness
    punish_lambda = 0.10 + 0.45 * exploration_need * (1.0 - correctness)
    multipliers = []
    for chain in range(BRANCHES):
        for depth in range(DEPTH):
            layer = model._layer(chain, depth)
            depth_boost = 1.0 + 0.35 * depth
            stabilize = reward_lambda * correctness * (1.0 - 0.15 * depth)
            explore = punish_lambda * (1.0 - correctness) * depth_boost
            mult = max(0.50, min(2.00, 1.0 + explore - stabilize))
            multipliers.append(mult)
            for p in layer.parameters():
                if p.grad is not None:
                    p.grad.mul_(mult)
    return {
        "correctness": correctness,
        "deep_route": deep_route,
        "exploration_need": exploration_need,
        "reward_lambda": reward_lambda,
        "punish_lambda": punish_lambda,
        "mean_layer_grad_mult": float(np.mean(multipliers)),
        "deep_layer_grad_mult": float(np.mean([multipliers[i] for i in range(len(multipliers)) if i % DEPTH == DEPTH - 1])),
    }


def make_fresh(kind):
    if kind == "MLP":
        return ParamMatchedMLP()
    if kind in ("ONION_STATIC", "ONION_DYNAMIC"):
        return CorrectedOnionNet()
    raise ValueError(kind)


def train_one(kind, train, valid, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = make_fresh(kind).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    batch_rng = torch.Generator().manual_seed(200000 + seed)
    initial_points = None
    if isinstance(model, CorrectedOnionNet):
        initial_points = torch.stack([layer.manifold_point().detach().clone() for layer in model.layers], dim=0)
    best_score = float("inf")
    best_state = None
    curve = {}
    last_plasticity = None
    calm_activations = 0

    for step in range(1, STEPS + 1):
        ids = torch.randint(0, len(train.features), (BATCH,), generator=batch_rng)
        opt.zero_grad(set_to_none=True)
        loss = base.loss_fn(model, train.features[ids], train.roots[ids], train.degree[ids])
        loss.backward()
        if kind == "ONION_DYNAMIC":
            last_plasticity = dynamic_plasticity(model, loss)
        elif isinstance(model, CorrectedOnionNet):
            last_plasticity = {"correctness": correctness_score(loss), "deep_route": float(np.mean(model.last_route_means[1:])), "exploration_need": float("nan"), "reward_lambda": 0.0, "punish_lambda": 0.0, "mean_layer_grad_mult": 1.0, "deep_layer_grad_mult": 1.0}
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        if isinstance(model, CorrectedOnionNet):
            model.update_stability(grad_norm)
            if model.last_alpha > 1e-6:
                calm_activations += 1
        opt.step()

        if step in CHECKPOINTS:
            metrics = base.evaluate(model, valid)
            route = model.route_stats() if isinstance(model, CorrectedOnionNet) else None
            curve[step] = metrics
            score = metrics["mae"] + 50.0 * (1.0 - metrics["count"]) + 20.0 * (1.0 - metrics["within1"])
            if score < best_score:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())
            print(f"{kind} seed={seed} step={step:4d} loss={float(loss.detach()):.6f} VALID={metrics} ROUTE={route} PLASTICITY={last_plasticity}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    final = base.evaluate(model, valid)
    route = model.route_stats() if isinstance(model, CorrectedOnionNet) else None
    geo = model.manifold_stats(initial_points) if isinstance(model, CorrectedOnionNet) else None
    return final, curve, route, geo, calm_activations, last_plasticity


def main():
    print("DEVICE=cpu")
    print("DATA_CONTRACT", OFFICIAL_DATA_CONTRACT)
    print("CORRECTED_ONION_CONTRACT branches=6 depth=3 intersection_only_routes triadic_boost layer_only_reward adaptive_calm")
    dm = base.prepare_official_deepmind()
    print("Building deterministic official DeepMind banks...", flush=True)
    train = base.build_bank(dm, base.TRAIN_COUNT, 7301)
    valid = base.build_bank(dm, base.VALID_COUNT, 8301)
    print("train degree distribution", {d: int((train.degree == d).sum()) for d in range(2, 6)})
    print("valid degree distribution", {d: int((valid.degree == d).sum()) for d in range(2, 6)})
    params = {k: parameter_count(make_fresh(k)) for k in ("MLP", "ONION_STATIC", "ONION_DYNAMIC")}
    print("PARAMS", params)
    ref = params["MLP"]
    for k, n in params.items():
        if abs(n - ref) / ref > 0.05:
            raise SystemExit(f"parameter budget mismatch >5% for {k}: {n} vs {ref}")

    rows = []
    for seed in SEEDS:
        results = {}
        diagnostics = {}
        curves = {}
        for kind in ("MLP", "ONION_STATIC", "ONION_DYNAMIC"):
            result, curve, route, geo, calm_count, plast = train_one(kind, train, valid, seed)
            results[kind] = result
            curves[kind] = curve
            diagnostics[kind] = {"route": route, "geometry": geo, "calm_active_steps": calm_count, "last_plasticity": plast}
        rows.append((seed, results, diagnostics))
        print(f"SEED_SUMMARY seed={seed} RESULTS={results} DIAGNOSTICS={diagnostics}", flush=True)
        print("LATE_CURVE", seed, {s: {"mlp_mae": curves["MLP"][s]["mae"], "static_mae": curves["ONION_STATIC"][s]["mae"], "dynamic_mae": curves["ONION_DYNAMIC"][s]["mae"], "mlp_count": curves["MLP"][s]["count"], "static_count": curves["ONION_STATIC"][s]["count"], "dynamic_count": curves["ONION_DYNAMIC"][s]["count"], "mlp_within1": curves["MLP"][s]["within1"], "static_within1": curves["ONION_STATIC"][s]["within1"], "dynamic_within1": curves["ONION_DYNAMIC"][s]["within1"]} for s in sorted(CHECKPOINTS)}, flush=True)

    def avg(kind, key):
        return float(np.mean([results[kind][key] for _, results, _ in rows]))

    summary = {kind: {key: avg(kind, key) for key in ("count", "mae", "within1", "merge")} for kind in ("MLP", "ONION_STATIC", "ONION_DYNAMIC")}
    summary["delta_dynamic_vs_mlp"] = {"count": summary["ONION_DYNAMIC"]["count"] - summary["MLP"]["count"], "mae": summary["ONION_DYNAMIC"]["mae"] - summary["MLP"]["mae"], "within1": summary["ONION_DYNAMIC"]["within1"] - summary["MLP"]["within1"], "merge": summary["ONION_DYNAMIC"]["merge"] - summary["MLP"]["merge"]}
    summary["delta_dynamic_vs_static"] = {"count": summary["ONION_DYNAMIC"]["count"] - summary["ONION_STATIC"]["count"], "mae": summary["ONION_DYNAMIC"]["mae"] - summary["ONION_STATIC"]["mae"], "within1": summary["ONION_DYNAMIC"]["within1"] - summary["ONION_STATIC"]["within1"], "merge": summary["ONION_DYNAMIC"]["merge"] - summary["ONION_STATIC"]["merge"]}
    wins_mlp = sum(results["ONION_DYNAMIC"]["mae"] < results["MLP"]["mae"] and results["ONION_DYNAMIC"]["count"] >= results["MLP"]["count"] for _, results, _ in rows)
    wins_static = sum(results["ONION_DYNAMIC"]["mae"] < results["ONION_STATIC"]["mae"] and results["ONION_DYNAMIC"]["count"] >= results["ONION_STATIC"]["count"] for _, results, _ in rows)
    print("FINAL_SUMMARY", summary, flush=True)
    print(f"DYNAMIC_WINS_VS_MLP={wins_mlp}/3", flush=True)
    print(f"DYNAMIC_WINS_VS_STATIC={wins_static}/3", flush=True)
    clear = wins_mlp >= 2 and wins_static >= 2 and summary["ONION_DYNAMIC"]["mae"] < summary["MLP"]["mae"] and summary["ONION_DYNAMIC"]["count"] >= summary["MLP"]["count"]
    print("V9_CORRECTED_ONION_DEEPMIND_CLEAR_SIGNAL" if clear else "V9_CORRECTED_ONION_DEEPMIND_NO_CLEAR_SIGNAL", flush=True)


if __name__ == "__main__":
    main()
