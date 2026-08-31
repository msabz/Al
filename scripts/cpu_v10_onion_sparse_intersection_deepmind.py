#!/usr/bin/env python3
"""CPU-only controlled DeepMind test for the corrected onion architecture.

This version fixes the v9 routing defect proven by evidence: v9 averaged every
layer's manifold point cloud into one representative point, which made all
90 cross-branch routes strongly active at initialization. Here routes are
STRICTLY geometric and dormant unless actual manifold points meet.

Contract:
- official google-deepmind/mathematics_dataset polynomial_roots only
- 6 symmetric branches x 3 manifold-weight layers
- branch routes have NO trainable gates and receive NO reward/punishment term
- a route activates only from mutual-nearest point coincidences
- a third/fourth/etc. point near the same meeting strengthens that route
- reward/punishment only rescales middle-layer plasticity
- adaptive calming is exact identity while stable and activates only on risk
"""
import copy
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cpu_v9_onion_corrected_deepmind as v9

base = v9.base
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
MEET_RADIUS = 0.035
MEET_TEMP = 0.010
PAIR_GAIN = 0.45
MULTIPOINT_BOOST = 0.65
DEVICE = torch.device("cpu")


class SpreadManifoldLinear(v9.ManifoldLinear):
    """Same infinite-solution manifolds, but points start spread over them."""
    def __init__(self, dim, family_offset):
        super().__init__(dim, family_offset)
        with torch.no_grad():
            self.theta.uniform_(-math.pi, math.pi)
            self.phi.uniform_(-math.pi, math.pi)


def _mutual_matches(pi, pj):
    """Return mutual-nearest (distance, midpoint) pairs; selected distances stay differentiable."""
    dist = torch.cdist(pi, pj)
    j_of_i = dist.argmin(dim=1)
    i_of_j = dist.argmin(dim=0)
    rows = torch.arange(len(pi), device=pi.device)
    mutual = i_of_j[j_of_i] == rows
    rows = rows[mutual]
    cols = j_of_i[mutual]
    if len(rows) == 0:
        return [], []
    ds = dist[rows, cols]
    mids = (pi[rows] + pj[cols]) * 0.5
    return list(ds.unbind(0)), list(mids.unbind(0))


def intersection_matrix_from_clouds(clouds):
    """Geometric route matrix.

    One close mutual-nearest point pair creates a weak/medium route. Additional
    branches with a point at that SAME midpoint strengthen the existing route.
    No trainable gate, reward, correctness or loss enters this function.
    """
    n = len(clouds)
    rows = []
    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                row.append(clouds[i].new_tensor(0.0))
                continue
            ds, mids = _mutual_matches(clouds[i], clouds[j])
            evidence = clouds[i].new_tensor(0.0)
            for d, mid in zip(ds, mids):
                pair = torch.sigmoid((MEET_RADIUS - d) / MEET_TEMP)
                support = clouds[i].new_tensor(0.0)
                for k in range(n):
                    if k == i or k == j:
                        continue
                    d3 = torch.linalg.vector_norm(clouds[k] - mid[None, :], dim=1).min()
                    support = support + torch.sigmoid((MEET_RADIUS - d3) / MEET_TEMP)
                evidence = evidence + pair * (1.0 + MULTIPOINT_BOOST * support)
            row.append(1.0 - torch.exp(-PAIR_GAIN * evidence))
        rows.append(torch.stack(row))
    G = torch.stack(rows)
    return 0.5 * (G + G.T)


class SparseIntersectionOnion(v9.CorrectedOnionNet):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            SpreadManifoldLinear(WIDTH, family_offset=(chain * DEPTH + depth) % 4)
            for chain in range(BRANCHES)
            for depth in range(DEPTH)
        ])

    def _intersection_matrix(self, depth):
        clouds = [
            self._layer(c, depth).manifold_point().reshape(-1, 3)
            for c in range(BRANCHES)
        ]
        return intersection_matrix_from_clouds(clouds)


def make_fresh(kind):
    if kind == "MLP":
        return v9.ParamMatchedMLP()
    if kind in ("ONION_STATIC", "ONION_DYNAMIC"):
        return SparseIntersectionOnion()
    raise ValueError(kind)


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def preflight():
    print("V10_PREFLIGHT_BEGIN")

    far = []
    for b in range(BRANCHES):
        x = torch.arange(8, dtype=torch.float32)[:, None]
        far.append(torch.cat((x + 10.0 * b, torch.zeros(8, 2)), dim=1))
    Gfar = intersection_matrix_from_clouds(far)
    assert float(Gfar.max()) < 1e-3, float(Gfar.max())

    pair_clouds = [x.clone() for x in far]
    pair_clouds[0] = torch.cat((torch.tensor([[0.0, 0.0, 0.0]]), far[0][1:]), dim=0)
    pair_clouds[1] = torch.cat((torch.tensor([[0.0, 0.0, 0.0]]), far[1][1:]), dim=0)
    G2 = intersection_matrix_from_clouds(pair_clouds)
    assert float(G2[0, 1]) > 0.20, float(G2[0, 1])

    triad_clouds = [x.clone() for x in pair_clouds]
    triad_clouds[2] = torch.cat((torch.tensor([[0.0, 0.0, 0.0]]), far[2][1:]), dim=0)
    G3 = intersection_matrix_from_clouds(triad_clouds)
    assert float(G3[0, 1]) > float(G2[0, 1]) + 0.05, (float(G2[0, 1]), float(G3[0, 1]))
    print("GEOMETRY_CONTRACT", {"far_max": float(Gfar.max()), "pair": float(G2[0,1]), "with_third": float(G3[0,1])})

    init_stats = []
    for seed in range(12):
        torch.manual_seed(9000 + seed)
        m = SparseIntersectionOnion()
        init_stats.append(m.route_stats())
    mean_route = float(np.mean([s["mean_route"] for s in init_stats]))
    max_edge_fraction = max(s["edges_gt_0.25"] for s in init_stats) / float(DEPTH * BRANCHES * BRANCHES)
    print("INITIAL_DORMANCY", {"mean_route": mean_route, "max_edge_fraction_gt_0.25": max_edge_fraction, "samples": init_stats})
    assert mean_route < 0.10, mean_route
    assert max_edge_fraction < 0.20, max_edge_fraction

    torch.manual_seed(123)
    m = SparseIntersectionOnion()
    normal = torch.zeros(4, base.FEATURES)
    a0 = m._adaptive_alpha(normal)
    assert float(a0.max()) == 0.0
    huge = torch.full((4, base.FEATURES), 100.0)
    a1 = m._adaptive_alpha(huge)
    assert float(a1.min()) > 0.0
    m.update_stability(10.0)
    assert float(m.risk_state) > 0.0
    for _ in range(100):
        m.update_stability(0.0)
    assert float(m.risk_state) == 0.0
    print("CALMING_CONTRACT", {"normal_alpha": float(a0.max()), "stress_alpha": float(a1.min()), "released_risk": float(m.risk_state)})

    names = [n for n, _ in m.named_parameters()]
    assert not any("gate" in n.lower() or "route" in n.lower() for n in names), names
    print("NO_TRAINABLE_ROUTE_GATES", True)
    print("V10_PREFLIGHT_PASS")


def train_one(kind, train, valid, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = make_fresh(kind).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    batch_rng = torch.Generator().manual_seed(300000 + seed)
    initial_points = None
    if isinstance(model, SparseIntersectionOnion):
        initial_points = torch.stack([layer.manifold_point().detach().clone() for layer in model.layers], dim=0)
    best_score = float("inf")
    best_state = None
    curve = {}
    last_plasticity = None
    calm_active_steps = 0
    initial_route = model.route_stats() if isinstance(model, SparseIntersectionOnion) else None

    for step in range(1, STEPS + 1):
        ids = torch.randint(0, len(train.features), (BATCH,), generator=batch_rng)
        opt.zero_grad(set_to_none=True)
        loss = base.loss_fn(model, train.features[ids], train.roots[ids], train.degree[ids])
        loss.backward()
        if kind == "ONION_DYNAMIC":
            last_plasticity = v9.dynamic_plasticity(model, loss)
        elif isinstance(model, SparseIntersectionOnion):
            last_plasticity = {
                "correctness": v9.correctness_score(loss),
                "reward_lambda": 0.0,
                "punish_lambda": 0.0,
                "mean_layer_grad_mult": 1.0,
                "deep_layer_grad_mult": 1.0,
            }
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        if isinstance(model, SparseIntersectionOnion):
            model.update_stability(grad_norm)
            if model.last_alpha > 1e-6:
                calm_active_steps += 1
        opt.step()

        if step in CHECKPOINTS:
            metrics = base.evaluate(model, valid)
            route = model.route_stats() if isinstance(model, SparseIntersectionOnion) else None
            curve[step] = metrics
            score = metrics["mae"] + 50.0 * (1.0 - metrics["count"]) + 20.0 * (1.0 - metrics["within1"])
            if score < best_score:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())
            print(f"{kind} seed={seed} step={step:4d} loss={float(loss.detach()):.6f} VALID={metrics} ROUTE={route} PLASTICITY={last_plasticity}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    final = base.evaluate(model, valid)
    final_route = model.route_stats() if isinstance(model, SparseIntersectionOnion) else None
    geo = model.manifold_stats(initial_points) if isinstance(model, SparseIntersectionOnion) else None
    return final, curve, initial_route, final_route, geo, calm_active_steps, last_plasticity


def main():
    if "--preflight" in sys.argv:
        preflight()
        return

    print("DEVICE=cpu")
    print("DATA_CONTRACT", OFFICIAL_DATA_CONTRACT)
    print("V10_CONTRACT branches=6 depth=3 sparse_point_intersections multipoint_strengthening layer_only_reward adaptive_calm")
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
        results, diagnostics, curves = {}, {}, {}
        for kind in ("MLP", "ONION_STATIC", "ONION_DYNAMIC"):
            result, curve, ri, rf, geo, calm, plast = train_one(kind, train, valid, seed)
            results[kind] = result
            curves[kind] = curve
            diagnostics[kind] = {
                "route_initial": ri,
                "route_final": rf,
                "geometry": geo,
                "calm_active_steps": calm,
                "last_plasticity": plast,
            }
        rows.append((seed, results, diagnostics))
        print(f"SEED_SUMMARY seed={seed} RESULTS={results} DIAGNOSTICS={diagnostics}", flush=True)

    def avg(kind, key):
        return float(np.mean([r[kind][key] for _, r, _ in rows]))

    summary = {
        k: {m: avg(k, m) for m in ("count", "mae", "within1", "merge")}
        for k in ("MLP", "ONION_STATIC", "ONION_DYNAMIC")
    }
    summary["delta_dynamic_vs_mlp"] = {
        "count": summary["ONION_DYNAMIC"]["count"] - summary["MLP"]["count"],
        "mae": summary["ONION_DYNAMIC"]["mae"] - summary["MLP"]["mae"],
        "within1": summary["ONION_DYNAMIC"]["within1"] - summary["MLP"]["within1"],
        "merge": summary["ONION_DYNAMIC"]["merge"] - summary["MLP"]["merge"],
    }
    summary["delta_dynamic_vs_static"] = {
        "count": summary["ONION_DYNAMIC"]["count"] - summary["ONION_STATIC"]["count"],
        "mae": summary["ONION_DYNAMIC"]["mae"] - summary["ONION_STATIC"]["mae"],
        "within1": summary["ONION_DYNAMIC"]["within1"] - summary["ONION_STATIC"]["within1"],
        "merge": summary["ONION_DYNAMIC"]["merge"] - summary["ONION_STATIC"]["merge"],
    }
    wins_mlp = sum(
        r["ONION_DYNAMIC"]["mae"] < r["MLP"]["mae"] and r["ONION_DYNAMIC"]["count"] >= r["MLP"]["count"]
        for _, r, _ in rows
    )
    wins_static = sum(
        r["ONION_DYNAMIC"]["mae"] < r["ONION_STATIC"]["mae"] and r["ONION_DYNAMIC"]["count"] >= r["ONION_STATIC"]["count"]
        for _, r, _ in rows
    )
    print("FINAL_SUMMARY", summary)
    print(f"DYNAMIC_WINS_VS_MLP={wins_mlp}/3")
    print(f"DYNAMIC_WINS_VS_STATIC={wins_static}/3")

    clear = (
        wins_mlp >= 2
        and summary["ONION_DYNAMIC"]["mae"] < summary["MLP"]["mae"]
        and summary["ONION_DYNAMIC"]["count"] >= summary["MLP"]["count"]
        and summary["ONION_DYNAMIC"]["within1"] >= summary["MLP"]["within1"]
    )
    print("V10_SPARSE_ONION_DEEPMIND_CLEAR_SIGNAL" if clear else "V10_SPARSE_ONION_DEEPMIND_NO_CLEAR_SIGNAL")


if __name__ == "__main__":
    main()
