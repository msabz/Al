#!/usr/bin/env python3
"""CPU-only controlled test of a 3x3x3 manifold-constrained neural lattice.

This experiment is deliberately isolated from production/MAI5 code.
Data: ONLY the official google-deepmind/mathematics_dataset polynomial_roots generator.
Task: predict sorted roots with multiplicity plus unique-start bits, using the same
loss/evaluation for both architectures.

Comparison:
  A) parameter-matched ordinary MLP (~same trainable parameter count)
  B) 3x3x3 lattice (27 functional cells). Each channel inside each cell owns a
     trainable point constrained to one of four infinite-solution manifolds.
     Information flows along +X,+Y,+Z directions through the cube.

The experiment records learning curves to test the hypothesis that the lattice's
advantage, if any, emerges later in training rather than at initialization.
"""
import copy
import math
import pathlib
import random
import re
import subprocess
import sys
from dataclasses import dataclass

import numpy as np
import sympy as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT_SCALE = 100.0
ROOT_SLOTS = 5
COEFF_SLOTS = 6
FEATURES = 7
TRAIN_COUNT = 1536
VALID_COUNT = 384
BATCH = 192
STEPS = 5000
LR = 1.2e-3
WEIGHT_DECAY = 1e-5
SEEDS = (11, 22, 33)
CHECKPOINTS = {1, 100, 250, 500, 1000, 2000, 3000, 4000, 5000}
DEVICE = torch.device("cpu")


def prepare_official_deepmind():
    root = pathlib.Path("/tmp/mathai-v6-manifold-dm")
    if not root.exists():
        subprocess.run([
            "git", "clone", "-q", "--depth", "1",
            "https://github.com/google-deepmind/mathematics_dataset.git", str(root)
        ], check=True)
    p = root / "mathematics_dataset/sample/polynomials.py"
    s = p.read_text()
    s = s.replace("coeffs.itemset(index, value)", "coeffs[index] = value")
    s = s.replace("expanded_coefficients.itemset(power, coeffs)", "expanded_coefficients[power] = coeffs")
    p.write_text(s)
    sys.path.insert(0, str(root))
    from mathematics_dataset.modules import algebra as dm_algebra
    return dm_algebra


PATTERNS = [
    r"^Let (.+?=.+?)\. (?:What is|Calculate) [A-Za-z][?.]$",
    r"^Suppose (.+?=.+?)\. (?:What is|Calculate) [A-Za-z][?.]$",
    r"^What is [A-Za-z] in (.+?=.+?)\?$",
    r"^Solve (.+?=.+?)(?: for [A-Za-z])?\.$",
    r"^Find [A-Za-z],? (?:such that|given that) (.+?=.+?)\.$",
    r"^Determine [A-Za-z],? (?:so that|given that) (.+?=.+?)\.$",
]


def extract_equality(question: str) -> str:
    if question.startswith("Factor "):
        raise ValueError("factor prompt")
    for pat in PATTERNS:
        m = re.match(pat, question)
        if m:
            return m.group(1)
    raise ValueError("unsupported polynomial prompt")


def parse_eq(text: str):
    a, b = text.split("=", 1)
    return sp.Eq(sp.sympify(a.replace("^", "**")), sp.sympify(b.replace("^", "**")))


def canonical_features(poly: sp.Poly):
    degree = int(poly.degree())
    if not 2 <= degree <= 5:
        raise ValueError("degree")
    raw = [float(sp.N(poly.nth(i))) for i in range(COEFF_SLOTS)]
    scaled = [raw[i] * (ROOT_SCALE ** i) for i in range(COEFF_SLOTS)]
    scale = max(abs(x) for x in scaled)
    if not math.isfinite(scale) or scale <= 1e-12:
        raise ValueError("zero polynomial")
    scaled = [x / scale for x in scaled]
    first = next((x for x in scaled if abs(x) > 1e-10), 0.0)
    if first < 0:
        scaled = [-x for x in scaled]
    scaled = [0.0 if abs(x) < 1e-10 else x for x in scaled]
    return np.asarray(scaled + [degree / 5.0], np.float32), degree


def roots_with_multiplicity(poly: sp.Poly):
    root_map = sp.roots(poly.as_expr(), poly.gens[0])
    if sum(int(m) for m in root_map.values()) != int(poly.degree()):
        raise ValueError("incomplete roots")
    values = []
    for root, mult in root_map.items():
        c = complex(sp.N(root, 18))
        if abs(c.imag) > 1e-8:
            raise ValueError("complex root")
        if not math.isfinite(c.real) or abs(c.real) > 300.0:
            raise ValueError("target_out_of_range")
        values.extend([float(c.real)] * int(mult))
    values.sort()
    if len(values) != int(poly.degree()) or not 2 <= len(values) <= ROOT_SLOTS:
        raise ValueError("root multiplicity count")
    return values


@dataclass
class Bank:
    features: torch.Tensor
    roots: torch.Tensor
    degree: torch.Tensor


def build_bank(dm_algebra, count: int, seed: int) -> Bank:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    modules = dm_algebra.train(lambda entropy_range: entropy_range)
    generator = modules["polynomial_roots"]
    xs, ys, ds = [], [], []
    attempts = 0
    while len(xs) < count:
        attempts += 1
        if attempts > count * 120:
            raise RuntimeError("official DeepMind bank exhausted")
        problem = generator()
        q = str(problem.question)
        try:
            eq = parse_eq(extract_equality(q))
            syms = sorted(eq.free_symbols, key=lambda z: str(z))
            if len(syms) != 1:
                raise ValueError("symbol count")
            poly = sp.Poly(sp.expand(eq.lhs - eq.rhs), syms[0])
            feat, degree = canonical_features(poly)
            roots = roots_with_multiplicity(poly)
            pad = np.zeros(ROOT_SLOTS, np.float32)
            pad[:degree] = np.asarray(roots, np.float32) / ROOT_SCALE
            xs.append(feat)
            ys.append(pad)
            ds.append(degree)
        except Exception:
            continue
    return Bank(
        torch.tensor(np.stack(xs), dtype=torch.float32),
        torch.tensor(np.stack(ys), dtype=torch.float32),
        torch.tensor(ds, dtype=torch.long),
    )


def unique_start_target(target_roots, degree):
    out = torch.zeros_like(target_roots)
    for d in range(2, ROOT_SLOTS + 1):
        ids = torch.where(degree == d)[0]
        if not len(ids):
            continue
        vals = target_roots[ids, :d]
        out[ids, 0] = 1.0
        if d > 1:
            out[ids, 1:d] = (torch.abs(vals[:, 1:d] - vals[:, :d-1]) > 1e-7).float()
    return out


class OrdinaryMLP(nn.Module):
    def __init__(self, hidden=52):
        super().__init__()
        self.fc1 = nn.Linear(FEATURES, hidden)
        self.fc2 = nn.Linear(hidden, ROOT_SLOTS * 2)

    def forward(self, x, degree):
        del degree
        out = self.fc2(torch.tanh(self.fc1(x)))
        return out[:, :ROOT_SLOTS], out[:, ROOT_SLOTS:]


class ManifoldLattice3D(nn.Module):
    """3x3x3=27-cell feed-forward lattice with manifold-constrained cell weights."""
    SIDE = 3
    WIDTH = 8

    def __init__(self):
        super().__init__()
        self.input = nn.Linear(FEATURES, self.WIDTH)
        self.theta = nn.Parameter(torch.randn(27, self.WIDTH) * 0.35)
        self.phi = nn.Parameter(torch.randn(27, self.WIDTH) * 0.35)
        self.bias = nn.Parameter(torch.zeros(27, self.WIDTH))
        self.readout = nn.Linear(self.WIDTH * 3, ROOT_SLOTS * 2)

    @staticmethod
    def _point(theta, phi, family):
        if family == 0:
            cp = torch.cos(phi)
            return torch.stack((torch.cos(theta)*cp, torch.sin(theta)*cp, torch.sin(phi)), dim=-1)
        if family == 1:
            return torch.stack((torch.cos(theta), torch.sin(theta), torch.tanh(phi)), dim=-1)
        u = torch.tanh(theta)
        v = torch.tanh(phi)
        if family == 2:
            return torch.stack((u, v, u*v), dim=-1)
        z = torch.tanh(torch.sin(math.pi*u) + torch.cos(math.pi*v))
        return torch.stack((u, v, z), dim=-1)

    def forward(self, x, degree):
        del degree
        base = torch.tanh(self.input(x))
        states = {}
        flat_states = []
        idx = 0
        for z in range(self.SIDE):
            for y in range(self.SIDE):
                for xx in range(self.SIDE):
                    preds = []
                    if xx > 0:
                        preds.append(states[(xx-1, y, z)])
                    if y > 0:
                        preds.append(states[(xx, y-1, z)])
                    if z > 0:
                        preds.append(states[(xx, y, z-1)])
                    if preds:
                        ps = torch.stack(preds, dim=0)
                        pmean = ps.mean(dim=0)
                        pmax = ps.max(dim=0).values
                    else:
                        pmean = torch.zeros_like(base)
                        pmax = torch.zeros_like(base)
                    family = (xx + 2*y + 3*z) % 4
                    point = self._point(self.theta[idx], self.phi[idx], family)
                    h = torch.tanh(
                        base * point[:, 0]
                        + pmean * point[:, 1]
                        + pmax * point[:, 2]
                        + self.bias[idx]
                    )
                    states[(xx, y, z)] = h
                    flat_states.append(h)
                    idx += 1
        stacked = torch.stack(flat_states, dim=1)
        pooled = torch.cat((stacked.mean(dim=1), stacked.max(dim=1).values, states[(2,2,2)]), dim=1)
        out = self.readout(pooled)
        return out[:, :ROOT_SLOTS], out[:, ROOT_SLOTS:]

    @torch.no_grad()
    def geometry_snapshot(self):
        points = []
        idx = 0
        for z in range(self.SIDE):
            for y in range(self.SIDE):
                for xx in range(self.SIDE):
                    family = (xx + 2*y + 3*z) % 4
                    points.append(self._point(self.theta[idx], self.phi[idx], family).cpu())
                    idx += 1
        return torch.stack(points, dim=0)


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def sorted_active(raw_roots, raw_unique, degree):
    sorted_roots = torch.zeros_like(raw_roots)
    sorted_unique = torch.zeros_like(raw_unique)
    for d in range(2, ROOT_SLOTS + 1):
        ids = torch.where(degree == d)[0]
        if not len(ids):
            continue
        values, order = torch.sort(raw_roots[ids, :d], dim=1)
        sorted_roots[ids, :d] = values
        sorted_unique[ids, :d] = torch.gather(raw_unique[ids, :d], 1, order)
    return sorted_roots, sorted_unique


def poly_residual(features, roots, degree):
    coeff = features[:, :COEFF_SLOTS]
    losses = []
    for d in range(2, ROOT_SLOTS + 1):
        ids = torch.where(degree == d)[0]
        if not len(ids):
            continue
        z = roots[ids, :d]
        c = coeff[ids]
        q = c[:, -1, None].expand(-1, d)
        for power in range(COEFF_SLOTS - 2, -1, -1):
            q = q * z + c[:, power, None]
        losses.append(F.smooth_l1_loss(q, torch.zeros_like(q), beta=0.25))
    return torch.stack(losses).mean() if losses else roots.sum() * 0


def loss_fn(model, x, roots, degree):
    raw, unique_logits = model(x, degree)
    pred, unique_logits = sorted_active(raw, unique_logits, degree)
    unique_target = unique_start_target(roots, degree)
    root_losses, unique_losses = [], []
    for d in range(2, ROOT_SLOTS + 1):
        ids = torch.where(degree == d)[0]
        if not len(ids):
            continue
        root_losses.append(F.smooth_l1_loss(pred[ids, :d], roots[ids, :d], beta=0.08))
        unique_losses.append(F.binary_cross_entropy_with_logits(unique_logits[ids, :d], unique_target[ids, :d]))
    root_loss = torch.stack(root_losses).mean()
    unique_loss = torch.stack(unique_losses).mean()
    residual = poly_residual(x, pred, degree)
    return root_loss + 0.35*unique_loss + 0.15*residual


@torch.no_grad()
def evaluate(model, bank: Bank):
    raw, logits = model(bank.features, bank.degree)
    pred, logits = sorted_active(raw, logits, bank.degree)
    target_unique = unique_start_target(bank.roots, bank.degree)
    total = count_ok = root_n = within1 = merge_correct = merge_n = missing = extra = 0
    root_abs = 0.0
    for i in range(len(bank.features)):
        d = int(bank.degree[i])
        pvals = pred[i, :d] * ROOT_SCALE
        tvals = bank.roots[i, :d] * ROOT_SCALE
        pstart = (torch.sigmoid(logits[i, :d]) >= 0.5).clone()
        pstart[0] = True
        tstart = target_unique[i, :d] > 0.5
        puniq = pvals[pstart]
        tuniq = tvals[tstart]
        count_ok += int(len(puniq) == len(tuniq))
        if len(puniq) < len(tuniq):
            missing += int(len(tuniq)-len(puniq))
        elif len(puniq) > len(tuniq):
            extra += int(len(puniq)-len(tuniq))
        n = max(len(puniq), len(tuniq))
        for j in range(n):
            err = abs(float(puniq[j]-tuniq[j])) if j < len(puniq) and j < len(tuniq) else 300.0
            root_abs += err
            root_n += 1
            within1 += int(err <= 1.0)
        merge_correct += int((pstart == tstart).sum())
        merge_n += d
        total += 1
    return {
        "count": count_ok/max(total,1),
        "mae": root_abs/max(root_n,1),
        "within1": within1/max(root_n,1),
        "merge": merge_correct/max(merge_n,1),
        "missing": missing,
        "extra": extra,
    }


def train_one(model, train, valid, seed, label):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = OrdinaryMLP() if isinstance(model, OrdinaryMLP) else ManifoldLattice3D()
    model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    batch_rng = torch.Generator().manual_seed(100000 + seed)
    initial_geo = model.geometry_snapshot().clone() if isinstance(model, ManifoldLattice3D) else None
    curve = {}
    best_score = float("inf")
    best_state = None
    for step in range(1, STEPS+1):
        ids = torch.randint(0, len(train.features), (BATCH,), generator=batch_rng)
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model, train.features[ids], train.roots[ids], train.degree[ids])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if step in CHECKPOINTS:
            m = evaluate(model, valid)
            curve[step] = m
            score = m["mae"] + 50.0*(1.0-m["count"]) + 20.0*(1.0-m["within1"])
            if score < best_score:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())
            print(f"{label} seed={seed} step={step:4d} loss={float(loss):.6f} VALID={m}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    final = evaluate(model, valid)
    geo = None
    if isinstance(model, ManifoldLattice3D):
        current = model.geometry_snapshot()
        displacement = torch.linalg.vector_norm(current-initial_geo, dim=-1)
        pts = current.mean(dim=1)
        dist = torch.cdist(pts, pts)
        mask = torch.triu(torch.ones_like(dist, dtype=torch.bool), diagonal=1)
        pair = dist[mask]
        geo = {
            "mean_point_displacement": float(displacement.mean()),
            "max_point_displacement": float(displacement.max()),
            "close_cell_pairs_lt_0.25": int((pair < 0.25).sum()),
            "median_cell_distance": float(pair.median()),
        }
    return final, curve, geo


def main():
    print("DEVICE=cpu")
    dm = prepare_official_deepmind()
    print("Building official DeepMind banks...", flush=True)
    train = build_bank(dm, TRAIN_COUNT, 7301)
    valid = build_bank(dm, VALID_COUNT, 8301)
    print("train degree distribution", {d:int((train.degree==d).sum()) for d in range(2,6)})
    print("valid degree distribution", {d:int((valid.degree==d).sum()) for d in range(2,6)})

    base_params = parameter_count(OrdinaryMLP())
    lattice_params = parameter_count(ManifoldLattice3D())
    print(f"PARAMS ordinary={base_params} lattice={lattice_params} ratio={lattice_params/base_params:.4f}")
    if abs(lattice_params-base_params)/base_params > 0.08:
        raise SystemExit("parameter budget mismatch > 8%")

    rows = []
    for seed in SEEDS:
        a, acurve, _ = train_one(OrdinaryMLP(), train, valid, seed, "MLP")
        b, bcurve, geo = train_one(ManifoldLattice3D(), train, valid, seed, "LATTICE")
        rows.append((seed,a,b,geo))
        print(f"SEED_SUMMARY seed={seed} MLP={a} LATTICE={b} GEOMETRY={geo}", flush=True)
        print("LATE_CURVE", seed, {
            s: {"mlp_mae":acurve[s]["mae"], "lat_mae":bcurve[s]["mae"],
                "mlp_count":acurve[s]["count"], "lat_count":bcurve[s]["count"],
                "mlp_within1":acurve[s]["within1"], "lat_within1":bcurve[s]["within1"]}
            for s in sorted(CHECKPOINTS)
        }, flush=True)

    def mean(key, which):
        return float(np.mean([row[which][key] for row in rows]))
    summary = {
        "mlp": {k:mean(k,1) for k in ("count","mae","within1","merge")},
        "lattice": {k:mean(k,2) for k in ("count","mae","within1","merge")},
    }
    summary["delta"] = {
        "count": summary["lattice"]["count"]-summary["mlp"]["count"],
        "mae": summary["lattice"]["mae"]-summary["mlp"]["mae"],
        "within1": summary["lattice"]["within1"]-summary["mlp"]["within1"],
        "merge": summary["lattice"]["merge"]-summary["mlp"]["merge"],
    }
    wins = sum(
        (b["mae"] < a["mae"] and b["count"] >= a["count"])
        for _,a,b,_ in rows
    )
    print("FINAL_SUMMARY", summary, flush=True)
    print(f"LATTICE_WINS_STRICT={wins}/{len(SEEDS)}", flush=True)

    clear = (
        wins >= 2
        and summary["delta"]["count"] >= 0.03
        and summary["delta"]["mae"] <= -3.0
    )
    if clear:
        print("V6_MANIFOLD_LATTICE_DEEPMIND_CLEAR_SIGNAL")
    else:
        print("V6_MANIFOLD_LATTICE_DEEPMIND_NO_CLEAR_SIGNAL")


if __name__ == "__main__":
    main()
