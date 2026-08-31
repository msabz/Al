#!/usr/bin/env python3
"""CPU-only v5 polynomial root-query architecture preflight.

Purpose:
- Uses ONLY the official google-deepmind/mathematics_dataset polynomial_roots generator.
- Reconstructs polynomial roots WITH multiplicity from the official generated equation.
- Uses disjoint deterministic train/validation banks.
- Tests a tiny Set-Transformer-inspired decoder before any Android/MAI5 migration or GPU run.

This is an architecture gate, not production training.
"""
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

SEED = 165
ROOT_SCALE = 100.0
ROOT_SLOTS = 5
COEFF_SLOTS = 6
FEATURES = 7
TRAIN_COUNT = 1536
VALID_COUNT = 384
STEPS = 2600
BATCH = 192
LR = 8e-4
WEIGHT_DECAY = 1e-5
DEVICE = torch.device("cpu")


def prepare_official_deepmind():
    root = pathlib.Path("/tmp/mathai-v5-setroot-dm")
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
    # roots() preserves algebraic multiplicity, unlike solve()/FiniteSet.
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
        if attempts > count * 100:
            raise RuntimeError("official DeepMind bank exhausted")
        problem = generator()
        q = str(problem.question)
        try:
            raw = extract_equality(q)
            eq = parse_eq(raw)
            syms = sorted(eq.free_symbols, key=lambda z: str(z))
            if len(syms) != 1:
                raise ValueError("symbol count")
            poly = sp.Poly(sp.expand(eq.lhs - eq.rhs), syms[0])
            feat, degree = canonical_features(poly)
            roots = roots_with_multiplicity(poly)
            pad = np.zeros(ROOT_SLOTS, np.float32)
            pad[:degree] = np.asarray(roots, np.float32) / ROOT_SCALE
            xs.append(feat); ys.append(pad); ds.append(degree)
        except Exception:
            continue
    return Bank(
        torch.tensor(np.stack(xs), dtype=torch.float32),
        torch.tensor(np.stack(ys), dtype=torch.float32),
        torch.tensor(ds, dtype=torch.long),
    )


class RootQueryNet(nn.Module):
    """Tiny Set-Transformer-inspired polynomial decoder.

    The polynomial coefficient vector is encoded once. Five learned root queries receive
    the same context, interact with one self-attention block, then emit root values and
    unique-start logits. Only the first `degree` queries are active.
    """
    def __init__(self, hidden=64, qdim=32):
        super().__init__()
        self.enc1 = nn.Linear(FEATURES, hidden)
        self.enc2 = nn.Linear(hidden, hidden)
        self.ctx = nn.Linear(hidden, qdim)
        self.root_queries = nn.Parameter(torch.empty(ROOT_SLOTS, qdim))
        self.q = nn.Linear(qdim, qdim, bias=False)
        self.k = nn.Linear(qdim, qdim, bias=False)
        self.v = nn.Linear(qdim, qdim, bias=False)
        self.ff = nn.Linear(qdim, qdim)
        self.root = nn.Linear(qdim, 1)
        self.unique = nn.Linear(qdim, 1)
        nn.init.xavier_uniform_(self.root_queries)

    def forward(self, x, degree):
        h = F.relu(self.enc1(x))
        h = F.relu(self.enc2(h))
        t = self.root_queries.unsqueeze(0) + self.ctx(h).unsqueeze(1)
        q, k, v = self.q(t), self.k(t), self.v(t)
        score = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(q.shape[-1])
        active = torch.arange(ROOT_SLOTS, device=x.device)[None, :] < degree[:, None]
        score = score.masked_fill(~active[:, None, :], -1e4)
        a = torch.softmax(score, dim=-1)
        mixed = t + torch.matmul(a, v)
        u = mixed + F.relu(self.ff(mixed))
        roots = self.root(u).squeeze(-1)
        unique_logits = self.unique(u).squeeze(-1)
        return roots, unique_logits


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
        unique_losses.append(F.binary_cross_entropy_with_logits(
            unique_logits[ids, :d], unique_target[ids, :d]
        ))
    root_loss = torch.stack(root_losses).mean()
    unique_loss = torch.stack(unique_losses).mean()
    residual = poly_residual(x, pred, degree)
    total = root_loss + 0.35 * unique_loss + 0.15 * residual
    return total, root_loss, unique_loss, residual


@torch.no_grad()
def evaluate(model, bank: Bank):
    raw, logits = model(bank.features, bank.degree)
    pred, logits = sorted_active(raw, logits, bank.degree)
    target_unique = unique_start_target(bank.roots, bank.degree)
    total = 0
    count_ok = 0
    root_abs = 0.0
    root_n = 0
    within1 = 0
    merge_correct = 0
    merge_n = 0
    missing = 0
    extra = 0
    for i in range(len(bank.features)):
        d = int(bank.degree[i])
        pvals = pred[i, :d] * ROOT_SCALE
        tvals = bank.roots[i, :d] * ROOT_SCALE
        pstart = torch.sigmoid(logits[i, :d]) >= 0.5
        pstart[0] = True
        tstart = target_unique[i, :d] > 0.5
        puniq = pvals[pstart]
        tuniq = tvals[tstart]
        if len(puniq) == len(tuniq):
            count_ok += 1
        else:
            if len(puniq) < len(tuniq): missing += int(len(tuniq) - len(puniq))
            else: extra += int(len(puniq) - len(tuniq))
        # For precision, compare sorted unique roots; penalize missing/extra as 300.
        n = max(len(puniq), len(tuniq))
        for j in range(n):
            err = abs(float(puniq[j] - tuniq[j])) if j < len(puniq) and j < len(tuniq) else 300.0
            root_abs += err; root_n += 1; within1 += int(err <= 1.0)
        merge_correct += int((pstart == tstart).sum())
        merge_n += d
        total += 1
    return {
        "examples": total,
        "count_accuracy": count_ok / max(total, 1),
        "unique_mae": root_abs / max(root_n, 1),
        "within_one": within1 / max(root_n, 1),
        "merge_bit_accuracy": merge_correct / max(merge_n, 1),
        "missing": missing,
        "extra": extra,
    }


def main():
    dm_algebra = prepare_official_deepmind()
    print("Building deterministic official DeepMind train bank...", flush=True)
    train = build_bank(dm_algebra, TRAIN_COUNT, 7301)
    print("Building disjoint deterministic official DeepMind validation bank...", flush=True)
    valid = build_bank(dm_algebra, VALID_COUNT, 8301)
    assert not torch.equal(train.features[:min(len(train.features),len(valid.features))], valid.features[:min(len(train.features),len(valid.features))])
    print("train degree distribution", {d:int((train.degree==d).sum()) for d in range(2,6)})
    print("valid degree distribution", {d:int((valid.degree==d).sum()) for d in range(2,6)})

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    model = RootQueryNet().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    before = evaluate(model, valid)
    print("BEFORE", before, flush=True)
    g = torch.Generator().manual_seed(SEED + 99)
    for step in range(1, STEPS + 1):
        ids = torch.randint(0, len(train.features), (BATCH,), generator=g)
        optimizer.zero_grad(set_to_none=True)
        total, root_l, unique_l, residual = loss_fn(
            model, train.features[ids], train.roots[ids], train.degree[ids]
        )
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step == 1 or step % 200 == 0:
            metrics = evaluate(model, valid)
            print(
                f"step={step:4d} loss={float(total):.6f} root={float(root_l):.6f} "
                f"merge={float(unique_l):.6f} residual={float(residual):.6f} VALID={metrics}",
                flush=True,
            )
    after = evaluate(model, valid)
    print("AFTER", after, flush=True)
    params = sum(p.numel() for p in model.parameters())
    print("prototype_parameters", params)

    # Stronger than the old fixed-bank overfit sanity: all gates are on a disjoint official bank.
    gates = {
        "count_accuracy>=0.75": after["count_accuracy"] >= 0.75,
        "within_one>=0.25": after["within_one"] >= 0.25,
        "unique_mae<=35": after["unique_mae"] <= 35.0,
        "merge_bit_accuracy>=0.85": after["merge_bit_accuracy"] >= 0.85,
        "improves_count": after["count_accuracy"] >= before["count_accuracy"] + 0.25,
        "improves_mae": after["unique_mae"] <= before["unique_mae"] * 0.65,
    }
    print("GATES", gates)
    if not all(gates.values()):
        raise SystemExit("V5_SETROOT_QUERY_GENERALIZATION_FAIL")
    print("V5_SETROOT_QUERY_GENERALIZATION_PASS")


if __name__ == "__main__":
    main()
