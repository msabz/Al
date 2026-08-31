#!/usr/bin/env python3
"""CPU-only capacity sanity for the MAI5-v4 polynomial cardinality head.

This is deliberately NOT a generalization test and NOT a production training run.
It repeatedly trains on one tiny fixed bank from the official DeepMind polynomial
generator to answer one question only: can the exact production architecture/loss
learn polynomial cardinality cleanly when asked to overfit?

No project synthetic examples are used.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import pathlib
import random
import tempfile


def load_runtime(repo: pathlib.Path):
    trainer = repo / "colab/train_v5_deepmind.py"
    text = trainer.read_text().replace("/content", tempfile.mkdtemp(prefix="mai5-v4-cpu-") )
    text = text.replace("RESUME_FROM_MAI5 = False", "RESUME_FROM_MAI5 = False", 1)
    prefix = text.split("# ========================= TRAIN =========================", 1)[0]
    ns = {"__name__": "mai5_v4_cpu_overfit", "__file__": str(trainer)}
    exec(compile(prefix, str(trainer), "exec"), ns)
    return ns


def reseed(ns, seed: int):
    random.seed(seed)
    ns["np"].random.seed(seed & 0xFFFFFFFF)
    ns["torch"].manual_seed(seed)


def official_polynomial_bank(ns, count: int, seed: int):
    old_modules = ns["dm_modules"]
    old_names = list(ns["DM_NAMES"])
    old_synthetic = ns["synthetic"]

    def forbidden(*args, **kwargs):
        raise RuntimeError("PROJECT_SYNTHETIC_FORBIDDEN")

    reseed(ns, seed)
    ns["dm_modules"] = ns["dm_algebra"].test()
    ns["DM_NAMES"] = ["polynomial_roots"]
    ns["synthetic"] = forbidden
    rng = random.Random(seed ^ 0x51A7)
    rows = []
    seen = set()
    attempts = 0
    try:
        while len(rows) < count and attempts < count * 3000:
            attempts += 1
            try:
                ex = ns["deepmind_example"](rng, allow_synthetic_fallback=False)
            except Exception:
                continue
            if ex["f"] != ns["POLYNOMIAL"] or ex["state"] != ns["FINITE"]:
                continue
            if not 1 <= len(ex["roots"]) <= ns["ROOT_SLOTS"]:
                continue
            if max(abs(float(v)) for v in ex["roots"]) > 300.0:
                continue
            if ex["eq"] in seen:
                continue
            seen.add(ex["eq"])
            rows.append(ex)
    finally:
        ns["dm_modules"] = old_modules
        ns["DM_NAMES"] = old_names
        ns["synthetic"] = old_synthetic
    if len(rows) != count:
        raise RuntimeError(f"Official DeepMind polynomial bank short: {len(rows)}/{count}")
    return rows


def metrics(ns, batch):
    torch = ns["torch"]
    F = ns["F"]
    k,n,d,f,r,rc,sy,st,eqv = ns["collate"](batch)
    model = ns["model"]
    model.eval()
    with torch.no_grad():
        out = model(k,n,d,f)
        count_target = rc.clamp(1, ns["ROOT_SLOTS"]) - 1
        count_ce = float(F.cross_entropy(out[:,5:10], count_target))
        count_pred = out[:,5:10].argmax(-1) + 1
        count_acc = float((count_pred == rc).float().mean())
        total_loss = float(ns["loss_fn"](out,r,rc,sy,st,f,n,None))

        abs_errors = []
        for i, ex in enumerate(batch):
            active = ns["polynomial_active_indices"](out[i], n[i])
            predicted = (out[i,:5][active] * ns["ROOT_SCALE"]).cpu().numpy().astype(float).tolist()
            expected = [float(x) for x in ex["roots"]]
            used = set()
            for value in expected:
                candidates = [(abs(p - value), j) for j,p in enumerate(predicted) if j not in used]
                if candidates:
                    err,j = min(candidates)
                    used.add(j)
                    abs_errors.append(err)
                else:
                    abs_errors.append(300.0)
        mae = sum(abs_errors) / max(len(abs_errors), 1)
    model.train()
    return {"count_ce":count_ce,"count_accuracy":count_acc,"total_loss":total_loss,"decoded_mae":mae}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=pathlib.Path, default=pathlib.Path(__file__).resolve().parents[1])
    ap.add_argument("--examples", type=int, default=32)
    ap.add_argument("--max-steps", type=int, default=1400)
    args = ap.parse_args()

    repo = args.repo_root.resolve()
    ns = load_runtime(repo)
    torch = ns["torch"]
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    if torch.cuda.is_available() or str(ns["device"]) != "cpu":
        raise RuntimeError("CPU_ONLY_SANITY_REFUSES_GPU")
    if ns["VERSION"] != 4:
        raise RuntimeError(f"Expected MAI5 v4 runtime, got v{ns['VERSION']}")

    bank = official_polynomial_bank(ns, args.examples, seed=0xA1654401)
    counts = [len(e["roots"]) for e in bank]
    represented = sorted(set(counts))
    if len(represented) < 3:
        raise RuntimeError(f"Cardinality bank lacks diversity: {represented}")
    print("OFFICIAL_DEEPMIND_ONLY examples=", len(bank), "counts=", {k:counts.count(k) for k in represented})

    before = metrics(ns, bank)
    print("BEFORE", before)

    k,n,d,f,r,rc,sy,st,eqv = ns["collate"](bank)
    model = ns["model"]
    model.train()
    success_step = None
    for step in range(1, args.max_steps + 1):
        out = model(k,n,d,f)
        loss = ns["loss_fn"](out,r,rc,sy,st,f,n,None)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}: {float(loss)}")
        loss.backward()
        ns["android_adam_step"](8e-4)
        if step == 1 or step % 50 == 0:
            now = metrics(ns, bank)
            print(f"STEP {step:4d}", now)
            if (
                now["count_accuracy"] >= 0.97
                and now["count_ce"] <= 0.08
                and now["total_loss"] <= before["total_loss"] * 0.35
            ):
                success_step = step
                break

    after = metrics(ns, bank)
    print("AFTER", after, "success_step=", success_step)

    if success_step is None:
        raise RuntimeError("V4_CARDINALITY_SANITY_FAIL: exact production loss did not overfit cardinality cleanly")
    if after["decoded_mae"] >= before["decoded_mae"]:
        raise RuntimeError(
            f"V4_CARDINALITY_SANITY_FAIL: decoded root MAE did not improve ({before['decoded_mae']} -> {after['decoded_mae']})"
        )
    print("V4_CARDINALITY_SANITY_PASS")


if __name__ == "__main__":
    main()
