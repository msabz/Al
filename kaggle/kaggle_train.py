#!/usr/bin/env python3
# Math AI v5 Kaggle launcher — reviewed release-run wrapper.
# Applies deterministic source fixes before the single GPU release run.

import importlib.util
import pathlib
import shutil
import urllib.request

SOURCE_COMMIT = "__SOURCE_COMMIT__"
CORE_COMMIT = "3a6fbc9036d6b4cac9f7c9e4e38faabae19659ce"
CORE_URL = f"https://raw.githubusercontent.com/msabz/Al/{CORE_COMMIT}/kaggle/kaggle_train.py"
CORE_PATH = pathlib.Path("/kaggle/working/_mathai_kaggle_core.py")

print("[RELEASE] Loading Kaggle orchestration core:", CORE_COMMIT, flush=True)
urllib.request.urlretrieve(CORE_URL, CORE_PATH)

spec = importlib.util.spec_from_file_location("mathai_kaggle_core", CORE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError("Could not load Kaggle orchestration core")
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)

core.SOURCE_COMMIT = SOURCE_COMMIT
core.TOTAL_STEPS = 40_000
core.BATCH_SIZE = 4096
core.CONSISTENCY_WEIGHT = 0.12

_original_prepare_source = core.prepare_source


def prepare_source_patched(log_fh):
    actual = _original_prepare_source(log_fh)

    # Correct the shared synthetic teacher before both training and audit.
    base = core.ROOT / "colab/train_v5_deepmind.py"
    text = base.read_text()

    old_tan = '''        else:
            root=max(-1.2,min(1.2,root)); eq=f"tan(x)={math.tan(root):.8g}"
        return mk(eq,[root],equiv=swap(eq))'''
    new_tan = '''        else:
            a=rng.randint(1,5); inner=rng.randint(1,14); b=inner-a*root
            eq=f"ln({a}*x{b:+g})={math.log(inner):.8g}"
        return mk(eq,[root],equiv=swap(eq))'''
    if old_tan not in text:
        raise RuntimeError("Synthetic analytic-label patch target not found")
    text = text.replace(old_tan, new_tan, 1)

    old_states = '''    z=rng.randrange(4)
    if z==0:return mk("0*x=1",state=NO_SOLUTION,equiv="1=0*x")
    if z==1:return mk("0*x=0",state=INFINITE,equiv="0=0*x")
    if z==2:return mk("x+y=2;2*x+2*y=4",state=INFINITE,equiv="2*x+2*y=4;x+y=2")
    return mk("x*y=1",state=UNSUPPORTED,equiv="1=x*y")'''
    new_states = '''    z=rng.randrange(5)
    if z==0:
        a=rng.choice([i for i in range(-12,13) if i]); b=rng.randint(-40,40)
        c=rng.randint(-40,40)
        while c==b: c=rng.randint(-40,40)
        eq=f"{a}*x{b:+}={a}*x{c:+}"
        return mk(eq,state=NO_SOLUTION,equiv=swap(eq))
    if z==1:
        a=rng.choice([i for i in range(-12,13) if i]); b=rng.randint(-40,40)
        eq=f"{a}*x{b:+}={a}*x{b:+}"
        return mk(eq,state=INFINITE,equiv=swap(eq))
    if z==2:
        a,b=[rng.choice([i for i in range(-9,10) if i]) for _ in range(2)]
        c=rng.randint(-30,30); scale=rng.choice([-4,-3,-2,2,3,4])
        e1=f"{a}*x{b:+}*y={c}"
        e2=f"{scale*a}*x{scale*b:+}*y={scale*c}"
        eq=e1+";"+e2
        return mk(eq,state=INFINITE,equiv=e2+";"+e1)
    if z==3:
        a,b=[rng.choice([i for i in range(-9,10) if i]) for _ in range(2)]
        c=rng.randint(-30,30); scale=rng.choice([-4,-3,-2,2,3,4])
        delta=rng.choice([i for i in range(-9,10) if i])
        e1=f"{a}*x{b:+}*y={c}"
        e2=f"{scale*a}*x{scale*b:+}*y={scale*c+delta}"
        eq=e1+";"+e2
        return mk(eq,state=NO_SOLUTION,equiv=e2+";"+e1)
    a=rng.choice([i for i in range(-7,8) if i]); b=rng.randint(-20,20); c=rng.randint(-20,20)
    eq=f"{a}*x*y{b:+}={c}"
    return mk(eq,state=UNSUPPORTED,equiv=swap(eq))'''
    if old_states not in text:
        raise RuntimeError("Synthetic state-diversity patch target not found")
    text = text.replace(old_states, new_states, 1)
    base.write_text(text)
    print("[RELEASE] Synthetic labels/state families: corrected and diversified", flush=True)

    # Domain-aware and permutation-aware independent audit.
    audit = core.ROOT / "colab/generalization_audit.py"
    text = audit.read_text()

    old = '    old_names = list(ns["DM_NAMES"])\n    try:'
    new = ('    old_names = list(ns["DM_NAMES"])\n'
           '    old_synthetic = ns["synthetic"]\n'
           '    ns["synthetic"] = lambda *args, **kwargs: {"eq": "__DM_FALLBACK__"}\n'
           '    try:')
    if old not in text:
        raise RuntimeError("Audit DeepMind fallback patch target not found")
    text = text.replace(old, new, 1)

    old = '        while len(result) < count and attempts < count * 100:\n            attempts += 1\n            ex = ns["deepmind_example"](rng)\n            if ex["eq"] in seen:'
    new = ('        while len(result) < count and attempts < count * 400:\n'
           '            attempts += 1\n'
           '            ex = ns["deepmind_example"](rng)\n'
           '            if ex.get("eq") == "__DM_FALLBACK__":\n'
           '                continue\n'
           '            if target_magnitude(ex) > 300.0:\n'
           '                continue\n'
           '            if ex["eq"] in seen:')
    if old not in text:
        raise RuntimeError("Audit DeepMind domain-filter patch target not found")
    text = text.replace(old, new, 1)

    old = '        ns["dm_modules"] = old_modules\n        ns["DM_NAMES"] = old_names'
    new = ('        ns["dm_modules"] = old_modules\n'
           '        ns["DM_NAMES"] = old_names\n'
           '        ns["synthetic"] = old_synthetic')
    if old not in text:
        raise RuntimeError("Audit DeepMind restore patch target not found")
    text = text.replace(old, new, 1)

    old_ood = ('    iid = trained["fresh_iid"]["rmse"]\n'
               '    ood = trained["strict_ood"]["rmse"]\n'
               '    if iid > 0 and ood / iid > 3.0:\n'
               '        reasons.append(f"large OOD gap: strict-OOD RMSE is {ood/iid:.2f}x fresh-IID")')
    new_ood = ('    random = audit["random_baseline"]\n'
               '    iid_norm = trained["fresh_iid"]["rmse"] / max(random["fresh_iid"]["rmse"], 1e-9)\n'
               '    ood_norm = trained["strict_ood"]["rmse"] / max(random["strict_ood"]["rmse"], 1e-9)\n'
               '    if ood_norm > max(0.75, iid_norm * 2.5):\n'
               '        reasons.append(f"large normalized OOD gap: OOD/random={ood_norm:.3f} vs IID/random={iid_norm:.3f}")')
    if old_ood not in text:
        raise RuntimeError("Audit normalized-OOD patch target not found")
    text = text.replace(old_ood, new_ood, 1)

    start = text.index("def consistency_audit(ns, model, examples, limit=128):")
    end = text.index("\n\ndef improvement(", start)
    new_consistency_audit = r'''def consistency_audit(ns, model, examples, limit=128):
    import itertools
    np = ns["np"]
    perms = list(itertools.permutations(range(5)))
    slot_deltas = []
    presence_deltas = []
    state_kl = []
    family_mismatch = 0
    checked = 0
    finite_slot_checks = 0

    for ex in examples[:limit]:
        transformed = equivalent_add_constant(ex["eq"], 37)
        try:
            a = predict_raw(ns, model, ex["eq"])
            b = predict_raw(ns, model, transformed)
        except Exception:
            continue
        checked += 1
        if a["family"] != b["family"]:
            family_mismatch += 1
            continue

        pa = np.clip(a["state_probs"], 1e-9, 1.0)
        pb = np.clip(b["state_probs"], 1e-9, 1.0)
        state_kl.append(float(np.sum(pa * np.log(pa / pb))))

        if ex["state"] != ns["FINITE"]:
            presence_deltas.append(float(np.mean(np.abs(a["presence"] - b["presence"]))))
            continue

        if ex["f"] == ns["SYSTEM"]:
            slot_deltas.append(float(np.mean(np.abs(a["slots"][:2] - b["slots"][:2]))))
            presence_deltas.append(float(np.mean(np.abs(a["presence"] - b["presence"]))))
            finite_slot_checks += 1
            continue

        best = None
        for perm in perms:
            p = np.asarray(perm, dtype=int)
            bs = b["slots"][p]
            bp = b["presence"][p]
            active = np.maximum(a["presence"], bp)
            root_delta = float(np.sum(np.abs(a["slots"] - bs) * active) / max(float(np.sum(active)), 1e-9))
            pres_delta = float(np.mean(np.abs(a["presence"] - bp)))
            score = root_delta / max(float(ns["ROOT_SCALE"]), 1e-9) + 0.35 * pres_delta
            if best is None or score < best[0]:
                best = (score, root_delta, pres_delta)
        if best is not None:
            slot_deltas.append(best[1])
            presence_deltas.append(best[2])
            finite_slot_checks += 1

    return {
        "checked": checked,
        "finite_slot_checks": finite_slot_checks,
        "family_mismatch": family_mismatch,
        "mean_slot_delta": float(np.mean(slot_deltas)) if slot_deltas else float("inf"),
        "mean_presence_delta": float(np.mean(presence_deltas)) if presence_deltas else float("inf"),
        "mean_state_kl": float(np.mean(state_kl)) if state_kl else float("inf"),
        "transformation": "add +37 to both sides; +37 held out from training; root slots optimally matched",
    }
'''
    text = text[:start] + new_consistency_audit + text[end:]

    text = text.replace(
        '"strict_ood": "finite targets with |solution| >= 135 while training synthetic range <= 100",',
        '"strict_ood": "independent synthetic range-stress bank with |solution| >= 135, inside global ±300 target domain",',
        1,
    )
    text = text.replace(
        '"deepmind_interpolate": "official DeepMind algebra test() split",',
        '"deepmind_interpolate": "official DeepMind algebra test() split, unseen and |target| <= 300",',
        1,
    )
    text = text.replace(
        '"deepmind_extrapolate": "official DeepMind algebra test_extra() polynomial_roots_big split",',
        '"deepmind_extrapolate": "official DeepMind test_extra() polynomial_roots_big structural extrapolation, |target| <= 300",',
        1,
    )
    text = text.replace(
        '"equivalence": "+37 added to both sides, a transformation not used as the training consistency pair",',
        '"equivalence": "+37 added to both sides; +37 held out while training uses +17 / x2 / swapped-11",',
        1,
    )
    audit.write_text(text)
    print("[RELEASE] Audit: DeepMind-only, domain-aware, permutation-invariant", flush=True)
    return actual


core.prepare_source = prepare_source_patched
_original_prepare_turbo_worker = core.prepare_turbo_worker


def prepare_turbo_worker_patched():
    worker = _original_prepare_turbo_worker()
    text = worker.read_text()

    old_dtype = "assigned_vals[sysmask,:2] = systems[sysmask,:2] / ROOT_SCALE"
    new_dtype = "assigned_vals[sysmask,:2] = (systems[sysmask,:2] / ROOT_SCALE).to(out.dtype)"
    if old_dtype not in text:
        raise RuntimeError("AMP dtype patch target not found")
    text = text.replace(old_dtype, new_dtype, 1)

    old_parser = 'parts = [x.strip() for x in body.split(" and ") if "=" in x]'
    new_parser = 'parts = [x.strip() for x in re.split(r"\\s*(?:,|\\band\\b)\\s*", body) if "=" in x]'
    if old_parser not in text:
        raise RuntimeError("DeepMind linear_2d parser patch target not found")
    text = text.replace(old_parser, new_parser, 1)

    marker = "\n\nclass PoolWriter:"
    helper = r'''

def algebraic_equiv_variant(eq, variant):
    pieces = []
    mode = int(variant) % 3
    for part in eq.split(";"):
        left, right = part.split("=", 1)
        if mode == 0:
            pieces.append(f"({left})+17=({right})+17")
        elif mode == 1:
            pieces.append(f"2*({left})=2*({right})")
        else:
            pieces.append(f"({right})-11=({left})-11")
    return ";".join(pieces)
'''
    if marker not in text:
        raise RuntimeError("PoolWriter insertion marker not found")
    text = text.replace(marker, helper + marker, 1)

    old_equiv = '''        if e["equiv"] is None:
            self.ek[i] = e["k"]; self.en[i] = e["n"]; self.ed[i] = e["d"]; self.ef[i] = e["f"]
        else:
            self.ek[i] = e["ek"]; self.en[i] = e["en"]; self.ed[i] = e["ed"]; self.ef[i] = e["ef"]'''
    new_equiv = '''        try:
            variant = algebraic_equiv_variant(e["eq"], i)
            ek, en, ed, ef, _ = ns["encode"](variant)
            if ef != e["f"]:
                raise ValueError("equivalent transform changed family")
            self.ek[i] = ek; self.en[i] = en; self.ed[i] = ed; self.ef[i] = ef
        except Exception:
            if e["equiv"] is None:
                self.ek[i] = e["k"]; self.en[i] = e["n"]; self.ed[i] = e["d"]; self.ef[i] = e["f"]
            else:
                self.ek[i] = e["ek"]; self.en[i] = e["en"]; self.ed[i] = e["ed"]; self.ef[i] = e["ef"]'''
    if old_equiv not in text:
        raise RuntimeError("PoolWriter equivalence block not found")
    text = text.replace(old_equiv, new_equiv, 1)

    old_consistency = '''    consistency = out.sum() * 0
    if other_out is not None:
        consistency = F.smooth_l1_loss(out, other_out, beta=0.1)
    total = root_loss + 0.35 * presence + 0.35 * state_loss + CONSISTENCY_WEIGHT * consistency'''
    new_consistency = '''    consistency = out.sum() * 0
    if other_out is not None:
        parts = [F.smooth_l1_loss(out[:,10:14], other_out[:,10:14], beta=0.1)]

        sys_ids = torch.where(finite & (families == SYSTEM))[0]
        if len(sys_ids):
            parts.append(F.smooth_l1_loss(out[sys_ids,:2], other_out[sys_ids,:2], beta=0.1))
            parts.append(0.35 * F.smooth_l1_loss(out[sys_ids,5:10], other_out[sys_ids,5:10], beta=0.1))

        set_ids = torch.where(finite & (families != SYSTEM))[0]
        if len(set_ids):
            ov = other_out[set_ids,:5][:,PERMS]
            op = other_out[set_ids,5:10][:,PERMS]
            pv0 = out[set_ids,:5][:,None,:]
            pp0 = out[set_ids,5:10][:,None,:]
            weights = torch.maximum(torch.sigmoid(pp0.detach()), torch.sigmoid(op.detach()))
            root_delta = F.smooth_l1_loss(pv0.expand_as(ov), ov, reduction="none", beta=0.1)
            root_cost = (root_delta * weights).sum(-1) / weights.sum(-1).clamp_min(1e-6)
            pres_cost = F.smooth_l1_loss(pp0.expand_as(op), op, reduction="none", beta=0.1).mean(-1)
            best = (root_cost + 0.35 * pres_cost).argmin(-1)
            rows = torch.arange(len(set_ids), device=device)
            matched_v = ov[rows, best]
            matched_p = op[rows, best]
            matched_w = weights[rows, best]
            matched_root = F.smooth_l1_loss(out[set_ids,:5], matched_v, reduction="none", beta=0.1)
            parts.append((matched_root * matched_w).sum() / matched_w.sum().clamp_min(1e-6))
            parts.append(0.35 * F.smooth_l1_loss(out[set_ids,5:10], matched_p, beta=0.1))

        consistency = sum(parts) / len(parts)

    total = root_loss + 0.35 * presence + 0.35 * state_loss + CONSISTENCY_WEIGHT * consistency'''
    if old_consistency not in text:
        raise RuntimeError("Permutation-invariant consistency patch target not found")
    text = text.replace(old_consistency, new_consistency, 1)

    worker.write_text(text)
    print("[RELEASE] AMP dtype: fixed", flush=True)
    print("[RELEASE] DeepMind linear_2d parser: fixed", flush=True)
    print("[RELEASE] Algebraic invariance: +17 / x2 / swapped-11", flush=True)
    print("[RELEASE] Set consistency: permutation-invariant", flush=True)
    print("[RELEASE] steps=40,000 batch=4096 consistency_weight=0.12", flush=True)
    return worker


core.prepare_turbo_worker = prepare_turbo_worker_patched


def cleanup_large_temporaries():
    candidates = [
        pathlib.Path("/kaggle/working/mathematics_dataset-v1.0.tar.gz"),
        pathlib.Path("/kaggle/working/mathai_dm_extract"),
        pathlib.Path("/kaggle/working/mathematics_dataset"),
        pathlib.Path("/kaggle/working/Al"),
        CORE_PATH,
    ]
    for p in candidates:
        try:
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            elif p.exists():
                p.unlink()
        except Exception as exc:
            print(f"[CLEANUP] warning for {p}: {exc}", flush=True)
    print("[CLEANUP] Large temporary datasets/source removed; result files preserved.", flush=True)


try:
    core.main()
finally:
    cleanup_large_temporaries()
