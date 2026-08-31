#!/usr/bin/env python3
import argparse, json, math, pathlib, random, re, statistics


def replace_setting(text, name, value_repr):
    pattern = rf"(?m)^{re.escape(name)}\s*=\s*.*$"
    out, n = re.subn(pattern, f"{name} = {value_repr}", text, count=1)
    if n != 1:
        raise RuntimeError(f"setting not found: {name}")
    return out


def load_runtime(root: pathlib.Path, work: pathlib.Path):
    src_path = root / "colab/train_v5_deepmind.py"
    src = src_path.read_text().replace("/content", str(work))
    src = replace_setting(src, "RESUME_FROM_MAI5", "False")
    src = replace_setting(src, "AUTO_DOWNLOAD_AT_END", "False")
    prefix = src.split("# ========================= TRAIN =========================", 1)[0]
    ns = {"__name__": "run8_diag", "__file__": str(src_path)}
    exec(compile(prefix, str(src_path), "exec"), ns)
    return ns


def target_magnitude(ex):
    vals = list(ex.get("roots", ())) + list(ex.get("system", ()))
    return max((abs(float(v)) for v in vals), default=0.0)


def make_bank(ns, split, count, seed, family_name=None):
    old_modules = ns["dm_modules"]
    old_names = list(ns["DM_NAMES"])
    try:
        if split == "interpolate":
            ns["dm_modules"] = ns["dm_algebra"].test()
            names = ["linear_1d", "linear_2d", "polynomial_roots"]
        elif split == "extrapolate":
            ns["dm_modules"] = ns["dm_algebra"].test_extra()
            names = ["polynomial_roots_big"]
        else:
            raise ValueError(split)
        if family_name is not None:
            names = [family_name]
        ns["DM_NAMES"] = names
        rng = random.Random(seed)
        result, seen = [], set()
        attempts = 0
        while len(result) < count and attempts < count * 3000:
            attempts += 1
            try:
                ex = ns["deepmind_example"](rng, allow_synthetic_fallback=False)
            except Exception:
                continue
            if target_magnitude(ex) > 300.0:
                continue
            if ex["eq"] in seen:
                continue
            seen.add(ex["eq"])
            result.append(ex)
        if len(result) < count:
            raise RuntimeError(f"{split}/{family_name or 'mixed'} only produced {len(result)}/{count}")
        return result
    finally:
        ns["dm_modules"] = old_modules
        ns["DM_NAMES"] = old_names


def predict_raw(ns, model, equation):
    torch = ns["torch"]
    np = ns["np"]
    k, n, d, fam, _ = ns["encode"](equation)
    device = ns["device"]
    with torch.no_grad():
        out = model(
            torch.tensor(k[None, :], device=device, dtype=torch.long),
            torch.tensor(n[None, :], device=device, dtype=torch.float32),
            torch.tensor(d[None, :], device=device, dtype=torch.float32),
            torch.tensor([fam], device=device, dtype=torch.long),
        )[0]
        return {
            "family": int(fam),
            "slots": (out[:5] * ns["ROOT_SCALE"]).detach().cpu().numpy().astype(float),
            "presence": torch.sigmoid(out[5:10]).detach().cpu().numpy().astype(float),
            "state": int(torch.argmax(out[10:14]).item()),
        }


def evaluate(ns, model, examples, keep_worst=False):
    np = ns["np"]
    sq = ae = target_sq = 0.0
    count = within = state_ok = missing = 0
    worst = []
    model.eval()
    for ex in examples:
        p = predict_raw(ns, model, ex["eq"])
        state_ok += int(p["state"] == ex["state"])
        if ex["state"] != ns["FINITE"]:
            continue
        if ex["f"] == ns["SYSTEM"]:
            expected = np.asarray(ex["system"], dtype=float)
            predicted = p["slots"][:len(expected)]
            errors = np.abs(predicted - expected)
        else:
            expected = np.asarray(ex["roots"], dtype=float)
            predicted = p["slots"][p["presence"] >= 0.5]
            used, errs = set(), []
            for value in expected:
                candidates = [(abs(float(v)-float(value)), j) for j,v in enumerate(predicted) if j not in used]
                if candidates:
                    er,j = min(candidates); used.add(j); errs.append(er)
                else:
                    errs.append(float(ns["ROOT_SCALE"])); missing += 1
            errors = np.asarray(errs, dtype=float)
        if len(errors):
            sq += float((errors**2).sum())
            ae += float(errors.sum())
            target_sq += float((expected**2).sum())
            count += len(errors)
            within += int((errors <= 1.0).sum())
            if keep_worst:
                worst.append((float(errors.max()), ex["eq"], [float(x) for x in expected], [float(x) for x in predicted]))
    rmse = math.sqrt(sq/max(count,1))
    target_rms = math.sqrt(target_sq/max(count,1))
    out = {
        "examples": len(examples), "value_count": count,
        "rmse": rmse, "mae": ae/max(count,1),
        "nrmse_target_rms": rmse/max(target_rms,1e-9),
        "target_rms": target_rms,
        "within_one_ratio": within/max(count,1),
        "state_accuracy": state_ok/max(len(examples),1),
        "missing_value_slots": missing,
    }
    if keep_worst:
        out["worst"] = [
            {"max_abs_error": e, "equation": q, "expected": ev, "predicted": pv}
            for e,q,ev,pv in sorted(worst, reverse=True)[:12]
        ]
    return out


def median_baseline(ns, examples, seeds):
    torch = ns["torch"]
    rows=[]
    for seed in seeds:
        torch.manual_seed(seed)
        m = ns["MAI5"]().to(ns["device"])
        rows.append(evaluate(ns,m,examples))
    keys=["rmse","mae","nrmse_target_rms","within_one_ratio","state_accuracy","missing_value_slots"]
    return {
        "seeds": list(seeds),
        "median": {k: statistics.median(r[k] for r in rows) for k in keys},
        "min_rmse": min(r["rmse"] for r in rows),
        "max_rmse": max(r["rmse"] for r in rows),
    }


def gains(base, trained):
    b=base["median"]
    return {
        "rmse_gain": (b["rmse"]-trained["rmse"])/max(b["rmse"],1e-9),
        "mae_gain": (b["mae"]-trained["mae"])/max(b["mae"],1e-9),
        "within_one_delta": trained["within_one_ratio"]-b["within_one_ratio"],
        "missing_delta": trained["missing_value_slots"]-b["missing_value_slots"],
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", default="run8_diagnostic.json")
    args=ap.parse_args()
    root=pathlib.Path(args.root).resolve()
    work=pathlib.Path("/tmp/mathai_run8_diag")
    work.mkdir(parents=True,exist_ok=True)
    ns=load_runtime(root,work)
    ns["load_mai5"](args.model)
    model=ns["model"]

    banks={
        "interp_linear_1d": make_bank(ns,"interpolate",192,0x810001,"linear_1d"),
        "interp_linear_2d": make_bank(ns,"interpolate",192,0x810002,"linear_2d"),
        "interp_polynomial": make_bank(ns,"interpolate",192,0x810003,"polynomial_roots"),
        "extra_polynomial": make_bank(ns,"extrapolate",192,0x810004,"polynomial_roots_big"),
    }
    seeds=[101,211,307,401,503,601,701,809]
    report={"model": args.model, "banks":{}}
    for name,bank in banks.items():
        trained=evaluate(ns,model,bank,keep_worst=True)
        base=median_baseline(ns,bank,seeds)
        report["banks"][name]={"trained":trained,"random_baseline":base,"gains":gains(base,trained)}
        print(name, json.dumps(report["banks"][name]["gains"]))
    pathlib.Path(args.output).write_text(json.dumps(report,indent=2))
    print("WROTE",args.output)

if __name__ == "__main__":
    main()
