#!/usr/bin/env python3
"""Seven-hour three-way RSNN experiment with no validation early stopping.

Each variant gets exactly two 3.5-hour training phases on GitHub-hosted runners.
The split avoids the per-job hosted-runner ceiling while preserving model,
optimizer, LR scheduler and structural-plasticity state across phases.

Variants:
  adopted_p30 : adopted P30 + INT8-QAT + fixed T=25
  open_growth : P30 start + unconstrained growth + 2% protect / 2% prune cycles
  dense_long  : same RSNN/QAT recipe with no pruning at all

There is no plateau/degradation stop. Training stops only when the two timed
phases are exhausted. Validation is used only to remember the best checkpoint.
Official/stress/ill banks are evaluated only after all seven training hours.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

import deepmind_rsnn_open_ended_three_way as base
import deepmind_rsnn_open_ended_three_way_fixed as fixedmod

PHASE_BUDGET_DEFAULT = 12600.0  # 3h30m x 2 = 7 hours total training time.
TOTAL_TRAIN_BUDGET = PHASE_BUDGET_DEFAULT * 2
VARIANTS = ("adopted_p30", "open_growth", "dense_long")


def bank_payload(bank: base.dm.Bank) -> Dict[str, object]:
    return {
        "name": bank.name,
        "x": bank.x.cpu(),
        "y_raw": bank.y_raw.cpu(),
        "condition": bank.condition.cpu(),
        "sample_questions": list(bank.sample_questions),
        "attempts": int(bank.attempts),
    }


def bank_from_payload(d: Dict[str, object]) -> base.dm.Bank:
    return base.dm.Bank(
        name=str(d["name"]),
        x=d["x"],
        y_raw=d["y_raw"],
        condition=d["condition"],
        sample_questions=list(d["sample_questions"]),
        attempts=int(d["attempts"]),
    )


def prepare_shared_banks(path: Path) -> None:
    _, fixed, _ = base.build_fixed_banks(None)
    fingerprints = {k: base.bank_fingerprint(v) for k, v in fixed.items()}
    payload = {"deepmind_commit": base.dm.DM_COMMIT,
               "fingerprints": fingerprints,
               "banks": {k: bank_payload(v) for k, v in fixed.items()}}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    path.with_suffix(".fingerprints.json").write_text(
        json.dumps(fingerprints, indent=2, sort_keys=True), encoding="utf-8")
    print("SEVEN_HOUR_SHARED_BANKS_READY " + json.dumps(fingerprints, sort_keys=True), flush=True)


def load_shared_banks(path: Path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["deepmind_commit"] != base.dm.DM_COMMIT:
        raise RuntimeError("DeepMind commit mismatch in shared banks")
    fixed = {k: bank_from_payload(v) for k, v in payload["banks"].items()}
    fingerprints = {k: base.bank_fingerprint(v) for k, v in fixed.items()}
    if fingerprints != payload["fingerprints"]:
        raise RuntimeError("shared-bank fingerprint corruption")
    fixed_seen = set()
    for bank in fixed.values():
        fixed_seen.update(base.bank_keys(bank))
    return fixed, fingerprints, fixed_seen


def make_model(variant: str):
    base.set_seed(base.SEED)
    if variant == "open_growth":
        return fixedmod.FixedOpenGrowthRSNN()
    return base.qmod.QATP30RSNN()


def topology_meta(model) -> Dict[str, object]:
    if not isinstance(model, fixedmod.FixedOpenGrowthRSNN):
        return {"phase": "fixed", "topology_stable": True}
    return {
        "phase": model.phase,
        "structural_cycle": model.structural_cycle,
        "growth_streak": model.growth_streak,
        "selection_streak": model.selection_streak,
        "selection_cycles": model.selection_cycles,
        "topology_stable": model.topology_stable,
        "prev_important": sorted(model.prev_important) if model.prev_important is not None else None,
        "events": list(model.events),
    }


def restore_topology_meta(model, meta: Dict[str, object]) -> None:
    if not isinstance(model, fixedmod.FixedOpenGrowthRSNN):
        return
    model.phase = str(meta["phase"])
    model.structural_cycle = int(meta["structural_cycle"])
    model.growth_streak = int(meta["growth_streak"])
    model.selection_streak = int(meta["selection_streak"])
    model.selection_cycles = int(meta["selection_cycles"])
    model.topology_stable = bool(meta["topology_stable"])
    prev = meta.get("prev_important")
    model.prev_important = set(int(x) for x in prev) if prev is not None else None
    model.events = list(meta.get("events", []))


def clone_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def warmup_no_early_stop(model, variant, warm, optimizer, criterion, phase_start, phase_budget):
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=300, eta_min=base.arch.ETA_MIN)
    hist = []
    for epoch in range(1, base.WARMUP_EPOCHS + 1):
        if time.monotonic() - phase_start >= phase_budget:
            raise RuntimeError("phase budget exhausted during warmup")
        if variant != "dense_long":
            target = base.sweep.pruning_target(epoch, base.FINAL_SPARSITY)
            if target is not None:
                model.prune_synapses(target, optimizer)
        model.quant_enabled = False
        loss = base.train_batches(model, warm, optimizer, criterion,
                                  seed=base.SEED, cycle=epoch, passes=1)
        scheduler.step()
        if epoch in (1, 50, 100, 150):
            hist.append({"epoch": epoch, "loss": loss, "active": base.count_active(model)})
    expected = base.TOTAL_WEIGHTS if variant == "dense_long" else base.P30_ACTIVE
    if base.count_active(model) != expected:
        raise AssertionError(f"warmup active={base.count_active(model)} expected={expected}")
    model.quant_enabled = True
    return hist


def independent_stream_bank(alg, fixed_seen, cycle: int):
    # Each cycle depends only on fixed held-out exclusions + its cycle seed.
    # This makes STREAM_N bit-identical across variants even when variants
    # progress at different speeds. Repetition across different stream cycles
    # is allowed; within each cycle and against held-out banks it is excluded.
    local_seen = set(fixed_seen)
    return base.dm.build_bank(
        alg,
        name=f"STREAM_{cycle:06d}",
        kind="train",
        count=base.STREAM_CHUNK,
        seed=15000 + cycle,
        global_seen=local_seen,
    )


def init_training(variant: str, fixed, phase_start, phase_budget):
    model = make_model(variant).to(base.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base.arch.LR,
                                  weight_decay=base.arch.WEIGHT_DECAY)
    criterion = nn.SmoothL1Loss(beta=1.0)
    warm_hist = warmup_no_early_stop(model, variant, fixed["WARMUP_TRAIN"],
                                     optimizer, criterion, phase_start, phase_budget)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5)
    val = base.eval_mae(model, fixed["EARLYSTOP_VALIDATION"])
    stable = variant != "open_growth" or model.topology_stable
    return {
        "model": model,
        "optimizer": optimizer,
        "criterion": criterion,
        "scheduler": scheduler,
        "cycle": 0,
        "examples_seen": 8192 * base.WARMUP_EPOCHS,
        "history": [],
        "warmup_history": warm_hist,
        "best_state": clone_state(model),
        "best_mae": val,
        "best_cycle": 0,
        "best_meta": topology_meta(model),
        "best_stable_state": clone_state(model) if stable else None,
        "best_stable_mae": val if stable else math.inf,
        "best_stable_cycle": 0 if stable else None,
        "best_stable_meta": topology_meta(model) if stable else None,
        "completed_training_seconds": 0.0,
    }


def load_training(variant: str, checkpoint: Path):
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if ck["variant"] != variant:
        raise RuntimeError("checkpoint variant mismatch")
    model = make_model(variant).to(base.DEVICE)
    model.load_state_dict(ck["model_state"], strict=True)
    restore_topology_meta(model, ck["topology_meta"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=base.arch.LR,
                                  weight_decay=base.arch.WEIGHT_DECAY)
    optimizer.load_state_dict(ck["optimizer_state"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5)
    scheduler.load_state_dict(ck["scheduler_state"])
    return {
        "model": model,
        "optimizer": optimizer,
        "criterion": nn.SmoothL1Loss(beta=1.0),
        "scheduler": scheduler,
        "cycle": int(ck["cycle"]),
        "examples_seen": int(ck["examples_seen"]),
        "history": list(ck["history"]),
        "warmup_history": list(ck["warmup_history"]),
        "best_state": ck["best_state"],
        "best_mae": float(ck["best_mae"]),
        "best_cycle": int(ck["best_cycle"]),
        "best_meta": ck["best_meta"],
        "best_stable_state": ck.get("best_stable_state"),
        "best_stable_mae": float(ck.get("best_stable_mae", math.inf)),
        "best_stable_cycle": ck.get("best_stable_cycle"),
        "best_stable_meta": ck.get("best_stable_meta"),
        "completed_training_seconds": float(ck.get("completed_training_seconds", 0.0)),
    }


def run_timed_phase(state, variant: str, fixed, fixed_seen, phase_budget: float, phase_number: int):
    alg = base.dm.prepare_deepmind()
    model = state["model"]
    optimizer = state["optimizer"]
    criterion = state["criterion"]
    scheduler = state["scheduler"]
    phase_start = time.monotonic()

    while time.monotonic() - phase_start < phase_budget:
        state["cycle"] += 1
        cycle = state["cycle"]
        chunk = independent_stream_bank(alg, fixed_seen, cycle)
        utility_cb = (model.update_train_utility_
                      if variant == "open_growth" and not model.topology_stable else None)
        loss = base.train_batches(model, chunk, optimizer, criterion,
                                  seed=base.SEED, cycle=1000 + cycle,
                                  passes=base.PASSES_PER_CHUNK, utility_cb=utility_cb)
        state["examples_seen"] += base.STREAM_CHUNK * base.PASSES_PER_CHUNK

        event = None
        if variant == "open_growth" and not model.topology_stable:
            event = model.structural_step(
                optimizer, fixed["STRUCTURE_VALIDATION"], criterion, seed=base.SEED)

        val = base.eval_mae(model, fixed["EARLYSTOP_VALIDATION"])
        scheduler.step(val)
        if val < state["best_mae"]:
            state["best_mae"] = val
            state["best_state"] = clone_state(model)
            state["best_cycle"] = cycle
            state["best_meta"] = topology_meta(model)

        stable = variant != "open_growth" or model.topology_stable
        if stable and val < state["best_stable_mae"]:
            state["best_stable_mae"] = val
            state["best_stable_state"] = clone_state(model)
            state["best_stable_cycle"] = cycle
            state["best_stable_meta"] = topology_meta(model)

        elapsed = time.monotonic() - phase_start
        row = {
            "phase": phase_number,
            "cycle": cycle,
            "train_loss": loss,
            "val_mae": val,
            "best_val_mae": state["best_mae"],
            "best_stable_val_mae": (None if not math.isfinite(state["best_stable_mae"])
                                     else state["best_stable_mae"]),
            "lr": optimizer.param_groups[0]["lr"],
            "active": base.count_active(model),
            "topology_phase": getattr(model, "phase", "fixed"),
            "topology_stable": bool(getattr(model, "topology_stable", True)),
            "phase_elapsed_s": elapsed,
            "event": event,
        }
        state["history"].append(row)
        print("SEVEN_HOUR_CYCLE " + json.dumps({k: v for k, v in row.items() if k != "event"}, sort_keys=True), flush=True)
        if event is not None:
            print("SEVEN_HOUR_STRUCTURE " + json.dumps(event, sort_keys=True), flush=True)

    state["completed_training_seconds"] += time.monotonic() - phase_start
    return state


def save_checkpoint(state, variant: str, fingerprints, path: Path) -> None:
    model = state["model"]
    payload = {
        "format": "RSNN_SEVEN_HOUR_CHECKPOINT_V1",
        "variant": variant,
        "deepmind_commit": base.dm.DM_COMMIT,
        "bank_fingerprints": fingerprints,
        "model_state": clone_state(model),
        "topology_meta": topology_meta(model),
        "optimizer_state": state["optimizer"].state_dict(),
        "scheduler_state": state["scheduler"].state_dict(),
        "cycle": state["cycle"],
        "examples_seen": state["examples_seen"],
        "history": state["history"],
        "warmup_history": state["warmup_history"],
        "best_state": state["best_state"],
        "best_mae": state["best_mae"],
        "best_cycle": state["best_cycle"],
        "best_meta": state["best_meta"],
        "best_stable_state": state["best_stable_state"],
        "best_stable_mae": state["best_stable_mae"],
        "best_stable_cycle": state["best_stable_cycle"],
        "best_stable_meta": state["best_stable_meta"],
        "completed_training_seconds": state["completed_training_seconds"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def finalize(state, variant: str, fixed, fingerprints, out: Path) -> None:
    model = state["model"]
    if variant == "open_growth" and state["best_stable_state"] is not None:
        chosen_state = state["best_stable_state"]
        chosen_mae = state["best_stable_mae"]
        chosen_cycle = state["best_stable_cycle"]
        chosen_meta = state["best_stable_meta"]
        chosen_kind = "best_after_topology_stability"
    else:
        chosen_state = state["best_state"]
        chosen_mae = state["best_mae"]
        chosen_cycle = state["best_cycle"]
        chosen_meta = state["best_meta"]
        chosen_kind = ("best_overall_topology_never_stabilized"
                       if variant == "open_growth" else "best_overall")

    model.load_state_dict(chosen_state, strict=True)
    restore_topology_meta(model, chosen_meta)
    model.quant_enabled = True
    model.eval()
    exported = base.qmod.Int8WeightRSNN(model).to(base.DEVICE)
    eval_names = ("OFFICIAL_INTERPOLATE", "STRESS_ENTROPY_10",
                  "STRESS_ENTROPY_12", "ILL_CONDITIONED")
    evaluations = {name: base.dm.evaluate_raw(exported, fixed[name]) for name in eval_names}
    protected_total = 0
    if variant == "open_growth":
        protected_total = sum(int(getattr(model, f"protected_{n}").sum())
                              for n, _, _ in base.matrix_triplets(model))

    result = {
        "experiment": "RSNN_SEVEN_HOUR_THREE_WAY",
        "variant": variant,
        "seed": base.SEED,
        "deepmind_commit": base.dm.DM_COMMIT,
        "training_policy": "no validation early stopping; two timed 3.5h phases; best checkpoint remembered on EARLYSTOP_VALIDATION",
        "stream_contract": "STREAM_N uses fixed held-out exclusions and seed=15000+N independently, so the same cycle is identical across variants; variants may consume different prefix lengths under equal seven-hour wall-time budgets",
        "target_training_seconds": TOTAL_TRAIN_BUDGET,
        "actual_completed_training_seconds": state["completed_training_seconds"],
        "stop_reason": "seven_hour_budget_complete",
        "stream_cycles": state["cycle"],
        "examples_seen_with_repeats": state["examples_seen"],
        "best_overall_validation_mae": state["best_mae"],
        "best_overall_cycle": state["best_cycle"],
        "best_after_topology_stability_validation_mae": (None if not math.isfinite(state["best_stable_mae"]) else state["best_stable_mae"]),
        "best_after_topology_stability_cycle": state["best_stable_cycle"],
        "selected_checkpoint_kind": chosen_kind,
        "selected_validation_mae": chosen_mae,
        "selected_cycle": chosen_cycle,
        "active_weights_selected": base.count_active(model),
        "protected_total_selected": protected_total,
        "topology_phase_selected": getattr(model, "phase", "fixed"),
        "topology_stable_selected": bool(getattr(model, "topology_stable", True)),
        "bank_fingerprints": fingerprints,
        "storage": exported.storage_report(),
        "evaluations": evaluations,
        "warmup_history": state["warmup_history"],
        "history": state["history"],
        "structural_events": getattr(model, "events", []),
    }

    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    torch.save({"variant": variant, "state_dict": chosen_state, "topology_meta": chosen_meta}, out / "best_model_state.pt")
    lines = [
        "RSNN SEVEN-HOUR THREE-WAY",
        f"variant={variant}",
        "stop_reason=seven_hour_budget_complete",
        f"target_training_seconds={TOTAL_TRAIN_BUDGET:.0f}",
        f"actual_completed_training_seconds={state['completed_training_seconds']:.2f}",
        f"stream_cycles={state['cycle']}",
        f"examples_seen={state['examples_seen']}",
        f"selected_checkpoint_kind={chosen_kind}",
        f"selected_cycle={chosen_cycle}",
        f"selected_validation_mae={chosen_mae:.8f}",
        f"active_weights={base.count_active(model)}",
        f"protected_total={protected_total}",
        f"topology_phase={getattr(model, 'phase', 'fixed')}",
        f"topology_stable={getattr(model, 'topology_stable', True)}",
    ]
    for name, m in evaluations.items():
        lines.append(
            f"{name}: MAE={m['mae']:.8f} RMSE={m['rmse']:.8f} "
            f"strict1={100*m['strict_1_0']:.4f}% within5={100*m['within_5_0']:.4f}% "
            f"residual={m['equation_residual_mean']:.8f}")
    (out / "SUMMARY.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "bank_fingerprints.json").write_text(json.dumps(fingerprints, indent=2, sort_keys=True), encoding="utf-8")
    print("\n" + "\n".join(lines), flush=True)
    print("SEVEN_HOUR_VARIANT_COMPLETE", flush=True)


def run_stage(args) -> None:
    fixed, fingerprints, fixed_seen = load_shared_banks(Path(args.banks_file))
    phase_start = time.monotonic()
    if args.mode == "stage1":
        state = init_training(args.variant, fixed, phase_start, args.phase_budget_seconds)
        elapsed_init = time.monotonic() - phase_start
        remaining = max(1.0, args.phase_budget_seconds - elapsed_init)
        state = run_timed_phase(state, args.variant, fixed, fixed_seen, remaining, 1)
        save_checkpoint(state, args.variant, fingerprints, Path(args.checkpoint_out))
        print(f"SEVEN_HOUR_STAGE1_COMPLETE variant={args.variant} cycles={state['cycle']} seconds={state['completed_training_seconds']:.2f}", flush=True)
        return

    state = load_training(args.variant, Path(args.resume_checkpoint))
    if state["completed_training_seconds"] < 1.0:
        raise RuntimeError("stage1 training-time accounting missing")
    state = run_timed_phase(state, args.variant, fixed, fixed_seen,
                            args.phase_budget_seconds, 2)
    finalize(state, args.variant, fixed, fingerprints, Path(args.output_dir))


def preflight(variant: str) -> None:
    base.set_seed(base.SEED)
    model = make_model(variant).to(base.DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=base.arch.LR,
                            weight_decay=base.arch.WEIGHT_DECAY)
    if variant == "dense_long":
        expected = base.TOTAL_WEIGHTS
    else:
        model.prune_synapses(base.FINAL_SPARSITY, opt)
        expected = base.P30_ACTIVE
    if base.count_active(model) != expected:
        raise AssertionError((variant, base.count_active(model), expected))
    print(f"SEVEN_HOUR_PREFLIGHT_PASS variant={variant} active={expected}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["prepare", "stage1", "stage2", "preflight"], required=True)
    p.add_argument("--variant", choices=VARIANTS)
    p.add_argument("--banks-file", default="seven-hour-banks.pt")
    p.add_argument("--phase-budget-seconds", type=float, default=PHASE_BUDGET_DEFAULT)
    p.add_argument("--checkpoint-out", default="stage1-checkpoint.pt")
    p.add_argument("--resume-checkpoint")
    p.add_argument("--output-dir", default="seven-hour-output")
    args = p.parse_args()
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    if args.mode == "prepare":
        prepare_shared_banks(Path(args.banks_file)); return
    if args.mode == "preflight":
        if not args.variant: p.error("--variant required")
        preflight(args.variant); return
    if not args.variant: p.error("--variant required")
    if args.mode == "stage2" and not args.resume_checkpoint:
        p.error("--resume-checkpoint required for stage2")
    run_stage(args)


if __name__ == "__main__":
    main()
