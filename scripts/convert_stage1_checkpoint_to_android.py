#!/usr/bin/env python3
"""Convert the exact seven-hour stage-1 PyTorch Open-Growth checkpoint
into the Android RSNN Lab V2 checkpoint format.

This preserves weights, masks, Adam moments, structural utility/appearance,
protected/cooldown state, topology metadata and the exact optimizer step.
It does not initialize or retrain the network.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import torch

MODEL_MAGIC = 0x4D4F4456
WRAPPER_VERSION = 1
RSNN_MAGIC = 0x52534E33
RSNN_VERSION = 3


def write_i32(f, v): f.write(struct.pack(">i", int(v)))
def write_i64(f, v): f.write(struct.pack(">q", int(v)))
def write_bool(f, v): f.write(b"\x01" if v else b"\x00")
def write_utf(f, s: str):
    # All strings emitted here are ASCII, therefore Java modified UTF-8 is
    # identical to ordinary UTF-8 for this payload.
    b = s.encode("utf-8")
    if len(b) > 65535: raise ValueError("Java UTF field too long")
    f.write(struct.pack(">H", len(b)))
    f.write(b)


def float_bytes(t):
    return t.detach().cpu().contiguous().view(-1).numpy().astype(">f4", copy=False).tobytes()


def int32_bytes(t):
    return t.detach().cpu().contiguous().view(-1).numpy().astype(">i4", copy=False).tobytes()


def mask_bytes(t):
    return bytes((t.detach().cpu().contiguous().view(-1).numpy() > 0.5).astype("uint8").tolist())


def bool_bytes(t):
    return bytes(t.detach().cpu().contiguous().view(-1).numpy().astype("uint8").tolist())


def int8_raw_bytes(t):
    a = t.detach().cpu().contiguous().view(-1).numpy().astype("int8", copy=False)
    return a.view("uint8").tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--meta", type=Path)
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if ck.get("format") != "RSNN_SEVEN_HOUR_CHECKPOINT_V1":
        raise RuntimeError("unexpected checkpoint format")
    if ck.get("variant") != "open_growth":
        raise RuntimeError("checkpoint is not open_growth")
    if ck.get("deepmind_commit") != "427f45075f84b8b9774950196ad63867ca20ffb3":
        raise RuntimeError("DeepMind source commit mismatch")

    state = ck["model_state"]
    topo = ck["topology_meta"]
    opt = ck["optimizer_state"]["state"]
    params = {"in": opt[0], "rec": opt[1], "out": opt[2]}
    optimizer_steps = [int(float(params[n]["step"])) for n in ("in", "rec", "out")]
    if len(set(optimizer_steps)) != 1:
        raise RuntimeError(f"optimizer step mismatch: {optimizer_steps}")
    optimizer_step = optimizer_steps[0]

    # The cloud scheduler had already reached 1e-5 at this checkpoint. Resuming
    # at the original 0.003 would be a discontinuity, so the Android continuation
    # starts from the checkpoint's current LR.
    config = {
        "hiddenDim": 160,
        "timeSteps": 25,
        "decay": 0.88,
        "threshold": 1.0,
        "initialSparsity": 0.30,
        "learningRate": float(ck["optimizer_state"]["param_groups"][0]["lr"]),
        "weightDecay": 1e-4,
        "adamBeta1": 0.9,
        "adamBeta2": 0.999,
        "adamEps": 1e-8,
        "gradientClip": 5.0,
        "utilityBeta": 0.95,
        "importantFraction": 0.20,
        "protectFraction": 0.02,
        "pruneFraction": 0.02,
        "noveltyLimit": 0.01,
        "stableCycles": 3,
        "regrowInitScale": 0.01,
        "structureEveryBatches": 32,
        "batchSize": 256,
        "inputClip": 1.0,
        "membraneClip": 8.0,
        "checkpointMinutes": 5,
        "minBatteryPercent": 20,
        "coreRefreshMs": 350,
    }
    config_json = json.dumps(config, separators=(",", ":"))

    def write_state(f, name: str):
        w = state[f"W_{name}"]
        mask = state[f"M_{name}"]
        utility = state[f"utility_{name}"]
        appearance = state[f"appearance_{name}"]
        protected = state[f"protected_{name}"]
        cooldown = state[f"cooldown_{name}"]
        mom = params[name]["exp_avg"]
        vel = params[name]["exp_avg_sq"]
        n = w.numel()
        for x in (mask, utility, appearance, protected, cooldown, mom, vel):
            if x.numel() != n: raise RuntimeError(f"state size mismatch: {name}")
        write_i32(f, n)
        f.write(float_bytes(w))
        f.write(mask_bytes(mask))
        f.write(float_bytes(mom))
        f.write(float_bytes(vel))
        f.write(float_bytes(utility))
        f.write(int32_bytes(appearance))
        f.write(bool_bytes(protected))
        f.write(int8_raw_bytes(cooldown))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as f:
        write_i32(f, MODEL_MAGIC)
        write_i32(f, WRAPPER_VERSION)
        write_utf(f, config_json)
        write_i32(f, RSNN_MAGIC)
        write_i32(f, RSNN_VERSION)
        write_i64(f, optimizer_step)
        write_utf(f, str(topo["phase"]))
        write_i32(f, topo["structural_cycle"])
        write_i32(f, topo["growth_streak"])
        write_i32(f, topo["selection_streak"])
        write_i32(f, topo["selection_cycles"])
        write_bool(f, topo["topology_stable"])
        write_state(f, "in")
        write_state(f, "rec")
        write_state(f, "out")
        total = 160 * 6 + 160 * 160 + 2 * 160
        prev = bytearray(total)
        for idx in topo.get("prev_important") or []:
            idx = int(idx)
            if 0 <= idx < total: prev[idx] = 1
        f.write(prev)

    source_sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    output_sha = hashlib.sha256(args.output.read_bytes()).hexdigest()
    active = sum(int((state[f"M_{n}"] > 0.5).sum()) for n in ("in", "rec", "out"))
    protected_count = sum(int(state[f"protected_{n}"].sum()) for n in ("in", "rec", "out"))
    meta = {
        "source_checkpoint_sha256": source_sha,
        "android_checkpoint_sha256": output_sha,
        "source_format": ck["format"],
        "source_variant": ck["variant"],
        "deepmind_commit": ck["deepmind_commit"],
        "completed_training_seconds": ck["completed_training_seconds"],
        "examples_seen": ck["examples_seen"],
        "optimizer_step": optimizer_step,
        "structural_cycle": topo["structural_cycle"],
        "phase": topo["phase"],
        "selection_cycles": topo["selection_cycles"],
        "topology_stable": topo["topology_stable"],
        "active_weights": active,
        "protected_weights": protected_count,
        "best_validation_mae": ck["best_mae"],
        "config": config,
        "note": "Converted from the exact stage-1 open_growth checkpoint; not reinitialized.",
    }
    meta_path = args.meta or args.output.with_suffix(args.output.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(meta, sort_keys=True))


if __name__ == "__main__":
    main()
