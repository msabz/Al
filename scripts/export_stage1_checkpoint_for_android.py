#!/usr/bin/env python3
"""Convert the already-trained Open-Growth stage-1 PyTorch checkpoint to the Android checkpoint format.

This does NOT initialize or retrain a model. It transfers the exact current model weights/masks,
Open-Growth structural state, AdamW first/second moments, utility/appearance/protection/cooldown state,
and the previous-important set from the existing stage-1 checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import torch

MODEL_MAGIC = 0x4D4F4456
WRAPPER_VERSION = 1
RSNN_MAGIC = 0x52534E33
RSNN_VERSION = 3
EXPECTED_DEEPMIND_COMMIT = "427f45075f84b8b9774950196ad63867ca20ffb3"
EXPECTED_SHA256 = "e4bf889fe38f11a9f6eff4b144731443013c60844d91920439f2ebef41c2270e"
TOTAL_WEIGHTS = 26880

CONFIG = {
    "hiddenDim": 160,
    "timeSteps": 25,
    "decay": 0.88,
    "threshold": 1.0,
    "initialSparsity": 0.30,
    # Stage-1 scheduler had already reached eta_min. Continue from that LR rather than jumping back to 0.003.
    "learningRate": 1e-5,
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
    # Android batch=16; 512 batches = 8192 examples, matching one Python structural cycle.
    "structureEveryBatches": 512,
    "batchSize": 16,
    "inputClip": 1.0,
    "membraneClip": 8.0,
    "checkpointMinutes": 5,
    "minBatteryPercent": 20,
    "coreRefreshMs": 350,
}


def write_utf(f, text: str) -> None:
    raw = text.encode("utf-8")
    if len(raw) > 65535:
        raise ValueError("DataOutputStream.writeUTF length exceeded")
    f.write(struct.pack(">H", len(raw)))
    f.write(raw)


def write_i32(f, value: int) -> None:
    f.write(struct.pack(">i", int(value)))


def write_i64(f, value: int) -> None:
    f.write(struct.pack(">q", int(value)))


def write_bool(f, value: bool) -> None:
    f.write(b"\x01" if value else b"\x00")


def write_f32_array(f, value) -> None:
    f.write(np.asarray(value, dtype=">f4").reshape(-1).tobytes())


def write_i32_array(f, value) -> None:
    f.write(np.asarray(value, dtype=">i4").reshape(-1).tobytes())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta")
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if ck.get("variant") != "open_growth":
        raise SystemExit(f"wrong variant: {ck.get('variant')!r}")
    if ck.get("deepmind_commit") != EXPECTED_DEEPMIND_COMMIT:
        raise SystemExit("DeepMind commit mismatch")

    ms = ck["model_state"]
    opt = ck["optimizer_state"]
    topo = ck["topology_meta"]
    param_map = [("in", 0), ("rec", 1), ("out", 2)]
    optimizer_steps = {int(float(opt["state"][pid]["step"])) for _, pid in param_map}
    if len(optimizer_steps) != 1:
        raise SystemExit(f"optimizer step mismatch: {optimizer_steps}")
    optimizer_step = optimizer_steps.pop()

    active = sum(int((ms[f"M_{n}"] > 0.5).sum().item()) for n, _ in param_map)
    if active != 25815:
        raise SystemExit(f"unexpected active count {active}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    config_json = json.dumps(CONFIG, separators=(",", ":"))

    with out.open("wb") as f:
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
        write_bool(f, bool(topo["topology_stable"]))

        for name, pid in param_map:
            w = ms[f"W_{name}"].detach().cpu().contiguous().numpy()
            mask = (ms[f"M_{name}"].detach().cpu().contiguous().numpy() > 0.5).astype(np.uint8)
            moment1 = opt["state"][pid]["exp_avg"].detach().cpu().contiguous().numpy()
            moment2 = opt["state"][pid]["exp_avg_sq"].detach().cpu().contiguous().numpy()
            utility = ms[f"utility_{name}"].detach().cpu().contiguous().numpy()
            appearance = ms[f"appearance_{name}"].detach().cpu().contiguous().numpy().astype(np.int32)
            protected = ms[f"protected_{name}"].detach().cpu().contiguous().numpy().astype(np.uint8)
            cooldown = ms[f"cooldown_{name}"].detach().cpu().contiguous().numpy().astype(np.int8)
            write_i32(f, w.size)
            write_f32_array(f, w)
            f.write(mask.reshape(-1).tobytes())
            write_f32_array(f, moment1)
            write_f32_array(f, moment2)
            write_f32_array(f, utility)
            write_i32_array(f, appearance)
            f.write(protected.reshape(-1).tobytes())
            f.write(cooldown.reshape(-1).tobytes())

        prev = np.zeros(TOTAL_WEIGHTS, dtype=np.uint8)
        for idx in topo.get("prev_important") or []:
            prev[int(idx)] = 1
        f.write(prev.tobytes())

    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise SystemExit(f"converted checkpoint SHA mismatch: {digest} != {EXPECTED_SHA256}")

    meta = {
        "source": "msabz/Al GitHub Actions artifact seven-hour-stage1-open_growth",
        "source_checkpoint": str(args.checkpoint),
        "trained_seconds": float(ck["completed_training_seconds"]),
        "examples_seen": int(ck["examples_seen"]),
        "python_optimizer_step": optimizer_step,
        "stream_cycle": int(ck["cycle"]),
        "active_weights": active,
        "phase": topo["phase"],
        "topology_stable": bool(topo["topology_stable"]),
        "best_validation_mae": float(ck["best_mae"]),
        "deepmind_commit": ck["deepmind_commit"],
        "android_checkpoint_sha256": digest,
        "resume_learning_rate": CONFIG["learningRate"],
        "transferred_optimizer_moments": True,
    }
    if args.meta:
        Path(args.meta).write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("ANDROID_PRETRAINED_CHECKPOINT_READY " + json.dumps(meta, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
