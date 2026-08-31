#!/usr/bin/env python3
"""Apply release-only MAI5 v4 plumbing after the model change is already validated.

No architecture or loss changes are made here. This updates the Kaggle entrypoint,
registered workflow labels/contracts, and makes official DeepMind audit-bank seeding
fully deterministic across bank construction order.
"""

from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one anchor, found {count}")
    return text.replace(old, new, 1)


def patch_entrypoint() -> None:
    path = ROOT / "kaggle/kaggle_train.py"
    text = path.read_text()
    text = replace_once(
        text,
        '"""Thin Kaggle entrypoint for the reviewed MAI5 v3 DeepMind-only trainer."""',
        '"""Thin Kaggle entrypoint for the reviewed MAI5 v4 DeepMind-only trainer."""',
        "entrypoint docstring",
    )
    text = replace_once(
        text,
        'kaggle/deepmind_only_train_v3.py',
        'kaggle/deepmind_only_train_v4.py',
        "entrypoint wrapper URL",
    )
    text = replace_once(
        text,
        '_deepmind_only_train_v3.py',
        '_deepmind_only_train_v4.py',
        "entrypoint local wrapper",
    )
    path.write_text(text)


def patch_registered_workflow() -> None:
    path = ROOT / ".github/workflows/kaggle-train.yml"
    text = path.read_text()
    if text.count("version != 3") != 1:
        raise RuntimeError("workflow MAI5 header version anchor changed")
    text = text.replace("version != 3", "version != 4", 1)
    # All remaining v3 tokens in this workflow are release names/messages/slug contracts.
    text = text.replace("MAI5-v3", "MAI5-v4")
    text = text.replace("MAI5 v3", "MAI5 v4")
    text = text.replace("mai5-v3", "mai5-v4")
    text = text.replace("V3", "V4")
    text = text.replace("v3", "v4")
    required = (
        "Kaggle GPU Train v5 MAI5-v4",
        "math-ai-v5-mai5-v4-trainer",
        "version != 4",
        "Report model format mismatch",
        "MathAI-v5-MAI5-v4-Kaggle-evidence",
        "MathAI-v5-MAI5-v4-VERIFIED",
    )
    missing = [marker for marker in required if marker not in text]
    if missing:
        raise RuntimeError(f"v4 workflow missing markers after migration: {missing}")
    forbidden = ("MAI5-v3", "MAI5 v3", "mai5-v3", "deepmind_only_train_v3")
    present = [marker for marker in forbidden if marker in text]
    if present:
        raise RuntimeError(f"legacy v3 workflow tokens remain: {present}")
    path.write_text(text)


def patch_audit_reseed() -> None:
    path = ROOT / "colab/generalization_audit.py"
    text = path.read_text()
    anchor = '''    try:\n        ns["synthetic"] = project_synthetic_forbidden\n        if spec["split"] == "interpolate":\n'''
    replacement = '''    try:\n        ns["synthetic"] = project_synthetic_forbidden\n        # DEEPMIND_BANK_GLOBAL_RESEED: official generator modules also consume\n        # process-global RNG state, so local random.Random(spec["seed"]) alone is\n        # insufficient to make a bank reproducible across construction order.\n        bank_seed = int(spec["seed"])\n        random.seed(bank_seed)\n        ns["np"].random.seed(bank_seed & 0xFFFFFFFF)\n        ns["torch"].manual_seed(bank_seed)\n        if spec["split"] == "interpolate":\n'''
    text = replace_once(text, anchor, replacement, "audit global reseed insertion")
    text = replace_once(
        text,
        '        rng = random.Random(int(spec["seed"]))',
        '        rng = random.Random(bank_seed)',
        "audit local RNG",
    )
    path.write_text(text)


def main() -> None:
    patch_entrypoint()
    patch_registered_workflow()
    patch_audit_reseed()
    print("MAI5_V4_RELEASE_PLUMBING_APPLIED")


if __name__ == "__main__":
    main()
