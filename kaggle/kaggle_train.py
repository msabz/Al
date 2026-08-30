#!/usr/bin/env python3
"""Math AI v5 Kaggle launcher — patched runtime wrapper.

Loads the proven autonomous Kaggle orchestration core from the last known-good
pipeline commit, then patches the current Turbo worker before training:
1) AMP dtype-safe system targets.
2) DeepMind linear_2d comma-separated equation parsing.
3) Cleanup of large temporary datasets so Kaggle/GitHub outputs stay small.
"""

import importlib.util
import pathlib
import shutil
import urllib.request

SOURCE_COMMIT = "__SOURCE_COMMIT__"  # GitHub Actions replaces this before push.
CORE_COMMIT = "3a6fbc9036d6b4cac9f7c9e4e38faabae19659ce"
CORE_URL = f"https://raw.githubusercontent.com/msabz/Al/{CORE_COMMIT}/kaggle/kaggle_train.py"
CORE_PATH = pathlib.Path("/kaggle/working/_mathai_kaggle_core.py")

print("[PATCH] Loading autonomous Kaggle core:", CORE_COMMIT, flush=True)
urllib.request.urlretrieve(CORE_URL, CORE_PATH)

spec = importlib.util.spec_from_file_location("mathai_kaggle_core", CORE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError("Could not load Kaggle orchestration core")
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)

# Pin the cloned app/trainer source to the exact commit that launched this run.
core.SOURCE_COMMIT = SOURCE_COMMIT

_original_prepare_turbo_worker = core.prepare_turbo_worker


def prepare_turbo_worker_patched():
    worker = _original_prepare_turbo_worker()
    text = worker.read_text()

    # AMP fix: under autocast, model outputs/assigned targets are FP16 while the
    # stored system labels are FP32. Advanced indexing assignment requires exact
    # dtype equality in PyTorch.
    old_dtype = "assigned_vals[sysmask,:2] = systems[sysmask,:2] / ROOT_SCALE"
    new_dtype = "assigned_vals[sysmask,:2] = (systems[sysmask,:2] / ROOT_SCALE).to(out.dtype)"
    if old_dtype not in text:
        raise RuntimeError("AMP dtype patch target not found")
    text = text.replace(old_dtype, new_dtype, 1)

    # Official DeepMind linear_2d renders systems as: eq1, eq2 (comma separated),
    # not `eq1 and eq2`. Accept both formats without changing the math contract.
    old_parser = 'parts = [x.strip() for x in body.split(" and ") if "=" in x]'
    new_parser = 'parts = [x.strip() for x in re.split(r"\\s*(?:,|\\band\\b)\\s*", body) if "=" in x]'
    if old_parser not in text:
        raise RuntimeError("DeepMind linear_2d parser patch target not found")
    text = text.replace(old_parser, new_parser, 1)

    worker.write_text(text)
    print("[PATCH] AMP dtype target fix: applied", flush=True)
    print("[PATCH] DeepMind linear_2d comma parser: applied", flush=True)
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
