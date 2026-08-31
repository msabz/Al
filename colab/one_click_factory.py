# Math AI v5 — one-click Google Colab factory
# Run this whole file in ONE Colab cell. It trains, selects the best checkpoint,
# injects it into the real Android app, verifies Python<->Kotlin parity, builds,
# and downloads one output ZIP.

# ========================= USER SETTINGS =========================
REPO_URL = "https://github.com/msabz/Al.git"
BRANCH = "feat/v5-deepmind-colab"
TOTAL_STEPS = 30_000
BATCH_SIZE = 128
LEARNING_RATE = 6e-4
CONSISTENCY_WEIGHT = 0.05
DEEPMIND_RATIO = 0.60
CHECKPOINT_EVERY = 1_000
RESUME_FROM_MAI5 = False   # True -> upload a previous .mai5 before training
BUILD_SIGNED_RELEASE = False  # True -> asks for your stable v5 JKS + passwords

# ========================= BOOTSTRAP =========================
import os, re, sys, json, time, shutil, hashlib, pathlib, subprocess, getpass
from datetime import datetime, timezone

ROOT = pathlib.Path("/content/MathAI-v5-factory")
WORK_MODEL = pathlib.Path("/content/math_ai_v5_working.mai5")
BEST_MODEL = pathlib.Path("/content/math_ai_v5_best.mai5")
REPORT = pathlib.Path("/content/training_report.json")
BUNDLE = pathlib.Path("/content/MathAI-v5-factory-output.zip")

print("=== Math AI v5 Automated AI App Factory ===")
print("Repository:", REPO_URL)
print("Branch:", BRANCH)

if ROOT.exists():
    shutil.rmtree(ROOT)
subprocess.run(["git", "clone", "-q", "--branch", BRANCH, "--single-branch", REPO_URL, str(ROOT)], check=True)
commit_sha = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
print("App source commit:", commit_sha)

trainer_src = (ROOT / "colab/train_v5_deepmind.py").read_text()

def replace_setting(text, name, value_repr):
    pattern = rf"(?m)^{re.escape(name)}\s*=\s*.*$"
    updated, count = re.subn(pattern, f"{name} = {value_repr}", text, count=1)
    if count != 1:
        raise RuntimeError(f"Could not patch trainer setting: {name}")
    return updated

trainer = trainer_src
trainer = replace_setting(trainer, "TOTAL_STEPS", str(int(TOTAL_STEPS)))
trainer = replace_setting(trainer, "BATCH_SIZE", str(int(BATCH_SIZE)))
trainer = replace_setting(trainer, "LEARNING_RATE", repr(float(LEARNING_RATE)))
trainer = replace_setting(trainer, "CONSISTENCY_WEIGHT", repr(float(CONSISTENCY_WEIGHT)))
trainer = replace_setting(trainer, "DEEPMIND_RATIO", repr(float(DEEPMIND_RATIO)))
trainer = replace_setting(trainer, "CHECKPOINT_EVERY", str(int(CHECKPOINT_EVERY)))
# Resume upload must happen in the notebook kernel, not inside a shell subprocess.
trainer = replace_setting(trainer, "RESUME_FROM_MAI5", "False")
trainer = replace_setting(trainer, "AUTO_DOWNLOAD_AT_END", "False")
trainer = replace_setting(trainer, "OUTPUT_FILE", repr(str(WORK_MODEL)))

resume_source = None
if RESUME_FROM_MAI5:
    from google.colab import files
    print("Upload a compatible MAI5 checkpoint to resume training.")
    uploaded = files.upload()
    resume_name = next((name for name in uploaded if name.lower().endswith(".mai5")), None)
    if not resume_name:
        raise RuntimeError("No .mai5 checkpoint uploaded")
    resume_source = pathlib.Path("/content/resume_input.mai5")
    resume_source.write_bytes(uploaded[resume_name])
    marker = "# Fixed external holdout: larger coefficient/solution range, never fed into training."
    if marker not in trainer:
        raise RuntimeError("Trainer resume injection point not found")
    trainer = trainer.replace(marker, f"load_mai5({repr(str(resume_source))})\n\n{marker}", 1)

worker = pathlib.Path("/content/math_ai_v5_worker.py")
worker.write_text(trainer)

for p in (WORK_MODEL, BEST_MODEL, REPORT, BUNDLE):
    if p.exists(): p.unlink()

# ========================= TRAIN + BEST CHECKPOINT SELECTION =========================
print("\n1) Training on Colab GPU/CPU and selecting best external-holdout checkpoint...")
start_time = time.time()
best_rmse = float("inf")
best_metrics = None
last_metrics = None
metric_re_checkpoint = re.compile(r"HOLDOUT rmse=([0-9.eE+-]+) mae=([0-9.eE+-]+) ±1=([0-9.eE+-]+)% state=([0-9.eE+-]+)%")
metric_re_final = re.compile(r"Holdout RMSE=([0-9.eE+-]+)\s+MAE=([0-9.eE+-]+)\s+within ±1=([0-9.eE+-]+)%\s+state accuracy=([0-9.eE+-]+)%")

proc = subprocess.Popen([sys.executable, "-u", str(worker)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
assert proc.stdout is not None
for line in proc.stdout:
    print(line, end="")
    m = metric_re_checkpoint.search(line) or metric_re_final.search(line)
    if m:
        metrics = {
            "rmse": float(m.group(1)),
            "mae": float(m.group(2)),
            "within_one_percent": float(m.group(3)),
            "state_accuracy_percent": float(m.group(4)),
        }
        last_metrics = metrics
        if WORK_MODEL.is_file() and metrics["rmse"] < best_rmse:
            best_rmse = metrics["rmse"]
            best_metrics = dict(metrics)
            shutil.copy2(WORK_MODEL, BEST_MODEL)
            print(f"  >>> NEW BEST MAI5: RMSE={best_rmse:.5f}")
rc = proc.wait()
if rc != 0:
    raise RuntimeError(f"Training worker failed with exit code {rc}")
if not BEST_MODEL.is_file():
    if not WORK_MODEL.is_file():
        raise RuntimeError("Trainer did not produce a MAI5 file")
    shutil.copy2(WORK_MODEL, BEST_MODEL)
    best_metrics = last_metrics

print("Best checkpoint:", BEST_MODEL, "bytes=", BEST_MODEL.stat().st_size)

# ========================= INJECT BEST MODEL =========================
print("\n2) Injecting best MAI5 into Android assets...")
assets = ROOT / "app/src/main/assets"
assets.mkdir(parents=True, exist_ok=True)
embedded_model = assets / "default_model.mai5"
shutil.copy2(BEST_MODEL, embedded_model)

# ========================= PYTHON REFERENCE INFERENCE =========================
# Reuse the unmodified trainer definitions only; do not run its training loop.
print("\n3) Generating Python reference predictions for Kotlin interop test...")
reference_source = replace_setting(trainer_src, "RESUME_FROM_MAI5", "False")
reference_source = replace_setting(reference_source, "AUTO_DOWNLOAD_AT_END", "False")
prefix = reference_source.split("# ========================= TRAIN =========================", 1)[0]
namespace = {"__name__": "mai5_interop_reference"}
exec(compile(prefix, "mai5_interop_reference.py", "exec"), namespace)
namespace["load_mai5"](str(BEST_MODEL))
torch = namespace["torch"]
np = namespace["np"]
model = namespace["model"]
device = namespace["device"]
encode = namespace["encode"]
ROOT_SCALE = namespace["ROOT_SCALE"]

interop_equations = [
    "2x+4=10",
    "(x-2)*(x+3)=0",
    "ln(2x+1)=1.60943791",
    "2x+3y=5;x-y=1",
    "0*x=1",
    "0*x=0",
]
sidecar = assets / "v5_interop_expected.tsv"
rows = ["# equation\tfamily\tstate\tslots\tpresence\tstate_probabilities"]
model.eval()
with torch.no_grad():
    for equation in interop_equations:
        k, n, d, fam, src = encode(equation)
        kt = torch.tensor(k[None, :], device=device, dtype=torch.long)
        nt = torch.tensor(n[None, :], device=device, dtype=torch.float32)
        dt = torch.tensor(d[None, :], device=device, dtype=torch.float32)
        ft = torch.tensor([fam], device=device, dtype=torch.long)
        out = model(kt, nt, dt, ft)[0]
        slots = (out[:5] * ROOT_SCALE).detach().cpu().numpy().astype(float)
        presence = torch.sigmoid(out[5:10]).detach().cpu().numpy().astype(float)
        state_probs = torch.softmax(out[10:14], dim=0).detach().cpu().numpy().astype(float)
        state = int(np.argmax(state_probs))
        csv = lambda arr: ",".join(f"{float(x):.9g}" for x in arr)
        rows.append("\t".join([src, str(int(fam)), str(state), csv(slots), csv(presence), csv(state_probs)]))
sidecar.write_text("\n".join(rows) + "\n")
print("Interop sidecar:", sidecar)

# ========================= ANDROID TOOLCHAIN =========================
print("\n4) Preparing Android/JDK build environment...")
subprocess.run("apt-get update -qq && apt-get install -y -qq openjdk-17-jdk-headless unzip wget > /dev/null", shell=True, check=True)
java_home = subprocess.check_output("dirname $(dirname $(readlink -f $(which javac)))", shell=True, text=True).strip()
os.environ["JAVA_HOME"] = java_home
os.environ["PATH"] = f"{java_home}/bin:" + os.environ.get("PATH", "")

sdk_root = pathlib.Path("/content/android-sdk")
sdkmanager = sdk_root / "cmdline-tools/latest/bin/sdkmanager"
if not sdkmanager.exists():
    sdk_zip = pathlib.Path("/content/android-commandline-tools.zip")
    subprocess.run(["wget", "-q", "https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip", "-O", str(sdk_zip)], check=True)
    tmp = pathlib.Path("/content/android-tools-unpack")
    if tmp.exists(): shutil.rmtree(tmp)
    tmp.mkdir()
    subprocess.run(["unzip", "-q", str(sdk_zip), "-d", str(tmp)], check=True)
    latest = sdk_root / "cmdline-tools/latest"
    latest.parent.mkdir(parents=True, exist_ok=True)
    if latest.exists(): shutil.rmtree(latest)
    shutil.move(str(tmp / "cmdline-tools"), str(latest))
os.environ["ANDROID_HOME"] = str(sdk_root)
os.environ["ANDROID_SDK_ROOT"] = str(sdk_root)
os.environ["PATH"] = f"{sdk_root}/platform-tools:{sdk_root}/cmdline-tools/latest/bin:" + os.environ["PATH"]
subprocess.run(f"yes | {sdkmanager} --licenses > /dev/null", shell=True, check=True)
subprocess.run([str(sdkmanager), "platform-tools", "platforms;android-33", "build-tools;33.0.2"], check=True, stdout=subprocess.DEVNULL)

# ========================= KOTLIN INTEROP GATE + BUILD =========================
print("\n5) Running Kotlin MAI5 compatibility gate...")
gradlew = ROOT / "gradlew"
gradlew.chmod(0o755)
subprocess.run([str(gradlew), "testDebugUnitTest", "--no-daemon", "--stacktrace"], cwd=ROOT, env=os.environ, check=True)
print("Python -> MAI5 -> Kotlin compatibility: PASS")

print("\n6) Building APK with trained model embedded...")
subprocess.run([str(gradlew), "assembleDebug", "--no-daemon", "--stacktrace"], cwd=ROOT, env=os.environ, check=True)
debug_apk = ROOT / "app/build/outputs/apk/debug/app-debug.apk"
if not debug_apk.is_file():
    raise RuntimeError("Debug APK not produced")
selected_apk = debug_apk
build_type = "debug"

if BUILD_SIGNED_RELEASE:
    from google.colab import files
    print("Upload the SAME stable v5 .jks used for all future release updates.")
    uploaded = files.upload()
    jks_name = next((name for name in uploaded if name.lower().endswith((".jks", ".keystore"))), None)
    if not jks_name:
        raise RuntimeError("No JKS/keystore uploaded")
    jks_path = pathlib.Path("/content") / jks_name
    jks_path.write_bytes(uploaded[jks_name])
    os.environ["V5_KEYSTORE_PATH"] = str(jks_path)
    os.environ["V5_KEYSTORE_PASSWORD"] = getpass.getpass("Keystore password: ")
    os.environ["V5_KEY_ALIAS"] = input("Key alias: ").strip()
    os.environ["V5_KEY_PASSWORD"] = getpass.getpass("Key password: ")
    subprocess.run([str(gradlew), "assembleRelease", "--no-daemon", "--stacktrace"], cwd=ROOT, env=os.environ, check=True)
    release_apk = ROOT / "app/build/outputs/apk/release/app-release.apk"
    if not release_apk.is_file():
        raise RuntimeError("Signed release APK not produced")
    selected_apk = release_apk
    build_type = "signed-release"

# ========================= REPORT + DELIVERY =========================
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

report = {
    "factory": "Math AI v5 Automated AI App Factory",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "repository": REPO_URL,
    "branch": BRANCH,
    "source_commit": commit_sha,
    "training": {
        "total_steps": TOTAL_STEPS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "consistency_weight": CONSISTENCY_WEIGHT,
        "deepmind_ratio": DEEPMIND_RATIO,
        "deepmind_modules": ["algebra__linear_1d", "algebra__linear_2d", "algebra__polynomial_roots"],
        "best_holdout": best_metrics,
        "resumed_from_mai5": bool(RESUME_FROM_MAI5),
        "elapsed_seconds": round(time.time() - start_time, 1),
    },
    "model": {
        "file": BEST_MODEL.name,
        "bytes": BEST_MODEL.stat().st_size,
        "sha256": sha256(BEST_MODEL),
        "format": "MAI5 v1",
        "embedded_asset": "app/src/main/assets/default_model.mai5",
    },
    "verification": {
        "python_kotlin_interop": "PASS",
        "equations_checked": interop_equations,
    },
    "apk": {
        "build_type": build_type,
        "file": selected_apk.name,
        "bytes": selected_apk.stat().st_size,
        "sha256": sha256(selected_apk),
    },
}
REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False))

out_dir = pathlib.Path("/content/MathAI-v5-output")
if out_dir.exists(): shutil.rmtree(out_dir)
out_dir.mkdir()
shutil.copy2(selected_apk, out_dir / ("MathAI-v5-release.apk" if build_type == "signed-release" else "MathAI-v5-debug.apk"))
shutil.copy2(BEST_MODEL, out_dir / "math_ai_v5_best.mai5")
shutil.copy2(REPORT, out_dir / "training_report.json")
shutil.copy2(sidecar, out_dir / "v5_interop_expected.tsv")
if BUNDLE.exists(): BUNDLE.unlink()
shutil.make_archive(str(BUNDLE.with_suffix("")), "zip", out_dir)

print("\n=== FACTORY COMPLETE ===")
print("Best holdout:", best_metrics)
print("MAI5:", BEST_MODEL)
print("APK:", selected_apk)
print("Bundle:", BUNDLE)
print("The APK starts from the embedded Colab model, then continues Adam training on-device.")

from google.colab import files
files.download(str(BUNDLE))
