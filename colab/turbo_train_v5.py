# Math AI v5 — TURBO Colab trainer
# Uses the official pre-generated DeepMind Mathematics Dataset, pre-encodes a large pool,
# keeps compact tensors on the GPU, uses AMP, warmup/cosine LR, and verbose live telemetry.

TOTAL_STEPS = 30000
BATCH_SIZE = 0                  # 0 = auto-benchmark; otherwise explicit batch
LEARNING_RATE = 2.0e-4
MIN_LEARNING_RATE = 2.0e-5
WARMUP_STEPS = 800
CONSISTENCY_WEIGHT = 0.03
DEEPMIND_RATIO = 0.65
CHECKPOINT_EVERY = 500
SEED = 165
RESUME_FROM_MAI5 = False
AUTO_DOWNLOAD_AT_END = False
OUTPUT_FILE = "/content/math_ai_v5_working.mai5"

# Dataset/resource knobs. These affect only the Colab training pipeline, not the APK architecture.
DEEPMIND_PER_FILE = 60000       # 9 files => up to 540k official DeepMind examples
SYNTHETIC_POOL_SIZE = 260000
TRAIN_TARGET_ABS = 300.0        # reject extreme labels that destabilize regression
LOG_EVERY = 50
TELEMETRY_EVERY = 200
USE_AMP = True
USE_TORCH_COMPILE = True

import ast
import gc
import math
import os
import pathlib
import random
import re
import shutil
import subprocess
import sys
import time
from fractions import Fraction

HERE = pathlib.Path(__file__).resolve().parent
BASE = HERE / "train_v5_deepmind.py"


def banner(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78, flush=True)


def human_bytes(n):
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < 1024 or u == units[-1]:
            return f"{x:.2f} {u}"
        x /= 1024


banner("[0/8] فحص موارد Google Colab")
cpu_count = os.cpu_count() or 1
ram_total = 0
try:
    import psutil
    ram_total = psutil.virtual_memory().total
except Exception:
    pass
print(f"CPU cores      : {cpu_count}")
if ram_total:
    print(f"System RAM     : {human_bytes(ram_total)}")
print(f"Python         : {sys.version.split()[0]}")

# Load EXACT Android-compatible architecture/encoder/MAI5 contract from the base trainer,
# but stop before its old on-the-fly training loop.
base_src = BASE.read_text()


def _replace_setting(text, name, value):
    pattern = rf"(?m)^{re.escape(name)}\s*=\s*.*$"
    out, n = re.subn(pattern, f"{name} = {value}", text, count=1)
    if n != 1:
        raise RuntimeError(f"Cannot override base setting {name}")
    return out


base_src = _replace_setting(base_src, "RESUME_FROM_MAI5", "False")
base_src = _replace_setting(base_src, "AUTO_DOWNLOAD_AT_END", "False")
base_prefix = base_src.split("# ========================= TRAIN =========================", 1)[0]
ns = {"__name__": "mathai_v5_turbo_base"}
exec(compile(base_prefix, str(BASE), "exec"), ns)

np = ns["np"]
torch = ns["torch"]
F = ns["F"]
model = ns["model"]
device = ns["device"]
params = ns["params"]
moments = ns["moments"]
velocities = ns["velocities"]
save_mai5 = ns["save_mai5"]
load_mai5 = ns["load_mai5"]
synthetic = ns["synthetic"]
mk = ns["mk"]
swap = ns["swap"]
evaluate = ns["evaluate"]

MAX_NODES = ns["MAX_NODES"]
ROOT_SLOTS = ns["ROOT_SLOTS"]
ROOT_SCALE = ns["ROOT_SCALE"]
FINITE = ns["FINITE"]
SYSTEM = ns["SYSTEM"]
PERMS = ns["PERMS"]

if device.type != "cuda":
    raise RuntimeError("TURBO mode requires a CUDA GPU. In Colab choose Runtime > Change runtime type > T4 GPU.")

gpu_name = torch.cuda.get_device_name(0)
gpu_mem = torch.cuda.get_device_properties(0).total_memory
print(f"GPU            : {gpu_name}")
print(f"GPU VRAM       : {human_bytes(gpu_mem)}")
print(f"CUDA           : {torch.version.cuda}")
print(f"PyTorch        : {torch.__version__}")
print("خطة الاستخدام : dataset pre-encode على CPU/RAM -> compact pool داخل VRAM -> GPU-only batches")
print("المعنى         : الـGPU لن ينتظر SymPy أو مولد DeepMind أثناء كل خطوة تدريب.")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

if RESUME_FROM_MAI5:
    from google.colab import files
    banner("[RESUME] رفع checkpoint سابق")
    uploaded = files.upload()
    resume_name = next((n for n in uploaded if n.lower().endswith(".mai5")), None)
    if not resume_name:
        raise RuntimeError("No .mai5 file uploaded")
    resume_path = pathlib.Path("/content/resume_input.mai5")
    resume_path.write_bytes(uploaded[resume_name])
    load_mai5(str(resume_path))

DATA_URL = "https://storage.googleapis.com/mathematics-dataset/mathematics_dataset-v1.0.tar.gz"
ARCHIVE = pathlib.Path("/content/mathematics_dataset-v1.0.tar.gz")
EXTRACT = pathlib.Path("/content/mathai_dm_extract")
SPLITS = ["train-easy", "train-medium", "train-hard"]
MODULES = ["algebra__linear_1d", "algebra__linear_2d", "algebra__polynomial_roots"]

banner("[1/8] تنزيل DeepMind Mathematics Dataset الرسمي")
print("المصدر         :", DATA_URL)
print("المطلوب        : ملفات algebra فقط المناسبة لمخرجات v5")
print("الوحدات        :", ", ".join(MODULES))
print("التقسيمات      :", ", ".join(SPLITS))
if not ARCHIVE.exists() or ARCHIVE.stat().st_size < 1_000_000_000:
    print("جاري تنزيل الأرشيف الرسمي (~2.3GB). التنزيل يحصل مرة واحدة داخل Runtime الحالي.")
    subprocess.run(
        ["wget", "-c", "--progress=bar:force:noscroll", DATA_URL, "-O", str(ARCHIVE)],
        check=True,
    )
else:
    print("الأرشيف موجود مسبقًا في /content؛ لن أعيد تنزيله.")
print("حجم الأرشيف    :", human_bytes(ARCHIVE.stat().st_size))

banner("[2/8] استخراج الملفات المطلوبة فقط")
if EXTRACT.exists():
    shutil.rmtree(EXTRACT)
EXTRACT.mkdir(parents=True, exist_ok=True)
patterns = [f"*{split_name}/{module}.txt" for split_name in SPLITS for module in MODULES]
cmd = ["tar", "-xzf", str(ARCHIVE), "-C", str(EXTRACT), "--wildcards"] + patterns
print(f"سنستخرج {len(patterns)} ملفات فقط بدل استخدام بقية مجالات dataset.")
subprocess.run(cmd, check=True)
located = {}
for split_name in SPLITS:
    for module in MODULES:
        candidates = [p for p in EXTRACT.rglob(f"{module}.txt") if p.parent.name == split_name]
        if not candidates:
            raise RuntimeError(f"Missing extracted file: {split_name}/{module}.txt")
        located[(split_name, module)] = candidates[0]
        print(f"  ✓ {split_name:12s} {module:28s} {human_bytes(candidates[0].stat().st_size)}")


def strip_prefix(s, prefix):
    s = s.strip()
    return s[len(prefix):].strip() if s.startswith(prefix) else s


def parse_scalar(text):
    s = text.strip().replace("−", "-")
    while len(s) >= 2 and s[0] == "(" and s[-1] == ")":
        s = s[1:-1].strip()
    try:
        return float(Fraction(s))
    except Exception:
        val = ns["sp"].N(ns["sp"].sympify(s))
        c = complex(val)
        if abs(c.imag) > 1e-9:
            raise ValueError("non-real answer")
        return float(c.real)


def rename_var(text, old, new):
    return re.sub(rf"\b{re.escape(old)}\b", new, text)


def parse_linear_1d(question, answer):
    m = re.match(r"^Solve (.+) for ([A-Za-z])\.$", question)
    if not m:
        raise ValueError("linear1d template")
    equation, var = m.group(1), m.group(2)
    if " and " in equation:
        raise ValueError("unexpected system")
    value = parse_scalar(answer)
    equation = rename_var(equation, var, "x")
    return mk(equation, [value], equiv=swap(equation))


def eval_ast(node, env):
    if isinstance(node, ast.Expression):
        return eval_ast(node.body, env)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise ValueError("unknown symbol")
        return float(env[node.id])
    if isinstance(node, ast.UnaryOp):
        v = eval_ast(node.operand, env)
        if isinstance(node.op, ast.USub): return -v
        if isinstance(node.op, ast.UAdd): return v
        raise ValueError("bad unary")
    if isinstance(node, ast.BinOp):
        a, b = eval_ast(node.left, env), eval_ast(node.right, env)
        if isinstance(node.op, ast.Add): return a + b
        if isinstance(node.op, ast.Sub): return a - b
        if isinstance(node.op, ast.Mult): return a * b
        if isinstance(node.op, ast.Div): return a / b
        if isinstance(node.op, ast.Pow):
            if abs(b) > 4: raise ValueError("power too large")
            return a ** b
        raise ValueError("bad operator")
    raise ValueError("unsupported expression")


def affine_coeff(expr, variables):
    tree = ast.parse(expr.replace("^", "**"), mode="eval")
    zero = {v: 0.0 for v in variables}
    c = eval_ast(tree, zero)
    coeffs = []
    for v in variables:
        env = dict(zero); env[v] = 1.0
        coeffs.append(eval_ast(tree, env) - c)
    probe = {v: float(i + 2) for i, v in enumerate(variables)}
    expected = c + sum(coeffs[i] * probe[v] for i, v in enumerate(variables))
    actual = eval_ast(tree, probe)
    if not math.isfinite(actual) or abs(actual - expected) > 1e-6 * max(1.0, abs(actual), abs(expected)):
        raise ValueError("not affine")
    return coeffs, c


def equation_affine(eq, variables):
    left, right = eq.split("=", 1)
    ac, a0 = affine_coeff(left, variables)
    bc, b0 = affine_coeff(right, variables)
    return [ac[i] - bc[i] for i in range(len(variables))], a0 - b0


def parse_linear_2d(question, answer):
    m = re.match(r"^Solve (.+) for ([A-Za-z])\.$", question)
    if not m:
        raise ValueError("linear2d template")
    body, asked = m.group(1), m.group(2)
    parts = [x.strip() for x in body.split(" and ") if "=" in x]
    if len(parts) != 2:
        raise ValueError("linear2d equations")
    variables = sorted(set(re.findall(r"\b[A-Za-z]\b", body)))
    if len(variables) != 2 or asked not in variables:
        raise ValueError("linear2d variables")
    known = parse_scalar(answer)
    other = variables[1] if variables[0] == asked else variables[0]
    sol = {asked: known}
    solved = False
    for eq in parts:
        co, c = equation_affine(eq, variables)
        ia = variables.index(asked); io = variables.index(other)
        if abs(co[io]) > 1e-12:
            sol[other] = -(co[ia] * known + c) / co[io]
            solved = True
            break
    if not solved:
        raise ValueError("cannot solve second variable")
    for eq in parts:
        co, c = equation_affine(eq, variables)
        residual = sum(co[i] * sol[variables[i]] for i in range(2)) + c
        if abs(residual) > 1e-5 * max(1.0, abs(c)):
            raise ValueError("system verification")
    repl = {variables[0]: "x", variables[1]: "y"}
    renamed = []
    for eq in parts:
        for old, new in repl.items():
            eq = rename_var(eq, old, new)
        renamed.append(eq)
    system = [float(sol[variables[0]]), float(sol[variables[1]])]
    text = ";".join(renamed)
    return mk(text, system=system, equiv=";".join(reversed(renamed)))


POLY_PATTERNS = [
    r"^Let (.+?=.+?)\. (?:What is|Calculate) ([A-Za-z])\??$",
    r"^Suppose (.+?=.+?)\. (?:What is|Calculate) ([A-Za-z])\??$",
    r"^What is ([A-Za-z]) in (.+?=.+?)\?$",
    r"^Solve (.+?=.+?)(?: for ([A-Za-z]))?\.$",
    r"^Find ([A-Za-z]),? (?:such that|given that) (.+?=.+?)\.$",
    r"^Determine ([A-Za-z]),? (?:so that|given that) (.+?=.+?)\.$",
]


def parse_polynomial(question, answer):
    if question.startswith("Factor "):
        raise ValueError("factor prompt")
    equation = None; var = None
    m = re.match(POLY_PATTERNS[0], question) or re.match(POLY_PATTERNS[1], question)
    if m:
        equation, var = m.group(1), m.group(2)
    else:
        m = re.match(POLY_PATTERNS[2], question)
        if m:
            var, equation = m.group(1), m.group(2)
    if equation is None:
        m = re.match(POLY_PATTERNS[3], question)
        if m:
            equation, var = m.group(1), m.group(2)
            if not var:
                vars_found = sorted(set(re.findall(r"\b[A-Za-z]\b", equation)))
                if len(vars_found) == 1: var = vars_found[0]
    if equation is None:
        for pat in POLY_PATTERNS[4:]:
            m = re.match(pat, question)
            if m:
                var, equation = m.group(1), m.group(2)
                break
    if equation is None or not var:
        raise ValueError("polynomial template")
    roots = [parse_scalar(x) for x in answer.split(",")]
    if not (1 <= len(roots) <= ROOT_SLOTS):
        raise ValueError("root count")
    equation = rename_var(equation, var, "x")
    return mk(equation, sorted(set(roots)), equiv=swap(equation))


def parse_deepmind(module, q, a):
    if module.endswith("linear_1d"):
        return parse_linear_1d(q, a)
    if module.endswith("linear_2d"):
        return parse_linear_2d(q, a)
    if module.endswith("polynomial_roots"):
        return parse_polynomial(q, a)
    raise ValueError("module")


def acceptable(e):
    vals = list(e["roots"]) + list(e["system"])
    return all(math.isfinite(float(v)) and abs(float(v)) <= TRAIN_TARGET_ABS for v in vals)


class PoolWriter:
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.count = 0
        self.k = np.empty((capacity, MAX_NODES), dtype=np.uint8)
        self.n = np.empty((capacity, MAX_NODES), dtype=np.float16)
        self.d = np.empty((capacity, MAX_NODES), dtype=np.float16)
        self.f = np.empty(capacity, dtype=np.uint8)
        self.r = np.zeros((capacity, ROOT_SLOTS), dtype=np.float32)
        self.rc = np.zeros(capacity, dtype=np.uint8)
        self.sy = np.zeros((capacity, 2), dtype=np.float32)
        self.st = np.empty(capacity, dtype=np.uint8)
        self.ek = np.empty((capacity, MAX_NODES), dtype=np.uint8)
        self.en = np.empty((capacity, MAX_NODES), dtype=np.float16)
        self.ed = np.empty((capacity, MAX_NODES), dtype=np.float16)
        self.ef = np.empty(capacity, dtype=np.uint8)

    def add(self, e):
        if self.count >= self.capacity:
            return False
        i = self.count
        self.k[i] = e["k"]; self.n[i] = e["n"]; self.d[i] = e["d"]; self.f[i] = e["f"]
        roots = e["roots"][:ROOT_SLOTS]
        self.rc[i] = len(roots)
        if roots: self.r[i, :len(roots)] = roots
        system = e["system"][:2]
        if system: self.sy[i, :len(system)] = system
        self.st[i] = e["state"]
        if e["equiv"] is None:
            self.ek[i] = e["k"]; self.en[i] = e["n"]; self.ed[i] = e["d"]; self.ef[i] = e["f"]
        else:
            self.ek[i] = e["ek"]; self.en[i] = e["en"]; self.ed[i] = e["ed"]; self.ef[i] = e["ef"]
        self.count += 1
        return True

    def trim(self):
        for name in ("k","n","d","f","r","rc","sy","st","ek","en","ed","ef"):
            setattr(self, name, getattr(self, name)[:self.count])


def iter_qa(path):
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        while True:
            q = fh.readline()
            if not q: break
            a = fh.readline()
            if not a: break
            q = strip_prefix(q, "Question:")
            a = strip_prefix(a, "Answer:")
            if q and a:
                yield q, a


banner("[3/8] تحويل DeepMind إلى tensors بنيوية — مرة واحدة فقط")
print("ليش؟          : حتى نلغي SymPy/Parser من الحلقة الساخنة ونخلي GPU يتدرب بلا انتظار.")
print("الهدف         :", DEEPMIND_PER_FILE, "مثال مقبول من كل ملف كحد أقصى")
print("فلتر الاستقرار: |target| <=", TRAIN_TARGET_ABS, "حتى لا تدمر جذور شاذة ضخمة الـregression.")
dm_capacity = DEEPMIND_PER_FILE * len(located)
dm_writer = PoolWriter(dm_capacity)
rejects = {}
pre_start = time.time()
for file_idx, ((split_name, module), path) in enumerate(located.items(), 1):
    accepted_here = 0; seen_here = 0
    t0 = time.time()
    print(f"\n[{file_idx}/{len(located)}] {split_name}/{module}")
    for q, a in iter_qa(path):
        seen_here += 1
        try:
            e = parse_deepmind(module, q, a)
            if not acceptable(e):
                raise ValueError("target_out_of_range")
            dm_writer.add(e)
            accepted_here += 1
        except Exception as exc:
            key = str(exc) or type(exc).__name__
            rejects[key] = rejects.get(key, 0) + 1
        if accepted_here and accepted_here % 10000 == 0:
            rate = accepted_here / max(time.time() - t0, 1e-6)
            print(f"  accepted={accepted_here:6d} seen={seen_here:7d} rate={rate:7.0f} ex/s", flush=True)
        if accepted_here >= DEEPMIND_PER_FILE:
            break
    print(f"  ✓ accepted={accepted_here} / seen={seen_here} in {time.time()-t0:.1f}s")
dm_writer.trim()
print(f"\nDeepMind pool : {dm_writer.count:,} أمثلة")
print(f"Preprocess time: {time.time()-pre_start:.1f}s")
print("أكثر أسباب الرفض:", sorted(rejects.items(), key=lambda kv: -kv[1])[:8])

banner("[4/8] بناء curriculum اصطناعي كبير ومتنوع")
syn_writer = PoolWriter(SYNTHETIC_POOL_SIZE)
syn_rng = random.Random(SEED + 7001)
t0 = time.time()
last_reported = -1
while syn_writer.count < SYNTHETIC_POOL_SIZE:
    try:
        e = synthetic(syn_rng, max_abs=100)
        if acceptable(e):
            syn_writer.add(e)
    except Exception:
        pass
    milestone = syn_writer.count // 50000
    if milestone != last_reported and syn_writer.count > 0 and syn_writer.count % 50000 == 0:
        last_reported = milestone
        rate = syn_writer.count / max(time.time() - t0, 1e-6)
        print(f"  synthetic={syn_writer.count:7,d}/{SYNTHETIC_POOL_SIZE:,} rate={rate:7.0f} ex/s", flush=True)
syn_writer.trim()
print(f"✓ Synthetic pool: {syn_writer.count:,} in {time.time()-t0:.1f}s")


def pool_to_gpu(w, label):
    print(f"نقل {label} إلى VRAM...", flush=True)
    out = {}
    for name in ("k","n","d","f","r","rc","sy","st","ek","en","ed","ef"):
        out[name] = torch.from_numpy(getattr(w, name)).to(device=device, non_blocking=True)
    out["size"] = w.count
    return out


banner("[5/8] تحميل الـtraining pool إلى ذاكرة الـGPU")
dm_pool = pool_to_gpu(dm_writer, "DeepMind")
syn_pool = pool_to_gpu(syn_writer, "Synthetic")
del dm_writer, syn_writer
gc.collect()
torch.cuda.synchronize()
free_mem, total_mem = torch.cuda.mem_get_info()
print(f"GPU pool ready : DeepMind={dm_pool['size']:,} + Synthetic={syn_pool['size']:,}")
print(f"VRAM used      : {human_bytes(total_mem-free_mem)} / {human_bytes(total_mem)}")
print("من الآن        : اختيار الدفعات، الإدخال، Forward، Loss، Backward، Adam كلها على GPU.")


def take(pool, count):
    idx = torch.randint(0, pool["size"], (count,), device=device)
    return (
        pool["k"][idx].long(),
        pool["n"][idx].float(),
        pool["d"][idx].float(),
        pool["f"][idx].long(),
        pool["r"][idx],
        pool["rc"][idx].long(),
        pool["sy"][idx],
        pool["st"][idx].long(),
        (
            pool["ek"][idx].long(),
            pool["en"][idx].float(),
            pool["ed"][idx].float(),
            pool["ef"][idx].long(),
        ),
    )


def mixed_batch(batch_size):
    ndm = int(round(batch_size * DEEPMIND_RATIO))
    nsyn = batch_size - ndm
    a = take(dm_pool, ndm)
    b = take(syn_pool, nsyn)
    merged = [torch.cat((a[i], b[i]), dim=0) for i in range(8)]
    equiv = tuple(torch.cat((a[8][i], b[8][i]), dim=0) for i in range(4))
    return (*merged, equiv)


def stable_loss(out, roots, root_count, systems, states, families, other_out=None):
    state_loss = F.cross_entropy(out[:,10:14], states)
    assigned_vals = torch.zeros((len(out), ROOT_SLOTS), device=device, dtype=out.dtype)
    assigned_pres = torch.zeros_like(assigned_vals)
    finite = states == FINITE
    sysmask = finite & (families == SYSTEM)
    nonsys = finite & (families != SYSTEM)
    if sysmask.any():
        assigned_vals[sysmask,:2] = systems[sysmask,:2] / ROOT_SCALE
        assigned_pres[sysmask,:2] = 1
    ids = torch.where(nonsys)[0]
    if len(ids):
        tv = (roots[ids] / ROOT_SCALE).to(out.dtype)
        tc = root_count[ids]
        basepres = (torch.arange(ROOT_SLOTS, device=device)[None,:] < tc[:,None]).to(out.dtype)
        pv = tv[:,PERMS]
        pp = basepres[:,PERMS]
        predv = out[ids,:5][:,None,:]
        predlog = out[ids,5:10][:,None,:].expand(-1, len(PERMS), -1)
        active = pp.sum(-1).clamp_min(1)
        root_cost = F.smooth_l1_loss(predv.expand_as(pv), pv, reduction="none", beta=0.1)
        root_cost = (root_cost * pp).sum(-1) / active
        pres_cost = F.binary_cross_entropy_with_logits(predlog, pp, reduction="none").mean(-1)
        cost = root_cost + 0.35 * pres_cost
        best = cost.argmin(-1)
        rows = torch.arange(len(ids), device=device)
        assigned_vals[ids] = pv[rows, best]
        assigned_pres[ids] = pp[rows, best]
    if finite.any():
        per = F.smooth_l1_loss(out[:,:5], assigned_vals, reduction="none", beta=0.1)
        active = assigned_pres.sum(-1).clamp_min(1)
        root_loss = (((per * assigned_pres).sum(-1) / active)[finite]).mean()
    else:
        root_loss = out.sum() * 0
    presence = F.binary_cross_entropy_with_logits(out[:,5:10], assigned_pres)
    consistency = out.sum() * 0
    if other_out is not None:
        consistency = F.smooth_l1_loss(out, other_out, beta=0.1)
    total = root_loss + 0.35 * presence + 0.35 * state_loss + CONSISTENCY_WEIGHT * consistency
    return total, root_loss, presence, state_loss, consistency


def turbo_adam_step(lr):
    ns["adam_step"] += 1
    step = ns["adam_step"]
    b1, b2 = 0.9, 0.999
    grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
    norm_sq = torch.zeros((), device=device)
    for g in grads:
        norm_sq += (g.float() * g.float()).sum()
    norm = torch.sqrt(norm_sq)
    raw_norm = float(norm.detach())
    if not math.isfinite(raw_norm):
        model.zero_grad(set_to_none=True)
        return raw_norm, 0.0, False
    clip = min(1.0, 5.0 / (raw_norm + 1e-30))
    c1 = 1 - b1 ** step; c2 = 1 - b2 ** step
    with torch.no_grad():
        for p, g, m, v in zip(params, grads, moments, velocities):
            g = g * clip
            m.mul_(b1).add_(g, alpha=1-b1)
            v.mul_(b2).addcmul_(g, g, value=1-b2)
            p.addcdiv_(m / c1, (v / c2).sqrt().add_(1e-8), value=-lr)
        model.embedding.weight[ns["PAD"]].zero_()
        moments[0][ns["PAD"]].zero_()
        velocities[0][ns["PAD"]].zero_()
    model.zero_grad(set_to_none=True)
    return raw_norm, clip, True


def current_lr(step):
    if step <= WARMUP_STEPS:
        return LEARNING_RATE * max(step, 1) / max(WARMUP_STEPS, 1)
    frac = (step - WARMUP_STEPS) / max(TOTAL_STEPS - WARMUP_STEPS, 1)
    frac = min(max(frac, 0.0), 1.0)
    return MIN_LEARNING_RATE + 0.5 * (LEARNING_RATE - MIN_LEARNING_RATE) * (1 + math.cos(math.pi * frac))


train_model = model
if USE_TORCH_COMPILE and hasattr(torch, "compile"):
    try:
        print("تفعيل torch.compile لتقليل Python overhead...")
        train_model = torch.compile(model, mode="reduce-overhead")
        print("✓ torch.compile enabled")
    except Exception as exc:
        print("⚠ torch.compile غير متاح، سنكمل eager:", exc)

banner("[6/8] Benchmark تلقائي لاختيار Batch يشبع الـGPU")
if not BATCH_SIZE:
    candidates = [512, 1024, 2048, 4096] if gpu_mem >= 10 * 1024**3 else [256, 512, 1024, 2048]
    results = []
    for bsz in candidates:
        try:
            torch.cuda.empty_cache()
            k,n,d,f,r,rc,sy,st,eqv = mixed_batch(bsz)
            for _ in range(2):
                model.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16, enabled=USE_AMP):
                    o = train_model(k,n,d,f); oo = train_model(*eqv)
                    loss, *_ = stable_loss(o,r,rc,sy,st,f,oo)
                loss.backward()
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            t0 = time.time(); repeats = 3
            for _ in range(repeats):
                with torch.autocast("cuda", dtype=torch.float16, enabled=USE_AMP):
                    o = train_model(k,n,d,f); oo = train_model(*eqv)
                    loss, *_ = stable_loss(o,r,rc,sy,st,f,oo)
                loss.backward(); model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            dt = (time.time()-t0)/repeats
            sps = bsz/dt
            results.append((sps, bsz, dt))
            print(f"  batch={bsz:4d}: {dt*1000:7.1f} ms/step  {sps:10.0f} examples/s")
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                print(f"  batch={bsz:4d}: OOM")
                torch.cuda.empty_cache(); model.zero_grad(set_to_none=True)
                continue
            raise
    if not results:
        raise RuntimeError("No batch size fit GPU memory")
    results.sort(reverse=True)
    BATCH_SIZE = results[0][1]
else:
    print("Batch ثابت من الإعدادات:", BATCH_SIZE)
print(f"✓ Selected batch = {BATCH_SIZE}")
print("ملاحظة: اختيار الأسرع يحاول رفع utilization، لكن لا نختار batch ضخم فقط لأنه يدخل بالذاكرة.")


def gpu_telemetry():
    try:
        q = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=utilization.gpu,temperature.gpu,power.draw,memory.used,memory.total",
            "--format=csv,noheader,nounits"
        ], text=True).strip().split(",")
        util, temp, power, mused, mtotal = [x.strip() for x in q]
        return f"GPU={util}% temp={temp}C power={power}W VRAM={mused}/{mtotal}MiB"
    except Exception:
        free, total = torch.cuda.mem_get_info()
        return f"VRAM={human_bytes(total-free)}/{human_bytes(total)}"


banner("[7/8] بدء التدريب الحقيقي")
print(f"Steps          : {TOTAL_STEPS:,}")
print(f"Batch          : {BATCH_SIZE:,}")
print(f"Samples/step   : {BATCH_SIZE:,} ({DEEPMIND_RATIO*100:.0f}% DeepMind / {(1-DEEPMIND_RATIO)*100:.0f}% synthetic)")
print(f"Peak LR        : {LEARNING_RATE:g}")
print(f"Warmup         : {WARMUP_STEPS} steps")
print(f"Final LR       : {MIN_LEARNING_RATE:g}")
print(f"AMP FP16       : {USE_AMP}")
print("Grad clip      : global norm <= 5.0")
print(f"Checkpoints    : every {CHECKPOINT_EVERY} steps")
print("كل سطر TRAIN سيعرض loss/grad/سرعة/ETA. وكل Telemetry سيعرض استخدام GPU الفعلي.")
print("أي loss غير finite أو انفجار شاذ سيُرفض بدل تلويث الأوزان.", flush=True)

start_step = ns["adam_step"] + 1
train_start = time.time()
skipped = 0
best_seen = float("inf")
model.train()

for step_idx in range(start_step, TOTAL_STEPS + 1):
    lr = current_lr(step_idx)
    k,n,d,f,r,rc,sy,st,eqv = mixed_batch(BATCH_SIZE)
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16, enabled=USE_AMP):
        out = train_model(k,n,d,f)
        other = train_model(*eqv)
        loss, root_l, pres_l, state_l, cons_l = stable_loss(out,r,rc,sy,st,f,other)

    loss_value = float(loss.detach())
    if (not math.isfinite(loss_value)) or loss_value > 500.0:
        skipped += 1
        model.zero_grad(set_to_none=True)
        if skipped <= 10 or skipped % 20 == 0:
            print(f"⚠ SKIP step={step_idx}: abnormal loss={loss_value}; weights not updated.", flush=True)
        continue

    loss.backward()
    grad_norm, clip_scale, updated = turbo_adam_step(lr)
    if not updated:
        skipped += 1
        print(f"⚠ SKIP step={step_idx}: non-finite gradient norm={grad_norm}; weights not updated.", flush=True)
        continue

    if step_idx % LOG_EVERY == 0:
        elapsed = time.time() - train_start
        done = step_idx - start_step + 1
        samples_sec = done * BATCH_SIZE / max(elapsed, 1e-6)
        eta = max(TOTAL_STEPS - step_idx, 0) * BATCH_SIZE / max(samples_sec, 1e-6)
        print(
            f"TRAIN step={step_idx:6d}/{TOTAL_STEPS} "
            f"loss={loss_value:8.5f} root={float(root_l.detach()):7.4f} "
            f"pres={float(pres_l.detach()):6.4f} state={float(state_l.detach()):6.4f} "
            f"cons={float(cons_l.detach()):6.4f} lr={lr:.2e} "
            f"grad={grad_norm:8.3f} clip={clip_scale:5.3f} "
            f"throughput={samples_sec:,.0f} ex/s ETA={eta/60:.1f}m",
            flush=True,
        )
    if step_idx % TELEMETRY_EVERY == 0:
        print("  TELEMETRY", gpu_telemetry(), flush=True)

    if step_idx % CHECKPOINT_EVERY == 0:
        save_mai5(OUTPUT_FILE)
        rmse, mae, acc, sacc = evaluate()
        print(
            f"  HOLDOUT rmse={rmse:.3f} mae={mae:.3f} "
            f"±1={acc*100:.1f}% state={sacc*100:.1f}%  saved={OUTPUT_FILE}",
            flush=True,
        )
        if rmse < best_seen:
            best_seen = rmse
            print(f"  ★ تحسن حقيقي على Holdout: best RMSE أصبح {best_seen:.4f}", flush=True)

save_mai5(OUTPUT_FILE)
rmse, mae, acc, sacc = evaluate()

banner("[8/8] انتهاء التدريب")
elapsed = time.time() - train_start
print(f"Training time   : {elapsed/60:.2f} min")
print(f"Optimizer step  : {ns['adam_step']:,}")
print(f"Skipped updates : {skipped}")
print(f"Final Holdout   : RMSE={rmse:.4f} MAE={mae:.4f} ±1={acc*100:.2f}% state={sacc*100:.2f}%")
print(f"MAI5            : {OUTPUT_FILE} ({human_bytes(os.path.getsize(OUTPUT_FILE))})")
print("المرحلة التالية في factory: اختيار أفضل checkpoint، audit ضد الحفظ، حقن MAI5 داخل APK، ثم Build.")
print(f"Holdout RMSE={rmse:.4f}  MAE={mae:.4f}  within ±1={acc*100:.2f}%  state accuracy={sacc*100:.2f}%")

if AUTO_DOWNLOAD_AT_END:
    from google.colab import files
    files.download(OUTPUT_FILE)
