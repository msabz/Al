#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path, old, new, label):
    p = ROOT / path
    text = p.read_text()
    if old not in text:
        raise RuntimeError(f"{label}: target not found in {path}")
    if text.count(old) != 1:
        raise RuntimeError(f"{label}: target count={text.count(old)} in {path}")
    p.write_text(text.replace(old, new, 1))
    print(f"OK {label}")


def insert_once(path, marker, addition, label):
    p = ROOT / path
    text = p.read_text()
    if marker not in text:
        raise RuntimeError(f"{label}: marker not found in {path}")
    if text.count(marker) != 1:
        raise RuntimeError(f"{label}: marker count={text.count(marker)} in {path}")
    p.write_text(text.replace(marker, addition + marker, 1))
    print(f"OK {label}")


# ---------------------------------------------------------------------------
# Android model contract: fresh MAI5 v2, ~44% fewer parameters, AdamW decay.
# ---------------------------------------------------------------------------
replace_once(
    "app/src/main/java/com/example/equationsolver/ai/V5ModelSpec.kt",
    "    const val FILE_VERSION = 1",
    "    const val FILE_VERSION = 2",
    "MAI5 format v2",
)
replace_once(
    "app/src/main/java/com/example/equationsolver/ai/V5ModelSpec.kt",
    "    const val SHARED_1 = 160\n    const val SHARED_2 = 128\n    const val HEAD_HIDDEN = 64",
    "    const val SHARED_1 = 96\n    const val SHARED_2 = 64\n    const val HEAD_HIDDEN = 48",
    "shrink network trunk and heads",
)
replace_once(
    "app/src/main/java/com/example/equationsolver/ai/V5ModelSpec.kt",
    "    const val MAX_GRADIENT_NORM = 5.0",
    "    const val MAX_GRADIENT_NORM = 5.0\n    const val WEIGHT_DECAY = 1e-5\n    const val CANONICAL_COEFF_SLOTS = 6",
    "add weight decay and coefficient slots",
)

# ---------------------------------------------------------------------------
# Android structural encoder: canonical numeric tensors for linear/poly/system.
# Text RPN remains only for families that cannot be reduced safely (analytic or
# unsupported/non-canonical fallbacks). Equivalent algebraic forms collapse to
# the same six-number representation before the network sees them.
# ---------------------------------------------------------------------------
replace_once(
    "app/src/main/java/com/example/equationsolver/ai/StructuralMathEncoder.kt",
    "import kotlin.math.abs\nimport kotlin.math.ln\nimport kotlin.math.sign",
    "import kotlin.math.*",
    "encoder math imports",
)
replace_once(
    "app/src/main/java/com/example/equationsolver/ai/StructuralMathEncoder.kt",
    "        require(equations.size <= 2) { \"v5 يدعم معادلة واحدة أو نظامًا من معادلتين\" }\n\n        val nodes = ArrayList<Pair<Int, Double>>(V5ModelSpec.MAX_NODES)",
    "        require(equations.size <= 2) { \"v5 يدعم معادلة واحدة أو نظامًا من معادلتين\" }\n\n        val family = classify(source)\n        canonicalNumericEncoding(source, family)?.let { return it }\n\n        val nodes = ArrayList<Pair<Int, Double>>(V5ModelSpec.MAX_NODES)",
    "route canonical numeric encoding",
)
replace_once(
    "app/src/main/java/com/example/equationsolver/ai/StructuralMathEncoder.kt",
    "            family = classify(source)",
    "            family = family",
    "reuse classified family",
)

KOTLIN_CANONICAL = r'''
    private const val CANONICAL_EPS = 1e-10
    private data class Affine(val x: Double, val y: Double, val c: Double)

    /**
     * Convert the algebraic families used by the DeepMind training contract into
     * fixed numeric coefficients. This removes spelling/order/side shortcuts.
     * LINEAR/POLYNOMIAL: [c0,c1,c2,c3,c4,c5] for p(x)=0.
     * SYSTEM: [a1,b1,c1,a2,b2,c2] for a*x+b*y=c, with canonical row order.
     */
    private fun canonicalNumericEncoding(source: String, family: EquationFamily): Encoding? = when (family) {
        EquationFamily.LINEAR, EquationFamily.POLYNOMIAL -> canonicalPolynomialEncoding(source, family)
        EquationFamily.SYSTEM -> canonicalSystemEncoding(source)
        EquationFamily.ANALYTIC -> null
    }

    private fun coefficientEncoding(source: String, family: EquationFamily, values: DoubleArray): Encoding {
        require(values.size == V5ModelSpec.CANONICAL_COEFF_SLOTS)
        val kinds = IntArray(V5ModelSpec.MAX_NODES)
        val numeric = FloatArray(V5ModelSpec.MAX_NODES)
        val depth = FloatArray(V5ModelSpec.MAX_NODES)
        for (i in values.indices) {
            kinds[i] = Kind.NUMBER
            numeric[i] = values[i].toFloat()
            // Reuse the third scalar feature as a strong coefficient-slot identity.
            depth[i] = (i + 1).toFloat() / values.size.toFloat()
        }
        return Encoding(source, kinds, numeric, depth, values.size, false, family)
    }

    private fun canonicalPolynomialEncoding(source: String, family: EquationFamily): Encoding? {
        if (';' in source) return null
        val at = topLevelEquals(source)
        if (at < 0) return null
        val hasX = source.contains('x')
        val hasY = source.contains('y')
        if (hasX && hasY) return null
        val variableKind = if (hasY) Kind.Y else Kind.X
        val left = polynomialOf(toRpn(source.substring(0, at)), variableKind) ?: return null
        val right = polynomialOf(toRpn(source.substring(at + 1)), variableKind) ?: return null
        val coeff = DoubleArray(V5ModelSpec.CANONICAL_COEFF_SLOTS) { left[it] - right[it] }
        val degree = polynomialDegree(coeff)
        if (family == EquationFamily.LINEAR && degree > 1) return null
        if (family == EquationFamily.POLYNOMIAL && degree < 2) return null
        canonicalizePolynomial(coeff)
        return coefficientEncoding(source, family, coeff)
    }

    private fun canonicalSystemEncoding(source: String): Encoding? {
        val equations = source.split(';').filter { it.isNotBlank() }
        if (equations.size != 2) return null
        val rows = equations.map { canonicalAffineRow(it) ?: return null }.toMutableList()
        rows.sortWith(Comparator { a, b ->
            var result = 0
            for (i in 0..2) {
                result = a[i].compareTo(b[i])
                if (result != 0) break
            }
            result
        })
        val values = doubleArrayOf(
            rows[0][0], rows[0][1], rows[0][2],
            rows[1][0], rows[1][1], rows[1][2]
        )
        return coefficientEncoding(source, EquationFamily.SYSTEM, values)
    }

    private fun canonicalAffineRow(equation: String): DoubleArray? {
        val at = topLevelEquals(equation)
        if (at < 0) return null
        val left = affineOf(toRpn(equation.substring(0, at))) ?: return null
        val right = affineOf(toRpn(equation.substring(at + 1))) ?: return null
        var a = left.x - right.x
        var b = left.y - right.y
        var c = -(left.c - right.c)
        val scale = maxOf(abs(a), abs(b), abs(c))
        if (scale <= CANONICAL_EPS) return doubleArrayOf(0.0, 0.0, 0.0)
        a /= scale; b /= scale; c /= scale
        val first = listOf(a, b, c).firstOrNull { abs(it) > CANONICAL_EPS } ?: 0.0
        if (first < 0.0) { a = -a; b = -b; c = -c }
        return doubleArrayOf(cleanZero(a), cleanZero(b), cleanZero(c))
    }

    private fun affineOf(nodes: List<Pair<Int, Double>>): Affine? {
        val stack = ArrayDeque<Affine>()
        fun pop(): Affine? = if (stack.isEmpty()) null else stack.removeLast()
        for ((kind, value) in nodes) {
            when (kind) {
                Kind.NUMBER -> stack.addLast(Affine(0.0, 0.0, value))
                Kind.X -> stack.addLast(Affine(1.0, 0.0, 0.0))
                Kind.Y -> stack.addLast(Affine(0.0, 1.0, 0.0))
                Kind.PI -> stack.addLast(Affine(0.0, 0.0, Math.PI))
                Kind.E -> stack.addLast(Affine(0.0, 0.0, Math.E))
                Kind.NEG -> { val a = pop() ?: return null; stack.addLast(Affine(-a.x, -a.y, -a.c)) }
                Kind.ADD, Kind.SUB -> {
                    val b = pop() ?: return null; val a = pop() ?: return null
                    val s = if (kind == Kind.ADD) 1.0 else -1.0
                    stack.addLast(Affine(a.x + s*b.x, a.y + s*b.y, a.c + s*b.c))
                }
                Kind.MUL -> {
                    val b = pop() ?: return null; val a = pop() ?: return null
                    val av = abs(a.x) > CANONICAL_EPS || abs(a.y) > CANONICAL_EPS
                    val bv = abs(b.x) > CANONICAL_EPS || abs(b.y) > CANONICAL_EPS
                    if (av && bv) return null
                    stack.addLast(if (av) Affine(a.x*b.c, a.y*b.c, a.c*b.c) else Affine(b.x*a.c, b.y*a.c, b.c*a.c))
                }
                Kind.DIV -> {
                    val b = pop() ?: return null; val a = pop() ?: return null
                    if (abs(b.x) > CANONICAL_EPS || abs(b.y) > CANONICAL_EPS || abs(b.c) <= CANONICAL_EPS) return null
                    stack.addLast(Affine(a.x/b.c, a.y/b.c, a.c/b.c))
                }
                Kind.POW -> {
                    val exponent = pop() ?: return null; val base = pop() ?: return null
                    if (abs(exponent.x) > CANONICAL_EPS || abs(exponent.y) > CANONICAL_EPS) return null
                    val e = exponent.c.roundToInt()
                    if (abs(exponent.c - e) > 1e-9) return null
                    val baseHasVar = abs(base.x) > CANONICAL_EPS || abs(base.y) > CANONICAL_EPS
                    when {
                        e == 0 -> stack.addLast(Affine(0.0, 0.0, 1.0))
                        e == 1 -> stack.addLast(base)
                        !baseHasVar -> stack.addLast(Affine(0.0, 0.0, base.c.pow(e.toDouble())))
                        else -> return null
                    }
                }
                Kind.SIN, Kind.COS, Kind.TAN, Kind.SQRT, Kind.LOG, Kind.LN, Kind.EXP, Kind.ABS -> {
                    val a = pop() ?: return null
                    if (abs(a.x) > CANONICAL_EPS || abs(a.y) > CANONICAL_EPS) return null
                    val v = constantFunction(kind, a.c) ?: return null
                    stack.addLast(Affine(0.0, 0.0, v))
                }
                else -> return null
            }
        }
        return if (stack.size == 1) stack.removeLast() else null
    }

    private fun polynomialOf(nodes: List<Pair<Int, Double>>, variableKind: Int): DoubleArray? {
        val stack = ArrayDeque<DoubleArray>()
        fun pop(): DoubleArray? = if (stack.isEmpty()) null else stack.removeLast()
        for ((kind, value) in nodes) {
            when (kind) {
                Kind.NUMBER -> stack.addLast(constantPolynomial(value))
                variableKind -> { val p = constantPolynomial(0.0); p[1] = 1.0; stack.addLast(p) }
                Kind.X, Kind.Y -> return null
                Kind.PI -> stack.addLast(constantPolynomial(Math.PI))
                Kind.E -> stack.addLast(constantPolynomial(Math.E))
                Kind.NEG -> { val a = pop() ?: return null; stack.addLast(DoubleArray(a.size) { -a[it] }) }
                Kind.ADD, Kind.SUB -> {
                    val b = pop() ?: return null; val a = pop() ?: return null
                    val s = if (kind == Kind.ADD) 1.0 else -1.0
                    stack.addLast(DoubleArray(a.size) { a[it] + s*b[it] })
                }
                Kind.MUL -> {
                    val b = pop() ?: return null; val a = pop() ?: return null
                    stack.addLast(polynomialMultiply(a, b) ?: return null)
                }
                Kind.DIV -> {
                    val b = pop() ?: return null; val a = pop() ?: return null
                    if (polynomialDegree(b) > 0 || abs(b[0]) <= CANONICAL_EPS) return null
                    stack.addLast(DoubleArray(a.size) { a[it] / b[0] })
                }
                Kind.POW -> {
                    val exponent = pop() ?: return null; val base = pop() ?: return null
                    if (polynomialDegree(exponent) > 0) return null
                    val e = exponent[0].roundToInt()
                    if (abs(exponent[0] - e) > 1e-9 || e !in 0..5) return null
                    stack.addLast(polynomialPower(base, e) ?: return null)
                }
                Kind.SIN, Kind.COS, Kind.TAN, Kind.SQRT, Kind.LOG, Kind.LN, Kind.EXP, Kind.ABS -> {
                    val a = pop() ?: return null
                    if (polynomialDegree(a) > 0) return null
                    stack.addLast(constantPolynomial(constantFunction(kind, a[0]) ?: return null))
                }
                else -> return null
            }
        }
        return if (stack.size == 1) stack.removeLast() else null
    }

    private fun constantPolynomial(v: Double): DoubleArray = DoubleArray(V5ModelSpec.CANONICAL_COEFF_SLOTS).also { it[0] = v }

    private fun polynomialMultiply(a: DoubleArray, b: DoubleArray): DoubleArray? {
        val out = DoubleArray(V5ModelSpec.CANONICAL_COEFF_SLOTS)
        for (i in a.indices) for (j in b.indices) {
            val term = a[i] * b[j]
            if (abs(term) <= CANONICAL_EPS) continue
            if (i + j >= out.size) return null
            out[i + j] += term
        }
        return out
    }

    private fun polynomialPower(base: DoubleArray, exponent: Int): DoubleArray? {
        var out = constantPolynomial(1.0)
        repeat(exponent) { out = polynomialMultiply(out, base) ?: return null }
        return out
    }

    private fun polynomialDegree(p: DoubleArray): Int = (p.lastIndex downTo 0).firstOrNull { abs(p[it]) > CANONICAL_EPS } ?: 0

    private fun canonicalizePolynomial(p: DoubleArray) {
        val scale = p.maxOf { abs(it) }
        if (scale <= CANONICAL_EPS) { p.fill(0.0); return }
        for (i in p.indices) p[i] /= scale
        val degree = polynomialDegree(p)
        if (p[degree] < 0.0) for (i in p.indices) p[i] = -p[i]
        for (i in p.indices) p[i] = cleanZero(p[i])
    }

    private fun constantFunction(kind: Int, v: Double): Double? = runCatching {
        when (kind) {
            Kind.SIN -> sin(v)
            Kind.COS -> cos(v)
            Kind.TAN -> tan(v)
            Kind.SQRT -> sqrt(v)
            Kind.LOG -> log10(v)
            Kind.LN -> ln(v)
            Kind.EXP -> exp(v)
            Kind.ABS -> abs(v)
            else -> return null
        }
    }.getOrNull()?.takeIf { it.isFinite() }

    private fun cleanZero(v: Double): Double = if (abs(v) < 1e-12) 0.0 else v

'''
insert_once(
    "app/src/main/java/com/example/equationsolver/ai/StructuralMathEncoder.kt",
    "    private fun normalize(raw: String): String =",
    KOTLIN_CANONICAL,
    "insert canonical coefficient algebra",
)

# Padding position is not information. Zero it so canonical inputs have only a
# tiny active subspace instead of 80 constant positional signals.
replace_once(
    "app/src/main/java/com/example/equationsolver/ai/NeuralNetwork.kt",
    "            input[base + V5ModelSpec.EMBEDDING_SIZE + 1] = if (V5ModelSpec.MAX_NODES <= 1) 0f else p.toFloat() / (V5ModelSpec.MAX_NODES - 1).toFloat()",
    "            input[base + V5ModelSpec.EMBEDDING_SIZE + 1] = if (kind == StructuralMathEncoder.Kind.PAD || V5ModelSpec.MAX_NODES <= 1) 0f else p.toFloat() / (V5ModelSpec.MAX_NODES - 1).toFloat()",
    "mask PAD positional features",
)
replace_once(
    "app/src/main/java/com/example/equationsolver/ai/NeuralNetwork.kt",
    "        for (i in p.indices) for (j in p[i].indices) {\n            val grad = g[i][j] * scale",
    "        val decay = (1f - lr * V5ModelSpec.WEIGHT_DECAY.toFloat()).coerceAtLeast(0f)\n        for (i in p.indices) for (j in p[i].indices) {\n            val grad = g[i][j] * scale",
    "prepare Android AdamW decay",
)
replace_once(
    "app/src/main/java/com/example/equationsolver/ai/NeuralNetwork.kt",
    "            p[i][j] -= lr * mh / (sqrt(vh.toDouble()).toFloat() + eps)",
    "            p[i][j] *= decay\n            p[i][j] -= lr * mh / (sqrt(vh.toDouble()).toFloat() + eps)",
    "apply Android AdamW decay",
)

# ---------------------------------------------------------------------------
# Python mirror: same MAI5 v2 dimensions, canonical numeric encoder, PAD mask,
# AdamW. This is the source of truth for Kaggle + Python/Kotlin interop.
# ---------------------------------------------------------------------------
replace_once(
    "colab/train_v5_deepmind.py",
    "LEARNING_RATE = 6e-4\nCONSISTENCY_WEIGHT = 0.05",
    "LEARNING_RATE = 6e-4\nWEIGHT_DECAY = 1e-5\nCONSISTENCY_WEIGHT = 0.05",
    "base Python weight decay setting",
)
replace_once(
    "colab/train_v5_deepmind.py",
    "MAGIC=0x4D414935; VERSION=1\nMAX_NODES=80; TOKEN_VOCAB=22; EMB=16; EXTRA=3; NODE_FEATURES=19; INPUT=1520\nSHARED1=160; SHARED2=128; HEAD_HIDDEN=64; HEADS=4; ROOT_SLOTS=5; STATES=4; HEAD_OUT=14\nROOT_SCALE=100.0; MAX_GRAD_NORM=5.0",
    "MAGIC=0x4D414935; VERSION=2\nMAX_NODES=80; TOKEN_VOCAB=22; EMB=16; EXTRA=3; NODE_FEATURES=19; INPUT=1520\nSHARED1=96; SHARED2=64; HEAD_HIDDEN=48; HEADS=4; ROOT_SLOTS=5; STATES=4; HEAD_OUT=14\nROOT_SCALE=100.0; MAX_GRAD_NORM=5.0; CANONICAL_COEFF_SLOTS=6",
    "Python MAI5 v2 dimensions",
)

PY_CANONICAL = r'''
CANONICAL_EPS=1e-10

def _eq_at(s):
    depth=0
    for i,c in enumerate(s):
        if c=='(': depth+=1
        elif c==')': depth-=1
        elif c=='=' and depth==0: return i
    return -1

def _poly_const(v):
    out=[0.0]*CANONICAL_COEFF_SLOTS; out[0]=float(v); return out

def _poly_degree(p):
    for i in range(len(p)-1,-1,-1):
        if abs(p[i])>CANONICAL_EPS: return i
    return 0

def _poly_mul(a,b):
    out=[0.0]*CANONICAL_COEFF_SLOTS
    for i,av in enumerate(a):
        for j,bv in enumerate(b):
            term=av*bv
            if abs(term)<=CANONICAL_EPS: continue
            if i+j>=len(out): return None
            out[i+j]+=term
    return out

def _poly_pow(base,e):
    out=_poly_const(1.0)
    for _ in range(e):
        out=_poly_mul(out,base)
        if out is None:return None
    return out

def _const_fn(kind,v):
    try:
        value={SIN:math.sin,COS:math.cos,TAN:math.tan,SQRT:math.sqrt,
               LOG:math.log10,LN:math.log,EXP:math.exp,ABS:abs}[kind](v)
        return value if math.isfinite(value) else None
    except Exception:
        return None

def _rpn_poly(nodes,var_kind):
    st=[]
    for k,v in nodes:
        if k==NUMBER: st.append(_poly_const(v))
        elif k==var_kind:
            p=_poly_const(0); p[1]=1.0; st.append(p)
        elif k in (X,Y): return None
        elif k==PI: st.append(_poly_const(math.pi))
        elif k==E: st.append(_poly_const(math.e))
        elif k==NEG:
            if not st:return None
            a=st.pop(); st.append([-x for x in a])
        elif k in (ADD,SUB,MUL,DIV,POW):
            if len(st)<2:return None
            b=st.pop(); a=st.pop()
            if k==ADD: st.append([a[i]+b[i] for i in range(len(a))])
            elif k==SUB: st.append([a[i]-b[i] for i in range(len(a))])
            elif k==MUL:
                p=_poly_mul(a,b)
                if p is None:return None
                st.append(p)
            elif k==DIV:
                if _poly_degree(b)>0 or abs(b[0])<=CANONICAL_EPS:return None
                st.append([x/b[0] for x in a])
            else:
                if _poly_degree(b)>0:return None
                e=round(b[0])
                if abs(b[0]-e)>1e-9 or not 0<=e<=5:return None
                p=_poly_pow(a,int(e))
                if p is None:return None
                st.append(p)
        elif k in (SIN,COS,TAN,SQRT,LOG,LN,EXP,ABS):
            if not st:return None
            a=st.pop()
            if _poly_degree(a)>0:return None
            val=_const_fn(k,a[0])
            if val is None:return None
            st.append(_poly_const(val))
        else:return None
    return st[0] if len(st)==1 else None

def _rpn_affine(nodes):
    st=[]
    for k,v in nodes:
        if k==NUMBER:st.append((0.0,0.0,float(v)))
        elif k==X:st.append((1.0,0.0,0.0))
        elif k==Y:st.append((0.0,1.0,0.0))
        elif k==PI:st.append((0.0,0.0,math.pi))
        elif k==E:st.append((0.0,0.0,math.e))
        elif k==NEG:
            if not st:return None
            x,y,c=st.pop();st.append((-x,-y,-c))
        elif k in (ADD,SUB,MUL,DIV,POW):
            if len(st)<2:return None
            bx,by,bc=st.pop(); ax,ay,ac=st.pop()
            if k in (ADD,SUB):
                s=1.0 if k==ADD else -1.0; st.append((ax+s*bx,ay+s*by,ac+s*bc))
            elif k==MUL:
                av=abs(ax)>CANONICAL_EPS or abs(ay)>CANONICAL_EPS
                bv=abs(bx)>CANONICAL_EPS or abs(by)>CANONICAL_EPS
                if av and bv:return None
                st.append((ax*bc,ay*bc,ac*bc) if av else (bx*ac,by*ac,bc*ac))
            elif k==DIV:
                if abs(bx)>CANONICAL_EPS or abs(by)>CANONICAL_EPS or abs(bc)<=CANONICAL_EPS:return None
                st.append((ax/bc,ay/bc,ac/bc))
            else:
                if abs(bx)>CANONICAL_EPS or abs(by)>CANONICAL_EPS:return None
                e=round(bc)
                if abs(bc-e)>1e-9:return None
                av=abs(ax)>CANONICAL_EPS or abs(ay)>CANONICAL_EPS
                if e==0:st.append((0.0,0.0,1.0))
                elif e==1:st.append((ax,ay,ac))
                elif av:return None
                else:st.append((0.0,0.0,ac**e))
        elif k in (SIN,COS,TAN,SQRT,LOG,LN,EXP,ABS):
            if not st:return None
            x,y,c=st.pop()
            if abs(x)>CANONICAL_EPS or abs(y)>CANONICAL_EPS:return None
            val=_const_fn(k,c)
            if val is None:return None
            st.append((0.0,0.0,val))
        else:return None
    return st[0] if len(st)==1 else None

def _canonical_poly(src,fam):
    if ';' in src:return None
    at=_eq_at(src)
    if at<0:return None
    has_x='x' in src; has_y='y' in src
    if has_x and has_y:return None
    var_kind=Y if has_y else X
    left=_rpn_poly(rpn(src[:at]),var_kind); right=_rpn_poly(rpn(src[at+1:]),var_kind)
    if left is None or right is None:return None
    p=[left[i]-right[i] for i in range(CANONICAL_COEFF_SLOTS)]
    degree=_poly_degree(p)
    if fam==LINEAR and degree>1:return None
    if fam==POLYNOMIAL and degree<2:return None
    scale=max(abs(x) for x in p)
    if scale<=CANONICAL_EPS:return [0.0]*CANONICAL_COEFF_SLOTS
    p=[x/scale for x in p]
    degree=_poly_degree(p)
    if p[degree]<0:p=[-x for x in p]
    return [0.0 if abs(x)<1e-12 else x for x in p]

def _canonical_row(eq):
    at=_eq_at(eq)
    if at<0:return None
    left=_rpn_affine(rpn(eq[:at])); right=_rpn_affine(rpn(eq[at+1:]))
    if left is None or right is None:return None
    a=left[0]-right[0]; b=left[1]-right[1]; c=-(left[2]-right[2])
    scale=max(abs(a),abs(b),abs(c))
    if scale<=CANONICAL_EPS:return [0.0,0.0,0.0]
    row=[a/scale,b/scale,c/scale]
    first=next((x for x in row if abs(x)>CANONICAL_EPS),0.0)
    if first<0:row=[-x for x in row]
    return [0.0 if abs(x)<1e-12 else x for x in row]

def _canonical_numeric(src,fam):
    if fam==SYSTEM:
        parts=[q for q in src.split(';') if q]
        if len(parts)!=2:return None
        rows=[_canonical_row(q) for q in parts]
        if any(r is None for r in rows):return None
        rows=sorted(rows,key=lambda r:(r[0],r[1],r[2]))
        return rows[0]+rows[1]
    if fam in (LINEAR,POLYNOMIAL):return _canonical_poly(src,fam)
    return None

'''
insert_once(
    "colab/train_v5_deepmind.py",
    "def encode(raw):\n",
    PY_CANONICAL,
    "insert Python canonical coefficient algebra",
)
replace_once(
    "colab/train_v5_deepmind.py",
    "def encode(raw):\n    src=normalize_text(raw); equations=[q for q in src.split(';') if q]\n    if not 1<=len(equations)<=2: raise ValueError(\"equation count\")\n    nodes=[]",
    "def encode(raw):\n    src=normalize_text(raw); equations=[q for q in src.split(';') if q]\n    if not 1<=len(equations)<=2: raise ValueError(\"equation count\")\n    fam=family_of(src)\n    canonical=_canonical_numeric(src,fam)\n    if canonical is not None:\n        kinds=np.zeros(MAX_NODES,np.int64); numeric=np.zeros(MAX_NODES,np.float32); depth=np.zeros(MAX_NODES,np.float32)\n        for i,v in enumerate(canonical):\n            kinds[i]=NUMBER; numeric[i]=np.float32(v); depth[i]=np.float32((i+1)/len(canonical))\n        return kinds,numeric,depth,fam,src\n    nodes=[]",
    "activate Python canonical encoding",
)
replace_once(
    "colab/train_v5_deepmind.py",
    "        pos=torch.linspace(0,1,MAX_NODES,device=kinds.device).view(1,MAX_NODES,1).expand(b,-1,-1)\n        feats=torch.cat([emb,numeric.unsqueeze(-1),pos,depth.unsqueeze(-1)],-1).reshape(b,-1)",
    "        active=(kinds != PAD).to(numeric.dtype).unsqueeze(-1)\n        pos=torch.linspace(0,1,MAX_NODES,device=kinds.device).view(1,MAX_NODES,1).expand(b,-1,-1)*active\n        feats=torch.cat([emb,numeric.unsqueeze(-1),pos,depth.unsqueeze(-1)],-1).reshape(b,-1)",
    "mask Python PAD positional features",
)
replace_once(
    "colab/train_v5_deepmind.py",
    "assert sum(p.numel() for p in model.parameters())==300984",
    "assert sum(p.numel() for p in model.parameters())==167800",
    "Python parameter assertion",
)
replace_once(
    "colab/train_v5_deepmind.py",
    "            g=g*scale; m.mul_(b1).add_(g,alpha=1-b1); v.mul_(b2).addcmul_(g,g,value=1-b2)\n            p.addcdiv_(m/c1,(v/c2).sqrt().add_(1e-8),value=-lr)",
    "            g=g*scale; m.mul_(b1).add_(g,alpha=1-b1); v.mul_(b2).addcmul_(g,g,value=1-b2)\n            if p.ndim > 1: p.mul_(1.0-lr*WEIGHT_DECAY)\n            p.addcdiv_(m/c1,(v/c2).sqrt().add_(1e-8),value=-lr)",
    "Python AdamW decay",
)

# Official DeepMind polynomial prompt adapter: accept both '?' and '.' endings
# from algebra.py. Factor prompts remain deliberately excluded because their
# target is a factorized expression, not roots; they are valid data for a
# different output contract, not bad examples.
replace_once(
    "colab/train_v5_deepmind.py",
    '      r"^Let (.+?=.+?)\\. (?:What is|Calculate) [A-Za-z]\\??$", r"^Suppose (.+?=.+?)\\. (?:What is|Calculate) [A-Za-z]\\??$",',
    '      r"^Let (.+?=.+?)\\. (?:What is|Calculate) [A-Za-z][?.]$", r"^Suppose (.+?=.+?)\\. (?:What is|Calculate) [A-Za-z][?.]$",',
    "base DeepMind polynomial punctuation",
)
replace_once(
    "colab/train_v5_deepmind.py",
    "def deepmind_example(rng):",
    "def deepmind_example(rng, allow_synthetic_fallback=True):",
    "DeepMind fallback control",
)
replace_once(
    "colab/train_v5_deepmind.py",
    "    return synthetic(rng)\n\n# ========================= BATCHING",
    "    if allow_synthetic_fallback:\n        return synthetic(rng)\n    raise RuntimeError(\"DeepMind parser exhausted without a compatible equation example\")\n\n# ========================= BATCHING",
    "forbid DeepMind fallback when requested",
)

# Replace synthetic checkpoint holdout with an official DeepMind interpolation
# bank whenever the worker is configured DeepMind-only.
OLD_HOLDOUT = '''# Fixed external holdout: larger coefficient/solution range, never fed into training.\nhold_rng=random.Random(0xA165)\nholdout=[synthetic(hold_rng,max_abs=240) for _ in range(160)]\n'''
NEW_HOLDOUT = '''# Fixed holdout. DeepMind-only runs use the official interpolate split and never\n# call the project synthetic generator; mixed/local experiments keep the legacy bank.\ndef _build_official_holdout(count=160):\n    global dm_modules, DM_NAMES\n    old_modules=dm_modules; old_names=list(DM_NAMES)\n    dm_modules=dm_algebra.test(); DM_NAMES=[\"linear_1d\",\"linear_2d\",\"polynomial_roots\"]\n    rng=random.Random(0xA165); out=[]; attempts=0\n    try:\n        while len(out)<count and attempts<count*500:\n            attempts+=1\n            try:e=deepmind_example(rng,allow_synthetic_fallback=False)\n            except Exception:continue\n            vals=list(e['roots'])+list(e['system'])\n            if any((not math.isfinite(float(v))) or abs(float(v))>300 for v in vals):continue\n            out.append(e)\n    finally:\n        dm_modules=old_modules; DM_NAMES=old_names\n    if len(out)!=count:raise RuntimeError(f\"official DeepMind holdout short: {len(out)}/{count}\")\n    return out\n\nhold_rng=random.Random(0xA165)\nholdout=_build_official_holdout(160) if DEEPMIND_RATIO>=0.999 else [synthetic(hold_rng,max_abs=240) for _ in range(160)]\n'''
replace_once("colab/train_v5_deepmind.py", OLD_HOLDOUT, NEW_HOLDOUT, "official DeepMind checkpoint holdout")

# ---------------------------------------------------------------------------
# Turbo worker: propagate DeepMind-only settings into base mirror, track split
# ranges for curriculum, improve official polynomial parser, and apply AdamW.
# ---------------------------------------------------------------------------
replace_once(
    "colab/turbo_train_v5.py",
    "LEARNING_RATE = 2.0e-4\nMIN_LEARNING_RATE = 2.0e-5",
    "LEARNING_RATE = 2.0e-4\nMIN_LEARNING_RATE = 2.0e-5\nWEIGHT_DECAY = 1e-5",
    "turbo weight decay setting",
)
replace_once(
    "colab/turbo_train_v5.py",
    "base_src = _replace_setting(base_src, \"RESUME_FROM_MAI5\", \"False\")\nbase_src = _replace_setting(base_src, \"AUTO_DOWNLOAD_AT_END\", \"False\")",
    "base_src = _replace_setting(base_src, \"RESUME_FROM_MAI5\", \"False\")\nbase_src = _replace_setting(base_src, \"AUTO_DOWNLOAD_AT_END\", \"False\")\nbase_src = _replace_setting(base_src, \"DEEPMIND_RATIO\", repr(DEEPMIND_RATIO))\nbase_src = _replace_setting(base_src, \"WEIGHT_DECAY\", repr(WEIGHT_DECAY))",
    "propagate DeepMind-only base settings",
)
replace_once(
    "colab/turbo_train_v5.py",
    '    r"^Let (.+?=.+?)\\. (?:What is|Calculate) ([A-Za-z])\\??$",\n    r"^Suppose (.+?=.+?)\\. (?:What is|Calculate) ([A-Za-z])\\??$",',
    '    r"^Let (.+?=.+?)\\. (?:What is|Calculate) ([A-Za-z])[?.]$",\n    r"^Suppose (.+?=.+?)\\. (?:What is|Calculate) ([A-Za-z])[?.]$",',
    "turbo polynomial punctuation",
)
replace_once(
    "colab/turbo_train_v5.py",
    "dm_writer = PoolWriter(dm_capacity)\nrejects = {}\npre_start = time.time()",
    "dm_writer = PoolWriter(dm_capacity)\nrejects = {}\ndm_split_ranges = {}\npre_start = time.time()",
    "initialize curriculum split ranges",
)
replace_once(
    "colab/turbo_train_v5.py",
    "for file_idx, ((split_name, module), path) in enumerate(located.items(), 1):\n    accepted_here = 0; seen_here = 0",
    "for file_idx, ((split_name, module), path) in enumerate(located.items(), 1):\n    if split_name not in dm_split_ranges:\n        dm_split_ranges[split_name] = [dm_writer.count, dm_writer.count]\n    accepted_here = 0; seen_here = 0",
    "start curriculum split ranges",
)
replace_once(
    "colab/turbo_train_v5.py",
    "    print(f\"  ✓ accepted={accepted_here} / seen={seen_here} in {time.time()-t0:.1f}s\")\ndm_writer.trim()",
    "    dm_split_ranges[split_name][1] = dm_writer.count\n    print(f\"  ✓ accepted={accepted_here} / seen={seen_here} in {time.time()-t0:.1f}s\")\ndm_writer.trim()\ndm_split_ends = {name: end for name, (_, end) in dm_split_ranges.items()}\nprint(\"DeepMind curriculum ranges:\", dm_split_ranges)",
    "finalize curriculum split ranges",
)
replace_once(
    "colab/turbo_train_v5.py",
    "        for p, g, m, v in zip(params, grads, moments, velocities):\n            g = g * clip\n            m.mul_(b1).add_(g, alpha=1-b1)\n            v.mul_(b2).addcmul_(g, g, value=1-b2)\n            p.addcdiv_(m / c1, (v / c2).sqrt().add_(1e-8), value=-lr)",
    "        for p, g, m, v in zip(params, grads, moments, velocities):\n            g = g * clip\n            m.mul_(b1).add_(g, alpha=1-b1)\n            v.mul_(b2).addcmul_(g, g, value=1-b2)\n            if p.ndim > 1:\n                p.mul_(1.0 - lr * WEIGHT_DECAY)\n            p.addcdiv_(m / c1, (v / c2).sqrt().add_(1e-8), value=-lr)",
    "turbo AdamW decay",
)
replace_once(
    "colab/turbo_train_v5.py",
    "print(f\"Final LR       : {MIN_LEARNING_RATE:g}\")",
    "print(f\"Final LR       : {MIN_LEARNING_RATE:g}\")\nprint(f\"Weight decay   : {WEIGHT_DECAY:g} (AdamW, matrices only)\")",
    "print turbo weight decay",
)

# ---------------------------------------------------------------------------
# DeepMind-only runtime wrapper: curriculum sampler (easy -> easy+medium -> all)
# without creating any project-generated training example.
# ---------------------------------------------------------------------------
OLD_TAKE = '''def take(pool, count):\n    idx = torch.randint(0, pool["size"], (count,), device=device)\n    return (\n        pool["k"][idx].long(),\n        pool["n"][idx].float(),\n        pool["d"][idx].float(),\n        pool["f"][idx].long(),\n        pool["r"][idx],\n        pool["rc"][idx].long(),\n        pool["sy"][idx],\n        pool["st"][idx].long(),\n        None,\n    )\n\n\ndef mixed_batch(batch_size):\n    return take(dm_pool, batch_size)\n'''
NEW_TAKE = '''def take(pool, count, upper=None):\n    upper = pool["size"] if upper is None else max(1, min(int(upper), int(pool["size"])))\n    idx = torch.randint(0, upper, (count,), device=device)\n    return (\n        pool["k"][idx].long(),\n        pool["n"][idx].float(),\n        pool["d"][idx].float(),\n        pool["f"][idx].long(),\n        pool["r"][idx],\n        pool["rc"][idx].long(),\n        pool["sy"][idx],\n        pool["st"][idx].long(),\n        None,\n    )\n\n\ndef curriculum_limit(step_idx=None):\n    if step_idx is None:\n        return dm_pool["size"], "all"\n    frac = float(step_idx) / max(float(TOTAL_STEPS), 1.0)\n    if frac <= 0.25:\n        return dm_split_ends["train-easy"], "easy"\n    if frac <= 0.60:\n        return dm_split_ends["train-medium"], "easy+medium"\n    return dm_split_ends["train-hard"], "easy+medium+hard"\n\n\ndef mixed_batch(batch_size, step_idx=None):\n    upper, _ = curriculum_limit(step_idx)\n    return take(dm_pool, batch_size, upper)\n'''
replace_once("kaggle/deepmind_only_train.py", OLD_TAKE, NEW_TAKE, "DeepMind-only curriculum sampler")
replace_once(
    "kaggle/deepmind_only_train.py",
    "    text = text.replace('other = train_model(*eqv)', 'other = None')",
    "    text = text.replace('other = train_model(*eqv)', 'other = None')\n    text = text.replace('k,n,d,f,r,rc,sy,st,eqv = mixed_batch(BATCH_SIZE)', 'k,n,d,f,r,rc,sy,st,eqv = mixed_batch(BATCH_SIZE, step_idx)', 1)",
    "pass curriculum step into sampler",
)
replace_once(
    "kaggle/deepmind_only_train.py",
    "    print(\"[DEEPMIND-ONLY] project augmentation: DISABLED\", flush=True)",
    "    print(\"[DEEPMIND-ONLY] project augmentation: DISABLED\", flush=True)\n    print(\"[DEEPMIND-ONLY] representation: canonical numeric coefficients for linear/poly/system\", flush=True)\n    print(\"[DEEPMIND-ONLY] curriculum: easy -> easy+medium -> easy+medium+hard\", flush=True)",
    "document DeepMind-only curriculum",
)

# GitHub MAI5 header gate must accept the new incompatible architecture version.
replace_once(
    ".github/workflows/kaggle-train.yml",
    "if magic != 0x4D414935 or version != 1:",
    "if magic != 0x4D414935 or version != 2:",
    "workflow MAI5 v2 gate",
)

# ---------------------------------------------------------------------------
# Canonicalization tests. They test invariance, not model accuracy.
# ---------------------------------------------------------------------------
test_path = ROOT / "app/src/test/java/com/example/equationsolver/ai/CanonicalNumericEncoderTest.kt"
test_path.parent.mkdir(parents=True, exist_ok=True)
if test_path.exists():
    raise RuntimeError("CanonicalNumericEncoderTest.kt already exists")
test_path.write_text(r'''package com.example.equationsolver.ai

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class CanonicalNumericEncoderTest {
    private fun coeffs(equation: String): FloatArray = StructuralMathEncoder.encode(equation).let { e ->
        assertEquals(V5ModelSpec.CANONICAL_COEFF_SLOTS, e.nodeCount)
        assertTrue((0 until V5ModelSpec.CANONICAL_COEFF_SLOTS).all { e.kinds[it] == StructuralMathEncoder.Kind.NUMBER })
        e.numeric.copyOfRange(0, V5ModelSpec.CANONICAL_COEFF_SLOTS)
    }

    @Test fun linearSideAndScaleAreCanonical() {
        assertArrayEquals(coeffs("2x+4=10"), coeffs("20=4x+8"), 1e-6f)
    }

    @Test fun polynomialFactoredAndExpandedAreCanonical() {
        assertArrayEquals(coeffs("x^2-1=0"), coeffs("0=(x-1)*(x+1)"), 1e-6f)
    }

    @Test fun systemOrderSideAndScaleAreCanonical() {
        val a = coeffs("8x+7y=251;9x=180")
        val b = coeffs("360=18x;502=16x+14y")
        assertArrayEquals(a, b, 1e-6f)
        assertEquals(EquationFamily.SYSTEM, StructuralMathEncoder.encode("8x+7y=251;9x=180").family)
    }
}
''')
print("OK add canonical encoder invariance tests")

print("MIGRATION_COMPLETE")
