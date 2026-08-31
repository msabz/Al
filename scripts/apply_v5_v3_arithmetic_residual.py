#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path, old, new):
    p = ROOT / path
    text = p.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected exactly one target, found {count}: {old[:100]!r}")
    p.write_text(text.replace(old, new, 1))
    print("patched", path)

# ---------------------------------------------------------------------------
# Android model contract: semantic encoding change => fresh MAI5 version.
# ---------------------------------------------------------------------------
replace_once(
    "app/src/main/java/com/example/equationsolver/ai/V5ModelSpec.kt",
    "    const val FILE_VERSION = 2",
    "    const val FILE_VERSION = 3",
)
replace_once(
    "app/src/main/java/com/example/equationsolver/ai/V5ModelSpec.kt",
    "    const val CANONICAL_COEFF_SLOTS = 6\n}",
    "    const val CANONICAL_COEFF_SLOTS = 6\n"
    "    const val POLYNOMIAL_FEATURE_SLOTS = 7 // six q(z)=P(ROOT_SCALE*z) coefficients + degree/5\n"
    "    const val SYSTEM_FEATURE_SLOTS = 9     // two scaled rows + normalized Cramer invariants (det,nx,ny)\n"
    "    const val POLYNOMIAL_RESIDUAL_WEIGHT = 0.15\n}"
)

# ---------------------------------------------------------------------------
# Kotlin encoder: root-scaled polynomial coordinate + root-scaled systems.
# ---------------------------------------------------------------------------
replace_once(
    "app/src/main/java/com/example/equationsolver/ai/StructuralMathEncoder.kt",
    "        require(values.size == V5ModelSpec.CANONICAL_COEFF_SLOTS)",
    "        require(values.isNotEmpty() && values.size <= V5ModelSpec.MAX_NODES)"
)
replace_once(
    "app/src/main/java/com/example/equationsolver/ai/StructuralMathEncoder.kt",
    "        canonicalizePolynomial(coeff)\n        return coefficientEncoding(source, family, coeff)",
    "        if (family == EquationFamily.POLYNOMIAL) {\n"
    "            canonicalizePolynomialForRootScale(coeff)\n"
    "            val features = DoubleArray(V5ModelSpec.POLYNOMIAL_FEATURE_SLOTS)\n"
    "            for (i in coeff.indices) features[i] = coeff[i]\n"
    "            features[V5ModelSpec.CANONICAL_COEFF_SLOTS] = degree.toDouble() / 5.0\n"
    "            return coefficientEncoding(source, family, features)\n"
    "        }\n"
    "        canonicalizePolynomial(coeff)\n"
    "        return coefficientEncoding(source, family, coeff)"
)
replace_once(
    "app/src/main/java/com/example/equationsolver/ai/StructuralMathEncoder.kt",
    "        val values = doubleArrayOf(\n            rows[0][0], rows[0][1], rows[0][2],\n            rows[1][0], rows[1][1], rows[1][2]\n        )\n        return coefficientEncoding(source, EquationFamily.SYSTEM, values)",
    "        val a1 = rows[0][0]; val b1 = rows[0][1]; val c1 = rows[0][2]\n"
    "        val a2 = rows[1][0]; val b2 = rows[1][1]; val c2 = rows[1][2]\n"
    "        val det = a1 * b2 - a2 * b1\n"
    "        val nx = c1 * b2 - c2 * b1\n"
    "        val ny = a1 * c2 - a2 * c1\n"
    "        val invariantScale = maxOf(abs(det), abs(nx), abs(ny))\n"
    "        val inv = if (invariantScale <= CANONICAL_EPS) doubleArrayOf(0.0, 0.0, 0.0)\n"
    "            else doubleArrayOf(det / invariantScale, nx / invariantScale, ny / invariantScale)\n"
    "        val values = doubleArrayOf(\n"
    "            a1, b1, c1, a2, b2, c2,\n"
    "            cleanZero(inv[0]), cleanZero(inv[1]), cleanZero(inv[2])\n"
    "        )\n"
    "        return coefficientEncoding(source, EquationFamily.SYSTEM, values)"
)
replace_once(
    "app/src/main/java/com/example/equationsolver/ai/StructuralMathEncoder.kt",
    "        var c = -(left.c - right.c)\n        val scale = maxOf(abs(a), abs(b), abs(c))",
    "        var c = -(left.c - right.c) / V5ModelSpec.ROOT_SCALE\n"
    "        // x=ROOT_SCALE*z, y=ROOT_SCALE*w => a*z+b*w=c/ROOT_SCALE.\n"
    "        // This keeps coefficients aligned with the network's normalized outputs.\n"
    "        val scale = maxOf(abs(a), abs(b), abs(c))"
)
replace_once(
    "app/src/main/java/com/example/equationsolver/ai/StructuralMathEncoder.kt",
    "    private fun canonicalizePolynomial(p: DoubleArray) {\n        val scale = p.maxOf { abs(it) }",
    "    private fun canonicalizePolynomialForRootScale(p: DoubleArray) {\n"
    "        // The network predicts z=root/ROOT_SCALE. Encode q(z)=P(ROOT_SCALE*z),\n"
    "        // then normalize q globally. Roots of q are exactly the normalized outputs.\n"
    "        for (power in 1 until p.size) p[power] *= V5ModelSpec.ROOT_SCALE.pow(power.toDouble())\n"
    "        canonicalizePolynomial(p)\n"
    "    }\n\n"
    "    private fun canonicalizePolynomial(p: DoubleArray) {\n        val scale = p.maxOf { abs(it) }"
)

# ---------------------------------------------------------------------------
# Kotlin training: add a bounded polynomial-equation residual loss signal.
# ---------------------------------------------------------------------------
replace_once(
    "app/src/main/java/com/example/equationsolver/ai/NeuralNetwork.kt",
    "            val supervised = supervisedGradient(cache.out, item.target)",
    "            val supervised = supervisedGradient(cache.out, item.target, item.input)"
)
replace_once(
    "app/src/main/java/com/example/equationsolver/ai/NeuralNetwork.kt",
    "    private fun supervisedGradient(out: FloatArray, target: V5Target): Pair<Double, FloatArray> {",
    "    private fun supervisedGradient(\n"
    "        out: FloatArray,\n"
    "        target: V5Target,\n"
    "        encoding: StructuralMathEncoder.Encoding\n"
    "    ): Pair<Double, FloatArray> {"
)
needle = "        val stateStart = V5ModelSpec.ROOT_SLOTS * 2\n"
insert = """        if (target.state == SolutionState.FINITE && target.family == EquationFamily.POLYNOMIAL) {
            val coeff = DoubleArray(V5ModelSpec.CANONICAL_COEFF_SLOTS) { encoding.numeric[it].toDouble() }
            val activePoly = assignedPresence.count { it }.coerceAtLeast(1)
            for (slot in 0 until V5ModelSpec.ROOT_SLOTS) {
                if (!assignedPresence[slot]) continue
                val z = out[slot].toDouble()
                var q = coeff.last()
                var dq = 0.0
                for (power in coeff.lastIndex - 1 downTo 0) {
                    dq = dq * z + q
                    q = q * z + coeff[power]
                }
                val absQ = kotlin.math.abs(q)
                val beta = 0.25
                val residualLoss = if (absQ < beta) 0.5 * q * q / beta else absQ - 0.5 * beta
                val dLossDq = if (absQ < beta) q / beta else if (q >= 0.0) 1.0 else -1.0
                loss += V5ModelSpec.POLYNOMIAL_RESIDUAL_WEIGHT * residualLoss / activePoly
                grad[slot] += (V5ModelSpec.POLYNOMIAL_RESIDUAL_WEIGHT * dLossDq * dq / activePoly).toFloat()
            }
        }

"""
p = ROOT / "app/src/main/java/com/example/equationsolver/ai/NeuralNetwork.kt"
text = p.read_text()
if text.count(needle) != 1:
    raise RuntimeError("NeuralNetwork.kt: stateStart target count != 1")
p.write_text(text.replace(needle, insert + needle, 1))
print("patched NeuralNetwork.kt residual")

# ---------------------------------------------------------------------------
# Python mirror: exact representation semantics and MAI5 version.
# ---------------------------------------------------------------------------
replace_once(
    "colab/train_v5_deepmind.py",
    "MAGIC=0x4D414935; VERSION=2",
    "MAGIC=0x4D414935; VERSION=3"
)
replace_once(
    "colab/train_v5_deepmind.py",
    "ROOT_SCALE=100.0; MAX_GRAD_NORM=5.0; CANONICAL_COEFF_SLOTS=6",
    "ROOT_SCALE=100.0; MAX_GRAD_NORM=5.0; CANONICAL_COEFF_SLOTS=6; POLYNOMIAL_FEATURE_SLOTS=7; SYSTEM_FEATURE_SLOTS=9; POLYNOMIAL_RESIDUAL_WEIGHT=0.15"
)
replace_once(
    "colab/train_v5_deepmind.py",
    "    scale=max(abs(x) for x in p)\n    if scale<=CANONICAL_EPS:return [0.0]*CANONICAL_COEFF_SLOTS\n    p=[x/scale for x in p]\n    degree=_poly_degree(p)\n    if p[degree]<0:p=[-x for x in p]\n    return [0.0 if abs(x)<1e-12 else x for x in p]",
    "    if fam==POLYNOMIAL:\n"
    "        p=[x*(ROOT_SCALE**i) for i,x in enumerate(p)]\n"
    "    scale=max(abs(x) for x in p)\n"
    "    if scale<=CANONICAL_EPS:\n"
    "        base=[0.0]*CANONICAL_COEFF_SLOTS\n"
    "    else:\n"
    "        base=[x/scale for x in p]\n"
    "        d=_poly_degree(base)\n"
    "        if base[d]<0:base=[-x for x in base]\n"
    "        base=[0.0 if abs(x)<1e-12 else x for x in base]\n"
    "    if fam==POLYNOMIAL:return base+[degree/5.0]\n"
    "    return base"
)
replace_once(
    "colab/train_v5_deepmind.py",
    "    a=left[0]-right[0]; b=left[1]-right[1]; c=-(left[2]-right[2])",
    "    a=left[0]-right[0]; b=left[1]-right[1]; c=-(left[2]-right[2])/ROOT_SCALE"
)
replace_once(
    "colab/train_v5_deepmind.py",
    "        rows=sorted(rows,key=lambda r:(r[0],r[1],r[2]))\n        return rows[0]+rows[1]",
    "        rows=sorted(rows,key=lambda r:(r[0],r[1],r[2]))\n"
    "        a1,b1,c1=rows[0]; a2,b2,c2=rows[1]\n"
    "        det=a1*b2-a2*b1; nx=c1*b2-c2*b1; ny=a1*c2-a2*c1\n"
    "        scale=max(abs(det),abs(nx),abs(ny))\n"
    "        inv=[0.0,0.0,0.0] if scale<=CANONICAL_EPS else [det/scale,nx/scale,ny/scale]\n"
    "        return rows[0]+rows[1]+[0.0 if abs(x)<1e-12 else x for x in inv]"
)
replace_once(
    "colab/train_v5_deepmind.py",
    "def loss_fn(out,roots,root_count,systems,states,families,other_out=None):",
    "def loss_fn(out,roots,root_count,systems,states,families,numeric,other_out=None):"
)
replace_once(
    "colab/train_v5_deepmind.py",
    "    presence=F.binary_cross_entropy_with_logits(out[:,5:10],assigned_pres)\n    total=root_loss + .35*presence + .35*state_loss",
    "    presence=F.binary_cross_entropy_with_logits(out[:,5:10],assigned_pres)\n"
    "    residual=out.sum()*0\n"
    "    poly_ids=torch.where(finite & (families==POLYNOMIAL))[0]\n"
    "    if len(poly_ids):\n"
    "        coeff=numeric[poly_ids,:CANONICAL_COEFF_SLOTS].to(out.dtype)\n"
    "        z=out[poly_ids,:ROOT_SLOTS]\n"
    "        q=coeff[:,-1,None].expand(-1,ROOT_SLOTS)\n"
    "        for power in range(CANONICAL_COEFF_SLOTS-2,-1,-1):q=q*z+coeff[:,power,None]\n"
    "        mask=assigned_pres[poly_ids]\n"
    "        per=F.smooth_l1_loss(q,torch.zeros_like(q),reduction='none',beta=0.25)\n"
    "        residual=((per*mask).sum(-1)/mask.sum(-1).clamp_min(1)).mean()\n"
    "    total=root_loss + .35*presence + .35*state_loss + POLYNOMIAL_RESIDUAL_WEIGHT*residual"
)
replace_once(
    "colab/train_v5_deepmind.py",
    "    loss=loss_fn(out,r,rc,sy,st,f,other)",
    "    loss=loss_fn(out,r,rc,sy,st,f,n,other)"
)

# ---------------------------------------------------------------------------
# Turbo GPU worker: same residual signal; still DeepMind-only via wrapper.
# ---------------------------------------------------------------------------
replace_once(
    "colab/turbo_train_v5.py",
    "SYSTEM = ns[\"SYSTEM\"]\nPERMS = ns[\"PERMS\"]",
    "SYSTEM = ns[\"SYSTEM\"]\nPOLYNOMIAL = ns[\"POLYNOMIAL\"]\nPOLYNOMIAL_RESIDUAL_WEIGHT = ns[\"POLYNOMIAL_RESIDUAL_WEIGHT\"]\nCANONICAL_COEFF_SLOTS = ns[\"CANONICAL_COEFF_SLOTS\"]\nPERMS = ns[\"PERMS\"]"
)
replace_once(
    "colab/turbo_train_v5.py",
    "def stable_loss(out, roots, root_count, systems, states, families, other_out=None):",
    "def stable_loss(out, roots, root_count, systems, states, families, numeric, other_out=None):"
)
replace_once(
    "colab/turbo_train_v5.py",
    "    presence = F.binary_cross_entropy_with_logits(out[:,5:10], assigned_pres)\n    consistency = out.sum() * 0",
    "    presence = F.binary_cross_entropy_with_logits(out[:,5:10], assigned_pres)\n"
    "    residual = out.sum() * 0\n"
    "    poly_ids = torch.where(finite & (families == POLYNOMIAL))[0]\n"
    "    if len(poly_ids):\n"
    "        coeff = numeric[poly_ids,:CANONICAL_COEFF_SLOTS].to(out.dtype)\n"
    "        z = out[poly_ids,:ROOT_SLOTS]\n"
    "        q = coeff[:,-1,None].expand(-1,ROOT_SLOTS)\n"
    "        for power in range(CANONICAL_COEFF_SLOTS-2,-1,-1):\n"
    "            q = q * z + coeff[:,power,None]\n"
    "        mask = assigned_pres[poly_ids]\n"
    "        per_res = F.smooth_l1_loss(q, torch.zeros_like(q), reduction=\"none\", beta=0.25)\n"
    "        residual = ((per_res * mask).sum(-1) / mask.sum(-1).clamp_min(1)).mean()\n"
    "    consistency = out.sum() * 0"
)
replace_once(
    "colab/turbo_train_v5.py",
    "    total = root_loss + 0.35 * presence + 0.35 * state_loss + CONSISTENCY_WEIGHT * consistency\n    return total, root_loss, presence, state_loss, consistency",
    "    total = root_loss + 0.35 * presence + 0.35 * state_loss + POLYNOMIAL_RESIDUAL_WEIGHT * residual + CONSISTENCY_WEIGHT * consistency\n"
    "    return total, root_loss, presence, state_loss, residual, consistency"
)
# Four stable_loss call sites: benchmark warmup, benchmark timed, train.
p = ROOT / "colab/turbo_train_v5.py"
text = p.read_text()
old = "stable_loss(o,r,rc,sy,st,f,oo)"
if text.count(old) != 2:
    raise RuntimeError(f"turbo benchmark stable_loss target count={text.count(old)}")
text = text.replace(old, "stable_loss(o,r,rc,sy,st,f,n,oo)")
old2 = "loss, root_l, pres_l, state_l, cons_l = stable_loss(out,r,rc,sy,st,f,other)"
if text.count(old2) != 1:
    raise RuntimeError("turbo train stable_loss target missing")
text = text.replace(old2, "loss, root_l, pres_l, state_l, residual_l, cons_l = stable_loss(out,r,rc,sy,st,f,n,other)", 1)
old3 = "f\"pres={float(pres_l.detach()):6.4f} state={float(state_l.detach()):6.4f} \"\n            f\"cons={float(cons_l.detach()):6.4f}"
new3 = "f\"pres={float(pres_l.detach()):6.4f} state={float(state_l.detach()):6.4f} \"\n            f\"res={float(residual_l.detach()):6.4f} cons={float(cons_l.detach()):6.4f}"
if text.count(old3) != 1:
    raise RuntimeError("turbo logging target missing")
text = text.replace(old3, new3, 1)
p.write_text(text)
print("patched turbo stable loss calls/logging")

# DeepMind-only wrapper's representation description.
replace_once(
    "kaggle/deepmind_only_train.py",
    "    print(\"[DEEPMIND-ONLY] representation: canonical numeric coefficients for linear/poly/system\", flush=True)",
    "    print(\"[DEEPMIND-ONLY] representation: root-scaled canonical poly/system + Cramer invariants\", flush=True)"
)

# ---------------------------------------------------------------------------
# Tests: verify semantic invariants rather than assuming every family has 6 nodes.
# ---------------------------------------------------------------------------
p = ROOT / "app/src/test/java/com/example/equationsolver/ai/CanonicalNumericEncoderTest.kt"
p.write_text(r'''package com.example.equationsolver.ai

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs

class CanonicalNumericEncoderTest {
    private fun encoding(equation: String) = StructuralMathEncoder.encode(equation).also { e ->
        assertTrue((0 until e.nodeCount).all { e.kinds[it] == StructuralMathEncoder.Kind.NUMBER })
        assertTrue(!e.truncated)
    }

    private fun first(equation: String, count: Int): FloatArray = encoding(equation).numeric.copyOfRange(0, count)

    @Test fun linearSideAndScaleAreCanonical() {
        assertArrayEquals(first("2x+4=10", 6), first("20=4x+8", 6), 1e-6f)
        assertEquals(6, encoding("2x+4=10").nodeCount)
    }

    @Test fun polynomialFactoredAndExpandedAreCanonicalAndRootScaled() {
        val a = encoding("x^2-1=0")
        val b = encoding("0=(x-1)*(x+1)")
        assertEquals(V5ModelSpec.POLYNOMIAL_FEATURE_SLOTS, a.nodeCount)
        assertArrayEquals(a.numeric.copyOfRange(0, 6), b.numeric.copyOfRange(0, 6), 1e-6f)
        assertEquals(2f / 5f, a.numeric[6], 1e-6f)
        // q(z)=P(100z) normalized: for x^2-1 the z^2 coefficient dominates.
        assertTrue(abs(a.numeric[2] - 1f) < 1e-6f)
        assertTrue(abs(a.numeric[0]) < 0.001f)
    }

    @Test fun systemOrderSideAndScaleAreCanonicalAndCramerAware() {
        val a = encoding("8x+7y=251;9x=180")
        val b = encoding("360=18x;502=16x+14y")
        assertEquals(V5ModelSpec.SYSTEM_FEATURE_SLOTS, a.nodeCount)
        assertArrayEquals(a.numeric.copyOfRange(0, 9), b.numeric.copyOfRange(0, 9), 1e-6f)
        val c = a.numeric
        val det = c[0] * c[4] - c[3] * c[1]
        val nx = c[2] * c[4] - c[5] * c[1]
        val ny = c[0] * c[5] - c[3] * c[2]
        assertTrue(abs(det) > 1e-6f)
        assertEquals(20f / 100f, nx / det, 1e-4f)
        assertEquals(13f / 100f, ny / det, 1e-4f)
        assertEquals(EquationFamily.SYSTEM, a.family)
    }
}
''')
print("rewrote CanonicalNumericEncoderTest.kt")

# Extend existing neural test with version semantics and finite polynomial residual path.
p = ROOT / "app/src/test/java/com/example/equationsolver/ai/NeuralNetworkTest.kt"
text = p.read_text()
anchor = "class NeuralNetworkTest {\n"
if text.count(anchor) != 1:
    raise RuntimeError("NeuralNetworkTest anchor missing")
extra = '''class NeuralNetworkTest {\n    @Test fun v3SemanticContractIsActive() {\n        assertEquals(3, V5ModelSpec.FILE_VERSION)\n        assertEquals(7, V5ModelSpec.POLYNOMIAL_FEATURE_SLOTS)\n        assertEquals(9, V5ModelSpec.SYSTEM_FEATURE_SLOTS)\n    }\n\n'''
p.write_text(text.replace(anchor, extra, 1))
print("patched NeuralNetworkTest.kt")

print("V5_V3_ARITHMETIC_RESIDUAL_MIGRATION_OK")
