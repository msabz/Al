#!/usr/bin/env python3
"""Apply the single reviewed MAI5-v4 polynomial cardinality-head migration.

This is intentionally a narrow semantic migration:
- LINEAR / ANALYTIC / SYSTEM keep the existing per-slot presence semantics.
- POLYNOMIAL reuses output logits 5..9 as a 5-way cardinality distribution for 1..5 roots.
- Polynomial matching stays permutation invariant but no longer lets cardinality logits affect assignment.
- Polynomial inference predicts k with argmax and selects the k root slots with smallest |q(z)| residual.
- Tensor shapes and parameter count stay unchanged; file version is bumped because output semantics changed.
"""

from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one anchor, found {count}")
    return text.replace(old, new, 1)


def regex_once(text: str, pattern: str, replacement: str, label: str) -> str:
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.S)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one regex match, found {count}")
    return updated


def patch_model_spec() -> None:
    path = ROOT / "app/src/main/java/com/example/equationsolver/ai/V5ModelSpec.kt"
    text = path.read_text()
    text = replace_once(text, "const val FILE_VERSION = 3", "const val FILE_VERSION = 4", "Kotlin FILE_VERSION")
    text = replace_once(
        text,
        "const val PRESENCE_THRESHOLD = 0.50",
        "const val PRESENCE_THRESHOLD = 0.50 // non-polynomial heads only; polynomial uses 5-way root-count logits",
        "Kotlin presence semantic comment",
    )
    path.write_text(text)


def patch_neural_network() -> None:
    path = ROOT / "app/src/main/java/com/example/equationsolver/ai/NeuralNetwork.kt"
    text = path.read_text()
    text = replace_once(
        text,
        " *   5..9   presence logits\n *   10..13 solution-state logits",
        " *   5..9   LINEAR/ANALYTIC/SYSTEM: presence logits; POLYNOMIAL: root-count logits (classes 1..5)\n *   10..13 solution-state logits",
        "Kotlin head layout comment",
    )

    supervised = r'''    private fun supervisedGradient(
        out: FloatArray,
        target: V5Target,
        encoding: StructuralMathEncoder.Encoding
    ): Pair<Double, FloatArray> {
        val grad = FloatArray(V5ModelSpec.HEAD_OUTPUT)
        var loss = 0.0
        val rootWeight = 1.0
        val cardinalityWeight = 0.35
        val stateWeight = 0.35

        val assignedValues = FloatArray(V5ModelSpec.ROOT_SLOTS)
        val assignedPresence = BooleanArray(V5ModelSpec.ROOT_SLOTS)

        if (target.state == SolutionState.FINITE) {
            if (target.family == EquationFamily.SYSTEM) {
                target.systemValues.take(V5ModelSpec.ROOT_SLOTS).forEachIndexed { index, value ->
                    assignedValues[index] = (value / V5ModelSpec.ROOT_SCALE).toFloat()
                    assignedPresence[index] = true
                }
            } else {
                val roots = target.canonicalRoots()
                val best = bestPermutation(out, roots, target.family)
                for (slot in 0 until V5ModelSpec.ROOT_SLOTS) {
                    val source = best[slot]
                    if (source < roots.size) {
                        assignedValues[slot] = (roots[source] / V5ModelSpec.ROOT_SCALE).toFloat()
                        assignedPresence[slot] = true
                    }
                }
            }
        }

        val activeCount = max(1, assignedPresence.count { it })
        for (slot in 0 until V5ModelSpec.ROOT_SLOTS) {
            if (assignedPresence[slot]) {
                val d = out[slot] - assignedValues[slot]
                loss += rootWeight * d * d / activeCount
                grad[slot] += (2.0 * rootWeight * d / activeCount).toFloat()
            }
        }

        if (target.family == EquationFamily.POLYNOMIAL) {
            if (target.state == SolutionState.FINITE) {
                val rootCount = target.canonicalRoots().size.coerceIn(1, V5ModelSpec.ROOT_SLOTS)
                val countStart = V5ModelSpec.ROOT_SLOTS
                val countLogits = FloatArray(V5ModelSpec.ROOT_SLOTS) { out[countStart + it] }
                val countProbs = softmax(countLogits)
                val countClass = rootCount - 1
                loss += -cardinalityWeight * ln(countProbs[countClass].coerceAtLeast(1e-9))
                for (i in countProbs.indices) {
                    val label = if (i == countClass) 1.0 else 0.0
                    grad[countStart + i] += (cardinalityWeight * (countProbs[i] - label)).toFloat()
                }
            }
        } else {
            for (slot in 0 until V5ModelSpec.ROOT_SLOTS) {
                val logitIndex = V5ModelSpec.ROOT_SLOTS + slot
                val p = sigmoid(out[logitIndex])
                val label = if (assignedPresence[slot]) 1.0 else 0.0
                loss += cardinalityWeight * binaryCrossEntropy(p, label) / V5ModelSpec.ROOT_SLOTS
                grad[logitIndex] += (cardinalityWeight * (p - label) / V5ModelSpec.ROOT_SLOTS).toFloat()
            }
        }

        if (target.state == SolutionState.FINITE && target.family == EquationFamily.POLYNOMIAL) {
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

        val stateStart = V5ModelSpec.ROOT_SLOTS * 2
        val stateLogits = FloatArray(V5ModelSpec.STATE_COUNT) { out[stateStart + it] }
        val probs = softmax(stateLogits)
        val stateId = target.state.id
        loss += -stateWeight * ln(probs[stateId].coerceAtLeast(1e-9))
        for (i in probs.indices) {
            val label = if (i == stateId) 1.0 else 0.0
            grad[stateStart + i] += (stateWeight * (probs[i] - label)).toFloat()
        }
        return loss to grad
    }

'''
    text = regex_once(
        text,
        r"    private fun supervisedGradient\(.*?\n    /\*\* Returns, for every prediction slot, the index into roots or roots\.size\+ for padding\. \*/\n",
        supervised + "    /** Returns, for every prediction slot, the index into roots or roots.size+ for padding. */\n",
        "Kotlin supervisedGradient",
    )

    best_perm = r'''    private fun bestPermutation(out: FloatArray, roots: DoubleArray, family: EquationFamily): IntArray {
        val paddedValues = FloatArray(V5ModelSpec.ROOT_SLOTS)
        val paddedPresent = BooleanArray(V5ModelSpec.ROOT_SLOTS)
        for (i in roots.indices) {
            paddedValues[i] = (roots[i] / V5ModelSpec.ROOT_SCALE).toFloat()
            paddedPresent[i] = true
        }
        var best = PERMUTATIONS[0]
        var bestCost = Double.POSITIVE_INFINITY
        for (perm in PERMUTATIONS) {
            var cost = 0.0
            for (slot in 0 until V5ModelSpec.ROOT_SLOTS) {
                val source = perm[slot]
                val present = paddedPresent[source]
                if (present) {
                    val d = (out[slot] - paddedValues[source]).toDouble()
                    cost += d * d
                }
                if (family != EquationFamily.POLYNOMIAL) {
                    val p = sigmoid(out[V5ModelSpec.ROOT_SLOTS + slot])
                    cost += 0.35 * binaryCrossEntropy(p, if (present) 1.0 else 0.0)
                }
            }
            if (cost < bestCost) {
                bestCost = cost
                best = perm
            }
        }
        return best
    }

'''
    text = regex_once(
        text,
        r"    private fun bestPermutation\(.*?\n    private fun backward\(",
        best_perm + "    private fun backward(",
        "Kotlin bestPermutation",
    )

    prediction = r'''    private fun polynomialResidual(encoding: StructuralMathEncoder.Encoding, normalizedRoot: Double): Double {
        var q = encoding.numeric[V5ModelSpec.CANONICAL_COEFF_SLOTS - 1].toDouble()
        for (power in V5ModelSpec.CANONICAL_COEFF_SLOTS - 2 downTo 0) {
            q = q * normalizedRoot + encoding.numeric[power].toDouble()
        }
        return kotlin.math.abs(q)
    }

    private fun predictionFromCache(c: Cache): V5Prediction {
        val values = DoubleArray(V5ModelSpec.ROOT_SLOTS) { c.out[it].toDouble() * V5ModelSpec.ROOT_SCALE }
        val stateStart = V5ModelSpec.ROOT_SLOTS * 2
        val stateProbs = softmax(FloatArray(V5ModelSpec.STATE_COUNT) { c.out[stateStart + it] })
        var bestState = 0
        for (i in 1 until stateProbs.size) if (stateProbs[i] > stateProbs[bestState]) bestState = i
        val state = SolutionState.fromId(bestState)

        val presence = if (c.encoding.family == EquationFamily.POLYNOMIAL) {
            if (state != SolutionState.FINITE) {
                DoubleArray(V5ModelSpec.ROOT_SLOTS)
            } else {
                val countLogits = FloatArray(V5ModelSpec.ROOT_SLOTS) { c.out[V5ModelSpec.ROOT_SLOTS + it] }
                val countProbs = softmax(countLogits)
                var countClass = 0
                for (i in 1 until countProbs.size) if (countProbs[i] > countProbs[countClass]) countClass = i
                val predictedCount = countClass + 1
                val ranked = (0 until V5ModelSpec.ROOT_SLOTS).sortedBy { slot ->
                    polynomialResidual(c.encoding, c.out[slot].toDouble())
                }
                val selected = ranked.take(predictedCount).toSet()
                DoubleArray(V5ModelSpec.ROOT_SLOTS) { if (it in selected) 1.0 else 0.0 }
            }
        } else {
            DoubleArray(V5ModelSpec.ROOT_SLOTS) { sigmoid(c.out[V5ModelSpec.ROOT_SLOTS + it]) }
        }
        return V5Prediction(c.encoding.family, state, stateProbs, values, presence)
    }

'''
    text = regex_once(
        text,
        r"    private fun predictionFromCache\(.*?\n    fun parameterCount\(\): Int \{",
        prediction + "    fun parameterCount(): Int {",
        "Kotlin predictionFromCache",
    )
    path.write_text(text)


def patch_python_base() -> None:
    path = ROOT / "colab/train_v5_deepmind.py"
    text = path.read_text()
    text = replace_once(text, "MAGIC=0x4D414935; VERSION=3", "MAGIC=0x4D414935; VERSION=4", "Python MAI5 VERSION")

    new_loss = r'''def polynomial_active_indices(out_row, numeric_row):
    """Return exactly k polynomial root slots: k is learned, ranking is by |q(z)|."""
    count = int(torch.argmax(out_row[5:10]).item()) + 1
    coeff = numeric_row[:CANONICAL_COEFF_SLOTS].to(device=out_row.device, dtype=out_row.dtype)
    z = out_row[:ROOT_SLOTS]
    q = coeff[-1].expand_as(z)
    for power in range(CANONICAL_COEFF_SLOTS - 2, -1, -1):
        q = q * z + coeff[power]
    return torch.topk(torch.abs(q), k=count, largest=False).indices


def loss_fn(out,roots,root_count,systems,states,families,numeric,other_out=None):
    state_loss=F.cross_entropy(out[:,10:14],states)
    assigned_vals=torch.zeros((len(out),ROOT_SLOTS),device=device,dtype=out.dtype)
    assigned_pres=torch.zeros_like(assigned_vals)
    finite=states==FINITE; sysmask=finite & (families==SYSTEM); nonsys=finite & (families!=SYSTEM)
    if sysmask.any():
        assigned_vals[sysmask,:2]=(systems[sysmask,:2]/ROOT_SCALE).to(out.dtype); assigned_pres[sysmask,:2]=1
    ids=torch.where(nonsys)[0]
    if len(ids):
        tv=(roots[ids]/ROOT_SCALE).to(out.dtype); tc=root_count[ids]
        basepres=(torch.arange(ROOT_SLOTS,device=device)[None,:] < tc[:,None]).to(out.dtype)
        pv=tv[:,PERMS]; pp=basepres[:,PERMS]
        predv=out[ids,:5][:,None,:]; predlog=out[ids,5:10][:,None,:].expand(-1,len(PERMS),-1)
        active=pp.sum(-1).clamp_min(1)
        root_cost=(((predv-pv)**2)*pp).sum(-1)/active
        pres_cost=F.binary_cross_entropy_with_logits(predlog,pp,reduction='none').mean(-1)
        nonpoly_match=(families[ids]!=POLYNOMIAL).to(out.dtype)[:,None]
        cost=root_cost + .35*pres_cost*nonpoly_match
        best=cost.argmin(-1); rows=torch.arange(len(ids),device=device)
        assigned_vals[ids]=pv[rows,best]; assigned_pres[ids]=pp[rows,best]
    active=assigned_pres.sum(-1).clamp_min(1)
    root_loss=((((out[:,:5]-assigned_vals)**2)*assigned_pres).sum(-1)/active)[finite].mean() if finite.any() else out.sum()*0

    cardinality_per=torch.zeros(len(out),device=device,dtype=out.dtype)
    cardinality_used=torch.zeros(len(out),device=device,dtype=torch.bool)
    nonpoly=families!=POLYNOMIAL
    if nonpoly.any():
        per=F.binary_cross_entropy_with_logits(out[nonpoly,5:10],assigned_pres[nonpoly],reduction='none').mean(-1)
        cardinality_per[nonpoly]=per; cardinality_used[nonpoly]=True
    poly_ids=torch.where(finite & (families==POLYNOMIAL))[0]
    if len(poly_ids):
        count_target=root_count[poly_ids].clamp(1,ROOT_SLOTS)-1
        cardinality_per[poly_ids]=F.cross_entropy(out[poly_ids,5:10],count_target,reduction='none')
        cardinality_used[poly_ids]=True
    cardinality=cardinality_per[cardinality_used].mean() if cardinality_used.any() else out.sum()*0

    residual=out.sum()*0
    if len(poly_ids):
        coeff=numeric[poly_ids,:CANONICAL_COEFF_SLOTS].to(out.dtype)
        z=out[poly_ids,:ROOT_SLOTS]
        q=coeff[:,-1,None].expand(-1,ROOT_SLOTS)
        for power in range(CANONICAL_COEFF_SLOTS-2,-1,-1):q=q*z+coeff[:,power,None]
        mask=assigned_pres[poly_ids]
        per=F.smooth_l1_loss(q,torch.zeros_like(q),reduction='none',beta=0.25)
        residual=((per*mask).sum(-1)/mask.sum(-1).clamp_min(1)).mean()
    total=root_loss + .35*cardinality + .35*state_loss + POLYNOMIAL_RESIDUAL_WEIGHT*residual
    if other_out is not None: total=total + CONSISTENCY_WEIGHT*F.mse_loss(out,other_out)
    return total

'''
    text = regex_once(
        text,
        r"def loss_fn\(out,roots,root_count,systems,states,families,numeric,other_out=None\):.*?\n# ========================= TRAINING EXAMPLE GENERATORS =========================\n",
        new_loss + "# ========================= TRAINING EXAMPLE GENERATORS =========================\n",
        "Python loss_fn",
    )

    old_eval = """                else:\n                    probs=torch.sigmoid(out[i,5:10]); pv=(out[i,:5][probs>=.5]*ROOT_SCALE).cpu().numpy(); ev=np.asarray(e['roots'])\n                    if len(ev)==0: continue\n"""
    new_eval = """                else:\n                    if e['f']==POLYNOMIAL:\n                        active_idx=polynomial_active_indices(out[i],n[i])\n                        pv=(out[i,:5][active_idx]*ROOT_SCALE).cpu().numpy()\n                    else:\n                        probs=torch.sigmoid(out[i,5:10]); pv=(out[i,:5][probs>=.5]*ROOT_SCALE).cpu().numpy()\n                    ev=np.asarray(e['roots'])\n                    if len(ev)==0: continue\n"""
    text = replace_once(text, old_eval, new_eval, "Python evaluate decoder")
    path.write_text(text)


def patch_turbo() -> None:
    path = ROOT / "colab/turbo_train_v5.py"
    text = path.read_text()
    stable = r'''def stable_loss(out, roots, root_count, systems, states, families, numeric, other_out=None):
    state_loss = F.cross_entropy(out[:,10:14], states)
    assigned_vals = torch.zeros((len(out), ROOT_SLOTS), device=device, dtype=out.dtype)
    assigned_pres = torch.zeros_like(assigned_vals)
    finite = states == FINITE
    sysmask = finite & (families == SYSTEM)
    nonsys = finite & (families != SYSTEM)
    if sysmask.any():
        assigned_vals[sysmask,:2] = (systems[sysmask,:2] / ROOT_SCALE).to(out.dtype)
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
        nonpoly_match = (families[ids] != POLYNOMIAL).to(out.dtype)[:,None]
        cost = root_cost + 0.35 * pres_cost * nonpoly_match
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

    cardinality_per = torch.zeros(len(out), device=device, dtype=out.dtype)
    cardinality_used = torch.zeros(len(out), device=device, dtype=torch.bool)
    nonpoly = families != POLYNOMIAL
    if nonpoly.any():
        per_card = F.binary_cross_entropy_with_logits(out[nonpoly,5:10], assigned_pres[nonpoly], reduction="none").mean(-1)
        cardinality_per[nonpoly] = per_card
        cardinality_used[nonpoly] = True
    poly_ids = torch.where(finite & (families == POLYNOMIAL))[0]
    if len(poly_ids):
        count_target = root_count[poly_ids].clamp(1, ROOT_SLOTS) - 1
        cardinality_per[poly_ids] = F.cross_entropy(out[poly_ids,5:10], count_target, reduction="none")
        cardinality_used[poly_ids] = True
    cardinality = cardinality_per[cardinality_used].mean() if cardinality_used.any() else out.sum() * 0

    residual = out.sum() * 0
    if len(poly_ids):
        coeff = numeric[poly_ids,:CANONICAL_COEFF_SLOTS].to(out.dtype)
        z = out[poly_ids,:ROOT_SLOTS]
        q = coeff[:,-1,None].expand(-1,ROOT_SLOTS)
        for power in range(CANONICAL_COEFF_SLOTS-2,-1,-1):
            q = q * z + coeff[:,power,None]
        mask = assigned_pres[poly_ids]
        per_res = F.smooth_l1_loss(q, torch.zeros_like(q), reduction="none", beta=0.25)
        residual = ((per_res * mask).sum(-1) / mask.sum(-1).clamp_min(1)).mean()
    consistency = out.sum() * 0
    if other_out is not None:
        consistency = F.smooth_l1_loss(out, other_out, beta=0.1)
    total = root_loss + 0.35 * cardinality + 0.35 * state_loss + POLYNOMIAL_RESIDUAL_WEIGHT * residual + CONSISTENCY_WEIGHT * consistency
    return total, root_loss, cardinality, state_loss, residual, consistency

'''
    text = regex_once(
        text,
        r"def stable_loss\(out, roots, root_count, systems, states, families, numeric, other_out=None\):.*?\n\ndef turbo_adam_step\(",
        stable + "def turbo_adam_step(",
        "Turbo stable_loss",
    )
    text = replace_once(
        text,
        "loss, root_l, pres_l, state_l, residual_l, cons_l = stable_loss(out,r,rc,sy,st,f,n,other)",
        "loss, root_l, card_l, state_l, residual_l, cons_l = stable_loss(out,r,rc,sy,st,f,n,other)",
        "Turbo loss unpack",
    )
    text = replace_once(
        text,
        'f"pres={float(pres_l.detach()):6.4f} state={float(state_l.detach()):6.4f} "',
        'f"card={float(card_l.detach()):6.4f} state={float(state_l.detach()):6.4f} "',
        "Turbo telemetry label",
    )
    path.write_text(text)


def patch_audit() -> None:
    path = ROOT / "colab/generalization_audit.py"
    text = path.read_text()
    predict = r'''def predict_raw(ns, model, equation):
    torch = ns["torch"]
    np = ns["np"]
    k, n, d, fam, normalized = ns["encode"](equation)
    device = ns["device"]
    with torch.no_grad():
        kinds = torch.tensor(k[None, :], device=device, dtype=torch.long)
        numeric = torch.tensor(n[None, :], device=device, dtype=torch.float32)
        depth = torch.tensor(d[None, :], device=device, dtype=torch.float32)
        out = model(kinds, numeric, depth, torch.tensor([fam], device=device, dtype=torch.long))[0]
        slots = (out[:5] * ns["ROOT_SCALE"]).detach().cpu().numpy().astype(float)
        if fam == ns["POLYNOMIAL"]:
            count_probs = torch.softmax(out[5:10], dim=0).detach().cpu().numpy().astype(float)
            active = ns["polynomial_active_indices"](out, numeric[0]).detach().cpu().numpy().astype(int).tolist()
            presence = np.zeros(5, dtype=float)
            presence[active] = 1.0
        else:
            count_probs = np.asarray([], dtype=float)
            presence = torch.sigmoid(out[5:10]).detach().cpu().numpy().astype(float)
        state_probs = torch.softmax(out[10:14], dim=0).detach().cpu().numpy().astype(float)
    return {
        "family": int(fam),
        "state": int(np.argmax(state_probs)),
        "slots": slots,
        "presence": presence,
        "root_count_probs": count_probs,
        "state_probs": state_probs,
        "normalized": normalized,
    }

'''
    text = regex_once(
        text,
        r"def predict_raw\(ns, model, equation\):.*?\n\ndef eval_examples\(",
        predict + "def eval_examples(",
        "Audit predict_raw",
    )
    path.write_text(text)


def main() -> None:
    patch_model_spec()
    patch_neural_network()
    patch_python_base()
    patch_turbo()
    patch_audit()
    print("MAI5_V4_CARDINALITY_MIGRATION_APPLIED")


if __name__ == "__main__":
    main()
