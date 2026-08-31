package com.example.equationsolver.ai

import java.io.DataInputStream
import java.io.DataOutputStream
import java.util.Arrays
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.pow
import kotlin.math.sqrt
import kotlin.random.Random

/**
 * v5 structural neural model:
 * RPN float32 features -> shared trunk -> deterministic hard-routed head.
 *
 * Head output layout:
 *   0..4   root/value slots (normalized)
 *   5..9   LINEAR/ANALYTIC/SYSTEM: presence logits; POLYNOMIAL: root-count logits (classes 1..5)
 *   10..13 solution-state logits
 *
 * Root heads use permutation-invariant masked set loss. The system head keeps
 * slot 0=x and slot 1=y because variable identity must not be permuted.
 */
class NeuralNetwork(private val random: Random = Random.Default) {
    private val embeddings = matrix(V5ModelSpec.TOKEN_VOCAB, V5ModelSpec.EMBEDDING_SIZE, 0.05)
    private val mE = zerosLike(embeddings)
    private val vE = zerosLike(embeddings)

    private val w1 = heMatrix(V5ModelSpec.INPUT_SIZE, V5ModelSpec.SHARED_1)
    private val b1 = FloatArray(V5ModelSpec.SHARED_1)
    private val mw1 = zerosLike(w1); private val vw1 = zerosLike(w1)
    private val mb1 = FloatArray(b1.size); private val vb1 = FloatArray(b1.size)

    private val w2 = heMatrix(V5ModelSpec.SHARED_1, V5ModelSpec.SHARED_2)
    private val b2 = FloatArray(V5ModelSpec.SHARED_2)
    private val mw2 = zerosLike(w2); private val vw2 = zerosLike(w2)
    private val mb2 = FloatArray(b2.size); private val vb2 = FloatArray(b2.size)

    private data class Head(
        val w1: Array<FloatArray>, val b1: FloatArray,
        val w2: Array<FloatArray>, val b2: FloatArray,
        val mw1: Array<FloatArray>, val vw1: Array<FloatArray>,
        val mb1: FloatArray, val vb1: FloatArray,
        val mw2: Array<FloatArray>, val vw2: Array<FloatArray>,
        val mb2: FloatArray, val vb2: FloatArray
    )

    private val heads = Array(V5ModelSpec.HEAD_COUNT) {
        val hw1 = heMatrix(V5ModelSpec.SHARED_2, V5ModelSpec.HEAD_HIDDEN)
        val hb1 = FloatArray(V5ModelSpec.HEAD_HIDDEN)
        val hw2 = heMatrix(V5ModelSpec.HEAD_HIDDEN, V5ModelSpec.HEAD_OUTPUT, linear = true)
        val hb2 = FloatArray(V5ModelSpec.HEAD_OUTPUT)
        Head(
            hw1, hb1, hw2, hb2,
            zerosLike(hw1), zerosLike(hw1), FloatArray(hb1.size), FloatArray(hb1.size),
            zerosLike(hw2), zerosLike(hw2), FloatArray(hb2.size), FloatArray(hb2.size)
        )
    }

    private val gE = zerosLike(embeddings)
    private val gw1 = zerosLike(w1); private val gb1 = FloatArray(b1.size)
    private val gw2 = zerosLike(w2); private val gb2 = FloatArray(b2.size)
    private data class HeadGrad(
        val w1: Array<FloatArray>, val b1: FloatArray,
        val w2: Array<FloatArray>, val b2: FloatArray
    )
    private val gh = Array(V5ModelSpec.HEAD_COUNT) { h ->
        HeadGrad(zerosLike(heads[h].w1), FloatArray(heads[h].b1.size), zerosLike(heads[h].w2), FloatArray(heads[h].b2.size))
    }

    private var step = 0

    @Volatile
    var lastGradientNorm: Double = 0.0
        private set

    private data class Cache(
        val encoding: StructuralMathEncoder.Encoding,
        val input: FloatArray,
        val z1: FloatArray,
        val a1: FloatArray,
        val z2: FloatArray,
        val a2: FloatArray,
        val zh: FloatArray,
        val ah: FloatArray,
        val out: FloatArray
    )

    data class Evaluation(
        val rootMeanSquaredError: Double,
        val rootMeanAbsoluteError: Double,
        val withinToleranceRatio: Double,
        val stateAccuracy: Double,
        val valueCount: Int
    )

    @Synchronized
    fun predict(encoding: StructuralMathEncoder.Encoding): V5Prediction {
        require(!encoding.truncated) { "المعادلة أطول من حد v5 البنيوي" }
        return predictionFromCache(forward(encoding))
    }

    @Synchronized
    fun trainBatch(
        items: Array<V5TrainItem>,
        learningRate: Double = 0.0007,
        consistencyWeight: Double = 0.05
    ): Double {
        require(items.isNotEmpty()) { "دفعة تدريب فارغة" }
        clearGradients()
        var loss = 0.0

        for (item in items) {
            require(!item.input.truncated) { "مثال تدريب مقطوع" }
            require(item.input.family == item.target.family) { "المسار البنيوي لا يطابق هدف التدريب" }
            val cache = forward(item.input)
            val supervised = supervisedGradient(cache.out, item.target, item.input)
            loss += supervised.first
            backward(cache, supervised.second)

            val equivalent = item.equivalent
            if (equivalent != null && consistencyWeight > 0.0 && !equivalent.truncated && equivalent.family == item.input.family) {
                val other = forward(equivalent)
                val gradA = FloatArray(V5ModelSpec.HEAD_OUTPUT)
                val gradB = FloatArray(V5ModelSpec.HEAD_OUTPUT)
                var consistency = 0.0
                for (i in gradA.indices) {
                    val d = (cache.out[i] - other.out[i]).toDouble()
                    consistency += d * d
                    val g = (2.0 * d / V5ModelSpec.HEAD_OUTPUT * consistencyWeight).toFloat()
                    gradA[i] = g
                    gradB[i] = -g
                }
                loss += consistency / V5ModelSpec.HEAD_OUTPUT * consistencyWeight
                backward(cache, gradA)
                backward(other, gradB)
            }
        }

        val invBatch = 1.0f / items.size.toFloat()
        val clip = gradientClipScale(invBatch)
        step++
        applyAdam(learningRate.toFloat(), invBatch * clip)
        return loss / items.size
    }

    @Synchronized
    fun evaluate(items: List<V5TrainItem>, tolerance: Double = 1.0): Evaluation {
        if (items.isEmpty()) return Evaluation(Double.NaN, Double.NaN, 0.0, Double.NaN, 0)
        var squared = 0.0
        var absolute = 0.0
        var within = 0
        var count = 0
        var stateCorrect = 0
        for (item in items) {
            val prediction = predictionFromCache(forward(item.input))
            if (prediction.state == item.target.state) stateCorrect++
            if (item.target.state != SolutionState.FINITE) continue
            if (item.target.family == EquationFamily.SYSTEM) {
                val expected = item.target.systemValues
                for (i in expected.indices.take(V5ModelSpec.ROOT_SLOTS)) {
                    val error = kotlin.math.abs(prediction.slotValues[i] - expected[i])
                    squared += error * error
                    absolute += error
                    if (error <= tolerance) within++
                    count++
                }
            } else {
                val expected = item.target.canonicalRoots()
                val predicted = prediction.roots
                val matching = minimumMatchingErrors(predicted, expected)
                for (error in matching) {
                    squared += error * error
                    absolute += error
                    if (error <= tolerance) within++
                    count++
                }
            }
        }
        return Evaluation(
            rootMeanSquaredError = if (count == 0) Double.NaN else sqrt(squared / count),
            rootMeanAbsoluteError = if (count == 0) Double.NaN else absolute / count,
            withinToleranceRatio = if (count == 0) 0.0 else within.toDouble() / count,
            stateAccuracy = stateCorrect.toDouble() / items.size,
            valueCount = count
        )
    }

    private fun forward(encoding: StructuralMathEncoder.Encoding): Cache {
        val input = buildInput(encoding)
        val z1 = dense(input, w1, b1)
        val a1 = relu(z1)
        val z2 = dense(a1, w2, b2)
        val a2 = relu(z2)
        val head = heads[encoding.family.id]
        val zh = dense(a2, head.w1, head.b1)
        val ah = relu(zh)
        val out = dense(ah, head.w2, head.b2)
        return Cache(encoding, input, z1, a1, z2, a2, zh, ah, out)
    }

    private fun buildInput(e: StructuralMathEncoder.Encoding): FloatArray {
        val input = FloatArray(V5ModelSpec.INPUT_SIZE)
        for (p in 0 until V5ModelSpec.MAX_NODES) {
            val kind = e.kinds[p].coerceIn(0, V5ModelSpec.TOKEN_VOCAB - 1)
            val base = p * V5ModelSpec.NODE_FEATURES
            for (d in 0 until V5ModelSpec.EMBEDDING_SIZE) input[base + d] = embeddings[kind][d]
            input[base + V5ModelSpec.EMBEDDING_SIZE] = e.numeric[p]
            input[base + V5ModelSpec.EMBEDDING_SIZE + 1] = if (kind == StructuralMathEncoder.Kind.PAD || V5ModelSpec.MAX_NODES <= 1) 0f else p.toFloat() / (V5ModelSpec.MAX_NODES - 1).toFloat()
            input[base + V5ModelSpec.EMBEDDING_SIZE + 2] = e.depth[p]
        }
        return input
    }

    private fun supervisedGradient(
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

    /** Returns, for every prediction slot, the index into roots or roots.size+ for padding. */
    private fun bestPermutation(out: FloatArray, roots: DoubleArray, family: EquationFamily): IntArray {
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

    private fun backward(c: Cache, dOut: FloatArray) {
        val family = c.encoding.family.id
        val head = heads[family]
        val hg = gh[family]

        for (i in c.ah.indices) for (j in dOut.indices) hg.w2[i][j] += c.ah[i] * dOut[j]
        for (j in dOut.indices) hg.b2[j] += dOut[j]

        val dAh = FloatArray(c.ah.size)
        for (i in dAh.indices) {
            var sum = 0f
            for (j in dOut.indices) sum += head.w2[i][j] * dOut[j]
            dAh[i] = sum
        }
        val dZh = FloatArray(dAh.size) { i -> if (c.zh[i] > 0f) dAh[i] else 0f }
        for (i in c.a2.indices) for (j in dZh.indices) hg.w1[i][j] += c.a2[i] * dZh[j]
        for (j in dZh.indices) hg.b1[j] += dZh[j]

        val dA2 = FloatArray(c.a2.size)
        for (i in dA2.indices) {
            var sum = 0f
            for (j in dZh.indices) sum += head.w1[i][j] * dZh[j]
            dA2[i] = sum
        }
        val dZ2 = FloatArray(dA2.size) { i -> if (c.z2[i] > 0f) dA2[i] else 0f }
        for (i in c.a1.indices) for (j in dZ2.indices) gw2[i][j] += c.a1[i] * dZ2[j]
        for (j in dZ2.indices) gb2[j] += dZ2[j]

        val dA1 = FloatArray(c.a1.size)
        for (i in dA1.indices) {
            var sum = 0f
            for (j in dZ2.indices) sum += w2[i][j] * dZ2[j]
            dA1[i] = sum
        }
        val dZ1 = FloatArray(dA1.size) { i -> if (c.z1[i] > 0f) dA1[i] else 0f }
        for (i in c.input.indices) for (j in dZ1.indices) gw1[i][j] += c.input[i] * dZ1[j]
        for (j in dZ1.indices) gb1[j] += dZ1[j]

        for (p in 0 until V5ModelSpec.MAX_NODES) {
            val kind = c.encoding.kinds[p].coerceIn(0, V5ModelSpec.TOKEN_VOCAB - 1)
            if (kind == StructuralMathEncoder.Kind.PAD) continue
            val base = p * V5ModelSpec.NODE_FEATURES
            for (d in 0 until V5ModelSpec.EMBEDDING_SIZE) {
                val inputIndex = base + d
                var sum = 0f
                for (j in dZ1.indices) sum += w1[inputIndex][j] * dZ1[j]
                gE[kind][d] += sum
            }
        }
    }

    private fun clearGradients() {
        clear(gE); clear(gw1); Arrays.fill(gb1, 0f); clear(gw2); Arrays.fill(gb2, 0f)
        for (h in gh) {
            clear(h.w1); Arrays.fill(h.b1, 0f); clear(h.w2); Arrays.fill(h.b2, 0f)
        }
    }

    private fun gradientClipScale(invBatch: Float): Float {
        var squared = 0.0
        fun addMatrix(m: Array<FloatArray>) { for (row in m) for (v in row) { val g = v * invBatch; squared += g * g } }
        fun addVector(v: FloatArray) { for (x in v) { val g = x * invBatch; squared += g * g } }
        addMatrix(gE); addMatrix(gw1); addVector(gb1); addMatrix(gw2); addVector(gb2)
        for (h in gh) { addMatrix(h.w1); addVector(h.b1); addMatrix(h.w2); addVector(h.b2) }
        lastGradientNorm = sqrt(squared)
        return if (lastGradientNorm > V5ModelSpec.MAX_GRADIENT_NORM) (V5ModelSpec.MAX_GRADIENT_NORM / lastGradientNorm).toFloat() else 1f
    }

    private fun applyAdam(lr: Float, gradientScale: Float) {
        val beta1 = 0.9f
        val beta2 = 0.999f
        val eps = 1e-8f
        val c1 = (1.0 - 0.9.pow(step.toDouble())).toFloat()
        val c2 = (1.0 - 0.999.pow(step.toDouble())).toFloat()
        adamMatrix(embeddings, mE, vE, gE, lr, gradientScale, beta1, beta2, eps, c1, c2)
        // Keep PAD embedding exactly zero so padding never becomes a learned signal.
        Arrays.fill(embeddings[StructuralMathEncoder.Kind.PAD], 0f)
        Arrays.fill(mE[StructuralMathEncoder.Kind.PAD], 0f)
        Arrays.fill(vE[StructuralMathEncoder.Kind.PAD], 0f)

        adamMatrix(w1, mw1, vw1, gw1, lr, gradientScale, beta1, beta2, eps, c1, c2)
        adamVector(b1, mb1, vb1, gb1, lr, gradientScale, beta1, beta2, eps, c1, c2)
        adamMatrix(w2, mw2, vw2, gw2, lr, gradientScale, beta1, beta2, eps, c1, c2)
        adamVector(b2, mb2, vb2, gb2, lr, gradientScale, beta1, beta2, eps, c1, c2)
        for (i in heads.indices) {
            val h = heads[i]; val g = gh[i]
            adamMatrix(h.w1, h.mw1, h.vw1, g.w1, lr, gradientScale, beta1, beta2, eps, c1, c2)
            adamVector(h.b1, h.mb1, h.vb1, g.b1, lr, gradientScale, beta1, beta2, eps, c1, c2)
            adamMatrix(h.w2, h.mw2, h.vw2, g.w2, lr, gradientScale, beta1, beta2, eps, c1, c2)
            adamVector(h.b2, h.mb2, h.vb2, g.b2, lr, gradientScale, beta1, beta2, eps, c1, c2)
        }
    }

    private fun adamMatrix(
        p: Array<FloatArray>, m: Array<FloatArray>, v: Array<FloatArray>, g: Array<FloatArray>,
        lr: Float, scale: Float, beta1: Float, beta2: Float, eps: Float, c1: Float, c2: Float
    ) {
        val decay = (1f - lr * V5ModelSpec.WEIGHT_DECAY.toFloat()).coerceAtLeast(0f)
        for (i in p.indices) for (j in p[i].indices) {
            val grad = g[i][j] * scale
            m[i][j] = beta1 * m[i][j] + (1f - beta1) * grad
            v[i][j] = beta2 * v[i][j] + (1f - beta2) * grad * grad
            val mh = m[i][j] / c1
            val vh = v[i][j] / c2
            p[i][j] *= decay
            p[i][j] -= lr * mh / (sqrt(vh.toDouble()).toFloat() + eps)
        }
    }

    private fun adamVector(
        p: FloatArray, m: FloatArray, v: FloatArray, g: FloatArray,
        lr: Float, scale: Float, beta1: Float, beta2: Float, eps: Float, c1: Float, c2: Float
    ) {
        for (i in p.indices) {
            val grad = g[i] * scale
            m[i] = beta1 * m[i] + (1f - beta1) * grad
            v[i] = beta2 * v[i] + (1f - beta2) * grad * grad
            val mh = m[i] / c1
            val vh = v[i] / c2
            p[i] -= lr * mh / (sqrt(vh.toDouble()).toFloat() + eps)
        }
    }

    private fun polynomialResidual(encoding: StructuralMathEncoder.Encoding, normalizedRoot: Double): Double {
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

    fun parameterCount(): Int {
        var count = embeddings.sumOf { it.size } + w1.sumOf { it.size } + b1.size + w2.sumOf { it.size } + b2.size
        for (h in heads) count += h.w1.sumOf { it.size } + h.b1.size + h.w2.sumOf { it.size } + h.b2.size
        return count
    }

    @Synchronized fun optimizerStep(): Int = step

    /** Same MAI5 file is produced by Colab and by the Android app. */
    @Synchronized
    fun saveState(out: DataOutputStream) {
        out.writeInt(V5ModelSpec.FILE_MAGIC)
        out.writeInt(V5ModelSpec.FILE_VERSION)
        out.writeInt(V5ModelSpec.MAX_NODES)
        out.writeInt(V5ModelSpec.TOKEN_VOCAB)
        out.writeInt(V5ModelSpec.EMBEDDING_SIZE)
        out.writeInt(V5ModelSpec.INPUT_SIZE)
        out.writeInt(V5ModelSpec.SHARED_1)
        out.writeInt(V5ModelSpec.SHARED_2)
        out.writeInt(V5ModelSpec.HEAD_HIDDEN)
        out.writeInt(V5ModelSpec.HEAD_COUNT)
        out.writeInt(V5ModelSpec.HEAD_OUTPUT)
        out.writeInt(step)
        out.writeFloat(V5ModelSpec.ROOT_SCALE.toFloat())

        writeTriplet(out, embeddings, mE, vE)
        writeTriplet(out, w1, mw1, vw1); writeTriplet(out, b1, mb1, vb1)
        writeTriplet(out, w2, mw2, vw2); writeTriplet(out, b2, mb2, vb2)
        for (h in heads) {
            writeTriplet(out, h.w1, h.mw1, h.vw1); writeTriplet(out, h.b1, h.mb1, h.vb1)
            writeTriplet(out, h.w2, h.mw2, h.vw2); writeTriplet(out, h.b2, h.mb2, h.vb2)
        }
    }

    @Synchronized
    fun loadState(input: DataInputStream) {
        require(input.readInt() == V5ModelSpec.FILE_MAGIC) { "هذا ليس ملف MAI5" }
        require(input.readInt() == V5ModelSpec.FILE_VERSION) { "إصدار ملف الأوزان غير مدعوم" }
        require(input.readInt() == V5ModelSpec.MAX_NODES) { "MAX_NODES مختلف" }
        require(input.readInt() == V5ModelSpec.TOKEN_VOCAB) { "قاموس RPN مختلف" }
        require(input.readInt() == V5ModelSpec.EMBEDDING_SIZE) { "Embedding مختلف" }
        require(input.readInt() == V5ModelSpec.INPUT_SIZE) { "حجم الإدخال مختلف" }
        require(input.readInt() == V5ModelSpec.SHARED_1 && input.readInt() == V5ModelSpec.SHARED_2) { "Shared trunk مختلف" }
        require(input.readInt() == V5ModelSpec.HEAD_HIDDEN) { "حجم الرأس مختلف" }
        require(input.readInt() == V5ModelSpec.HEAD_COUNT && input.readInt() == V5ModelSpec.HEAD_OUTPUT) { "عقد الرؤوس مختلف" }
        step = input.readInt().coerceAtLeast(0)
        require(kotlin.math.abs(input.readFloat() - V5ModelSpec.ROOT_SCALE.toFloat()) < 1e-4f) { "ROOT_SCALE مختلف" }

        readTriplet(input, embeddings, mE, vE)
        readTriplet(input, w1, mw1, vw1); readTriplet(input, b1, mb1, vb1)
        readTriplet(input, w2, mw2, vw2); readTriplet(input, b2, mb2, vb2)
        for (h in heads) {
            readTriplet(input, h.w1, h.mw1, h.vw1); readTriplet(input, h.b1, h.mb1, h.vb1)
            readTriplet(input, h.w2, h.mw2, h.vw2); readTriplet(input, h.b2, h.mb2, h.vb2)
        }
        Arrays.fill(embeddings[StructuralMathEncoder.Kind.PAD], 0f)
    }

    private fun writeTriplet(out: DataOutputStream, p: Array<FloatArray>, m: Array<FloatArray>, v: Array<FloatArray>) {
        writeMatrix(out, p); writeMatrix(out, m); writeMatrix(out, v)
    }
    private fun writeTriplet(out: DataOutputStream, p: FloatArray, m: FloatArray, v: FloatArray) {
        writeVector(out, p); writeVector(out, m); writeVector(out, v)
    }
    private fun readTriplet(input: DataInputStream, p: Array<FloatArray>, m: Array<FloatArray>, v: Array<FloatArray>) {
        readMatrix(input, p); readMatrix(input, m); readMatrix(input, v)
    }
    private fun readTriplet(input: DataInputStream, p: FloatArray, m: FloatArray, v: FloatArray) {
        readVector(input, p); readVector(input, m); readVector(input, v)
    }
    private fun writeMatrix(out: DataOutputStream, matrix: Array<FloatArray>) { for (row in matrix) for (value in row) out.writeFloat(value) }
    private fun readMatrix(input: DataInputStream, matrix: Array<FloatArray>) { for (row in matrix) for (i in row.indices) row[i] = input.readFloat() }
    private fun writeVector(out: DataOutputStream, vector: FloatArray) { for (value in vector) out.writeFloat(value) }
    private fun readVector(input: DataInputStream, vector: FloatArray) { for (i in vector.indices) vector[i] = input.readFloat() }

    private fun dense(input: FloatArray, weights: Array<FloatArray>, bias: FloatArray): FloatArray {
        val out = bias.copyOf()
        for (i in input.indices) {
            val x = input[i]
            if (x == 0f) continue
            val row = weights[i]
            for (j in out.indices) out[j] += x * row[j]
        }
        return out
    }
    private fun relu(v: FloatArray) = FloatArray(v.size) { max(0f, v[it]) }
    private fun sigmoid(v: Float): Double = if (v >= 0f) 1.0 / (1.0 + exp(-v.toDouble())) else {
        val e = exp(v.toDouble()); e / (1.0 + e)
    }
    private fun softmax(logits: FloatArray): DoubleArray {
        val max = logits.maxOrNull()?.toDouble() ?: 0.0
        val exps = DoubleArray(logits.size) { exp(logits[it] - max) }
        val sum = exps.sum().coerceAtLeast(1e-12)
        return DoubleArray(logits.size) { exps[it] / sum }
    }
    private fun binaryCrossEntropy(p: Double, y: Double): Double = -(y * ln(p.coerceIn(1e-9, 1.0 - 1e-9)) + (1.0 - y) * ln((1.0 - p).coerceIn(1e-9, 1.0 - 1e-9)))

    private fun minimumMatchingErrors(predicted: DoubleArray, expected: DoubleArray): DoubleArray {
        if (expected.isEmpty()) return doubleArrayOf()
        if (predicted.isEmpty()) return DoubleArray(expected.size) { V5ModelSpec.ROOT_SCALE }
        val used = BooleanArray(predicted.size)
        val errors = DoubleArray(expected.size)
        for (i in expected.indices) {
            var best = -1
            var bestError = Double.POSITIVE_INFINITY
            for (j in predicted.indices) if (!used[j]) {
                val e = kotlin.math.abs(predicted[j] - expected[i])
                if (e < bestError) { bestError = e; best = j }
            }
            if (best >= 0) { used[best] = true; errors[i] = bestError } else errors[i] = V5ModelSpec.ROOT_SCALE
        }
        return errors
    }

    private fun matrix(rows: Int, cols: Int, scale: Double): Array<FloatArray> = Array(rows) { row ->
        if (row == StructuralMathEncoder.Kind.PAD) FloatArray(cols)
        else FloatArray(cols) { ((random.nextDouble() * 2.0 - 1.0) * scale).toFloat() }
    }
    private fun heMatrix(rows: Int, cols: Int, linear: Boolean = false): Array<FloatArray> {
        val scale = sqrt((if (linear) 1.0 else 2.0) / rows)
        return Array(rows) { FloatArray(cols) { ((random.nextDouble() * 2.0 - 1.0) * scale).toFloat() } }
    }
    private fun zerosLike(matrix: Array<FloatArray>) = Array(matrix.size) { FloatArray(matrix[it].size) }
    private fun clear(matrix: Array<FloatArray>) { matrix.forEach { Arrays.fill(it, 0f) } }

    companion object {
        private val PERMUTATIONS: Array<IntArray> = run {
            val result = ArrayList<IntArray>(120)
            fun rec(prefix: IntArray, used: BooleanArray, depth: Int) {
                if (depth == V5ModelSpec.ROOT_SLOTS) { result += prefix.copyOf(); return }
                for (i in 0 until V5ModelSpec.ROOT_SLOTS) if (!used[i]) {
                    used[i] = true; prefix[depth] = i; rec(prefix, used, depth + 1); used[i] = false
                }
            }
            rec(IntArray(V5ModelSpec.ROOT_SLOTS), BooleanArray(V5ModelSpec.ROOT_SLOTS), 0)
            result.toTypedArray()
        }
    }
}
