package com.example.equationsolver.ai

import java.io.DataInputStream
import java.io.DataOutputStream
import java.util.Arrays
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.pow
import kotlin.math.sqrt
import kotlin.random.Random

/**
 * Small sequence model tuned for on-device training.
 * Tokens -> learned positional embeddings -> dense 128 -> 128 -> 64 -> [x,y].
 */
class NeuralNetwork(
    private val vocabSize: Int = MathTokenizer.vocabSize,
    private val maxTokens: Int = MathTokenizer.MAX_TOKENS,
    private val embeddingSize: Int = 24,
    private val hiddenSizes: IntArray = intArrayOf(128, 128, 64),
    private val outputSize: Int = 2,
    private val random: Random = Random.Default
) {
    private val inputSize = maxTokens * embeddingSize

    private val embeddings = Array(vocabSize) { token ->
        if (token == 0) DoubleArray(embeddingSize)
        else DoubleArray(embeddingSize) { (random.nextDouble() * 2.0 - 1.0) * 0.05 }
    }
    private val mE = Array(vocabSize) { DoubleArray(embeddingSize) }
    private val vE = Array(vocabSize) { DoubleArray(embeddingSize) }

    private val weights = mutableListOf<Array<DoubleArray>>()
    private val biases = mutableListOf<DoubleArray>()
    private val mW = mutableListOf<Array<DoubleArray>>()
    private val vW = mutableListOf<Array<DoubleArray>>()
    private val mB = mutableListOf<DoubleArray>()
    private val vB = mutableListOf<DoubleArray>()
    private val gradE: Array<DoubleArray>
    private val gradW: List<Array<DoubleArray>>
    private val gradB: List<DoubleArray>
    private var step = 0

    @Volatile
    var lastGradientNorm: Double = 0.0
        private set

    init {
        require(vocabSize > 1 && maxTokens > 0 && embeddingSize > 0 && outputSize > 0) { "بنية النموذج غير صالحة" }
        require(hiddenSizes.isNotEmpty() && hiddenSizes.all { it > 0 }) { "أحجام الطبقات غير صالحة" }
        var prev = inputSize
        for (size in hiddenSizes + outputSize) {
            val scale = sqrt(2.0 / prev)
            weights += Array(prev) { DoubleArray(size) { (random.nextDouble() * 2.0 - 1.0) * scale } }
            biases += DoubleArray(size)
            mW += Array(prev) { DoubleArray(size) }
            vW += Array(prev) { DoubleArray(size) }
            mB += DoubleArray(size)
            vB += DoubleArray(size)
            prev = size
        }
        gradE = Array(vocabSize) { DoubleArray(embeddingSize) }
        gradW = weights.map { layer -> Array(layer.size) { DoubleArray(layer[0].size) } }
        gradB = biases.map { DoubleArray(it.size) }
    }

    @Synchronized
    fun predict(tokens: IntArray): DoubleArray {
        require(tokens.size == maxTokens) { "طول تسلسل الإدخال غير صحيح" }
        return forwardWithCache(tokens).prediction.copyOf()
    }

    private data class Cache(
        val prediction: DoubleArray,
        val activations: Array<DoubleArray>,
        val preActivations: Array<DoubleArray>,
        val tokens: IntArray
    )

    private fun embeddedInput(tokens: IntArray): DoubleArray {
        val input = DoubleArray(inputSize)
        for (p in 0 until maxTokens) {
            val token = tokens[p].coerceIn(0, vocabSize - 1)
            if (token == 0) continue
            val base = p * embeddingSize
            for (d in 0 until embeddingSize) input[base + d] = embeddings[token][d]
        }
        return input
    }

    private fun forwardWithCache(tokens: IntArray): Cache {
        var activation = embeddedInput(tokens)
        val activations = Array(weights.size + 1) { DoubleArray(0) }
        val preActivations = Array(weights.size) { DoubleArray(0) }
        activations[0] = activation
        for (l in weights.indices) {
            val z = DoubleArray(biases[l].size)
            for (j in z.indices) {
                var sum = biases[l][j]
                for (i in activation.indices) sum += activation[i] * weights[l][i][j]
                z[j] = sum
            }
            preActivations[l] = z
            activation = DoubleArray(z.size) { j -> if (l < weights.lastIndex) max(0.0, z[j]) else z[j] }
            activations[l + 1] = activation
        }
        return Cache(activation, activations, preActivations, tokens)
    }

    fun train(tokens: IntArray, target: DoubleArray, learningRate: Double = 0.001): Double =
        trainBatch(arrayOf(tokens), arrayOf(target), learningRate)

    @Synchronized
    fun trainBatch(inputs: Array<IntArray>, targets: Array<DoubleArray>, learningRate: Double = 0.001): Double {
        require(inputs.isNotEmpty() && inputs.size == targets.size) { "دفعة التدريب غير صحيحة" }
        require(learningRate.isFinite() && learningRate > 0.0) { "معدل التعلم يجب أن يكون عددًا موجبًا ومحدودًا" }
        clearGradients()
        var loss = 0.0

        for (sample in inputs.indices) {
            require(inputs[sample].size == maxTokens && targets[sample].size == outputSize) { "أبعاد التدريب غير صحيحة" }
            require(targets[sample].all { it.isFinite() }) { "قيم الهدف يجب أن تكون محدودة" }
            val cache = forwardWithCache(inputs[sample])
            val prediction = cache.prediction
            require(prediction.all { it.isFinite() }) { "خرج النموذج غير محدود قبل التحديث" }
            for (j in prediction.indices) {
                val error = prediction[j] - targets[sample][j]
                require(error.isFinite()) { "خطأ التدريب خارج النطاق العددي" }
                loss += error * error
                require(loss.isFinite()) { "خسارة التدريب خارج النطاق العددي" }
            }

            val deltas = Array(weights.size) { DoubleArray(0) }
            deltas[weights.lastIndex] = DoubleArray(outputSize) { j -> prediction[j] - targets[sample][j] }
            for (l in weights.lastIndex - 1 downTo 0) {
                deltas[l] = DoubleArray(weights[l][0].size) { i ->
                    var sum = 0.0
                    for (j in deltas[l + 1].indices) sum += weights[l + 1][i][j] * deltas[l + 1][j]
                    if (cache.preActivations[l][i] > 0.0) sum else 0.0
                }
            }

            for (l in weights.indices) {
                for (i in weights[l].indices) for (j in weights[l][i].indices) {
                    gradW[l][i][j] += deltas[l][j] * cache.activations[l][i]
                }
                for (j in biases[l].indices) gradB[l][j] += deltas[l][j]
            }

            // Gradient reaching the learned token embeddings through the first dense layer.
            for (p in 0 until maxTokens) {
                val token = cache.tokens[p].coerceIn(0, vocabSize - 1)
                if (token == 0) continue
                val base = p * embeddingSize
                for (d in 0 until embeddingSize) {
                    val inputIndex = base + d
                    var g = 0.0
                    for (j in deltas[0].indices) g += weights[0][inputIndex][j] * deltas[0][j]
                    gradE[token][d] += g
                }
            }
        }

        val invBatch = 1.0 / inputs.size
        val clipScale = gradientClipScale(invBatch)
        require(step < Int.MAX_VALUE) { "عداد خطوات المحسن وصل إلى الحد الأقصى" }
        step++
        val beta1 = 0.9
        val beta2 = 0.999
        val epsilon = 1e-8
        val correction1 = 1.0 - beta1.pow(step)
        val correction2 = 1.0 - beta2.pow(step)

        for (token in 1 until vocabSize) {
            for (d in 0 until embeddingSize) {
                val g = gradE[token][d] * invBatch * clipScale
                mE[token][d] = beta1 * mE[token][d] + (1.0 - beta1) * g
                vE[token][d] = beta2 * vE[token][d] + (1.0 - beta2) * g * g
                val mh = mE[token][d] / correction1
                val vh = vE[token][d] / correction2
                embeddings[token][d] -= learningRate * mh / (sqrt(vh) + epsilon)
            }
        }

        for (l in weights.indices) {
            for (i in weights[l].indices) for (j in weights[l][i].indices) {
                val g = gradW[l][i][j] * invBatch * clipScale
                mW[l][i][j] = beta1 * mW[l][i][j] + (1.0 - beta1) * g
                vW[l][i][j] = beta2 * vW[l][i][j] + (1.0 - beta2) * g * g
                val mh = mW[l][i][j] / correction1
                val vh = vW[l][i][j] / correction2
                weights[l][i][j] -= learningRate * mh / (sqrt(vh) + epsilon)
            }
            for (j in biases[l].indices) {
                val g = gradB[l][j] * invBatch * clipScale
                mB[l][j] = beta1 * mB[l][j] + (1.0 - beta1) * g
                vB[l][j] = beta2 * vB[l][j] + (1.0 - beta2) * g * g
                val mh = mB[l][j] / correction1
                val vh = vB[l][j] / correction2
                biases[l][j] -= learningRate * mh / (sqrt(vh) + epsilon)
            }
        }
        assertFiniteState()
        return loss / (inputs.size * outputSize)
    }

    private fun clearGradients() {
        gradE.forEach { Arrays.fill(it, 0.0) }
        gradW.forEach { layer -> layer.forEach { Arrays.fill(it, 0.0) } }
        gradB.forEach { Arrays.fill(it, 0.0) }
    }

    /** BLAS-style scaled sum-of-squares avoids overflow while measuring the norm. */
    private fun gradientClipScale(invBatch: Double): Double {
        var scale = 0.0
        var sumSquares = 1.0

        fun accumulate(raw: Double) {
            val gradient = raw * invBatch
            require(gradient.isFinite()) { "تدرج التدريب غير محدود" }
            val magnitude = abs(gradient)
            if (magnitude == 0.0) return
            if (scale < magnitude) {
                val ratio = if (scale == 0.0) 0.0 else scale / magnitude
                sumSquares = 1.0 + sumSquares * ratio * ratio
                scale = magnitude
            } else {
                val ratio = magnitude / scale
                sumSquares += ratio * ratio
            }
        }

        for (token in 1 until vocabSize) for (value in gradE[token]) accumulate(value)
        for (layer in gradW) for (row in layer) for (value in row) accumulate(value)
        for (layer in gradB) for (value in layer) accumulate(value)

        lastGradientNorm = if (scale == 0.0) 0.0 else scale * sqrt(sumSquares)
        require(lastGradientNorm.isFinite()) { "معيار التدرج خارج النطاق العددي" }
        return if (lastGradientNorm > MAX_GRADIENT_NORM) MAX_GRADIENT_NORM / lastGradientNorm else 1.0
    }

    data class Evaluation(
        val meanSquaredError: Double,
        val meanAbsoluteError: Double,
        val withinToleranceRatio: Double,
        val valueCount: Int
    )

    /** Evaluates only meaningful x/y outputs selected by each sample mask. */
    @Synchronized
    fun evaluate(
        inputs: List<IntArray>,
        targets: List<DoubleArray>,
        activeOutputs: List<BooleanArray>,
        tolerance: Double
    ): Evaluation {
        require(inputs.size == targets.size && inputs.size == activeOutputs.size) { "بيانات التحقق غير متطابقة" }
        require(tolerance.isFinite() && tolerance >= 0.0) { "حد التحقق غير صالح" }
        var squared = 0.0
        var absolute = 0.0
        var within = 0
        var count = 0
        for (sample in inputs.indices) {
            require(inputs[sample].size == maxTokens && targets[sample].size == outputSize) { "أبعاد بيانات التحقق غير صحيحة" }
            require(targets[sample].all { it.isFinite() }) { "قيم التحقق يجب أن تكون محدودة" }
            val prediction = forwardWithCache(inputs[sample]).prediction
            require(prediction.all { it.isFinite() }) { "خرج النموذج غير محدود" }
            for (output in prediction.indices) {
                if (!activeOutputs[sample].getOrElse(output) { false }) continue
                val error = kotlin.math.abs(prediction[output] - targets[sample][output])
                squared += error * error
                absolute += error
                if (error <= tolerance) within++
                count++
            }
        }
        if (count == 0) return Evaluation(Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY, 0.0, 0)
        require(squared.isFinite() && absolute.isFinite()) { "مقاييس التحقق خارج النطاق العددي" }
        return Evaluation(squared / count, absolute / count, within.toDouble() / count, count)
    }

    fun parameterCount(): Int = embeddings.sumOf { it.size } +
        weights.sumOf { layer -> layer.sumOf { it.size } } + biases.sumOf { it.size }

    @Synchronized
    fun optimizerStep(): Int = step

    @Synchronized
    fun meanSquaredError(inputs: List<IntArray>, targets: List<DoubleArray>): Double {
        if (inputs.isEmpty()) return 0.0
        require(inputs.size == targets.size) { "بيانات التحقق غير متطابقة" }
        var total = 0.0
        for (i in inputs.indices) {
            require(inputs[i].size == maxTokens && targets[i].size == outputSize) { "أبعاد بيانات التحقق غير صحيحة" }
            require(targets[i].all { it.isFinite() }) { "قيم التحقق يجب أن تكون محدودة" }
            val prediction = forwardWithCache(inputs[i]).prediction
            require(prediction.all { it.isFinite() }) { "خرج النموذج غير محدود" }
            for (j in prediction.indices) {
                val error = prediction[j] - targets[i][j]
                total += error * error
                require(total.isFinite()) { "مقياس الخطأ خارج النطاق العددي" }
            }
        }
        return total / (inputs.size * outputSize)
    }

    @Synchronized
    fun saveState(out: DataOutputStream) {
        assertFiniteState()
        out.writeInt(MAGIC)
        out.writeInt(vocabSize)
        out.writeInt(maxTokens)
        out.writeInt(embeddingSize)
        out.writeInt(hiddenSizes.size)
        hiddenSizes.forEach(out::writeInt)
        out.writeInt(outputSize)
        out.writeInt(step)
        writeMatrix(out, embeddings); writeMatrix(out, mE); writeMatrix(out, vE)
        for (l in weights.indices) {
            writeMatrix(out, weights[l]); writeMatrix(out, mW[l]); writeMatrix(out, vW[l])
            writeVector(out, biases[l]); writeVector(out, mB[l]); writeVector(out, vB[l])
        }
    }

    @Synchronized
    fun loadState(input: DataInputStream) {
        require(input.readInt() == MAGIC) { "ملف النموذج غير متوافق" }
        require(input.readInt() == vocabSize && input.readInt() == maxTokens && input.readInt() == embeddingSize) { "بنية النموذج مختلفة" }
        val hiddenCount = input.readInt()
        require(hiddenCount == hiddenSizes.size) { "عدد الطبقات مختلف" }
        repeat(hiddenCount) { require(input.readInt() == hiddenSizes[it]) { "حجم طبقة مختلف" } }
        require(input.readInt() == outputSize) { "حجم الخرج مختلف" }
        val loadedStep = input.readInt()
        require(loadedStep >= 0) { "عداد المحسن غير صالح" }
        step = loadedStep
        readMatrix(input, embeddings); readMatrix(input, mE); readMatrix(input, vE)
        for (l in weights.indices) {
            readMatrix(input, weights[l]); readMatrix(input, mW[l]); readMatrix(input, vW[l])
            readVector(input, biases[l]); readVector(input, mB[l]); readVector(input, vB[l])
        }
        assertFiniteState()
    }

    private fun assertFiniteState() {
        fun finite(matrix: Array<DoubleArray>) = matrix.all { row -> row.all { it.isFinite() } }
        require(finite(embeddings) && finite(mE) && finite(vE)) { "حالة embeddings غير عددية" }
        require(weights.all(::finite) && mW.all(::finite) && vW.all(::finite)) { "حالة أوزان النموذج غير عددية" }
        require(biases.all { row -> row.all { it.isFinite() } } &&
            mB.all { row -> row.all { it.isFinite() } } &&
            vB.all { row -> row.all { it.isFinite() } }) { "حالة انحيازات النموذج غير عددية" }
    }

    private fun writeMatrix(out: DataOutputStream, matrix: Array<DoubleArray>) {
        for (row in matrix) for (value in row) out.writeDouble(value)
    }
    private fun readMatrix(input: DataInputStream, matrix: Array<DoubleArray>) {
        for (row in matrix) for (i in row.indices) row[i] = readFiniteDouble(input)
    }
    private fun writeVector(out: DataOutputStream, vector: DoubleArray) { for (value in vector) out.writeDouble(value) }
    private fun readVector(input: DataInputStream, vector: DoubleArray) { for (i in vector.indices) vector[i] = readFiniteDouble(input) }
    private fun readFiniteDouble(input: DataInputStream): Double {
        val value = input.readDouble()
        require(value.isFinite()) { "ملف النموذج يحتوي قيمة غير عددية" }
        return value
    }

    companion object {
        private const val MAGIC = 0x45514E34
        private const val MAX_GRADIENT_NORM = 5.0
    }
}
