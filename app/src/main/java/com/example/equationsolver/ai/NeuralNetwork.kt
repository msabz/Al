package com.example.equationsolver.ai

import kotlin.math.max
import kotlin.math.pow
import kotlin.math.sqrt
import kotlin.random.Random

class NeuralNetwork(
    private val inputSize: Int,
    private val hiddenSizes: IntArray = intArrayOf(32, 16),
    private val outputSize: Int = 2
) {
    private val weights = mutableListOf<Array<DoubleArray>>()
    private val biases = mutableListOf<DoubleArray>()
    private val mW = mutableListOf<Array<DoubleArray>>()
    private val vW = mutableListOf<Array<DoubleArray>>()
    private val mB = mutableListOf<DoubleArray>()
    private val vB = mutableListOf<DoubleArray>()
    private var step = 0

    init {
        var prev = inputSize
        for (size in hiddenSizes + outputSize) {
            val scale = sqrt(2.0 / prev)
            weights += Array(prev) { DoubleArray(size) { (Random.nextDouble() * 2.0 - 1.0) * scale } }
            biases += DoubleArray(size)
            mW += Array(prev) { DoubleArray(size) }
            vW += Array(prev) { DoubleArray(size) }
            mB += DoubleArray(size)
            vB += DoubleArray(size)
            prev = size
        }
    }

    fun predict(input: DoubleArray): DoubleArray {
        require(input.size == inputSize) { "حجم الإدخال غير صحيح" }
        return forwardWithCache(input).first
    }

    private fun forwardWithCache(input: DoubleArray): Pair<DoubleArray, Pair<Array<DoubleArray>, Array<DoubleArray>>> {
        var activation = input.copyOf()
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
        return activation to (activations to preActivations)
    }

    fun train(input: DoubleArray, target: DoubleArray, learningRate: Double = 0.001): Double =
        trainBatch(arrayOf(input), arrayOf(target), learningRate)

    fun trainBatch(inputs: Array<DoubleArray>, targets: Array<DoubleArray>, learningRate: Double = 0.001): Double {
        require(inputs.isNotEmpty() && inputs.size == targets.size) { "دفعة التدريب غير صحيحة" }
        val gradW = weights.map { layer -> Array(layer.size) { DoubleArray(layer[0].size) } }
        val gradB = biases.map { DoubleArray(it.size) }
        var loss = 0.0

        for (sample in inputs.indices) {
            require(inputs[sample].size == inputSize && targets[sample].size == outputSize) { "أبعاد التدريب غير صحيحة" }
            val (prediction, cache) = forwardWithCache(inputs[sample])
            val activations = cache.first
            val preActivations = cache.second
            for (j in prediction.indices) {
                val error = prediction[j] - targets[sample][j]
                loss += error * error
            }
            val deltas = Array(weights.size) { DoubleArray(0) }
            deltas[weights.lastIndex] = DoubleArray(outputSize) { j -> prediction[j] - targets[sample][j] }
            for (l in weights.lastIndex - 1 downTo 0) {
                deltas[l] = DoubleArray(weights[l][0].size) { i ->
                    var sum = 0.0
                    for (j in deltas[l + 1].indices) sum += weights[l + 1][i][j] * deltas[l + 1][j]
                    if (preActivations[l][i] > 0.0) sum else 0.0
                }
            }
            for (l in weights.indices) {
                for (i in weights[l].indices) for (j in weights[l][i].indices) {
                    gradW[l][i][j] += deltas[l][j] * activations[l][i]
                }
                for (j in biases[l].indices) gradB[l][j] += deltas[l][j]
            }
        }

        val invBatch = 1.0 / inputs.size
        step++
        val beta1 = 0.9
        val beta2 = 0.999
        val epsilon = 1e-8
        for (l in weights.indices) {
            for (i in weights[l].indices) for (j in weights[l][i].indices) {
                val g = gradW[l][i][j] * invBatch
                mW[l][i][j] = beta1 * mW[l][i][j] + (1 - beta1) * g
                vW[l][i][j] = beta2 * vW[l][i][j] + (1 - beta2) * g * g
                val mh = mW[l][i][j] / (1 - beta1.pow(step))
                val vh = vW[l][i][j] / (1 - beta2.pow(step))
                weights[l][i][j] -= learningRate * mh / (sqrt(vh) + epsilon)
            }
            for (j in biases[l].indices) {
                val g = gradB[l][j] * invBatch
                mB[l][j] = beta1 * mB[l][j] + (1 - beta1) * g
                vB[l][j] = beta2 * vB[l][j] + (1 - beta2) * g * g
                val mh = mB[l][j] / (1 - beta1.pow(step))
                val vh = vB[l][j] / (1 - beta2.pow(step))
                biases[l][j] -= learningRate * mh / (sqrt(vh) + epsilon)
            }
        }
        return loss / (inputs.size * outputSize)
    }

    fun meanSquaredError(inputs: List<DoubleArray>, targets: List<DoubleArray>): Double {
        if (inputs.isEmpty()) return 0.0
        require(inputs.size == targets.size) { "بيانات التحقق غير متطابقة" }
        var total = 0.0
        for (i in inputs.indices) {
            val prediction = predict(inputs[i])
            for (j in prediction.indices) {
                val error = prediction[j] - targets[i][j]
                total += error * error
            }
        }
        return total / (inputs.size * outputSize)
    }

    fun getWeights(): List<Array<DoubleArray>> = weights
    fun getBiases(): List<DoubleArray> = biases

    fun setWeights(newWeights: List<Array<DoubleArray>>, newBiases: List<DoubleArray>) {
        require(newWeights.size == weights.size && newBiases.size == biases.size)
        for (l in weights.indices) {
            require(newWeights[l].size == weights[l].size && newBiases[l].size == biases[l].size)
            for (i in weights[l].indices) {
                require(newWeights[l][i].size == weights[l][i].size)
                weights[l][i] = newWeights[l][i].copyOf()
            }
            biases[l] = newBiases[l].copyOf()
        }
    }
}
