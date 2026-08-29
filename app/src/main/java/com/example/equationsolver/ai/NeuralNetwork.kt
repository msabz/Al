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
        var activation = input.copyOf()
        for (l in weights.indices) {
            val out = DoubleArray(biases[l].size)
            for (j in out.indices) {
                var sum = biases[l][j]
                for (i in activation.indices) sum += activation[i] * weights[l][i][j]
                out[j] = if (l < weights.lastIndex) max(0.0, sum) else sum
            }
            activation = out
        }
        return activation
    }

    fun train(input: DoubleArray, target: DoubleArray, learningRate: Double = 0.001) {
        require(input.size == inputSize && target.size == outputSize) { "أبعاد التدريب غير صحيحة" }
        val activations = Array(weights.size + 1) { DoubleArray(0) }
        val preActivations = Array(weights.size) { DoubleArray(0) }
        activations[0] = input.copyOf()
        for (l in weights.indices) {
            val z = DoubleArray(biases[l].size)
            for (j in z.indices) {
                var sum = biases[l][j]
                for (i in activations[l].indices) sum += activations[l][i] * weights[l][i][j]
                z[j] = sum
            }
            preActivations[l] = z
            activations[l + 1] = DoubleArray(z.size) { j -> if (l < weights.lastIndex) max(0.0, z[j]) else z[j] }
        }

        val deltas = Array(weights.size) { DoubleArray(0) }
        deltas[weights.lastIndex] = DoubleArray(outputSize) { j -> activations.last()[j] - target[j] }
        for (l in weights.lastIndex - 1 downTo 0) {
            deltas[l] = DoubleArray(weights[l][0].size) { i ->
                var sum = 0.0
                for (j in deltas[l + 1].indices) sum += weights[l + 1][i][j] * deltas[l + 1][j]
                if (preActivations[l][i] > 0.0) sum else 0.0
            }
        }

        step++
        val beta1 = 0.9
        val beta2 = 0.999
        val epsilon = 1e-8
        for (l in weights.indices) {
            for (i in weights[l].indices) for (j in weights[l][i].indices) {
                val g = deltas[l][j] * activations[l][i]
                mW[l][i][j] = beta1 * mW[l][i][j] + (1 - beta1) * g
                vW[l][i][j] = beta2 * vW[l][i][j] + (1 - beta2) * g * g
                val mh = mW[l][i][j] / (1 - beta1.pow(step))
                val vh = vW[l][i][j] / (1 - beta2.pow(step))
                weights[l][i][j] -= learningRate * mh / (sqrt(vh) + epsilon)
            }
            for (j in biases[l].indices) {
                val g = deltas[l][j]
                mB[l][j] = beta1 * mB[l][j] + (1 - beta1) * g
                vB[l][j] = beta2 * vB[l][j] + (1 - beta2) * g * g
                val mh = mB[l][j] / (1 - beta1.pow(step))
                val vh = vB[l][j] / (1 - beta2.pow(step))
                biases[l][j] -= learningRate * mh / (sqrt(vh) + epsilon)
            }
        }
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
