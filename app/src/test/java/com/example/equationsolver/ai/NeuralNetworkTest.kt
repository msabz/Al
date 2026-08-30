package com.example.equationsolver.ai

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.DataInputStream
import java.io.DataOutputStream
import kotlin.random.Random

class NeuralNetworkTest {
    @Test
    fun tokenizerReportsUnknownAndTruncatedInputInsteadOfSilentlyHidingIt() {
        val tooLong = List(MathTokenizer.MAX_TOKENS + 5) { "x" }.joinToString("+") + "=1"
        val longEncoding = MathTokenizer.encode(tooLong)
        val unknownEncoding = MathTokenizer.encode("x@2=4")

        assertTrue(longEncoding.truncated)
        assertTrue(longEncoding.tokenCount > MathTokenizer.MAX_TOKENS)
        assertEquals(1, unknownEncoding.unknownCount)
    }

    @Test
    fun tokenizerAcceptsArabicDigitsAndCommonArabicVariables() {
        val encoding = MathTokenizer.encode("٢س + ٤ = ١٠")
        assertEquals(0, encoding.unknownCount)
        assertTrue(!encoding.truncated)
        assertTrue(encoding.tokenCount > 0)
    }

    @Test
    fun productionArchitectureRunsARealFiniteUpdate() {
        val network = NeuralNetwork(random = Random(42))
        val inputs = arrayOf(
            MathTokenizer.tokenize("2x+4=10"),
            MathTokenizer.tokenize("x+y=3;x-y=1")
        )
        val targets = arrayOf(doubleArrayOf(0.03, 0.0), doubleArrayOf(0.02, 0.01))

        val loss = network.trainBatch(inputs, targets, learningRate = 0.0007)

        assertTrue(loss.isFinite())
        assertTrue(network.predict(inputs[0]).all { it.isFinite() })
        assertEquals(247_026, network.parameterCount())
        assertEquals(1, network.optimizerStep())
    }

    @Test
    fun backpropagationChangesWeightsAndReducesLoss() {
        val network = smallNetwork(seed = 17)
        val input = intArrayOf(2, 3, 4, 0)
        val target = doubleArrayOf(0.25, -0.15)
        val beforePrediction = network.predict(input)
        val beforeLoss = network.meanSquaredError(listOf(input), listOf(target))

        repeat(250) { network.train(input, target, learningRate = 0.01) }

        val afterPrediction = network.predict(input)
        val afterLoss = network.meanSquaredError(listOf(input), listOf(target))
        assertTrue("The neural output must change after real gradient updates", beforePrediction.zip(afterPrediction).any { (a, b) -> a != b })
        assertTrue("Expected training loss to fall, before=$beforeLoss after=$afterLoss", afterLoss < beforeLoss * 0.05)
        assertEquals(250, network.optimizerStep())
        assertTrue(network.lastGradientNorm.isFinite())
    }

    @Test
    fun checkpointRestoresWeightsAndAdamStateExactly() {
        val original = smallNetwork(seed = 5)
        val input = intArrayOf(1, 2, 3, 0)
        val target = doubleArrayOf(-0.3, 0.2)
        repeat(12) { original.train(input, target, learningRate = 0.004) }

        val bytes = ByteArrayOutputStream().also { buffer ->
            DataOutputStream(buffer).use(original::saveState)
        }.toByteArray()
        val restored = smallNetwork(seed = 999)
        DataInputStream(ByteArrayInputStream(bytes)).use(restored::loadState)

        assertArrayEquals(original.predict(input), restored.predict(input), 0.0)
        assertEquals(original.optimizerStep(), restored.optimizerStep())

        original.train(input, target, learningRate = 0.004)
        restored.train(input, target, learningRate = 0.004)
        assertArrayEquals("Adam moments must resume, not restart", original.predict(input), restored.predict(input), 0.0)
    }

    @Test
    fun validationIgnoresOutputsThatAreNotPartOfTheEquation() {
        val network = smallNetwork(seed = 9)
        val first = intArrayOf(1, 2, 0, 0)
        val second = intArrayOf(3, 4, 0, 0)
        val firstPrediction = network.predict(first)
        val secondPrediction = network.predict(second)

        val result = network.evaluate(
            inputs = listOf(first, second),
            targets = listOf(
                doubleArrayOf(firstPrediction[0] + 0.01, firstPrediction[1] + 50.0),
                doubleArrayOf(secondPrediction[0] - 50.0, secondPrediction[1] - 0.02)
            ),
            activeOutputs = listOf(booleanArrayOf(true, false), booleanArrayOf(false, true)),
            tolerance = 0.015
        )

        assertEquals(0.00025, result.meanSquaredError, 1e-12)
        assertEquals(0.015, result.meanAbsoluteError, 1e-12)
        assertEquals(0.5, result.withinToleranceRatio, 0.0)
        assertEquals(2, result.valueCount)
    }

    private fun smallNetwork(seed: Int) = NeuralNetwork(
        vocabSize = 6,
        maxTokens = 4,
        embeddingSize = 4,
        hiddenSizes = intArrayOf(12, 8),
        outputSize = 2,
        random = Random(seed)
    )
}
