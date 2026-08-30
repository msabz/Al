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
    @Test fun structuralEncoderTreatsWholeNumberAsOneNode() {
        val e = StructuralMathEncoder.encode("12.5x+4=29")
        assertEquals(EquationFamily.LINEAR, e.family)
        assertEquals(3, e.kinds.take(e.nodeCount).count { it == StructuralMathEncoder.Kind.NUMBER })
        assertTrue(!e.truncated)
    }

    @Test fun factoredPolynomialUsesPolynomialHead() {
        assertEquals(EquationFamily.POLYNOMIAL, StructuralMathEncoder.encode("(x-2)*(x+3)=0").family)
        assertEquals(EquationFamily.POLYNOMIAL, StructuralMathEncoder.encode("x*x-1=0").family)
    }

    @Test fun hardRoutingAndGradientUpdateAreFinite() {
        val n = NeuralNetwork(Random(42))
        val a = StructuralMathEncoder.encode("2x+4=10")
        val b = StructuralMathEncoder.encode("x+y=3;x-y=1")
        val items = arrayOf(
            V5TrainItem(a, V5Target(EquationFamily.LINEAR, SolutionState.FINITE, roots=doubleArrayOf(3.0)), StructuralMathEncoder.encode("10=2x+4")),
            V5TrainItem(b, V5Target(EquationFamily.SYSTEM, SolutionState.FINITE, systemValues=doubleArrayOf(2.0,1.0)), StructuralMathEncoder.encode("x-y=1;x+y=3"))
        )
        val loss = n.trainBatch(items, 0.0006, 0.05)
        assertTrue(loss.isFinite())
        assertEquals(300_984, n.parameterCount())
        assertEquals(1, n.optimizerStep())
        assertTrue(n.lastGradientNorm.isFinite())
    }

    @Test fun checkpointRestoresWeightsAndAdamExactly() {
        val original = NeuralNetwork(Random(5))
        val e = StructuralMathEncoder.encode("(x-2)*(x+3)=0")
        val item = V5TrainItem(e, V5Target(EquationFamily.POLYNOMIAL, SolutionState.FINITE, roots=doubleArrayOf(-3.0,2.0)), StructuralMathEncoder.encode("0=(x-2)*(x+3)"))
        repeat(2) { original.trainBatch(arrayOf(item), 0.0006, 0.05) }
        val bytes = ByteArrayOutputStream().also { DataOutputStream(it).use(original::saveState) }.toByteArray()
        val restored = NeuralNetwork(Random(999))
        DataInputStream(ByteArrayInputStream(bytes)).use(restored::loadState)
        val p1 = original.predict(e)
        val p2 = restored.predict(e)
        assertArrayEquals(p1.slotValues, p2.slotValues, 0.0)
        assertArrayEquals(p1.presenceProbabilities, p2.presenceProbabilities, 0.0)
        assertArrayEquals(p1.stateProbabilities, p2.stateProbabilities, 0.0)
        assertEquals(original.optimizerStep(), restored.optimizerStep())
        original.trainBatch(arrayOf(item), 0.0006, 0.05)
        restored.trainBatch(arrayOf(item), 0.0006, 0.05)
        assertArrayEquals(original.predict(e).slotValues, restored.predict(e).slotValues, 0.0)
    }

    @Test fun solutionStateCanTrainWithoutNumericRoots() {
        val n = NeuralNetwork(Random(9))
        val e = StructuralMathEncoder.encode("0*x=1")
        val item = V5TrainItem(e, V5Target(EquationFamily.LINEAR, SolutionState.NO_SOLUTION))
        val before = n.predict(e).stateProbabilities[SolutionState.NO_SOLUTION.id]
        repeat(8) { n.trainBatch(arrayOf(item), 0.002, 0.0) }
        val after = n.predict(e).stateProbabilities[SolutionState.NO_SOLUTION.id]
        assertTrue(after > before)
    }
}
