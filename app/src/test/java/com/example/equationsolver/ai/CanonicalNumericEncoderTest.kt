package com.example.equationsolver.ai

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

    @Test fun coefficientFractionsDoNotRoutePolynomialToAnalytic() {
        val e = encoding("54*x^5-15556*x^4/3+55154*x^3-153764*x^2/3+1232*x=0")
        assertEquals(EquationFamily.POLYNOMIAL, e.family)
        assertEquals(V5ModelSpec.POLYNOMIAL_FEATURE_SLOTS, e.nodeCount)
        assertEquals(EquationFamily.ANALYTIC, StructuralMathEncoder.classify("1/x=2"))
        assertEquals(EquationFamily.ANALYTIC, StructuralMathEncoder.classify("1/(x+1)=2"))
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
