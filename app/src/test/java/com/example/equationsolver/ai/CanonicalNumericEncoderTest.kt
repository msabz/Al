package com.example.equationsolver.ai

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
