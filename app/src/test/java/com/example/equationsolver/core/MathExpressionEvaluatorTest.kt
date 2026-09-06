package com.example.equationsolver.core

import org.junit.Assert.assertEquals
import org.junit.Test
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sin

class MathExpressionEvaluatorTest {
    @Test fun evaluatesImplicitMultiplicationAndPowers() {
        assertEquals(0.0, MathExpressionEvaluator.residual("x^3-8=0", x = 2.0), 1e-9)
        assertEquals(0.0, MathExpressionEvaluator.residual("3x+2=11", x = 3.0), 1e-9)
    }

    @Test fun powerPrecedesUnarySignAndRemainsRightAssociative() {
        assertEquals(-4.0, MathExpressionEvaluator.sides("-2^2=0").first, 0.0)
        assertEquals(4.0, MathExpressionEvaluator.sides("(-2)^2=0").first, 0.0)
        assertEquals(512.0, MathExpressionEvaluator.sides("2^3^2=0").first, 0.0)
        assertEquals(0.25, MathExpressionEvaluator.sides("2^-2=0").first, 1e-15)
    }

    @Test fun acceptsSmallNonZeroDenominator() {
        val value = MathExpressionEvaluator.sides("1/0.000000000000001=0").first
        assertEquals(1.0e15, value, 1.0)
    }

    @Test fun evaluatesRationalRadicalAndFunctions() {
        assertEquals(0.0, MathExpressionEvaluator.residual("(2x+4)/(x+5)=1", x = 1.0), 1e-9)
        assertEquals(0.0, MathExpressionEvaluator.residual("sqrt(2x+5)=3", x = 2.0), 1e-9)
        assertEquals(0.0, MathExpressionEvaluator.residual("exp(2x)=${exp(2.0)}", x = 1.0), 1e-8)
        assertEquals(0.0, MathExpressionEvaluator.residual("ln(x)=${ln(4.0)}", x = 4.0), 1e-8)
        assertEquals(0.0, MathExpressionEvaluator.residual("sin(x)=${sin(0.5)}", x = 0.5), 1e-8)
    }
}
