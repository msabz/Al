package com.example.equationsolver.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Test

class MathTeacherTest {
    @Test fun solvesArabicDigitsAndVariableAlias() {
        val answer = MathTeacher.solve("٢س+٤=١٠")
        assertNotNull(answer.x)
        assertEquals(3.0, answer.x!!, 1e-9)
    }

    @Test fun findsCubicPrincipalRoot() {
        val answer = MathTeacher.solve("x^3-6x^2+11x-6=0")
        assertNotNull(answer.x)
        assertEquals(1.0, answer.x!!, 1e-4)
    }

    @Test fun findsSinePrincipalRootNearZero() {
        val answer = MathTeacher.solve("sin(x)=0.47942554")
        assertNotNull(answer.x)
        assertEquals(0.5, answer.x!!, 1e-3)
    }

    @Test fun cosineTieUsesNegativePrincipalRoot() {
        val answer = MathTeacher.solve("cos(x)=0.69670671")
        assertNotNull(answer.x)
        assertEquals(-0.8, answer.x!!, 1e-3)
    }

    @Test fun solvesBasePowerEquationNumerically() {
        val answer = MathTeacher.solve("2^x=8")
        assertNotNull(answer.x)
        assertEquals(3.0, answer.x!!, 1e-4)
    }

    @Test fun detectsEvenMultiplicityRootBetweenGridPoints() {
        val answer = MathTeacher.solve("(x-0.37)^2*(x+1)=0")
        assertNotNull(answer.x)
        assertEquals(0.37, answer.x!!, 1e-4)
    }
}
