package com.example.equationsolver.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class UniversalEquationSolverTest {
    @Test fun solvesSystemWithCorrectSigns() {
        val result = UniversalEquationSolver.solve("x+y=3;x-y=1")
        assertEquals(2.0, result.x!!, 1e-9)
        assertEquals(1.0, result.y!!, 1e-9)
    }

    @Test fun solvesGeneralTwoByTwoSystem() {
        val result = UniversalEquationSolver.solve("2x+3y=5;x-y=1")
        assertEquals(1.6, result.x!!, 1e-9)
        assertEquals(0.6, result.y!!, 1e-9)
    }

    @Test fun preservesYAsY() {
        val result = UniversalEquationSolver.solve("2y=4")
        assertNull(result.x)
        assertEquals(2.0, result.y!!, 1e-9)
    }

    @Test fun quadraticUsesDeterministicCanonicalRoot() {
        val result = UniversalEquationSolver.solve("x^2-4=0")
        assertTrue(result.summary.contains("2"))
        assertEquals(-2.0, result.x!!, 1e-9)
    }
}
