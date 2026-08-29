package com.example.equationsolver.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ExactSolverTest {
    private val solver = ExactSolver()

    @Test fun solvesSingleEquation() {
        val result = solver.solve("2x+4=10")
        assertTrue(result is SolutionResult.SingleVariable)
        assertEquals(3.0, (result as SolutionResult.SingleVariable).x, 1e-9)
    }

    @Test fun solvesTwoVariableSystem() {
        val result = solver.solve("2x+3y=5;x-y=1")
        assertTrue(result is SolutionResult.TwoVariables)
        val r = result as SolutionResult.TwoVariables
        assertEquals(1.6, r.x, 1e-9)
        assertEquals(0.6, r.y, 1e-9)
    }

    @Test fun detectsNoSolution() {
        assertTrue(solver.solve("x=1;x=2") === SolutionResult.NoSolution)
    }

    @Test fun detectsInfiniteSolutions() {
        assertTrue(solver.solve("x=1;2x=2") === SolutionResult.InfiniteSolutions)
    }

    @Test fun supportsImplicitAndSignedCoefficients() {
        val result = solver.solve("-x+y=2;x+y=4")
        assertTrue(result is SolutionResult.TwoVariables)
        val r = result as SolutionResult.TwoVariables
        assertEquals(1.0, r.x, 1e-9)
        assertEquals(3.0, r.y, 1e-9)
    }
}
