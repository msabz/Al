package com.example.equationsolver.core

import kotlin.math.abs

class ExactSolver(private val parser: EquationParser = EquationParser()) {
    companion object { private const val EPS = 1e-10 }

    fun solve(input: String): SolutionResult = try {
        val equations = input.split(';').map { it.trim() }.filter { it.isNotEmpty() }
        when (equations.size) {
            1 -> solveSingle(parser.parseLinearEquation(equations[0]))
            2 -> solveSystem(parser.parseLinearEquation(equations[0]), parser.parseLinearEquation(equations[1]))
            else -> SolutionResult.Error("أدخل معادلة واحدة أو معادلتين مفصولتين بـ ';'")
        }
    } catch (e: IllegalArgumentException) {
        SolutionResult.Error(e.message ?: "صياغة المعادلة غير صحيحة")
    }

    private fun solveSingle(eq: LinearEquation): SolutionResult {
        if (abs(eq.b) > EPS) return SolutionResult.Error("المعادلة تحتوي على مجهولين؛ أدخل معادلتين لحل النظام.")
        if (abs(eq.a) <= EPS) return if (abs(eq.c) <= EPS) SolutionResult.InfiniteSolutions else SolutionResult.NoSolution
        return SolutionResult.SingleVariable(eq.c / eq.a)
    }

    private fun solveSystem(e1: LinearEquation, e2: LinearEquation): SolutionResult {
        val det = e1.a * e2.b - e2.a * e1.b
        if (abs(det) > EPS) {
            val x = (e1.c * e2.b - e2.c * e1.b) / det
            val y = (e1.a * e2.c - e2.a * e1.c) / det
            return SolutionResult.TwoVariables(x, y)
        }
        val sameA = abs(e1.a * e2.c - e2.a * e1.c) <= EPS
        val sameB = abs(e1.b * e2.c - e2.b * e1.c) <= EPS
        return if (sameA && sameB) SolutionResult.InfiniteSolutions else SolutionResult.NoSolution
    }
}
