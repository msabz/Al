package com.example.equationsolver.data

import com.example.equationsolver.core.MathExpressionEvaluator
import com.example.equationsolver.core.UniversalEquationSolver
import kotlin.math.abs

object GeneratedEquationValidator {
    private const val EPS = 2e-6

    fun isValid(example: GeneratedExample): Boolean {
        if (!example.x.isFinite() || !example.y.isFinite()) return false
        return try {
            if (example.equation.contains(';')) validateSystem(example)
            else {
                val (left, right) = MathExpressionEvaluator.sides(example.equation, example.x, example.y)
                val scale = 1.0 + abs(left) + abs(right)
                abs(left - right) <= EPS * scale
            }
        } catch (_: Exception) { false }
    }

    private fun validateSystem(example: GeneratedExample): Boolean {
        val analysis = UniversalEquationSolver.analyze(example.equation)
        return satisfies(analysis.first, example.x, example.y) &&
            analysis.equationCount == 2 && satisfies(analysis.second, example.x, example.y)
    }

    private fun satisfies(p: UniversalEquationSolver.Polynomial, x: Double, y: Double): Boolean {
        val value = p.x2 * x * x + p.x * x + p.y * y + p.c
        val scale = 1.0 + abs(p.x2 * x * x) + abs(p.x * x) + abs(p.y * y) + abs(p.c)
        return abs(value) <= EPS * scale
    }
}
