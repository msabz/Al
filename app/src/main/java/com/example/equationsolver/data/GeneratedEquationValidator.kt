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
                val scale = maxOf(1.0, abs(left), abs(right))
                if (!scale.isFinite()) return false
                val normalizedResidual = left / scale - right / scale
                normalizedResidual.isFinite() && abs(normalizedResidual) <= EPS
            }
        } catch (_: Exception) { false }
    }

    private fun validateSystem(example: GeneratedExample): Boolean {
        val analysis = UniversalEquationSolver.analyze(example.equation)
        return satisfies(analysis.first, example.x, example.y) &&
            analysis.equationCount == 2 && satisfies(analysis.second, example.x, example.y)
    }

    private fun satisfies(p: UniversalEquationSolver.Polynomial, x: Double, y: Double): Boolean {
        val x2Term = p.x2 * x * x
        val xTerm = p.x * x
        val yTerm = p.y * y
        if (!x2Term.isFinite() || !xTerm.isFinite() || !yTerm.isFinite() || !p.c.isFinite()) return false
        val scale = maxOf(1.0, abs(x2Term), abs(xTerm), abs(yTerm), abs(p.c))
        val normalizedResidual = x2Term / scale + xTerm / scale + yTerm / scale + p.c / scale
        return normalizedResidual.isFinite() && abs(normalizedResidual) <= EPS
    }
}
