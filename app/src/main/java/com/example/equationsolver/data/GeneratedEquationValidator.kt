package com.example.equationsolver.data

import com.example.equationsolver.core.UniversalEquationSolver
import kotlin.math.abs

object GeneratedEquationValidator {
    private const val EPS = 1e-7

    fun isValid(example: GeneratedExample): Boolean {
        if (!example.x.isFinite() || !example.y.isFinite()) return false
        return try {
            val a = UniversalEquationSolver.analyze(example.equation)
            satisfies(a.first, example.x, example.y) &&
                (a.equationCount == 1 || satisfies(a.second, example.x, example.y))
        } catch (_: Exception) { false }
    }

    private fun satisfies(p: UniversalEquationSolver.Polynomial, x: Double, y: Double): Boolean {
        val value = p.x2 * x * x + p.x * x + p.y * y + p.c
        val scale = 1.0 + abs(p.x2 * x * x) + abs(p.x * x) + abs(p.y * y) + abs(p.c)
        return abs(value) <= EPS * scale
    }
}
