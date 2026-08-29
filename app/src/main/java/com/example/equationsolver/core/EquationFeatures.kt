package com.example.equationsolver.core

data class EquationSystemFeatures(val values: DoubleArray)

object EquationFeatures {
    // Stable symbolic features instead of character one-hot encoding.
    // Values are normalized to keep gradients well-conditioned.
    fun fromInput(input: String, solver: ExactSolver = ExactSolver()): EquationSystemFeatures {
        val equations = input.split(';').map { it.trim() }.filter { it.isNotEmpty() }
        require(equations.size in 1..2) { "يجب إدخال معادلة واحدة أو معادلتين" }
        val parser = EquationParser()
        val first = parser.parseLinearEquation(equations[0])
        val second = if (equations.size == 2) parser.parseLinearEquation(equations[1]) else LinearEquation(0.0, 0.0, 0.0)
        return EquationSystemFeatures(doubleArrayOf(
            first.a / 10.0, first.b / 10.0, first.c / 50.0,
            second.a / 10.0, second.b / 10.0, second.c / 50.0,
            equations.size.toDouble()
        ))
    }
}
