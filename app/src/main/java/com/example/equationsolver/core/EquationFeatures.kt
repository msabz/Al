package com.example.equationsolver.core

data class EquationSystemFeatures(val values: DoubleArray)

object EquationFeatures {
    /** Seven stable polynomial features used by the current lightweight neural network. */
    fun fromInput(input: String): EquationSystemFeatures {
        val analysis = UniversalEquationSolver.analyze(input)
        val a = analysis.first
        val b = analysis.second
        return EquationSystemFeatures(doubleArrayOf(
            a.x2 / 10.0,
            a.x / 10.0,
            a.y / 10.0,
            a.c / 50.0,
            b.x / 10.0,
            b.y / 10.0,
            b.c / 50.0
        ))
    }
}
