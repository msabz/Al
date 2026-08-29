package com.example.equationsolver.core

sealed class SolutionResult {
    data class SingleVariable(val x: Double) : SolutionResult()
    data class TwoVariables(val x: Double, val y: Double) : SolutionResult()
    object NoSolution : SolutionResult()
    object InfiniteSolutions : SolutionResult()
    data class Error(val message: String) : SolutionResult()
}
