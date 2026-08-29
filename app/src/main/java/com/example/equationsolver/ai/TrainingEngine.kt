package com.example.equationsolver.ai

import com.example.equationsolver.data.EquationGenerator
import com.example.equationsolver.core.ExactSolver

object TrainingEngine {
    fun trainRandom(samples: Int, progress: (Int) -> Unit = {}) {
        val solver = ExactSolver()
        for (i in 1..samples) {
            val sample = EquationGenerator.generate()
            val result = solver.solve(sample.equation)
            ModelManager.trainOnSolution(sample.equation, result, repeats = 1, learningRate = 0.001)
            if (i % 500 == 0) progress(i)
        }
    }

    fun trainFile(examples: List<Pair<String, DoubleArray>>, progress: (Int) -> Unit = {}) {
        val solver = ExactSolver()
        for ((index, pair) in examples.withIndex()) {
            val (_, expected) = pair
            val result = solver.solve(pair.first)
            val expectedResult = when (expected.size) {
                1 -> com.example.equationsolver.core.SolutionResult.SingleVariable(expected[0])
                else -> com.example.equationsolver.core.SolutionResult.TwoVariables(expected[0], expected[1])
            }
            // ExactSolver remains the authoritative target. The supplied answer is validated first.
            val target = when (result) {
                is com.example.equationsolver.core.SolutionResult.SingleVariable -> result
                is com.example.equationsolver.core.SolutionResult.TwoVariables -> result
                else -> expectedResult
            }
            ModelManager.trainOnSolution(pair.first, target, repeats = 1, learningRate = 0.001)
            if ((index + 1) % 100 == 0) progress(index + 1)
        }
    }
}
