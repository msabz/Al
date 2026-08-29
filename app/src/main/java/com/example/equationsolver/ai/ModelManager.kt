package com.example.equationsolver.ai

import android.content.Context
import com.example.equationsolver.core.EquationFeatures
import com.example.equationsolver.core.ExactSolver
import com.example.equationsolver.core.SolutionResult
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken

object ModelManager {
    private const val PREFS = "equation_solver_model_v2"
    lateinit var nn: NeuralNetwork
        private set

    fun init(context: Context) {
        if (::nn.isInitialized) return
        nn = NeuralNetwork(7)
        load(context)
    }

    fun predict(input: String): DoubleArray = nn.predict(EquationFeatures.fromInput(input).values)

    fun trainOnSolution(input: String, solution: SolutionResult, repeats: Int, learningRate: Double) {
        val target = when (solution) {
            is SolutionResult.SingleVariable -> doubleArrayOf(solution.x / 100.0, 0.0)
            is SolutionResult.TwoVariables -> doubleArrayOf(solution.x / 100.0, solution.y / 100.0)
            else -> return
        }
        val features = EquationFeatures.fromInput(input).values
        repeat(repeats) { nn.train(features, target, learningRate) }
    }

    fun save(context: Context) {
        val p = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        p.edit()
            .putString("weights", Gson().toJson(nn.getWeights()))
            .putString("biases", Gson().toJson(nn.getBiases()))
            .apply()
    }

    private fun load(context: Context) {
        val p = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        val w = p.getString("weights", null) ?: return
        val b = p.getString("biases", null) ?: return
        try {
            val gson = Gson()
            val wt = object : TypeToken<List<Array<DoubleArray>>>() {}.type
            val bt = object : TypeToken<List<DoubleArray>>() {}.type
            nn.setWeights(gson.fromJson(w, wt), gson.fromJson(b, bt))
        } catch (_: Exception) {
            // Invalid/old model is ignored; the fresh model remains usable.
        }
    }
}
