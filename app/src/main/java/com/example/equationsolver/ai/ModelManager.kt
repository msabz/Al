package com.example.equationsolver.ai

import android.content.Context
import com.example.equationsolver.core.EquationFeatures
import com.example.equationsolver.core.SolutionResult
import com.example.equationsolver.core.UniversalEquationSolver
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken

object ModelManager {
    private const val PREFS = "equation_solver_model_v3"
    private const val KEY_WEIGHTS = "weights"
    private const val KEY_BIASES = "biases"
    private const val KEY_SAMPLES = "training_samples"
    private const val KEY_BATCHES = "training_batches"
    private const val KEY_BEST_VAL = "best_validation_mse"
    private const val KEY_TRAINING_ENABLED = "training_enabled"
    private const val KEY_LAST_LOSS = "last_loss"

    lateinit var nn: NeuralNetwork
        private set

    fun init(context: Context) {
        if (::nn.isInitialized) return
        nn = NeuralNetwork(7)
        load(context)
    }

    fun predict(input: String): DoubleArray = nn.predict(EquationFeatures.fromInput(input).values)
    fun classify(input: String): String = UniversalEquationSolver.equationType(input)

    fun trainOnSolution(input: String, repeats: Int, learningRate: Double) {
        val result = UniversalEquationSolver.solve(input); val x = result.x ?: return
        trainWithTarget(input, doubleArrayOf(x / 100.0, (result.y ?: 0.0) / 100.0), repeats, learningRate)
    }

    fun trainOnSolution(input: String, solution: SolutionResult, repeats: Int, learningRate: Double) {
        val target = when (solution) {
            is SolutionResult.SingleVariable -> doubleArrayOf(solution.x / 100.0, 0.0)
            is SolutionResult.TwoVariables -> doubleArrayOf(solution.x / 100.0, solution.y / 100.0)
            else -> return
        }
        trainWithTarget(input, target, repeats, learningRate)
    }

    private fun trainWithTarget(input: String, target: DoubleArray, repeats: Int, learningRate: Double) {
        val features = EquationFeatures.fromInput(input).values
        repeat(repeats.coerceAtLeast(1)) { nn.train(features, target, learningRate) }
    }

    @Synchronized
    fun save(context: Context, samples: Long? = null, batches: Long? = null, bestValidationMse: Double? = null, lastLoss: Double? = null) {
        val p = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        val tempWeights = Gson().toJson(nn.getWeights())
        val tempBiases = Gson().toJson(nn.getBiases())
        val e = p.edit()
            .putString(KEY_WEIGHTS, tempWeights)
            .putString(KEY_BIASES, tempBiases)
        if (samples != null) e.putLong(KEY_SAMPLES, samples)
        if (batches != null) e.putLong(KEY_BATCHES, batches)
        if (bestValidationMse != null && bestValidationMse.isFinite()) e.putFloat(KEY_BEST_VAL, bestValidationMse.toFloat())
        if (lastLoss != null && lastLoss.isFinite()) e.putFloat(KEY_LAST_LOSS, lastLoss.toFloat())
        e.commit()
    }

    fun setTrainingEnabled(context: Context, enabled: Boolean) {
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit().putBoolean(KEY_TRAINING_ENABLED, enabled).commit()
    }

    fun isTrainingEnabled(context: Context): Boolean =
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getBoolean(KEY_TRAINING_ENABLED, false)

    fun trainingSamples(context: Context): Long = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getLong(KEY_SAMPLES, 0L)
    fun trainingBatches(context: Context): Long = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getLong(KEY_BATCHES, 0L)
    fun bestValidationMse(context: Context): Double = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getFloat(KEY_BEST_VAL, Float.POSITIVE_INFINITY).toDouble()
    fun lastLoss(context: Context): Double = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getFloat(KEY_LAST_LOSS, Float.NaN).toDouble()

    private fun load(context: Context) {
        val p = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        val w = p.getString(KEY_WEIGHTS, null) ?: return; val b = p.getString(KEY_BIASES, null) ?: return
        try {
            val gson = Gson(); val wt = object : TypeToken<List<Array<DoubleArray>>>() {}.type; val bt = object : TypeToken<List<DoubleArray>>() {}.type
            nn.setWeights(gson.fromJson(w, wt), gson.fromJson(b, bt))
        } catch (_: Exception) { }
    }
}
