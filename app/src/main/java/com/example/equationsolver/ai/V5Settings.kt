package com.example.equationsolver.ai

import android.content.Context

/** Settings that do not change the checkpoint architecture. */
object V5Settings {
    private const val PREFS = "math_ai_v5_settings"
    const val MIN_TRAIN_RANGE = 10
    const val MAX_TRAIN_RANGE = 120

    enum class PowerMode { ECO, BALANCED, FAST }

    data class Snapshot(
        val powerMode: PowerMode,
        val learningRate: Double,
        val consistencyWeight: Double,
        val maxAbsTrainingValue: Int,
        val validateEveryBatches: Int,
        val checkpointMinutes: Int,
        val enableLinear: Boolean,
        val enablePolynomial: Boolean,
        val enableAnalytic: Boolean,
        val enableSystem: Boolean
    ) {
        fun familyEnabled(family: EquationFamily): Boolean = when (family) {
            EquationFamily.LINEAR -> enableLinear
            EquationFamily.POLYNOMIAL -> enablePolynomial
            EquationFamily.ANALYTIC -> enableAnalytic
            EquationFamily.SYSTEM -> enableSystem
        }

        val batchSize: Int get() = when (powerMode) {
            PowerMode.ECO -> 6
            PowerMode.BALANCED -> 10
            PowerMode.FAST -> 16
        }
    }

    fun read(context: Context): Snapshot {
        val p = context.applicationContext.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        return Snapshot(
            powerMode = runCatching { PowerMode.valueOf(p.getString("power", PowerMode.BALANCED.name)!!) }.getOrDefault(PowerMode.BALANCED),
            learningRate = p.getFloat("lr", 0.0006f).toDouble().coerceIn(0.00001, 0.01),
            consistencyWeight = p.getFloat("consistency", 0.05f).toDouble().coerceIn(0.0, 1.0),
            maxAbsTrainingValue = p.getInt("range", 100).coerceIn(MIN_TRAIN_RANGE, MAX_TRAIN_RANGE),
            validateEveryBatches = p.getInt("validate", 40).coerceIn(5, 1000),
            checkpointMinutes = p.getInt("checkpoint", 5).coerceIn(1, 120),
            enableLinear = p.getBoolean("linear", true),
            enablePolynomial = p.getBoolean("poly", true),
            enableAnalytic = p.getBoolean("analytic", true),
            enableSystem = p.getBoolean("system", true)
        )
    }

    fun write(context: Context, value: Snapshot) {
        context.applicationContext.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit()
            .putString("power", value.powerMode.name)
            .putFloat("lr", value.learningRate.coerceIn(0.00001, 0.01).toFloat())
            .putFloat("consistency", value.consistencyWeight.coerceIn(0.0, 1.0).toFloat())
            .putInt("range", value.maxAbsTrainingValue.coerceIn(MIN_TRAIN_RANGE, MAX_TRAIN_RANGE))
            .putInt("validate", value.validateEveryBatches.coerceIn(5, 1000))
            .putInt("checkpoint", value.checkpointMinutes.coerceIn(1, 120))
            .putBoolean("linear", value.enableLinear)
            .putBoolean("poly", value.enablePolynomial)
            .putBoolean("analytic", value.enableAnalytic)
            .putBoolean("system", value.enableSystem)
            .apply()
    }
}
