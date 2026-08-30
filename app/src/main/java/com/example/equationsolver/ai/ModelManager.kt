package com.example.equationsolver.ai

import android.content.Context
import com.example.equationsolver.core.MathTeacher
import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.FileInputStream
import java.io.FileOutputStream

object ModelManager {
    private const val PREFS = "equation_solver_model_v4"
    private const val MODEL_FILE = "equation_model_v4.bin"
    private const val KEY_SAMPLES = "training_samples"
    private const val KEY_BATCHES = "training_batches"
    private const val KEY_BEST_VAL = "best_validation_mse"
    private const val KEY_TRAINING_ENABLED = "training_enabled"
    private const val KEY_LAST_LOSS = "last_loss"

    lateinit var nn: NeuralNetwork
        private set

    @Synchronized
    fun init(context: Context) {
        if (::nn.isInitialized) return
        nn = NeuralNetwork()
        load(context.applicationContext)
    }

    fun predict(input: String): DoubleArray = nn.predict(MathTokenizer.tokenize(input))

    fun trainWithTarget(input: String, x: Double, y: Double, repeats: Int, learningRate: Double) {
        val tokens = MathTokenizer.tokenize(input)
        val target = doubleArrayOf(x / 100.0, y / 100.0)
        repeat(repeats.coerceAtLeast(1)) { nn.train(tokens, target, learningRate) }
    }

    fun trainOnTeacherSolution(input: String, repeats: Int, learningRate: Double): Boolean {
        val answer = MathTeacher.solve(input)
        val x = answer.x ?: 0.0
        val y = answer.y ?: 0.0
        if (answer.x == null && answer.y == null) return false
        trainWithTarget(input, x, y, repeats, learningRate)
        return true
    }

    @Synchronized
    fun save(
        context: Context,
        samples: Long? = null,
        batches: Long? = null,
        bestValidationMse: Double? = null,
        lastLoss: Double? = null
    ) {
        if (!::nn.isInitialized) return
        val app = context.applicationContext
        val target = app.getFileStreamPath(MODEL_FILE)
        val temp = app.getFileStreamPath("$MODEL_FILE.tmp")
        try {
            DataOutputStream(BufferedOutputStream(FileOutputStream(temp))).use { nn.saveState(it) }
            if (target.exists() && !target.delete()) throw IllegalStateException("تعذر استبدال النموذج القديم")
            if (!temp.renameTo(target)) {
                FileInputStream(temp).use { input -> FileOutputStream(target).use { output -> input.copyTo(output) } }
                temp.delete()
            }
        } catch (e: Exception) {
            temp.delete()
            throw e
        }

        val p = app.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        val edit = p.edit()
        if (samples != null) edit.putLong(KEY_SAMPLES, samples)
        if (batches != null) edit.putLong(KEY_BATCHES, batches)
        if (bestValidationMse != null && bestValidationMse.isFinite()) edit.putFloat(KEY_BEST_VAL, bestValidationMse.toFloat())
        if (lastLoss != null && lastLoss.isFinite()) edit.putFloat(KEY_LAST_LOSS, lastLoss.toFloat())
        edit.commit()
    }

    fun setTrainingEnabled(context: Context, enabled: Boolean) {
        context.applicationContext.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            .edit().putBoolean(KEY_TRAINING_ENABLED, enabled).commit()
    }

    fun isTrainingEnabled(context: Context): Boolean =
        context.applicationContext.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getBoolean(KEY_TRAINING_ENABLED, false)

    fun trainingSamples(context: Context): Long = context.applicationContext.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getLong(KEY_SAMPLES, 0L)
    fun trainingBatches(context: Context): Long = context.applicationContext.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getLong(KEY_BATCHES, 0L)
    fun bestValidationMse(context: Context): Double = context.applicationContext.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getFloat(KEY_BEST_VAL, Float.POSITIVE_INFINITY).toDouble()
    fun lastLoss(context: Context): Double = context.applicationContext.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getFloat(KEY_LAST_LOSS, Float.NaN).toDouble()

    private fun load(context: Context) {
        val file = context.getFileStreamPath(MODEL_FILE)
        if (!file.exists()) return
        try {
            DataInputStream(BufferedInputStream(FileInputStream(file))).use { nn.loadState(it) }
        } catch (_: Exception) {
            // Keep the new random model if a previous checkpoint is incompatible/corrupt.
            file.renameTo(context.getFileStreamPath("$MODEL_FILE.corrupt"))
        }
    }
}
