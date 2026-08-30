package com.example.equationsolver.ai

import android.content.Context
import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.FileInputStream
import java.io.FileOutputStream
import kotlin.math.abs

object ModelManager {
    private const val PREFS = "equation_solver_model_v4"
    private const val MODEL_FILE = "equation_model_v4.bin"
    private const val KEY_SAMPLES = "training_samples"
    private const val KEY_BATCHES = "training_batches"
    private const val KEY_BEST_VAL = "best_validation_active_mse_v2"
    private const val KEY_LAST_VAL = "last_validation_active_mse_v2"
    private const val KEY_VAL_ACCURACY = "last_validation_within_one_v2"
    private const val KEY_TRAINING_ENABLED = "training_enabled"
    private const val KEY_LAST_LOSS = "last_loss"
    private const val KEY_CHECKPOINT_TIME = "checkpoint_time"
    private const val OUTPUT_SCALE = 100.0

    data class TrainingDelta(
        val before: DoubleArray,
        val after: DoubleArray,
        val meanAbsoluteErrorBefore: Double,
        val meanAbsoluteErrorAfter: Double,
        val optimizerSteps: Int
    )

    data class ModelInfo(
        val parameterCount: Int,
        val optimizerStep: Int,
        val checkpointBytes: Long,
        val checkpointSavedAt: Long,
        val hasRecoveryBackup: Boolean
    )

    lateinit var nn: NeuralNetwork
        private set

    @Synchronized
    fun init(context: Context) {
        if (::nn.isInitialized) return
        nn = NeuralNetwork()
        load(context.applicationContext)
    }

    fun predict(input: String): DoubleArray = nn.predict(MathTokenizer.tokenize(input))

    fun predictValues(input: String): DoubleArray =
        predict(input).map { it * OUTPUT_SCALE }.toDoubleArray()

    fun trainWithTarget(input: String, x: Double, y: Double, repeats: Int, learningRate: Double): TrainingDelta {
        require(x.isFinite() && y.isFinite()) { "قيم التدريب يجب أن تكون أرقامًا محدودة" }
        val encoding = MathTokenizer.encode(input)
        require(!encoding.truncated) { "المعادلة أطول من حد النموذج (${MathTokenizer.MAX_TOKENS} token)" }
        require(encoding.unknownCount == 0) { "المعادلة تحتوي رموزًا لا يعرفها النموذج" }
        val tokens = encoding.tokens
        val target = doubleArrayOf(x / OUTPUT_SCALE, y / OUTPUT_SCALE)
        val before = nn.predict(tokens)
        val count = repeats.coerceAtLeast(1)
        repeat(count) { nn.train(tokens, target, learningRate) }
        val after = nn.predict(tokens)
        return TrainingDelta(
            before = before.map { it * OUTPUT_SCALE }.toDoubleArray(),
            after = after.map { it * OUTPUT_SCALE }.toDoubleArray(),
            meanAbsoluteErrorBefore = meanAbsoluteError(before, target) * OUTPUT_SCALE,
            meanAbsoluteErrorAfter = meanAbsoluteError(after, target) * OUTPUT_SCALE,
            optimizerSteps = count
        )
    }

    @Synchronized
    fun save(
        context: Context,
        samples: Long? = null,
        batches: Long? = null,
        bestValidationMse: Double? = null,
        lastLoss: Double? = null,
        lastValidationMse: Double? = null,
        validationAccuracy: Double? = null
    ) {
        if (!::nn.isInitialized) return
        val app = context.applicationContext
        val target = app.getFileStreamPath(MODEL_FILE)
        val temp = app.getFileStreamPath("$MODEL_FILE.tmp")
        val backup = app.getFileStreamPath("$MODEL_FILE.bak")

        temp.delete()
        val fileOutput = FileOutputStream(temp)
        DataOutputStream(BufferedOutputStream(fileOutput)).use {
            nn.saveState(it)
            it.flush()
            fileOutput.fd.sync()
        }

        var oldMoved = false
        try {
            if (backup.exists()) backup.delete()
            if (target.exists()) {
                if (!target.renameTo(backup)) throw IllegalStateException("تعذر إنشاء نسخة احتياطية من النموذج")
                oldMoved = true
            }

            if (!temp.renameTo(target)) {
                FileInputStream(temp).use { input -> FileOutputStream(target).use { output -> input.copyTo(output) } }
                temp.delete()
            }
        } catch (e: Exception) {
            temp.delete()
            if (target.exists()) target.delete()
            if (oldMoved && backup.exists()) backup.renameTo(target)
            throw e
        }

        val p = app.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        val edit = p.edit()
        if (samples != null) edit.putLong(KEY_SAMPLES, samples)
        if (batches != null) edit.putLong(KEY_BATCHES, batches)
        if (bestValidationMse != null && bestValidationMse.isFinite()) edit.putFloat(KEY_BEST_VAL, bestValidationMse.toFloat())
        if (lastLoss != null && lastLoss.isFinite()) edit.putFloat(KEY_LAST_LOSS, lastLoss.toFloat())
        if (lastValidationMse != null && lastValidationMse.isFinite()) edit.putFloat(KEY_LAST_VAL, lastValidationMse.toFloat())
        if (validationAccuracy != null && validationAccuracy.isFinite()) edit.putFloat(KEY_VAL_ACCURACY, validationAccuracy.toFloat())
        edit.putLong(KEY_CHECKPOINT_TIME, System.currentTimeMillis())
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
    fun lastValidationMse(context: Context): Double = context.applicationContext.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getFloat(KEY_LAST_VAL, Float.NaN).toDouble()
    fun lastValidationAccuracy(context: Context): Double = context.applicationContext.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getFloat(KEY_VAL_ACCURACY, Float.NaN).toDouble()

    fun modelInfo(context: Context): ModelInfo {
        val app = context.applicationContext
        val prefs = app.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        val checkpoint = app.getFileStreamPath(MODEL_FILE)
        return ModelInfo(
            parameterCount = nn.parameterCount(),
            optimizerStep = nn.optimizerStep(),
            checkpointBytes = checkpoint.takeIf { it.exists() }?.length() ?: 0L,
            checkpointSavedAt = prefs.getLong(KEY_CHECKPOINT_TIME, 0L),
            hasRecoveryBackup = app.getFileStreamPath("$MODEL_FILE.bak").exists()
        )
    }

    private fun load(context: Context) {
        val file = context.getFileStreamPath(MODEL_FILE)
        val backup = context.getFileStreamPath("$MODEL_FILE.bak")
        if (tryLoad(file)) return

        if (file.exists()) {
            val corrupt = context.getFileStreamPath("$MODEL_FILE.corrupt")
            corrupt.delete()
            file.renameTo(corrupt)
        }

        nn = NeuralNetwork()
        if (tryLoad(backup)) {
            try {
                FileInputStream(backup).use { input -> FileOutputStream(file).use { output -> input.copyTo(output) } }
            } catch (_: Exception) { }
        } else {
            nn = NeuralNetwork()
        }
    }

    private fun tryLoad(file: java.io.File): Boolean {
        if (!file.exists()) return false
        return try {
            DataInputStream(BufferedInputStream(FileInputStream(file))).use { nn.loadState(it) }
            true
        } catch (_: Exception) {
            false
        }
    }

    private fun meanAbsoluteError(prediction: DoubleArray, target: DoubleArray): Double {
        var total = 0.0
        for (i in target.indices) total += abs(prediction.getOrElse(i) { 0.0 } - target[i])
        return total / target.size
    }
}
