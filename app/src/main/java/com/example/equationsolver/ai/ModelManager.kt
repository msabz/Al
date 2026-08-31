package com.example.equationsolver.ai

import android.content.Context
import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.InputStream
import java.io.OutputStream
import kotlin.math.abs

/** Owns the v5 checkpoint and provides a stable import/export bridge to Colab. */
object ModelManager {
    private const val PREFS = "math_ai_model_v5"
    private const val MODEL_FILE = "math_ai_model_v5.mai5"
    private const val DEFAULT_ASSET_MODEL = "default_model.mai5"
    private const val KEY_SAMPLES = "training_samples"
    private const val KEY_BATCHES = "training_batches"
    private const val KEY_BEST_VAL = "best_validation_mse"
    private const val KEY_LAST_VAL = "last_validation_mse"
    private const val KEY_VAL_ACCURACY = "last_validation_accuracy"
    private const val KEY_TRAINING_ENABLED = "training_enabled"
    private const val KEY_LAST_LOSS = "last_loss"
    private const val KEY_CHECKPOINT_TIME = "checkpoint_time"
    private const val KEY_IMPORTED_AT = "imported_at"
    private const val KEY_BOOTSTRAPPED_ASSET = "bootstrapped_from_asset"

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
        val hasRecoveryBackup: Boolean,
        val importedAt: Long = 0L,
        val bootstrappedFromAsset: Boolean = false
    )

    lateinit var nn: NeuralNetwork
        private set

    @Synchronized
    fun init(context: Context) {
        if (::nn.isInitialized) return
        nn = NeuralNetwork()
        load(context.applicationContext)
    }

    fun predictStructured(input: String): V5Prediction {
        val encoding = StructuralMathEncoder.encode(input)
        require(!encoding.truncated) { "المعادلة تتجاوز ${V5ModelSpec.MAX_NODES} عقدة RPN" }
        return nn.predict(encoding)
    }

    /** Compatibility helper for older UI code: [primary/x, y]. */
    fun predictValues(input: String): DoubleArray {
        val p = predictStructured(input)
        return if (p.family == EquationFamily.SYSTEM) {
            doubleArrayOf(p.x ?: p.slotValues[0], p.y ?: p.slotValues[1])
        } else {
            val value = p.roots.firstOrNull() ?: p.slotValues[0]
            if (input.lowercase().contains('y') && !input.lowercase().contains('x')) doubleArrayOf(0.0, value)
            else doubleArrayOf(value, 0.0)
        }
    }

    fun trainWithTarget(input: String, x: Double, y: Double, repeats: Int, learningRate: Double): TrainingDelta {
        require(x.isFinite() && y.isFinite()) { "قيم التدريب غير صالحة" }
        val encoding = StructuralMathEncoder.encode(input)
        require(!encoding.truncated) { "المعادلة أطول من حد v5" }
        val before = predictValues(input)
        val target = if (encoding.family == EquationFamily.SYSTEM) {
            V5Target(EquationFamily.SYSTEM, SolutionState.FINITE, systemValues = doubleArrayOf(x, y))
        } else {
            val root = if (input.lowercase().contains('y') && !input.lowercase().contains('x')) y else x
            V5Target(encoding.family, SolutionState.FINITE, roots = doubleArrayOf(root))
        }
        val item = V5TrainItem(encoding, target, equivalent = null)
        val count = repeats.coerceAtLeast(1)
        repeat(count) { nn.trainBatch(arrayOf(item), learningRate, 0.0) }
        val after = predictValues(input)
        return TrainingDelta(
            before = before,
            after = after,
            meanAbsoluteErrorBefore = meanAbsoluteError(before, doubleArrayOf(x, y)),
            meanAbsoluteErrorAfter = meanAbsoluteError(after, doubleArrayOf(x, y)),
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
        val raw = FileOutputStream(temp)
        DataOutputStream(BufferedOutputStream(raw)).use { out ->
            nn.saveState(out)
            out.flush()
            raw.fd.sync()
        }

        var oldMoved = false
        try {
            if (backup.exists()) backup.delete()
            if (target.exists()) {
                if (!target.renameTo(backup)) throw IllegalStateException("تعذر إنشاء نسخة احتياطية")
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

        val edit = app.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit()
        if (samples != null) edit.putLong(KEY_SAMPLES, samples)
        if (batches != null) edit.putLong(KEY_BATCHES, batches)
        if (bestValidationMse != null && bestValidationMse.isFinite()) edit.putFloat(KEY_BEST_VAL, bestValidationMse.toFloat())
        if (lastLoss != null && lastLoss.isFinite()) edit.putFloat(KEY_LAST_LOSS, lastLoss.toFloat())
        if (lastValidationMse != null && lastValidationMse.isFinite()) edit.putFloat(KEY_LAST_VAL, lastValidationMse.toFloat())
        if (validationAccuracy != null && validationAccuracy.isFinite()) edit.putFloat(KEY_VAL_ACCURACY, validationAccuracy.toFloat())
        edit.putLong(KEY_CHECKPOINT_TIME, System.currentTimeMillis()).commit()
    }

    /** Imports the exact MAI5 file emitted by Colab. */
    @Synchronized
    fun importWeights(context: Context, input: InputStream, resetTrainingStats: Boolean = true): ModelInfo {
        val candidate = NeuralNetwork()
        DataInputStream(BufferedInputStream(input)).use { candidate.loadState(it) }
        nn = candidate
        val app = context.applicationContext
        if (resetTrainingStats) {
            val enabled = isTrainingEnabled(app)
            app.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit().clear().putBoolean(KEY_TRAINING_ENABLED, enabled).commit()
        }
        app.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit()
            .putLong(KEY_IMPORTED_AT, System.currentTimeMillis())
            .putBoolean(KEY_BOOTSTRAPPED_ASSET, false)
            .commit()
        save(app)
        return modelInfo(app)
    }

    @Synchronized
    fun exportWeights(output: OutputStream) {
        DataOutputStream(BufferedOutputStream(output)).use { out -> nn.saveState(out); out.flush() }
    }

    @Synchronized
    fun resetModel(context: Context) {
        nn = NeuralNetwork()
        val app = context.applicationContext
        app.getFileStreamPath(MODEL_FILE).delete()
        app.getFileStreamPath("$MODEL_FILE.bak").delete()
        val enabled = isTrainingEnabled(app)
        app.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit().clear()
            .putBoolean(KEY_TRAINING_ENABLED, enabled)
            .putBoolean(KEY_BOOTSTRAPPED_ASSET, false)
            .commit()
        save(app, samples = 0, batches = 0)
    }

    fun setTrainingEnabled(context: Context, enabled: Boolean) {
        context.applicationContext.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit().putBoolean(KEY_TRAINING_ENABLED, enabled).commit()
    }

    fun isTrainingEnabled(context: Context): Boolean = context.applicationContext.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getBoolean(KEY_TRAINING_ENABLED, false)
    fun trainingSamples(context: Context): Long = context.applicationContext.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getLong(KEY_SAMPLES, 0L)
    fun trainingBatches(context: Context): Long = context.applicationContext.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getLong(KEY_BATCHES, 0L)
    fun bestValidationMse(context: Context): Double = context.applicationContext.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getFloat(KEY_BEST_VAL, Float.POSITIVE_INFINITY).toDouble()
    fun lastLoss(context: Context): Double = context.applicationContext.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getFloat(KEY_LAST_LOSS, Float.NaN).toDouble()
    fun lastValidationMse(context: Context): Double = context.applicationContext.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getFloat(KEY_LAST_VAL, Float.NaN).toDouble()
    fun lastValidationAccuracy(context: Context): Double = context.applicationContext.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getFloat(KEY_VAL_ACCURACY, Float.NaN).toDouble()

    fun modelInfo(context: Context): ModelInfo {
        val app = context.applicationContext
        val p = app.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        val checkpoint = app.getFileStreamPath(MODEL_FILE)
        return ModelInfo(
            parameterCount = nn.parameterCount(),
            optimizerStep = nn.optimizerStep(),
            checkpointBytes = checkpoint.takeIf { it.exists() }?.length() ?: 0L,
            checkpointSavedAt = p.getLong(KEY_CHECKPOINT_TIME, 0L),
            hasRecoveryBackup = app.getFileStreamPath("$MODEL_FILE.bak").exists(),
            importedAt = p.getLong(KEY_IMPORTED_AT, 0L),
            bootstrappedFromAsset = p.getBoolean(KEY_BOOTSTRAPPED_ASSET, false)
        )
    }

    /**
     * Load order:
     * 1) mutable internal checkpoint
     * 2) recovery backup
     * 3) immutable default_model.mai5 embedded by the Colab factory
     * 4) fresh random v5 model
     */
    private fun load(context: Context) {
        val file = context.getFileStreamPath(MODEL_FILE)
        val backup = context.getFileStreamPath("$MODEL_FILE.bak")
        if (tryLoadFile(file)) return
        if (file.exists()) {
            val corrupt = context.getFileStreamPath("$MODEL_FILE.corrupt")
            corrupt.delete(); file.renameTo(corrupt)
        }
        nn = NeuralNetwork()
        if (tryLoadFile(backup)) {
            runCatching { FileInputStream(backup).use { i -> FileOutputStream(file).use { o -> i.copyTo(o) } } }
            return
        }

        if (tryLoadEmbeddedAsset(context)) {
            context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit()
                .putBoolean(KEY_BOOTSTRAPPED_ASSET, true)
                .putLong(KEY_IMPORTED_AT, 0L)
                .commit()
            // Immediately promote the immutable asset to the normal mutable checkpoint,
            // so on-device Adam training can continue from exactly the Colab state.
            runCatching { save(context, samples = 0, batches = 0) }
            return
        }

        nn = NeuralNetwork()
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit().putBoolean(KEY_BOOTSTRAPPED_ASSET, false).apply()
    }

    private fun tryLoadFile(file: java.io.File): Boolean {
        if (!file.exists()) return false
        val candidate = NeuralNetwork()
        return try {
            DataInputStream(BufferedInputStream(FileInputStream(file))).use { candidate.loadState(it) }
            nn = candidate
            true
        } catch (_: Exception) { false }
    }

    private fun tryLoadEmbeddedAsset(context: Context): Boolean {
        val candidate = NeuralNetwork()
        return try {
            context.assets.open(DEFAULT_ASSET_MODEL).use { raw ->
                DataInputStream(BufferedInputStream(raw)).use { candidate.loadState(it) }
            }
            nn = candidate
            true
        } catch (_: Exception) {
            false
        }
    }

    private fun meanAbsoluteError(a: DoubleArray, b: DoubleArray): Double {
        val count = maxOf(a.size, b.size).coerceAtLeast(1)
        var total = 0.0
        for (i in 0 until count) total += abs(a.getOrElse(i) { 0.0 } - b.getOrElse(i) { 0.0 })
        return total / count
    }
}
