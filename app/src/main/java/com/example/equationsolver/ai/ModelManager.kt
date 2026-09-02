package com.example.equationsolver.ai

import android.content.Context
import android.util.Base64
import android.util.Base64InputStream
import java.util.zip.GZIPInputStream
import java.io.FileInputStream
import java.io.FileOutputStream
import kotlin.math.abs

object ModelManager {
    private const val PREFS = "open_growth_rsnn_resume_v1"
    private const val MODEL_FILE = "open_growth_rsnn_phone.bin"
    private const val ASSET_FILE = "open_growth_stage1_phone.b64"
    private const val KEY_SAMPLES = "training_samples"
    private const val KEY_BATCHES = "training_batches"
    private const val KEY_BEST_VAL = "best_validation_mse"
    private const val KEY_LAST_VAL = "last_validation_mse"
    private const val KEY_VAL_ACCURACY = "last_validation_within_one"
    private const val KEY_TRAINING_ENABLED = "training_enabled"
    private const val KEY_LAST_LOSS = "last_loss"
    private const val KEY_CHECKPOINT_TIME = "checkpoint_time"

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

    lateinit var nn: OpenGrowthRsnnPhone
        private set

    @Synchronized
    fun init(context: Context) {
        if (::nn.isInitialized) return
        val app = context.applicationContext
        nn = OpenGrowthRsnnPhone()
        val file = app.getFileStreamPath(MODEL_FILE)
        val backup = app.getFileStreamPath("$MODEL_FILE.bak")
        val loaded = tryLoad(file) || tryLoad(backup)
        if (!loaded) {
            app.assets.open(ASSET_FILE).use { encoded -> GZIPInputStream(Base64InputStream(encoded, Base64.DEFAULT)).use { nn.load(it) } }
            val p = app.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            if (!p.contains(KEY_SAMPLES)) {
                p.edit()
                    .putLong(KEY_SAMPLES, nn.examplesSeen)
                    .putLong(KEY_BATCHES, nn.optimizerStepLong())
                    .putLong(KEY_CHECKPOINT_TIME, System.currentTimeMillis())
                    .commit()
            }
            save(app)
        }
    }

    fun predict(input: String): DoubleArray = predictValues(input).map { it / OpenGrowthRsnnPhone.TARGET_SCALE }.toDoubleArray()

    fun predictValues(input: String): DoubleArray {
        val features = LinearSystemCodec.parseSystem(input).features
        return nn.predictRaw(features)
    }

    fun trainWithTarget(input: String, x: Double, y: Double, repeats: Int, learningRate: Double): TrainingDelta {
        require(x.isFinite() && y.isFinite()) { "قيم التدريب يجب أن تكون أرقامًا محدودة" }
        val features = LinearSystemCodec.parseSystem(input).features
        val before = nn.predictRaw(features)
        val count = repeats.coerceIn(1, 32)
        repeat(count) { nn.trainBatch(listOf(features), listOf(doubleArrayOf(x, y)), learningRate.coerceAtMost(1e-4)) }
        val after = nn.predictRaw(features)
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
        FileOutputStream(temp).use { out -> nn.save(out); out.fd.sync() }
        if (backup.exists()) backup.delete()
        if (target.exists() && !target.renameTo(backup)) error("تعذر إنشاء نسخة احتياطية من النموذج")
        if (!temp.renameTo(target)) {
            FileInputStream(temp).use { input -> FileOutputStream(target).use { output -> input.copyTo(output) } }
            temp.delete()
        }
        val p = app.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        val edit = p.edit()
        if (samples != null) edit.putLong(KEY_SAMPLES, samples)
        if (batches != null) edit.putLong(KEY_BATCHES, batches)
        if (bestValidationMse != null && bestValidationMse.isFinite()) edit.putFloat(KEY_BEST_VAL, bestValidationMse.toFloat())
        if (lastLoss != null && lastLoss.isFinite()) edit.putFloat(KEY_LAST_LOSS, lastLoss.toFloat())
        if (lastValidationMse != null && lastValidationMse.isFinite()) edit.putFloat(KEY_LAST_VAL, lastValidationMse.toFloat())
        if (validationAccuracy != null && validationAccuracy.isFinite()) edit.putFloat(KEY_VAL_ACCURACY, validationAccuracy.toFloat())
        edit.putLong(KEY_CHECKPOINT_TIME, System.currentTimeMillis()).commit()
    }

    fun setTrainingEnabled(context: Context, enabled: Boolean) {
        context.applicationContext.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            .edit().putBoolean(KEY_TRAINING_ENABLED, enabled).commit()
    }

    fun isTrainingEnabled(context: Context): Boolean = context.applicationContext.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getBoolean(KEY_TRAINING_ENABLED, false)
    fun trainingSamples(context: Context): Long = context.applicationContext.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getLong(KEY_SAMPLES, nn.examplesSeen)
    fun trainingBatches(context: Context): Long = context.applicationContext.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getLong(KEY_BATCHES, nn.optimizerStepLong())
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

    private fun tryLoad(file: java.io.File): Boolean {
        if (!file.exists()) return false
        return try { FileInputStream(file).use { nn.load(it) }; true } catch (_: Exception) { false }
    }

    private fun meanAbsoluteError(prediction: DoubleArray, target: DoubleArray): Double =
        (abs(prediction[0] - target[0]) + abs(prediction[1] - target[1])) / 2.0
}
