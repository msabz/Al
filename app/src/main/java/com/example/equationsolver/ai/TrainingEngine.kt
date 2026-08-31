package com.example.equationsolver.ai

import android.content.Context
import android.os.BatteryManager
import android.os.Build
import android.os.PowerManager
import android.os.SystemClock
import com.example.equationsolver.data.V5ExampleGenerator
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import java.util.ArrayDeque
import kotlin.math.max
import kotlin.math.sqrt
import kotlin.random.Random

/** Continuous v5 training. The network never uses the reference solver for inference. */
object TrainingEngine {
    const val BATCH_SIZE = 10
    const val EPOCHS = 3
    const val OUTPUT_SCALE = V5ModelSpec.ROOT_SCALE

    private const val CURRICULUM_ROUND_SAMPLES = 10_000L
    private const val PROGRESS_EVERY_BATCHES = 5L
    private const val VALIDATION_SET_SIZE = 160
    private const val CONSISTENCY_QUEUE_LIMIT = 48

    data class ValidationMetrics(
        val normalizedMse: Double,
        val rmse: Double,
        val meanAbsoluteError: Double,
        val withinOneUnitRatio: Double,
        val stateAccuracy: Double = Double.NaN,
        val valueCount: Int
    ) {
        companion object { val EMPTY = ValidationMetrics(Double.NaN, Double.NaN, Double.NaN, Double.NaN, Double.NaN, 0) }
    }

    data class Snapshot(
        val samples: Long = 0L,
        val batches: Long = 0L,
        val curriculumRound: Long = 0L,
        val loss: Double = Double.NaN,
        val bestValidationMse: Double = Double.POSITIVE_INFINITY,
        val validation: ValidationMetrics = ValidationMetrics.EMPTY,
        val lastEquation: String = "",
        val lastFamily: String = "",
        val gradientNorm: Double = Double.NaN,
        val paused: Boolean = false,
        val reason: String = ""
    )

    private data class ValidationBank(val items: List<V5TrainItem>, val equations: Set<String>)

    @Volatile private var currentSnapshot = Snapshot()
    @Volatile var lastValidation: ValidationMetrics = ValidationMetrics.EMPTY
        private set
    @Volatile private var externalFileSessionActive = false
    @Volatile private var externalFileStopRequested = false

    val lastValidationMse: Double get() = lastValidation.normalizedMse
    private val validationBank: ValidationBank by lazy { buildValidationBank() }

    fun snapshot(): Snapshot = currentSnapshot
    @Synchronized fun beginExternalFileSession(): Boolean { if (externalFileSessionActive) return false; externalFileSessionActive = true; externalFileStopRequested = false; return true }
    @Synchronized fun endExternalFileSession() { externalFileSessionActive = false; externalFileStopRequested = false }
    fun isExternalFileSessionActive(): Boolean = externalFileSessionActive
    fun requestExternalFileSessionStop() { externalFileStopRequested = true }
    fun isExternalFileStopRequested(): Boolean = externalFileStopRequested
    fun isReservedForValidation(equation: String): Boolean = equation in validationBank.equations

    fun trainRandom(samples: Int, progress: (Int) -> Unit = {}) {
        val settings = V5Settings.read(appContext ?: return)
        val batch = ArrayList<V5TrainItem>(settings.batchSize)
        repeat(samples) { index ->
            val generated = V5ExampleGenerator.generate(maxAbs = settings.maxAbsTrainingValue)
            if (!settings.familyEnabled(generated.target.family) || generated.equation in validationBank.equations) return@repeat
            val item = encode(generated.equation, generated.equivalentEquation, generated.target) ?: return@repeat
            batch += item
            if (batch.size >= settings.batchSize) {
                ModelManager.nn.trainBatch(batch.toTypedArray(), settings.learningRate, settings.consistencyWeight)
                batch.clear()
            }
            if ((index + 1) % 500 == 0) progress(index + 1)
        }
        if (batch.isNotEmpty()) ModelManager.nn.trainBatch(batch.toTypedArray(), settings.learningRate, settings.consistencyWeight)
    }

    @Volatile private var appContext: Context? = null

    suspend fun trainContinuous(context: Context, learningRate: Double = 0.0007, progress: (Snapshot) -> Unit = {}) {
        appContext = context.applicationContext
        var samples = ModelManager.trainingSamples(context)
        var batches = ModelManager.trainingBatches(context)
        var bestValidation = ModelManager.bestValidationMse(context)
        var lastLoss = ModelManager.lastLoss(context)
        var validation = restoredValidation(context)
        var lastEquation = ""
        var lastFamily = ""
        var lastCheckpointAt = SystemClock.elapsedRealtime()
        if (!bestValidation.isFinite()) bestValidation = Double.POSITIVE_INFINITY

        val batch = ArrayList<V5TrainItem>(16)
        val replay = ArrayDeque<V5TrainItem>(CONSISTENCY_QUEUE_LIMIT)
        publish(samples, batches, lastLoss, bestValidation, validation, lastEquation, lastFamily, false, "", progress)

        try {
            while (currentCoroutineContext().isActive) {
                val reason = pauseReason(context)
                if (reason != null) {
                    publish(samples, batches, lastLoss, bestValidation, validation, lastEquation, lastFamily, true, reason, progress)
                    delay(3_000L); continue
                }

                val settings = V5Settings.read(context)
                val generated = V5ExampleGenerator.generate(maxAbs = settings.maxAbsTrainingValue)
                if (!settings.familyEnabled(generated.target.family) || generated.equation in validationBank.equations) continue
                val item = encode(generated.equation, generated.equivalentEquation, generated.target) ?: continue
                batch += item
                replay.addLast(item)
                while (replay.size > CONSISTENCY_QUEUE_LIMIT) replay.removeFirst()
                samples++
                lastEquation = generated.equation
                lastFamily = generated.familyName

                if (batch.size >= settings.batchSize) {
                    // A tiny bounded replay sample helps consistency/generalisation without retaining the dataset.
                    if (replay.size > settings.batchSize * 2 && batch.size < 16) batch += replay.first
                    lastLoss = ModelManager.nn.trainBatch(batch.toTypedArray(), settings.learningRate, settings.consistencyWeight)
                    batches++
                    batch.clear()

                    if (batches % settings.validateEveryBatches.toLong() == 0L) {
                        validation = evaluateFixedHoldout()
                        lastValidation = validation
                        if (validation.normalizedMse.isFinite() && validation.normalizedMse < bestValidation) bestValidation = validation.normalizedMse
                    }
                    if (batches % PROGRESS_EVERY_BATCHES == 0L || batches == 1L) publish(samples, batches, lastLoss, bestValidation, validation, lastEquation, lastFamily, false, "", progress)
                    else currentSnapshot = snapshotOf(samples, batches, lastLoss, bestValidation, validation, lastEquation, lastFamily)

                    val checkpointMs = settings.checkpointMinutes.toLong() * 60_000L
                    val now = SystemClock.elapsedRealtime()
                    if (now - lastCheckpointAt >= checkpointMs) {
                        saveCheckpoint(context, samples, batches, bestValidation, lastLoss, validation)
                        lastCheckpointAt = now
                    }
                    delay(throttleDelay(context, settings.powerMode))
                }
            }
        } finally {
            currentSnapshot = snapshotOf(samples, batches, lastLoss, bestValidation, validation, lastEquation, lastFamily)
            runCatching { saveCheckpoint(context, samples, batches, bestValidation, lastLoss, validation) }
            appContext = null
        }
    }

    /** Compatibility with the existing file-training UI: equation -> [x,y]. */
    fun trainFile(examples: List<Pair<String, DoubleArray>>, progress: (Int) -> Unit = {}) {
        if (examples.isEmpty()) return
        val context = appContext
        val settings = if (context != null) V5Settings.read(context) else V5Settings.Snapshot(
            V5Settings.PowerMode.BALANCED, 0.0006, 0.05, 100, 40, 5, true, true, true, true
        )
        repeat(EPOCHS) { epoch ->
            val batch = ArrayList<V5TrainItem>(settings.batchSize)
            examples.forEachIndexed { index, pair ->
                if (externalFileStopRequested || Thread.currentThread().isInterrupted) return
                if (pair.first in validationBank.equations) return@forEachIndexed
                val encoding = runCatching { StructuralMathEncoder.encode(pair.first) }.getOrNull() ?: return@forEachIndexed
                if (encoding.truncated) return@forEachIndexed
                val values = pair.second
                val target = if (encoding.family == EquationFamily.SYSTEM) {
                    V5Target(EquationFamily.SYSTEM, SolutionState.FINITE, systemValues = doubleArrayOf(values.getOrElse(0) { 0.0 }, values.getOrElse(1) { 0.0 }))
                } else {
                    val root = if (pair.first.lowercase().contains('y') && !pair.first.lowercase().contains('x')) values.getOrElse(1) { 0.0 } else values.getOrElse(0) { 0.0 }
                    V5Target(encoding.family, SolutionState.FINITE, roots = doubleArrayOf(root))
                }
                batch += V5TrainItem(encoding, target)
                if (batch.size >= settings.batchSize) { ModelManager.nn.trainBatch(batch.toTypedArray(), settings.learningRate, 0.0); batch.clear() }
                if ((index + 1) % 1000 == 0) progress(epoch * examples.size + index + 1)
            }
            if (batch.isNotEmpty()) ModelManager.nn.trainBatch(batch.toTypedArray(), settings.learningRate, 0.0)
        }
        lastValidation = evaluateFixedHoldout()
    }

    fun evaluateFixedHoldout(): ValidationMetrics {
        val result = ModelManager.nn.evaluate(validationBank.items, tolerance = 1.0)
        val rmse = result.rootMeanSquaredError
        return ValidationMetrics(
            normalizedMse = if (rmse.isFinite()) (rmse / OUTPUT_SCALE) * (rmse / OUTPUT_SCALE) else Double.NaN,
            rmse = rmse,
            meanAbsoluteError = result.rootMeanAbsoluteError,
            withinOneUnitRatio = result.withinToleranceRatio,
            stateAccuracy = result.stateAccuracy,
            valueCount = result.valueCount
        )
    }

    private fun encode(equation: String, equivalent: String, target: V5Target): V5TrainItem? = try {
        val a = StructuralMathEncoder.encode(equation)
        val b = StructuralMathEncoder.encode(equivalent)
        if (a.truncated || b.truncated || a.family != target.family || b.family != target.family) null else V5TrainItem(a, target, b)
    } catch (_: Exception) { null }

    private fun buildValidationBank(): ValidationBank {
        val random = Random(0x0A165)
        val items = ArrayList<V5TrainItem>(VALIDATION_SET_SIZE)
        val equations = LinkedHashSet<String>()
        var attempts = 0
        while (items.size < VALIDATION_SET_SIZE && attempts < 20_000) {
            attempts++
            // Default training range is <=100; holdout deliberately pushes beyond it.
            val sample = V5ExampleGenerator.generate(random, maxAbs = 240)
            if (!equations.add(sample.equation)) continue
            val item = encode(sample.equation, sample.equivalentEquation, sample.target) ?: continue
            items += item
        }
        check(items.size == VALIDATION_SET_SIZE) { "تعذر إنشاء Holdout v5 الثابت" }
        return ValidationBank(items, equations)
    }

    private fun publish(samples: Long, batches: Long, loss: Double, best: Double, validation: ValidationMetrics, equation: String, family: String, paused: Boolean, reason: String, progress: (Snapshot) -> Unit) {
        val s = snapshotOf(samples, batches, loss, best, validation, equation, family, paused, reason)
        currentSnapshot = s; progress(s)
    }

    private fun snapshotOf(samples: Long, batches: Long, loss: Double, best: Double, validation: ValidationMetrics, equation: String, family: String, paused: Boolean = false, reason: String = "") = Snapshot(
        samples, batches, samples / CURRICULUM_ROUND_SAMPLES, loss, best, validation, equation, family, ModelManager.nn.lastGradientNorm, paused, reason
    )

    private fun restoredValidation(context: Context): ValidationMetrics {
        val mse = ModelManager.lastValidationMse(context)
        val accuracy = ModelManager.lastValidationAccuracy(context)
        return if (mse.isFinite() && mse >= 0.0) ValidationMetrics(mse, sqrt(mse) * OUTPUT_SCALE, Double.NaN, accuracy, Double.NaN, 0) else ValidationMetrics.EMPTY
    }

    private fun saveCheckpoint(context: Context, samples: Long, batches: Long, best: Double, loss: Double, validation: ValidationMetrics) {
        ModelManager.save(context, samples, batches, best, loss, validation.normalizedMse, validation.withinOneUnitRatio)
    }

    private fun pauseReason(context: Context): String? {
        val battery = context.getSystemService(Context.BATTERY_SERVICE) as BatteryManager
        val level = battery.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY)
        if (level in 0..20) return "البطارية منخفضة ($level%)"
        val power = context.getSystemService(Context.POWER_SERVICE) as PowerManager
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q && power.currentThermalStatus >= PowerManager.THERMAL_STATUS_MODERATE) return "حرارة الجهاز مرتفعة؛ سيستأنف التدريب بعد أن يبرد"
        if (power.isPowerSaveMode && level <= 35) return "وضع توفير الطاقة فعال"
        return null
    }

    private fun throttleDelay(context: Context, mode: V5Settings.PowerMode): Long {
        val battery = context.getSystemService(Context.BATTERY_SERVICE) as BatteryManager
        val charging = Build.VERSION.SDK_INT >= Build.VERSION_CODES.M && battery.isCharging
        val power = context.getSystemService(Context.POWER_SERVICE) as PowerManager
        var delayMs = when (mode) {
            V5Settings.PowerMode.ECO -> 220L
            V5Settings.PowerMode.BALANCED -> if (charging) 45L else 100L
            V5Settings.PowerMode.FAST -> if (charging) 10L else 55L
        }
        if (power.isPowerSaveMode) delayMs = max(delayMs, 300L)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q && power.currentThermalStatus >= PowerManager.THERMAL_STATUS_LIGHT) delayMs = max(delayMs, 180L)
        return delayMs
    }
}
