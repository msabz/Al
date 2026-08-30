package com.example.equationsolver.ai

import android.content.Context
import android.os.BatteryManager
import android.os.Build
import android.os.PowerManager
import android.os.SystemClock
import com.example.equationsolver.data.EquationGenerator
import com.example.equationsolver.data.GeneratedEquationValidator
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlin.math.max
import kotlin.math.sqrt
import kotlin.random.Random

/** Real on-device training loop with a fixed, never-trained validation holdout. */
object TrainingEngine {
    const val BATCH_SIZE = 16
    const val EPOCHS = 3
    const val OUTPUT_SCALE = 100.0

    private const val CURRICULUM_ROUND_SAMPLES = 10_000L
    private const val VALIDATE_EVERY_BATCHES = 25L
    private const val PROGRESS_EVERY_BATCHES = 5L
    private const val CHECKPOINT_INTERVAL_MS = 5L * 60L * 1000L
    private const val VALIDATION_SET_SIZE = 160
    private const val NORMALIZED_ONE_UNIT = 1.0 / OUTPUT_SCALE

    data class ValidationMetrics(
        val normalizedMse: Double,
        val rmse: Double,
        val meanAbsoluteError: Double,
        val withinOneUnitRatio: Double,
        val valueCount: Int
    ) {
        companion object {
            val EMPTY = ValidationMetrics(Double.NaN, Double.NaN, Double.NaN, Double.NaN, 0)
        }
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

    private data class ValidationItem(
        val equation: String,
        val tokens: IntArray,
        val target: DoubleArray,
        val activeOutputs: BooleanArray
    )

    private data class ValidationBank(
        val items: List<ValidationItem>,
        val equations: Set<String>
    )

    @Volatile
    private var currentSnapshot = Snapshot()

    @Volatile
    var lastValidation: ValidationMetrics = ValidationMetrics.EMPTY
        private set

    @Volatile
    private var externalFileSessionActive = false

    @Volatile
    private var externalFileStopRequested = false

    val lastValidationMse: Double get() = lastValidation.normalizedMse

    private val validationBank: ValidationBank by lazy { buildValidationBank() }

    fun snapshot(): Snapshot = currentSnapshot

    @Synchronized
    fun beginExternalFileSession(): Boolean {
        if (externalFileSessionActive) return false
        externalFileSessionActive = true
        externalFileStopRequested = false
        return true
    }

    @Synchronized
    fun endExternalFileSession() {
        externalFileSessionActive = false
        externalFileStopRequested = false
    }

    fun isExternalFileSessionActive(): Boolean = externalFileSessionActive

    fun requestExternalFileSessionStop() {
        externalFileStopRequested = true
    }

    fun isExternalFileStopRequested(): Boolean = externalFileStopRequested

    fun isReservedForValidation(equation: String): Boolean = equation in validationBank.equations

    fun trainRandom(samples: Int, progress: (Int) -> Unit = {}) {
        val inputs = ArrayList<IntArray>(BATCH_SIZE)
        val targets = ArrayList<DoubleArray>(BATCH_SIZE)
        for (i in 1..samples) {
            val sample = EquationGenerator.generate()
            if (!GeneratedEquationValidator.isValid(sample) || sample.equation in validationBank.equations) continue
            inputs += MathTokenizer.tokenize(sample.equation)
            targets += target(sample.x, sample.y)
            if (inputs.size == BATCH_SIZE || i == samples) {
                if (inputs.isNotEmpty()) ModelManager.nn.trainBatch(inputs.toTypedArray(), targets.toTypedArray(), 0.0007)
                inputs.clear()
                targets.clear()
            }
            if (i % 500 == 0) progress(i)
        }
    }

    suspend fun trainContinuous(
        context: Context,
        learningRate: Double = 0.0007,
        progress: (Snapshot) -> Unit = {}
    ) {
        var samples = ModelManager.trainingSamples(context)
        var batches = ModelManager.trainingBatches(context)
        var bestValidation = ModelManager.bestValidationMse(context)
        var lastLoss = ModelManager.lastLoss(context)
        var validation = restoredValidation(context)
        var lastEquation = ""
        var lastFamily = ""
        var lastCheckpointAt = SystemClock.elapsedRealtime()
        if (!bestValidation.isFinite()) bestValidation = Double.POSITIVE_INFINITY

        val inputs = ArrayList<IntArray>(BATCH_SIZE)
        val targets = ArrayList<DoubleArray>(BATCH_SIZE)
        publish(samples, batches, lastLoss, bestValidation, validation, lastEquation, lastFamily, false, "", progress)

        try {
            while (currentCoroutineContext().isActive) {
                val reason = pauseReason(context)
                if (reason != null) {
                    publish(samples, batches, lastLoss, bestValidation, validation, lastEquation, lastFamily, true, reason, progress)
                    delay(3_000L)
                    continue
                }

                val sample = EquationGenerator.generate()
                if (!GeneratedEquationValidator.isValid(sample) || sample.equation in validationBank.equations) continue
                inputs += MathTokenizer.tokenize(sample.equation)
                targets += target(sample.x, sample.y)
                samples++
                lastEquation = sample.equation
                lastFamily = sample.family

                if (inputs.size == BATCH_SIZE) {
                    lastLoss = ModelManager.nn.trainBatch(inputs.toTypedArray(), targets.toTypedArray(), learningRate)
                    batches++
                    inputs.clear()
                    targets.clear()

                    if (batches % VALIDATE_EVERY_BATCHES == 0L) {
                        validation = evaluateFixedHoldout()
                        lastValidation = validation
                        if (validation.normalizedMse.isFinite() && validation.normalizedMse < bestValidation) {
                            bestValidation = validation.normalizedMse
                        }
                    }

                    if (batches % PROGRESS_EVERY_BATCHES == 0L || batches == 1L) {
                        publish(samples, batches, lastLoss, bestValidation, validation, lastEquation, lastFamily, false, "", progress)
                    } else {
                        currentSnapshot = snapshotOf(samples, batches, lastLoss, bestValidation, validation, lastEquation, lastFamily)
                    }

                    val now = SystemClock.elapsedRealtime()
                    if (now - lastCheckpointAt >= CHECKPOINT_INTERVAL_MS) {
                        saveCheckpoint(context, samples, batches, bestValidation, lastLoss, validation)
                        lastCheckpointAt = now
                    }
                    delay(throttleDelay(context))
                }
            }
        } finally {
            currentSnapshot = snapshotOf(samples, batches, lastLoss, bestValidation, validation, lastEquation, lastFamily)
            try {
                saveCheckpoint(context, samples, batches, bestValidation, lastLoss, validation)
            } catch (_: Exception) {
                // Cancellation must still complete even if storage is unavailable.
            }
        }
    }

    fun trainFile(examples: List<Pair<String, DoubleArray>>, progress: (Int) -> Unit = {}) {
        val trainingExamples = examples.filterNot { it.first in validationBank.equations }
        if (trainingExamples.isEmpty()) return
        val indices = IntArray(trainingExamples.size) { it }
        indices.shuffle()
        val trainCount = trainingExamples.size

        repeat(EPOCHS) { epoch ->
            val inputs = ArrayList<IntArray>(BATCH_SIZE)
            val targets = ArrayList<DoubleArray>(BATCH_SIZE)
            for (position in 0 until trainCount) {
                if (externalFileStopRequested || Thread.currentThread().isInterrupted) return
                val pair = trainingExamples[indices[position]]
                inputs += MathTokenizer.tokenize(pair.first)
                targets += suppliedTarget(pair.second)
                if (inputs.size == BATCH_SIZE || position == trainCount - 1) {
                    ModelManager.nn.trainBatch(inputs.toTypedArray(), targets.toTypedArray(), 0.0007)
                    inputs.clear()
                    targets.clear()
                }
                if ((position + 1) % 1_000 == 0) progress(epoch * trainCount + position + 1)
            }
        }
        if (externalFileStopRequested || Thread.currentThread().isInterrupted) return
        lastValidation = evaluateFixedHoldout()
    }

    fun evaluateFixedHoldout(): ValidationMetrics {
        val items = validationBank.items
        val result = ModelManager.nn.evaluate(
            inputs = items.map { it.tokens },
            targets = items.map { it.target },
            activeOutputs = items.map { it.activeOutputs },
            tolerance = NORMALIZED_ONE_UNIT
        )
        return ValidationMetrics(
            normalizedMse = result.meanSquaredError,
            rmse = sqrt(result.meanSquaredError) * OUTPUT_SCALE,
            meanAbsoluteError = result.meanAbsoluteError * OUTPUT_SCALE,
            withinOneUnitRatio = result.withinToleranceRatio,
            valueCount = result.valueCount
        )
    }

    private fun buildValidationBank(): ValidationBank {
        val random = Random(0x0A16)
        val items = ArrayList<ValidationItem>(VALIDATION_SET_SIZE)
        val equations = LinkedHashSet<String>()
        var attempts = 0
        while (items.size < VALIDATION_SET_SIZE && attempts < VALIDATION_SET_SIZE * 20) {
            attempts++
            val sample = EquationGenerator.generate(random)
            if (!GeneratedEquationValidator.isValid(sample) || !equations.add(sample.equation)) continue
            items += ValidationItem(
                equation = sample.equation,
                tokens = MathTokenizer.tokenize(sample.equation),
                target = target(sample.x, sample.y),
                activeOutputs = booleanArrayOf(sample.xActive, sample.yActive)
            )
        }
        check(items.size == VALIDATION_SET_SIZE) { "تعذر بناء مجموعة التحقق الثابتة" }
        return ValidationBank(items, equations)
    }

    private fun publish(
        samples: Long,
        batches: Long,
        loss: Double,
        bestValidation: Double,
        validation: ValidationMetrics,
        equation: String,
        family: String,
        paused: Boolean,
        reason: String,
        progress: (Snapshot) -> Unit
    ) {
        val snapshot = snapshotOf(samples, batches, loss, bestValidation, validation, equation, family, paused, reason)
        currentSnapshot = snapshot
        progress(snapshot)
    }

    private fun snapshotOf(
        samples: Long,
        batches: Long,
        loss: Double,
        bestValidation: Double,
        validation: ValidationMetrics,
        equation: String,
        family: String,
        paused: Boolean = false,
        reason: String = ""
    ) = Snapshot(
        samples = samples,
        batches = batches,
        curriculumRound = samples / CURRICULUM_ROUND_SAMPLES,
        loss = loss,
        bestValidationMse = bestValidation,
        validation = validation,
        lastEquation = equation,
        lastFamily = family,
        gradientNorm = ModelManager.nn.lastGradientNorm,
        paused = paused,
        reason = reason
    )

    private fun restoredValidation(context: Context): ValidationMetrics {
        val mse = ModelManager.lastValidationMse(context)
        val accuracy = ModelManager.lastValidationAccuracy(context)
        return if (mse.isFinite() && mse >= 0.0) ValidationMetrics(
            normalizedMse = mse,
            rmse = sqrt(mse) * OUTPUT_SCALE,
            meanAbsoluteError = Double.NaN,
            withinOneUnitRatio = accuracy,
            valueCount = 0
        ) else ValidationMetrics.EMPTY
    }

    private fun saveCheckpoint(
        context: Context,
        samples: Long,
        batches: Long,
        bestValidation: Double,
        lastLoss: Double,
        validation: ValidationMetrics
    ) {
        ModelManager.save(
            context = context,
            samples = samples,
            batches = batches,
            bestValidationMse = bestValidation,
            lastLoss = lastLoss,
            lastValidationMse = validation.normalizedMse,
            validationAccuracy = validation.withinOneUnitRatio
        )
    }

    private fun pauseReason(context: Context): String? {
        val battery = context.getSystemService(Context.BATTERY_SERVICE) as BatteryManager
        val level = battery.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY)
        if (level in 0..20) return "البطارية منخفضة ($level%)"

        val power = context.getSystemService(Context.POWER_SERVICE) as PowerManager
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q &&
            power.currentThermalStatus >= PowerManager.THERMAL_STATUS_MODERATE
        ) return "حرارة الجهاز مرتفعة؛ سيستأنف التدريب تلقائيًا بعد أن يبرد"
        if (power.isPowerSaveMode && level <= 35) return "وضع توفير الطاقة فعال والبطارية أقل من 35%"
        return null
    }

    private fun throttleDelay(context: Context): Long {
        val battery = context.getSystemService(Context.BATTERY_SERVICE) as BatteryManager
        val charging = Build.VERSION.SDK_INT >= Build.VERSION_CODES.M && battery.isCharging
        val power = context.getSystemService(Context.POWER_SERVICE) as PowerManager
        var delayMs = if (charging) 15L else 80L
        if (power.isPowerSaveMode) delayMs = max(delayMs, 250L)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q &&
            power.currentThermalStatus >= PowerManager.THERMAL_STATUS_LIGHT
        ) delayMs = max(delayMs, 160L)
        return delayMs
    }

    private fun suppliedTarget(values: DoubleArray): DoubleArray {
        val x = values.getOrElse(0) { 0.0 }
        val y = values.getOrElse(1) { 0.0 }
        require(x.isFinite() && y.isFinite()) { "قيم تدريب غير عددية" }
        return target(x, y)
    }

    private fun target(x: Double, y: Double) = doubleArrayOf(x / OUTPUT_SCALE, y / OUTPUT_SCALE)
}
