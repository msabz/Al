package com.example.equationsolver.ai

import android.content.Context
import android.os.BatteryManager
import android.os.Build
import android.os.PowerManager
import com.example.equationsolver.data.EquationGenerator
import com.example.equationsolver.data.GeneratedEquationValidator
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive

object TrainingEngine {
    const val BATCH_SIZE = 24
    const val EPOCHS = 3
    const val VALIDATION_RATIO = 0.10
    private const val CHECKPOINT_EVERY_BATCHES = 50L

    data class Snapshot(
        val samples: Long,
        val batches: Long,
        val epoch: Long,
        val loss: Double,
        val bestValidationMse: Double
    )

    @Volatile private var currentSamples = 0L
    @Volatile private var currentBatches = 0L
    @Volatile private var currentEpoch = 0L
    @Volatile private var currentLoss = Double.NaN
    @Volatile private var currentBestValidation = Double.POSITIVE_INFINITY
    @Volatile var lastValidationMse: Double = Double.POSITIVE_INFINITY
        private set

    fun snapshot(): Snapshot = Snapshot(currentSamples, currentBatches, currentEpoch, currentLoss, currentBestValidation)

    fun trainRandom(samples: Int, progress: (Int) -> Unit = {}) {
        val inputs = ArrayList<IntArray>(BATCH_SIZE)
        val targets = ArrayList<DoubleArray>(BATCH_SIZE)
        for (i in 1..samples) {
            val sample = EquationGenerator.generate()
            if (!GeneratedEquationValidator.isValid(sample)) continue
            inputs += MathTokenizer.tokenize(sample.equation)
            targets += target(sample.x, sample.y)
            if (inputs.size == BATCH_SIZE || i == samples) {
                if (inputs.isNotEmpty()) ModelManager.nn.trainBatch(inputs.toTypedArray(), targets.toTypedArray(), 0.0007)
                inputs.clear(); targets.clear()
            }
            if (i % 500 == 0) progress(i)
        }
    }

    suspend fun trainContinuous(
        context: Context,
        learningRate: Double = 0.0007,
        progress: (samples: Long, batches: Long, epoch: Long, loss: Double, validationMse: Double, paused: Boolean, reason: String) -> Unit = { _, _, _, _, _, _, _ -> }
    ) {
        var samples = ModelManager.trainingSamples(context)
        var batches = ModelManager.trainingBatches(context)
        var epoch = samples / 1000L
        var bestValidation = ModelManager.bestValidationMse(context)
        var lastLoss = ModelManager.lastLoss(context)
        if (!bestValidation.isFinite()) bestValidation = Double.POSITIVE_INFINITY
        setSnapshot(samples, batches, epoch, lastLoss, bestValidation)

        val inputs = ArrayList<IntArray>(BATCH_SIZE)
        val targets = ArrayList<DoubleArray>(BATCH_SIZE)
        try {
            while (currentCoroutineContext().isActive) {
                val reason = pauseReason(context)
                if (reason != null) {
                    progress(samples, batches, epoch, lastLoss, bestValidation, true, reason)
                    delay(3000L)
                    continue
                }

                val sample = EquationGenerator.generate()
                if (!GeneratedEquationValidator.isValid(sample)) continue
                inputs += MathTokenizer.tokenize(sample.equation)
                targets += target(sample.x, sample.y)
                samples++

                if (inputs.size == BATCH_SIZE) {
                    lastLoss = ModelManager.nn.trainBatch(inputs.toTypedArray(), targets.toTypedArray(), learningRate)
                    batches++
                    epoch = samples / 1000L
                    inputs.clear(); targets.clear()

                    if (batches % 5L == 0L) {
                        lastValidationMse = validationOnFreshRandomSet(120)
                        if (lastValidationMse.isFinite() && lastValidationMse < bestValidation) bestValidation = lastValidationMse
                        progress(samples, batches, epoch, lastLoss, lastValidationMse, false, "")
                    }
                    setSnapshot(samples, batches, epoch, lastLoss, bestValidation)

                    if (batches % CHECKPOINT_EVERY_BATCHES == 0L) {
                        ModelManager.save(context, samples, batches, bestValidation, lastLoss)
                    }
                }
            }
        } finally {
            setSnapshot(samples, batches, epoch, lastLoss, bestValidation)
            try { ModelManager.save(context, samples, batches, bestValidation, lastLoss) } catch (_: Exception) { }
        }
    }

    private fun pauseReason(context: Context): String? {
        val battery = context.getSystemService(Context.BATTERY_SERVICE) as BatteryManager
        val level = battery.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY)
        if (level in 0..15) return "البطارية منخفضة ($level%)"

        val power = context.getSystemService(Context.POWER_SERVICE) as PowerManager
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q && power.currentThermalStatus >= PowerManager.THERMAL_STATUS_MODERATE) {
            return "حرارة الجهاز مرتفعة؛ التدريب متوقف لحماية الهاتف"
        }
        if (power.isPowerSaveMode && level <= 25) return "وضع توفير الطاقة فعال"
        return null
    }

    fun trainFile(examples: List<Pair<String, DoubleArray>>, progress: (Int) -> Unit = {}) {
        if (examples.isEmpty()) return
        val indices = IntArray(examples.size) { it }
        indices.shuffle()
        val validationCount = if (examples.size < 10) 0 else maxOf(1, (examples.size * VALIDATION_RATIO).toInt())
        val trainCount = examples.size - validationCount

        repeat(EPOCHS) { epoch ->
            val inputs = ArrayList<IntArray>(BATCH_SIZE)
            val targets = ArrayList<DoubleArray>(BATCH_SIZE)
            for (position in 0 until trainCount) {
                val pair = examples[indices[position]]
                inputs += MathTokenizer.tokenize(pair.first)
                targets += suppliedTarget(pair.second)
                if (inputs.size == BATCH_SIZE || position == trainCount - 1) {
                    ModelManager.nn.trainBatch(inputs.toTypedArray(), targets.toTypedArray(), 0.0007)
                    inputs.clear(); targets.clear()
                }
                if ((position + 1) % 1000 == 0) progress(epoch * trainCount + position + 1)
            }
            if (validationCount > 0) lastValidationMse = validationMse(examples, indices, trainCount)
        }
    }

    private fun validationOnFreshRandomSet(size: Int): Double {
        val inputs = ArrayList<IntArray>(size)
        val targets = ArrayList<DoubleArray>(size)
        var attempts = 0
        while (inputs.size < size && attempts < size * 5) {
            attempts++
            val sample = EquationGenerator.generate()
            if (!GeneratedEquationValidator.isValid(sample)) continue
            inputs += MathTokenizer.tokenize(sample.equation)
            targets += target(sample.x, sample.y)
        }
        return if (inputs.isEmpty()) Double.POSITIVE_INFINITY else ModelManager.nn.meanSquaredError(inputs, targets)
    }

    private fun validationMse(examples: List<Pair<String, DoubleArray>>, indices: IntArray, start: Int): Double {
        if (start >= indices.size) return Double.NaN
        val inputs = ArrayList<IntArray>()
        val targets = ArrayList<DoubleArray>()
        for (position in start until indices.size) {
            val pair = examples[indices[position]]
            inputs += MathTokenizer.tokenize(pair.first)
            targets += suppliedTarget(pair.second)
        }
        return ModelManager.nn.meanSquaredError(inputs, targets)
    }

    private fun suppliedTarget(values: DoubleArray): DoubleArray = doubleArrayOf(
        values.getOrElse(0) { 0.0 } / 100.0,
        values.getOrElse(1) { 0.0 } / 100.0
    )

    private fun target(x: Double, y: Double) = doubleArrayOf(x / 100.0, y / 100.0)

    private fun setSnapshot(samples: Long, batches: Long, epoch: Long, loss: Double, bestValidation: Double) {
        currentSamples = samples
        currentBatches = batches
        currentEpoch = epoch
        currentLoss = loss
        currentBestValidation = bestValidation
    }
}
