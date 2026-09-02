package com.example.equationsolver.ai

import android.content.Context
import android.os.BatteryManager
import android.os.Build
import android.os.PowerManager
import android.os.SystemClock
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt
import kotlin.random.Random

object TrainingEngine {
    const val BATCH_SIZE = 8
    const val EPOCHS = 3
    const val OUTPUT_SCALE = 100.0
    private const val VALIDATE_EVERY_BATCHES = 40L
    private const val PROGRESS_EVERY_BATCHES = 4L
    private const val CHECKPOINT_INTERVAL_MS = 3L * 60L * 1000L
    private const val VALIDATION_SET_SIZE = 160
    private const val CURRICULUM_ROUND_SAMPLES = 50_000L

    data class ValidationMetrics(
        val normalizedMse: Double,
        val rmse: Double,
        val meanAbsoluteError: Double,
        val withinOneUnitRatio: Double,
        val valueCount: Int
    ) { companion object { val EMPTY = ValidationMetrics(Double.NaN, Double.NaN, Double.NaN, Double.NaN, 0) } }

    data class Snapshot(
        val samples: Long = 0L,
        val batches: Long = 0L,
        val curriculumRound: Long = 0L,
        val loss: Double = Double.NaN,
        val bestValidationMse: Double = Double.POSITIVE_INFINITY,
        val validation: ValidationMetrics = ValidationMetrics.EMPTY,
        val lastEquation: String = "",
        val lastFamily: String = "linear-system",
        val gradientNorm: Double = Double.NaN,
        val paused: Boolean = false,
        val reason: String = ""
    )

    data class PhoneSample(val equation: String, val features: FloatArray, val target: DoubleArray)
    private data class ValidationBank(val items: List<PhoneSample>, val equations: Set<String>)

    @Volatile private var currentSnapshot = Snapshot()
    @Volatile var lastValidation: ValidationMetrics = ValidationMetrics.EMPTY
        private set
    @Volatile private var externalFileSessionActive = false
    @Volatile private var externalFileStopRequested = false
    private val validationBank: ValidationBank by lazy { buildValidationBank() }

    fun snapshot(): Snapshot = currentSnapshot
    fun isReservedForValidation(equation: String): Boolean = equation in validationBank.equations
    fun suggestEquation(): PhoneSample = generateSample(Random.Default)
    @Synchronized fun beginExternalFileSession(): Boolean { if (externalFileSessionActive) return false; externalFileSessionActive = true; externalFileStopRequested = false; return true }
    @Synchronized fun endExternalFileSession() { externalFileSessionActive = false; externalFileStopRequested = false }
    fun isExternalFileSessionActive(): Boolean = externalFileSessionActive
    fun requestExternalFileSessionStop() { externalFileStopRequested = true }
    fun isExternalFileStopRequested(): Boolean = externalFileStopRequested

    fun trainRandom(samples: Int, progress: (Int) -> Unit = {}) {
        val xs = ArrayList<FloatArray>(BATCH_SIZE); val ys = ArrayList<DoubleArray>(BATCH_SIZE)
        repeat(samples) { i ->
            val s = generateSample(Random.Default)
            if (s.equation in validationBank.equations) return@repeat
            xs += s.features; ys += s.target
            if (xs.size == BATCH_SIZE) { ModelManager.nn.trainBatch(xs, ys, 1e-5); xs.clear(); ys.clear() }
            if ((i + 1) % 500 == 0) progress(i + 1)
        }
    }

    suspend fun trainContinuous(context: Context, learningRate: Double = 1e-5, progress: (Snapshot) -> Unit = {}) {
        var samples = ModelManager.trainingSamples(context)
        var batches = ModelManager.trainingBatches(context)
        var bestValidation = ModelManager.bestValidationMse(context)
        var lastLoss = ModelManager.lastLoss(context)
        var validation = restoredValidation(context)
        var lastEquation = ""
        var lastCheckpointAt = SystemClock.elapsedRealtime()
        val xs = ArrayList<FloatArray>(BATCH_SIZE); val ys = ArrayList<DoubleArray>(BATCH_SIZE)
        publish(samples, batches, lastLoss, bestValidation, validation, lastEquation, false, "", progress)
        try {
            while (currentCoroutineContext().isActive) {
                val reason = pauseReason(context)
                if (reason != null) { publish(samples, batches, lastLoss, bestValidation, validation, lastEquation, true, reason, progress); delay(3000L); continue }
                val s = generateSample(Random.Default)
                if (s.equation in validationBank.equations) continue
                xs += s.features; ys += s.target; samples++; lastEquation = s.equation
                if (xs.size == BATCH_SIZE) {
                    lastLoss = ModelManager.nn.trainBatch(xs, ys, 1e-5)
                    xs.clear(); ys.clear(); batches++
                    if (batches % VALIDATE_EVERY_BATCHES == 0L) {
                        validation = evaluateFixedHoldout(); lastValidation = validation
                        if (validation.normalizedMse.isFinite()) bestValidation = min(bestValidation, validation.normalizedMse)
                    }
                    if (batches % PROGRESS_EVERY_BATCHES == 0L || batches == 1L) publish(samples, batches, lastLoss, bestValidation, validation, lastEquation, false, "", progress)
                    else currentSnapshot = snapshotOf(samples, batches, lastLoss, bestValidation, validation, lastEquation)
                    val now = SystemClock.elapsedRealtime()
                    if (now - lastCheckpointAt >= CHECKPOINT_INTERVAL_MS) { saveCheckpoint(context, samples, batches, bestValidation, lastLoss, validation); lastCheckpointAt = now }
                    delay(throttleDelay(context))
                }
            }
        } finally {
            currentSnapshot = snapshotOf(samples, batches, lastLoss, bestValidation, validation, lastEquation)
            try { saveCheckpoint(context, samples, batches, bestValidation, lastLoss, validation) } catch (_: Exception) { }
        }
    }

    fun trainFile(examples: List<Pair<String, DoubleArray>>, progress: (Int) -> Unit = {}) {
        val valid = examples.mapNotNull { pair ->
            try { PhoneSample(pair.first, LinearSystemCodec.parseSystem(pair.first).features, doubleArrayOf(pair.second.getOrElse(0){0.0}, pair.second.getOrElse(1){0.0})) }
            catch (_: Exception) { null }
        }.filterNot { it.equation in validationBank.equations }
        if (valid.isEmpty()) return
        repeat(EPOCHS) { epoch ->
            val shuffled = valid.shuffled()
            var pos = 0
            while (pos < shuffled.size) {
                if (externalFileStopRequested || Thread.currentThread().isInterrupted) return
                val chunk = shuffled.subList(pos, min(pos + BATCH_SIZE, shuffled.size))
                ModelManager.nn.trainBatch(chunk.map { it.features }, chunk.map { it.target }, 1e-5)
                pos += chunk.size
                if (pos % 1000 == 0) progress(epoch * shuffled.size + pos)
            }
        }
        lastValidation = evaluateFixedHoldout()
    }

    fun evaluateFixedHoldout(): ValidationMetrics {
        val items = validationBank.items
        val e = ModelManager.nn.evaluate(items.map { it.features }, items.map { it.target }, 1.0)
        return ValidationMetrics(
            normalizedMse = e.meanSquaredError / (OUTPUT_SCALE * OUTPUT_SCALE),
            rmse = sqrt(e.meanSquaredError),
            meanAbsoluteError = e.meanAbsoluteError,
            withinOneUnitRatio = e.withinToleranceRatio,
            valueCount = e.valueCount
        )
    }

    private fun buildValidationBank(): ValidationBank {
        val r = Random(0x0A16)
        val items = ArrayList<PhoneSample>(VALIDATION_SET_SIZE); val eqs = LinkedHashSet<String>()
        while (items.size < VALIDATION_SET_SIZE) { val s = generateSample(r); if (eqs.add(s.equation)) items += s }
        return ValidationBank(items, eqs)
    }

    private fun generateSample(r: Random): PhoneSample {
        while (true) {
            val denX = r.nextInt(1, 6); val denY = r.nextInt(1, 6)
            val x = r.nextInt(-80, 81) / denX.toDouble(); val y = r.nextInt(-80, 81) / denY.toDouble()
            var a = r.nextInt(-14, 15); var b = r.nextInt(-14, 15); var d = r.nextInt(-14, 15); var e = r.nextInt(-14, 15)
            if (a == 0 && b == 0 || d == 0 && e == 0) continue
            val det = a * e - b * d
            if (det == 0) continue
            val c = a * x + b * y; val f = d * x + e * y
            if (kotlin.math.abs(x) > 300 || kotlin.math.abs(y) > 300) continue
            val eq = formatEquation(a, b, c) + ";" + formatEquation(d, e, f)
            val features = LinearSystemCodec.featuresFromRows(a.toDouble(), b.toDouble(), c, d.toDouble(), e.toDouble(), f)
            return PhoneSample(eq, features, doubleArrayOf(x, y))
        }
    }

    private fun fmtTerm(v: Int, variable: String, first: Boolean = false): String {
        if (v == 0) return if (first) "0" else ""
        val sign = if (v < 0) "-" else if (first) "" else "+"
        val mag = kotlin.math.abs(v)
        return sign + (if (mag == 1) "" else mag.toString()) + variable
    }
    private fun formatEquation(a: Int, b: Int, c: Double): String {
        var lhs = fmtTerm(a, "x", first = true) + fmtTerm(b, "y")
        if (lhs == "0") lhs = "0"
        val rhs = if (kotlin.math.abs(c - kotlin.math.round(c)) < 1e-9) kotlin.math.round(c).toLong().toString()
        else "%.4f".format(java.util.Locale.US, c).trimEnd('0').trimEnd('.')
        return "$lhs=$rhs"
    }

    private fun publish(samples: Long, batches: Long, loss: Double, best: Double, validation: ValidationMetrics, equation: String, paused: Boolean, reason: String, progress: (Snapshot)->Unit) {
        val s = snapshotOf(samples,batches,loss,best,validation,equation,paused,reason); currentSnapshot=s; progress(s)
    }
    private fun snapshotOf(samples: Long,batches: Long,loss: Double,best: Double,validation: ValidationMetrics,equation: String,paused:Boolean=false,reason:String="") = Snapshot(
        samples=samples,batches=batches,curriculumRound=samples/CURRICULUM_ROUND_SAMPLES,loss=loss,bestValidationMse=best,validation=validation,lastEquation=equation,lastFamily="linear-system",gradientNorm=ModelManager.nn.lastGradientNorm,paused=paused,reason=reason)
    private fun restoredValidation(context: Context): ValidationMetrics {
        val mse=ModelManager.lastValidationMse(context); val acc=ModelManager.lastValidationAccuracy(context)
        return if (mse.isFinite() && mse>=0) ValidationMetrics(mse,sqrt(mse)*OUTPUT_SCALE,Double.NaN,acc,0) else ValidationMetrics.EMPTY
    }
    private fun saveCheckpoint(context: Context,samples:Long,batches:Long,best:Double,lastLoss:Double,v:ValidationMetrics) = ModelManager.save(context,samples,batches,best,lastLoss,v.normalizedMse,v.withinOneUnitRatio)
    private fun pauseReason(context: Context): String? {
        val battery=context.getSystemService(Context.BATTERY_SERVICE) as BatteryManager
        val level=battery.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY)
        if (level in 0..15) return "البطارية منخفضة ($level%)"
        val power=context.getSystemService(Context.POWER_SERVICE) as PowerManager
        if (Build.VERSION.SDK_INT>=Build.VERSION_CODES.Q && power.currentThermalStatus>=PowerManager.THERMAL_STATUS_MODERATE) return "حرارة الجهاز مرتفعة؛ سيستأنف التدريب بعد أن يبرد"
        return null
    }
    private fun throttleDelay(context: Context): Long {
        val battery=context.getSystemService(Context.BATTERY_SERVICE) as BatteryManager
        val charging=Build.VERSION.SDK_INT>=Build.VERSION_CODES.M && battery.isCharging
        val power=context.getSystemService(Context.POWER_SERVICE) as PowerManager
        var d=if(charging) 10L else 60L
        if(power.isPowerSaveMode) d=max(d,200L)
        if(Build.VERSION.SDK_INT>=Build.VERSION_CODES.Q && power.currentThermalStatus>=PowerManager.THERMAL_STATUS_LIGHT) d=max(d,140L)
        return d
    }
}
