package com.example.equationsolver.ai

import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.InputStream
import java.io.OutputStream
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sqrt

class OpenGrowthRsnnPhone {
    companion object {
        private const val MAGIC = 0x4F475231 // OGR1
        private const val VERSION = 1
        const val INPUT = 6
        const val HIDDEN = 160
        const val OUTPUT = 2
        const val TIME_STEPS = 25
        const val TOTAL_WEIGHTS = HIDDEN * INPUT + HIDDEN * HIDDEN + OUTPUT * HIDDEN
        const val TARGET_SCALE = 100.0
        private const val DECAY = 0.88f
        private const val THRESHOLD = 1.0f
        private const val SURROGATE_ALPHA = 2.0f
        private const val BETA1 = 0.9f
        private const val BETA2 = 0.999f
        private const val EPS = 1e-8f
        private const val WEIGHT_DECAY = 1e-4f
        private const val GRAD_CLIP = 5.0f
    }

    private val wIn = FloatArray(HIDDEN * INPUT)
    private val wRec = FloatArray(HIDDEN * HIDDEN)
    private val wOut = FloatArray(OUTPUT * HIDDEN)
    private val mIn = ByteArray(wIn.size)
    private val mRec = ByteArray(wRec.size)
    private val mOut = ByteArray(wOut.size)
    private val adamMIn = FloatArray(wIn.size)
    private val adamMRec = FloatArray(wRec.size)
    private val adamMOut = FloatArray(wOut.size)
    private val adamVIn = FloatArray(wIn.size)
    private val adamVRec = FloatArray(wRec.size)
    private val adamVOut = FloatArray(wOut.size)

    private val qIn = FloatArray(wIn.size)
    private val qRec = FloatArray(wRec.size)
    private val qOut = FloatArray(wOut.size)
    private var quantDirty = true

    private var step: Long = 0L
    var structuralCycle: Long = 0L
        private set
    var examplesSeen: Long = 0L
        private set
    var previousTrainingSeconds: Double = 0.0
        private set
    var sourceBestMae: Double = Double.NaN
        private set
    var sourceBestCycle: Long = 0L
        private set
    @Volatile var lastGradientNorm: Double = Double.NaN
        private set

    @Synchronized
    fun load(input: InputStream) {
        val d = DataInputStream(input.buffered())
        run {
            require(d.readInt() == MAGIC) { "Checkpoint Open-Growth غير صالح" }
            require(d.readInt() == VERSION) { "إصدار Checkpoint غير مدعوم" }
            require(d.readInt() == HIDDEN) { "حجم RSNN غير مطابق" }
            require(d.readInt() == TIME_STEPS) { "عدد خطوات RSNN غير مطابق" }
            val decay = d.readFloat(); val threshold = d.readFloat()
            require(abs(decay - DECAY) < 1e-6f && abs(threshold - THRESHOLD) < 1e-6f) { "ثوابت RSNN غير مطابقة" }
            step = d.readLong()
            structuralCycle = d.readLong()
            examplesSeen = d.readLong()
            previousTrainingSeconds = d.readDouble()
            sourceBestMae = d.readDouble()
            sourceBestCycle = d.readLong()
            readFloats(d, wIn); readFloats(d, wRec); readFloats(d, wOut)
            d.readFully(mIn); d.readFully(mRec); d.readFully(mOut)
            readFloats(d, adamMIn); readFloats(d, adamMRec); readFloats(d, adamMOut)
            readFloats(d, adamVIn); readFloats(d, adamVRec); readFloats(d, adamVOut)
        }
        applyMasks()
        quantDirty = true
    }

    @Synchronized
    fun save(output: OutputStream) {
        val d = DataOutputStream(output.buffered())
        run {
            d.writeInt(MAGIC); d.writeInt(VERSION); d.writeInt(HIDDEN); d.writeInt(TIME_STEPS)
            d.writeFloat(DECAY); d.writeFloat(THRESHOLD)
            d.writeLong(step); d.writeLong(structuralCycle); d.writeLong(examplesSeen)
            d.writeDouble(previousTrainingSeconds); d.writeDouble(sourceBestMae); d.writeLong(sourceBestCycle)
            writeFloats(d, wIn); writeFloats(d, wRec); writeFloats(d, wOut)
            d.write(mIn); d.write(mRec); d.write(mOut)
            writeFloats(d, adamMIn); writeFloats(d, adamMRec); writeFloats(d, adamMOut)
            writeFloats(d, adamVIn); writeFloats(d, adamVRec); writeFloats(d, adamVOut)
            d.flush()
        }
    }

    fun parameterCount(): Int = TOTAL_WEIGHTS
    fun activeWeightCount(): Int = mIn.count { it.toInt() != 0 } + mRec.count { it.toInt() != 0 } + mOut.count { it.toInt() != 0 }
    fun optimizerStep(): Int = min(step, Int.MAX_VALUE.toLong()).toInt()
    fun optimizerStepLong(): Long = step

    @Synchronized
    fun predictRaw(features: FloatArray): DoubleArray {
        require(features.size == INPUT)
        ensureQuantized()
        val pred = forward(features, null, null)
        return doubleArrayOf(pred[0] * TARGET_SCALE, pred[1] * TARGET_SCALE)
    }

    data class EvalResult(
        val meanSquaredError: Double,
        val meanAbsoluteError: Double,
        val withinToleranceRatio: Double,
        val valueCount: Int
    )

    @Synchronized
    fun evaluate(features: List<FloatArray>, targetsRaw: List<DoubleArray>, toleranceRaw: Double = 1.0): EvalResult {
        require(features.size == targetsRaw.size)
        if (features.isEmpty()) return EvalResult(Double.NaN, Double.NaN, Double.NaN, 0)
        ensureQuantized()
        var sq = 0.0; var ae = 0.0; var within = 0; var n = 0
        for (i in features.indices) {
            val p = forward(features[i], null, null)
            for (o in 0 until OUTPUT) {
                val raw = p[o] * TARGET_SCALE
                val e = raw - targetsRaw[i][o]
                sq += e * e; ae += abs(e); if (abs(e) <= toleranceRaw) within++; n++
            }
        }
        return EvalResult(sq / n, ae / n, within.toDouble() / n, n)
    }

    @Synchronized
    fun trainBatch(features: List<FloatArray>, targetsRaw: List<DoubleArray>, requestedLr: Double): Double {
        require(features.size == targetsRaw.size && features.isNotEmpty())
        ensureQuantized()
        val gIn = FloatArray(wIn.size); val gRec = FloatArray(wRec.size); val gOut = FloatArray(wOut.size)
        var lossSum = 0.0
        val batch = features.size
        for (b in features.indices) {
            val x = features[b]
            require(x.size == INPUT)
            val pre = FloatArray(TIME_STEPS * HIDDEN)
            val spikes = FloatArray(TIME_STEPS * HIDDEN)
            val pred = forward(x, pre, spikes)
            val gPred = FloatArray(OUTPUT)
            for (o in 0 until OUTPUT) {
                val target = (targetsRaw[b][o] / TARGET_SCALE).toFloat()
                val diff = pred[o] - target
                val ad = abs(diff)
                lossSum += if (ad < 1f) 0.5 * diff * diff else ad - 0.5
                val base = if (ad < 1f) diff else if (diff >= 0f) 1f else -1f
                gPred[o] = base / (batch * OUTPUT).toFloat()
            }
            backwardSample(x, pre, spikes, gPred, gIn, gRec, gOut)
        }
        var norm2 = 0.0
        for (v in gIn) norm2 += v.toDouble() * v
        for (v in gRec) norm2 += v.toDouble() * v
        for (v in gOut) norm2 += v.toDouble() * v
        val norm = sqrt(norm2)
        lastGradientNorm = norm
        if (norm.isFinite() && norm > GRAD_CLIP) {
            val s = (GRAD_CLIP / norm).toFloat()
            scaleInPlace(gIn, s); scaleInPlace(gRec, s); scaleInPlace(gOut, s)
        }
        maskGradient(gIn, mIn); maskGradient(gRec, mRec); maskGradient(gOut, mOut)
        val lr = requestedLr.coerceIn(1e-6, 1e-4).toFloat()
        step++
        adamw(wIn, gIn, adamMIn, adamVIn, mIn, lr, step)
        adamw(wRec, gRec, adamMRec, adamVRec, mRec, lr, step)
        adamw(wOut, gOut, adamMOut, adamVOut, mOut, lr, step)
        applyMasks(); quantDirty = true
        examplesSeen += batch.toLong()
        return lossSum / (batch * OUTPUT).toDouble()
    }

    private fun forward(x: FloatArray, preStore: FloatArray?, spikeStore: FloatArray?): FloatArray {
        val syn = FloatArray(HIDDEN)
        for (h in 0 until HIDDEN) {
            var s = 0f; val base = h * INPUT
            for (i in 0 until INPUT) s += qIn[base + i] * x[i]
            syn[h] = s
        }
        var mem = FloatArray(HIDDEN)
        var prevSpikes = FloatArray(HIDDEN)
        val out = FloatArray(OUTPUT)
        for (t in 0 until TIME_STEPS) {
            val current = FloatArray(HIDDEN)
            val currentSpikes = FloatArray(HIDDEN)
            for (h in 0 until HIDDEN) {
                var rec = 0f; val base = h * HIDDEN
                for (j in 0 until HIDDEN) rec += qRec[base + j] * prevSpikes[j]
                val p = DECAY * mem[h] + syn[h] + rec
                current[h] = p
                val sp = if (p - THRESHOLD >= 0f) 1f else 0f
                currentSpikes[h] = sp
                mem[h] = p - sp * THRESHOLD
                if (preStore != null) preStore[t * HIDDEN + h] = p
                if (spikeStore != null) spikeStore[t * HIDDEN + h] = sp
            }
            for (o in 0 until OUTPUT) {
                var s = 0f; val base = o * HIDDEN
                for (h in 0 until HIDDEN) s += qOut[base + h] * currentSpikes[h]
                out[o] += s
            }
            prevSpikes = currentSpikes
        }
        for (o in 0 until OUTPUT) out[o] /= TIME_STEPS.toFloat()
        return out
    }

    private fun backwardSample(
        x: FloatArray,
        pre: FloatArray,
        spikes: FloatArray,
        gPred: FloatArray,
        gIn: FloatArray,
        gRec: FloatArray,
        gOut: FloatArray
    ) {
        val gOutStep = FloatArray(OUTPUT) { gPred[it] / TIME_STEPS.toFloat() }
        for (t in 0 until TIME_STEPS) {
            val off = t * HIDDEN
            for (o in 0 until OUTPUT) {
                val base = o * HIDDEN
                val go = gOutStep[o]
                for (h in 0 until HIDDEN) if (spikes[off + h] != 0f) gOut[base + h] += go
            }
        }

        var gPost = FloatArray(HIDDEN)
        var gSpikeFuture = FloatArray(HIDDEN)
        val gSyn = FloatArray(HIDDEN)
        for (t in TIME_STEPS - 1 downTo 0) {
            val off = t * HIDDEN
            val gPre = FloatArray(HIDDEN)
            for (h in 0 until HIDDEN) {
                var fromOut = 0f
                for (o in 0 until OUTPUT) fromOut += qOut[o * HIDDEN + h] * gOutStep[o]
                val gSpike = fromOut + gSpikeFuture[h] - THRESHOLD * gPost[h]
                val z = pre[off + h] - THRESHOLD
                val surrogate = 1f / ((1f + SURROGATE_ALPHA * abs(z)) * (1f + SURROGATE_ALPHA * abs(z)))
                val gp = gPost[h] + gSpike * surrogate
                gPre[h] = gp
                gSyn[h] += gp
            }
            if (t > 0) {
                val prevOff = (t - 1) * HIDDEN
                for (h in 0 until HIDDEN) {
                    val gp = gPre[h]; val base = h * HIDDEN
                    if (gp != 0f) for (j in 0 until HIDDEN) {
                        val sp = spikes[prevOff + j]
                        if (sp != 0f) gRec[base + j] += gp * sp
                    }
                }
            }
            val nextSpikeFuture = FloatArray(HIDDEN)
            for (j in 0 until HIDDEN) {
                var s = 0f
                for (h in 0 until HIDDEN) s += qRec[h * HIDDEN + j] * gPre[h]
                nextSpikeFuture[j] = s
            }
            val nextPost = FloatArray(HIDDEN)
            for (h in 0 until HIDDEN) nextPost[h] = DECAY * gPre[h]
            gSpikeFuture = nextSpikeFuture
            gPost = nextPost
        }
        for (h in 0 until HIDDEN) {
            val gp = gSyn[h]; val base = h * INPUT
            for (i in 0 until INPUT) gIn[base + i] += gp * x[i]
        }
    }

    private fun ensureQuantized() {
        if (!quantDirty) return
        quantizeRows(wIn, mIn, qIn, HIDDEN, INPUT)
        quantizeRows(wRec, mRec, qRec, HIDDEN, HIDDEN)
        quantizeRows(wOut, mOut, qOut, OUTPUT, HIDDEN)
        quantDirty = false
    }

    private fun quantizeRows(source: FloatArray, mask: ByteArray, target: FloatArray, rows: Int, cols: Int) {
        for (r in 0 until rows) {
            val base = r * cols; var mx = 0f
            for (c in 0 until cols) if (mask[base + c].toInt() != 0) mx = max(mx, abs(source[base + c]))
            val scale = if (mx > 0f) mx / 127f else 1f
            for (c in 0 until cols) {
                val idx = base + c
                if (mask[idx].toInt() == 0) target[idx] = 0f
                else {
                    val q = (source[idx] / scale).toDouble().let { kotlin.math.round(it) }.toInt().coerceIn(-127, 127)
                    target[idx] = q * scale
                }
            }
        }
    }

    private fun adamw(w: FloatArray, g: FloatArray, m: FloatArray, v: FloatArray, mask: ByteArray, lr: Float, step: Long) {
        val b1corr = (1.0 - BETA1.toDouble().pow(step.toDouble())).toFloat()
        val b2corr = (1.0 - BETA2.toDouble().pow(step.toDouble())).toFloat()
        for (i in w.indices) {
            if (mask[i].toInt() == 0) { w[i] = 0f; m[i] = 0f; v[i] = 0f; continue }
            val gi = g[i]
            m[i] = BETA1 * m[i] + (1f - BETA1) * gi
            v[i] = BETA2 * v[i] + (1f - BETA2) * gi * gi
            val mh = m[i] / b1corr
            val vh = v[i] / b2corr
            w[i] *= (1f - lr * WEIGHT_DECAY)
            w[i] -= (lr * mh / (sqrt(vh.toDouble()).toFloat() + EPS))
        }
    }

    private fun applyMasks() {
        for (i in wIn.indices) if (mIn[i].toInt() == 0) wIn[i] = 0f
        for (i in wRec.indices) if (mRec[i].toInt() == 0) wRec[i] = 0f
        for (i in wOut.indices) if (mOut[i].toInt() == 0) wOut[i] = 0f
    }

    private fun maskGradient(g: FloatArray, mask: ByteArray) { for (i in g.indices) if (mask[i].toInt() == 0) g[i] = 0f }
    private fun scaleInPlace(a: FloatArray, s: Float) { for (i in a.indices) a[i] *= s }
    private fun readFloats(d: DataInputStream, a: FloatArray) { for (i in a.indices) a[i] = d.readFloat() }
    private fun writeFloats(d: DataOutputStream, a: FloatArray) { for (v in a) d.writeFloat(v) }
}
