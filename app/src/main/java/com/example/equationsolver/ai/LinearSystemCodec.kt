package com.example.equationsolver.ai

import com.example.equationsolver.core.EquationParser
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.round

object LinearSystemCodec {
    private val parser = EquationParser()

    data class Parsed(val features: FloatArray, val canonicalText: String)

    fun parseSystem(raw: String): Parsed {
        val text = raw.trim().replace('،', ';').replace('\n', ';')
        val parts = text.split(';', ',').map { it.trim() }.filter { it.isNotEmpty() }
        require(parts.size == 2) { "أدخل معادلتين خطيتين وبينهما ; مثل: 2x+y=7;x-y=2" }
        val e1 = parser.parseLinearEquation(parts[0])
        val e2 = parser.parseLinearEquation(parts[1])
        val r1 = canonicalRow(e1.a, e1.b, e1.c)
        val r2 = canonicalRow(e2.a, e2.b, e2.c)
        val rows = if (compareRows(r1, r2) <= 0) arrayOf(r1, r2) else arrayOf(r2, r1)
        val det = rows[0][0] * rows[1][1] - rows[0][1] * rows[1][0]
        require(abs(det) > 1e-10) { "النظام مفرد أو قريب جدًا من المفرد ولا يملك حلاً وحيدًا مستقراً" }
        val out = FloatArray(6)
        for (i in 0..2) out[i] = rows[0][i].toFloat()
        for (i in 0..2) out[i + 3] = rows[1][i].toFloat()
        return Parsed(out, parts.joinToString(";"))
    }

    fun featuresFromRows(a: Double, b: Double, c: Double, d: Double, e: Double, f: Double): FloatArray {
        val r1 = canonicalRow(a, b, c)
        val r2 = canonicalRow(d, e, f)
        val rows = if (compareRows(r1, r2) <= 0) arrayOf(r1, r2) else arrayOf(r2, r1)
        return floatArrayOf(
            rows[0][0].toFloat(), rows[0][1].toFloat(), rows[0][2].toFloat(),
            rows[1][0].toFloat(), rows[1][1].toFloat(), rows[1][2].toFloat()
        )
    }

    private fun canonicalRow(a0: Double, b0: Double, c0: Double): DoubleArray {
        val scale = max(abs(a0), max(abs(b0), abs(c0)))
        require(scale > 1e-12 && scale.isFinite()) { "معادلة صفرية أو غير صالحة" }
        var a = a0 / scale
        var b = b0 / scale
        var c = c0 / scale
        val first = if (abs(a) > 1e-12) a else b
        if (first < 0.0) { a = -a; b = -b; c = -c }
        return doubleArrayOf(a, b, c)
    }

    private fun compareRows(a: DoubleArray, b: DoubleArray): Int {
        for (i in 0..2) {
            val av = round(a[i] * 1e14) / 1e14
            val bv = round(b[i] * 1e14) / 1e14
            if (av < bv) return -1
            if (av > bv) return 1
        }
        return 0
    }
}
