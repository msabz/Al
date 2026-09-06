package com.example.equationsolver.core

import java.util.Locale
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Exact local solver for linear/quadratic equations and 2x2 linear systems.
 * It is used as ground truth/comparison for these supported families.
 */
object UniversalEquationSolver {
    /** Relative tolerance used only for cancellation-sensitive derived quantities. */
    private const val REL_EPS = 1e-14

    data class Polynomial(
        val x2: Double = 0.0,
        val x: Double = 0.0,
        val y: Double = 0.0,
        val c: Double = 0.0
    )

    data class Analysis(
        val equationCount: Int,
        val first: Polynomial,
        val second: Polynomial = Polynomial()
    ) {
        val maxDegree: Int = maxOf(
            if (first.x2 != 0.0) 2 else if (first.x != 0.0 || first.y != 0.0) 1 else 0,
            if (second.x2 != 0.0) 2 else if (second.x != 0.0 || second.y != 0.0) 1 else 0
        )
    }

    data class Result(
        val type: String,
        val summary: String,
        val steps: List<String>,
        val x: Double? = null,
        val y: Double? = null,
        val exact: Boolean = true
    )

    fun analyze(input: String): Analysis {
        val equations = normalize(input).split(';').map { it.trim() }.filter { it.isNotEmpty() }
        require(equations.size in 1..2) { "أدخل معادلة واحدة أو معادلتين مفصولتين بـ ';'" }
        val parsed = equations.map { parseEquation(it) }
        return Analysis(parsed.size, parsed[0], parsed.getOrElse(1) { Polynomial() })
    }

    fun solve(input: String): Result {
        return try {
            val a = analyze(input)
            when (a.equationCount) {
                1 -> solveSingle(a.first)
                2 -> solveSystem(a.first, a.second)
                else -> error("عدد المعادلات غير مدعوم")
            }
        } catch (e: IllegalArgumentException) {
            Result("خطأ", e.message ?: "صياغة المعادلة غير صحيحة", listOf("تحقق من كتابة المعادلة."), exact = false)
        }
    }

    fun equationType(input: String): String {
        return try {
            val a = analyze(input)
            when {
                a.equationCount == 2 && a.first.maxVariableDegree() <= 1 && a.second.maxVariableDegree() <= 1 -> "نظام معادلتين خطيتين"
                a.maxDegree >= 2 -> "معادلة تربيعية"
                a.first.y != 0.0 && a.first.x == 0.0 -> "معادلة خطية في y"
                else -> "معادلة خطية"
            }
        } catch (_: Exception) {
            "صيغة عامة"
        }
    }

    private fun solveSingle(p: Polynomial): Result {
        if (p.y != 0.0 && (p.x != 0.0 || p.x2 != 0.0)) {
            return Result("غير مدعوم", "المعادلة تحتوي x و y معًا؛ أدخل معادلتين خطيتين للنظام.", listOf("اكتب معادلتين مفصولتين بـ ';'."), exact = false)
        }
        val variable = if (p.x != 0.0 || p.x2 != 0.0) 'x' else 'y'
        if (p.x2 != 0.0 && variable == 'x') {
            // Normalize coefficients first so b² and 4ac do not overflow merely because
            // the same equation was multiplied by a large finite constant.
            val scale = maxOf(abs(p.x2), abs(p.x), abs(p.c))
            require(scale.isFinite() && scale > 0.0) { "معاملات المعادلة غير صالحة" }
            val qa = p.x2 / scale
            val qb = p.x / scale
            val qc = p.c / scale
            val b2 = qb * qb
            val fourAc = 4.0 * qa * qc
            val d = b2 - fourAc
            val dScale = abs(b2) + abs(fourAc)
            val dTolerance = if (dScale == 0.0) 0.0 else REL_EPS * dScale
            val originalD = p.x * p.x - 4.0 * p.x2 * p.c
            val dDescription = if (originalD.isFinite()) fmt(originalD)
            else "${fmt(d)} (بعد تطبيع المعاملات؛ Δ الأصلي خارج النطاق العددي)"

            if (d < -dTolerance) return Result("تربيعية", "لا يوجد حل حقيقي.", listOf("المميز Δ = $dDescription", "بما أن Δ < 0 فلا توجد جذور حقيقية."))
            if (abs(d) <= dTolerance) {
                val root = -qb / (2.0 * qa)
                return finiteRootResult(root) {
                    Result("تربيعية", "x = ${fmt(root)}", listOf(
                        "نستخدم الصيغة العامة.",
                        "المميز Δ = $dDescription",
                        "x = −b / 2a = ${fmt(root)}"
                    ), x = root)
                }
            }

            // Stable quadratic formula: using q avoids catastrophic cancellation in the
            // small root when |b| is much larger than |a| and |c|.
            val s = sqrt(d)
            val q = -0.5 * (qb + Math.copySign(s, qb))
            val r1 = q / qa
            val r2 = qc / q
            if (!r1.isFinite() || !r2.isFinite()) return nonFiniteSolution()
            val canonical = listOf(r1, r2).minWithOrNull(compareBy<Double> { abs(it) }.thenBy { it }) ?: r1
            return Result("تربيعية", "x = ${fmt(r1)} أو x = ${fmt(r2)}", listOf(
                "الصيغة: ax² + bx + c = 0",
                "a = ${fmt(p.x2)}, b = ${fmt(p.x)}, c = ${fmt(p.c)}",
                "المميز Δ = $dDescription",
                "x₁ = ${fmt(r1)}",
                "x₂ = ${fmt(r2)}"
            ), x = canonical)
        }
        val coefficient = if (variable == 'x') p.x else p.y
        if (coefficient == 0.0) {
            return if (p.c == 0.0) Result("خطية", "عدد لا نهائي من الحلول.", listOf("0 = 0، لذلك كل قيمة تحقق المعادلة."))
            else Result("خطية", "لا يوجد حل.", listOf("المعادلة تختزل إلى قيمة ثابتة غير صحيحة."))
        }
        val value = -p.c / coefficient
        if (!value.isFinite()) return nonFiniteSolution()
        return Result("خطية", "$variable = ${fmt(value)}", listOf(
            "نضع الحدود التي تحتوي $variable في جهة واحدة.",
            "${fmt(coefficient)}$variable = ${fmt(-p.c)}",
            "$variable = ${fmt(value)}"
        ), x = if (variable == 'x') value else null, y = if (variable == 'y') value else null)
    }

    private fun solveSystem(e1: Polynomial, e2: Polynomial): Result {
        require(e1.x2 == 0.0 && e2.x2 == 0.0) { "النظام التربيعي المتعدد غير مدعوم حاليًا." }

        // Normalize each row independently. This preserves the solution while avoiding
        // overflow in determinant/cross-product calculations for scaled equations.
        val n1 = normalizeRow(e1)
        val n2 = normalizeRow(e2)
        val detLeft = n1.x * n2.y
        val detRight = n2.x * n1.y
        val det = detLeft - detRight
        val detScale = abs(detLeft) + abs(detRight)
        if (nearlyZero(det, detScale)) {
            val crossXLeft = n1.x * n2.c
            val crossXRight = n2.x * n1.c
            val crossYLeft = n1.y * n2.c
            val crossYRight = n2.y * n1.c
            val same = nearlyZero(crossXLeft - crossXRight, abs(crossXLeft) + abs(crossXRight)) &&
                nearlyZero(crossYLeft - crossYRight, abs(crossYLeft) + abs(crossYRight))
            return if (same) Result("نظام خطي", "عدد لا نهائي من الحلول.", listOf("المعادلتان تمثلان نفس الخط."))
            else Result("نظام خطي", "لا يوجد حل.", listOf("المعادلتان متوازيتان ولا تتقاطعان."))
        }
        val x = (-n1.c * n2.y + n2.c * n1.y) / det
        val y = (-n1.x * n2.c + n2.x * n1.c) / det
        if (!x.isFinite() || !y.isFinite()) return nonFiniteSolution()
        val originalDet = e1.x * e2.y - e2.x * e1.y
        val determinantStep = if (originalDet.isFinite()) "نحسب المحدد Δ = ${fmt(originalDet)}."
        else "نحسب المحدد بعد تطبيع المعاملات Δ' = ${fmt(det)} لأن Δ الأصلي خارج النطاق العددي."
        return Result("نظام خطي", "x = ${fmt(x)}\ny = ${fmt(y)}", listOf(
            "لدينا معادلتان خطيتان.",
            determinantStep,
            "x = ${fmt(x)}",
            "y = ${fmt(y)}",
            "نراجع بالتعويض للتأكد من الحل."
        ), x = x, y = y)
    }

    private fun normalizeRow(p: Polynomial): Polynomial {
        val scale = maxOf(abs(p.x), abs(p.y), abs(p.c))
        if (scale == 0.0) return p
        require(scale.isFinite()) { "معاملات المعادلة غير صالحة" }
        return Polynomial(x = p.x / scale, y = p.y / scale, c = p.c / scale)
    }

    private fun nearlyZero(value: Double, scale: Double): Boolean {
        if (scale == 0.0) return value == 0.0
        return abs(value) <= REL_EPS * scale
    }

    private fun finiteRootResult(root: Double, result: () -> Result): Result =
        if (root.isFinite()) result() else nonFiniteSolution()

    private fun nonFiniteSolution(): Result = Result(
        "خطأ",
        "الحل خارج نطاق الأعداد المحدودة التي يمكن للتطبيق تمثيلها.",
        listOf("أعد كتابة المعادلة بمقياس عددي أصغر."),
        exact = false
    )

    private fun parseEquation(equation: String): Polynomial {
        require(equation.count { it == '=' } == 1) { "يجب أن تحتوي كل معادلة على علامة '=' واحدة" }
        val parts = equation.split('=')
        val left = parsePolynomial(parts[0])
        val right = parsePolynomial(parts[1])
        return Polynomial(
            x2 = cleanCancellation(left.x2 - right.x2, abs(left.x2) + abs(right.x2)),
            x = cleanCancellation(left.x - right.x, abs(left.x) + abs(right.x)),
            y = cleanCancellation(left.y - right.y, abs(left.y) + abs(right.y)),
            c = cleanCancellation(left.c - right.c, abs(left.c) + abs(right.c))
        )
    }

    private fun parsePolynomial(expression: String): Polynomial {
        var s = expression.replace("*", "").replace(" ", "").replace("²", "^2").lowercase()
        require(s.isNotEmpty()) { "طرف المعادلة فارغ" }
        if (!s.startsWith("+") && !s.startsWith("-")) s = "+$s"
        val terms = Regex("[+-][^+-]+").findAll(s).map { it.value }.toList()
        require(terms.isNotEmpty()) { "لا توجد حدود صالحة" }
        var x2 = 0.0
        var x = 0.0
        var y = 0.0
        var c = 0.0
        var x2Magnitude = 0.0
        var xMagnitude = 0.0
        var yMagnitude = 0.0
        var cMagnitude = 0.0
        for (term in terms) {
            require(term.length > 1) { "حد غير صالح: $term" }
            when {
                term.contains("xy") || term.contains("yx") -> throw IllegalArgumentException("الحدود xy غير مدعومة")
                term.endsWith("x^2") -> {
                    val value = coefficient(term.removeSuffix("x^2"))
                    x2 += value
                    x2Magnitude += abs(value)
                }
                term.endsWith("x") -> {
                    val value = coefficient(term.removeSuffix("x"))
                    x += value
                    xMagnitude += abs(value)
                }
                term.endsWith("y") -> {
                    val value = coefficient(term.removeSuffix("y"))
                    y += value
                    yMagnitude += abs(value)
                }
                else -> {
                    val value = signedNumber(term)
                    c += value
                    cMagnitude += abs(value)
                }
            }
        }
        require(x2.isFinite() && x.isFinite() && y.isFinite() && c.isFinite()) { "معاملات المعادلة غير محدودة" }
        return Polynomial(
            cleanCancellation(x2, x2Magnitude),
            cleanCancellation(x, xMagnitude),
            cleanCancellation(y, yMagnitude),
            cleanCancellation(c, cMagnitude)
        )
    }

    private fun cleanCancellation(value: Double, magnitude: Double): Double {
        if (value == 0.0 || magnitude == 0.0) return 0.0
        return if (abs(value) <= REL_EPS * magnitude) 0.0 else value
    }

    private fun coefficient(raw: String): Double = when (raw) {
        "+", "" -> 1.0
        "-" -> -1.0
        else -> signedNumber(raw)
    }

    private fun signedNumber(raw: String): Double {
        val value = raw.toDoubleOrNull() ?: throw IllegalArgumentException("قيمة غير صالحة: $raw")
        require(value.isFinite()) { "القيمة يجب أن تكون عددًا محدودًا: $raw" }
        return value
    }

    private fun normalize(value: String): String = ArabicEquationNormalizer.normalize(value)

    private fun Polynomial.maxVariableDegree(): Int = if (x2 != 0.0) 2 else if (x != 0.0 || y != 0.0) 1 else 0

    private fun fmt(value: Double): String {
        if (!value.isFinite()) return value.toString()
        if (value == 0.0) return "0"
        val magnitude = abs(value)
        if (magnitude >= 1e-7 && magnitude < 1e9) {
            return String.format(Locale.US, "%.8f", value).trimEnd('0').trimEnd('.')
        }
        val parts = String.format(Locale.US, "%.8e", value).split('e')
        val mantissa = parts[0].trimEnd('0').trimEnd('.')
        val exponent = parts[1].toInt().toString()
        return "${mantissa}e$exponent"
    }
}
