package com.example.equationsolver.core

import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Unified local equation engine.
 * Supports linear/quadratic single-variable equations and 2x2 linear systems.
 * It also produces human-readable solution steps and a simple equation type.
 */
object UniversalEquationSolver {
    private const val EPS = 1e-10

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
        val maxDegree: Int = maxOf(if (abs(first.x2) > EPS) 2 else if (abs(first.x) > EPS) 1 else 0,
            if (abs(second.x2) > EPS) 2 else if (abs(second.x) > EPS) 1 else 0)
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
            "صيغة غير معروفة"
        }
    }

    private fun solveSingle(p: Polynomial): Result {
        if (abs(p.y) > EPS && abs(p.x) > EPS) {
            return Result("غير مدعوم", "المعادلة تحتوي x و y معًا؛ أدخل معادلتين خطيتين للنظام.", listOf("اكتب معادلتين مفصولتين بـ ';'."), exact = false)
        }
        val variable = if (abs(p.x) > EPS || abs(p.x2) > EPS) 'x' else 'y'
        if (abs(p.x2) > EPS && variable == 'x') {
            val d = p.x * p.x - 4.0 * p.x2 * p.c
            if (d < -EPS) return Result("تربيعية", "لا يوجد حل حقيقي.", listOf("المميز Δ = %.6f".format(d), "بما أن Δ < 0 فلا توجد جذور حقيقية."))
            if (abs(d) <= EPS) {
                val root = -p.x / (2.0 * p.x2)
                return Result("تربيعية", "x = ${fmt(root)}", listOf(
                    "نستخدم الصيغة العامة.",
                    "Δ = b² − 4ac = 0",
                    "x = −b / 2a = ${fmt(root)}"
                ), x = root)
            }
            val s = sqrt(d)
            val r1 = (-p.x + s) / (2.0 * p.x2)
            val r2 = (-p.x - s) / (2.0 * p.x2)
            return Result("تربيعية", "x = ${fmt(r1)} أو x = ${fmt(r2)}", listOf(
                "الصيغة: ax² + bx + c = 0",
                "a = ${fmt(p.x2)}, b = ${fmt(p.x)}, c = ${fmt(p.c)}",
                "Δ = %.6f".format(d),
                "x₁ = ${fmt(r1)}",
                "x₂ = ${fmt(r2)}"
            ), x = r1)
        }
        val coefficient = if (variable == 'x') p.x else p.y
        if (abs(coefficient) <= EPS) {
            return if (abs(p.c) <= EPS) Result("خطية", "عدد لا نهائي من الحلول.", listOf("0 = 0، لذلك كل قيمة تحقق المعادلة."))
            else Result("خطية", "لا يوجد حل.", listOf("المعادلة تختزل إلى قيمة ثابتة غير صحيحة."))
        }
        val value = -p.c / coefficient
        return Result("خطية", "$variable = ${fmt(value)}", listOf(
            "نضع الحدود التي تحتوي $variable في جهة واحدة.",
            "${fmt(coefficient)}$variable = ${fmt(-p.c)}",
            "$variable = ${fmt(value)}"
        ), x = if (variable == 'x') value else null, y = if (variable == 'y') value else null)
    }

    private fun solveSystem(e1: Polynomial, e2: Polynomial): Result {
        require(abs(e1.x2) <= EPS && abs(e2.x2) <= EPS) { "النظام التربيعي المتعدد غير مدعوم حاليًا." }
        val det = e1.x * e2.y - e2.x * e1.y
        if (abs(det) <= EPS) {
            val same = abs(e1.x * e2.c - e2.x * e1.c) <= EPS && abs(e1.y * e2.c - e2.y * e1.c) <= EPS
            return if (same) Result("نظام خطي", "عدد لا نهائي من الحلول.", listOf("المعادلتان تمثلان نفس الخط."))
            else Result("نظام خطي", "لا يوجد حل.", listOf("المعادلتان متوازيتان ولا تتقاطعان."))
        }
        val x = (e1.c * e2.y - e2.c * e1.y) / det
        val y = (e1.x * e2.c - e2.x * e1.c) / det
        return Result("نظام خطي", "x = ${fmt(x)}\ny = ${fmt(y)}", listOf(
            "لدينا معادلتان خطيتان.",
            "نحسب المحدد Δ = ${fmt(det)}.",
            "x = ${fmt(x)}",
            "y = ${fmt(y)}",
            "نراجع بالتعويض للتأكد من الحل."
        ), x = x, y = y)
    }

    private fun parseEquation(equation: String): Polynomial {
        require(equation.count { it == '=' } == 1) { "يجب أن تحتوي كل معادلة على علامة '=' واحدة" }
        val parts = equation.split('=')
        val left = parsePolynomial(parts[0])
        val right = parsePolynomial(parts[1])
        return Polynomial(left.x2 - right.x2, left.x - right.x, left.y - right.y, left.c - right.c)
    }

    private fun parsePolynomial(expression: String): Polynomial {
        var s = expression.replace("*", "").replace(" ", "")
            .replace("²", "^2").lowercase()
        require(s.isNotEmpty()) { "طرف المعادلة فارغ" }
        if (!s.startsWith("+") && !s.startsWith("-")) s = "+$s"
        val terms = Regex("[+-][^+-]+?").findAll(s).map { it.value }.toList()
        var x2 = 0.0
        var x = 0.0
        var y = 0.0
        var c = 0.0
        for (term in terms) {
            require(term.length > 1) { "حد غير صالح: $term" }
            when {
                term.contains("xy") || term.contains("yx") -> error("الحدود xy غير مدعومة")
                term.endsWith("x^2") -> x2 += coefficient(term.removeSuffix("x^2"))
                term.endsWith("x") -> x += coefficient(term.removeSuffix("x"))
                term.endsWith("y") -> y += coefficient(term.removeSuffix("y"))
                else -> c += signedNumber(term)
            }
        }
        return Polynomial(x2, x, y, c)
    }

    private fun coefficient(raw: String): Double = when (raw) {
        "+", "" -> 1.0
        "-" -> -1.0
        else -> signedNumber(raw)
    }

    private fun signedNumber(raw: String): Double = raw.toDoubleOrNull()
        ?: throw IllegalArgumentException("قيمة غير صالحة: $raw")

    private fun normalize(value: String): String = value.trim()
        .replace('−', '-')
        .replace('×', '*')
        .replace("٫", ".")

    private fun Polynomial.maxVariableDegree(): Int = if (abs(x2) > EPS) 2 else if (abs(x) > EPS || abs(y) > EPS) 1 else 0

    private fun fmt(value: Double): String = if (abs(value) < 1e-9) "0" else "%.6f".format(value).trimEnd('0').trimEnd('.')
}
