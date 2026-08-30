package com.example.equationsolver.core

import kotlin.math.abs

/** Ground-truth/comparison engine. The neural model never calls this from ModelManager.predict(). */
object MathTeacher {
    data class Answer(
        val type: String,
        val summary: String,
        val x: Double? = null,
        val y: Double? = null,
        val roots: List<Double> = emptyList(),
        val steps: List<String> = emptyList(),
        val supported: Boolean = true
    )

    fun solve(input: String): Answer {
        val normalized = ArabicEquationNormalizer.normalize(input)
        if (normalized.contains(';')) return fromUniversal(UniversalEquationSolver.solve(normalized))

        try {
            val analysis = UniversalEquationSolver.analyze(normalized)
            if (analysis.equationCount == 1 && analysis.maxDegree <= 2) {
                val exact = UniversalEquationSolver.solve(normalized)
                if (exact.exact || exact.x != null || exact.y != null) return fromUniversal(exact)
            }
        } catch (_: Exception) { }

        val hasX = Regex("(^|[^a-z])x([^a-z]|$)").containsMatchIn(normalized.lowercase()) || normalized.lowercase().contains('x')
        val hasY = normalized.lowercase().contains('y')
        if (hasX && hasY) return Answer("غير مدعوم", "المعادلات غير الخطية متعددة المتغيرات غير مدعومة في المصحح الحالي.", supported = false)
        if (!hasX && !hasY) return Answer("غير مدعوم", "لا يوجد متغير x أو y في المعادلة.", supported = false)

        val roots = findRoots(normalized, solveY = hasY && !hasX)
        if (roots.isEmpty()) return Answer(
            "عددي",
            "لم يتم العثور على جذر حقيقي ضمن المجال [-100, 100].",
            steps = listOf("تم فحص المعادلة عدديًا ضمن المجال [-100, 100]."),
            supported = true
        )

        val primary = roots.minByOrNull { abs(it) } ?: roots.first()
        val label = if (hasY && !hasX) "y" else "x"
        val summary = if (roots.size == 1) "$label = ${fmt(primary)}"
        else roots.joinToString(prefix = "$label ∈ {", postfix = "}") { fmt(it) }
        return Answer(
            type = "معادلة عددية عامة",
            summary = summary,
            x = if (label == "x") primary else null,
            y = if (label == "y") primary else null,
            roots = roots,
            steps = listOf(
                "تم تقييم طرفي المعادلة عدديًا.",
                "تم البحث عن تغيرات الإشارة ونقاط قريبة من الصفر.",
                "الجذر الرئيسي للمقارنة مع النموذج هو الأقرب إلى الصفر: ${fmt(primary)}"
            )
        )
    }

    private fun fromUniversal(result: UniversalEquationSolver.Result): Answer {
        val roots = buildList { if (result.x != null) add(result.x) }
        return Answer(result.type, result.summary, result.x, result.y, roots, result.steps, result.exact || result.x != null || result.y != null)
    }

    private fun findRoots(equation: String, solveY: Boolean): List<Double> {
        val roots = mutableListOf<Double>()
        val min = -100.0
        val max = 100.0
        val step = 0.1
        var a = min
        var fa = safeResidual(equation, a, solveY)
        while (a < max) {
            val b = minOf(max, a + step)
            val fb = safeResidual(equation, b, solveY)
            if (fa != null && abs(fa) < 1e-7) addRoot(roots, refineNewton(equation, a, solveY))
            if (fa != null && fb != null && fa * fb < 0.0) {
                val root = bisect(equation, a, b, solveY)
                if (root != null) addRoot(roots, root)
            }
            if (roots.size >= 12) break
            a = b
            fa = fb
        }
        return roots.sorted()
    }

    private fun safeResidual(equation: String, value: Double, solveY: Boolean): Double? = try {
        val r = if (solveY) MathExpressionEvaluator.residual(equation, y = value)
        else MathExpressionEvaluator.residual(equation, x = value)
        r.takeIf { it.isFinite() && abs(it) < 1e12 }
    } catch (_: Exception) { null }

    private fun bisect(equation: String, left: Double, right: Double, solveY: Boolean): Double? {
        var lo = left
        var hi = right
        var flo = safeResidual(equation, lo, solveY) ?: return null
        repeat(60) {
            val mid = (lo + hi) * 0.5
            val fm = safeResidual(equation, mid, solveY) ?: return null
            if (abs(fm) < 1e-10) return mid
            if (flo * fm <= 0.0) hi = mid else { lo = mid; flo = fm }
        }
        val root = (lo + hi) * 0.5
        val residual = safeResidual(equation, root, solveY) ?: return null
        return root.takeIf { abs(residual) < 1e-5 }
    }

    private fun refineNewton(equation: String, start: Double, solveY: Boolean): Double {
        var x = start
        repeat(15) {
            val f = safeResidual(equation, x, solveY) ?: return x
            if (abs(f) < 1e-10) return x
            val h = 1e-5 * (1.0 + abs(x))
            val fp = safeResidual(equation, x + h, solveY) ?: return x
            val fm = safeResidual(equation, x - h, solveY) ?: return x
            val derivative = (fp - fm) / (2.0 * h)
            if (abs(derivative) < 1e-12) return x
            val next = x - f / derivative
            if (!next.isFinite() || next !in -100.0..100.0) return x
            x = next
        }
        return x
    }

    private fun addRoot(roots: MutableList<Double>, value: Double) {
        if (!value.isFinite() || value !in -100.0001..100.0001) return
        if (roots.none { abs(it - value) < 1e-4 }) roots += value
    }

    private fun fmt(value: Double): String = if (abs(value) < 1e-10) "0" else "%.8f".format(java.util.Locale.US, value).trimEnd('0').trimEnd('.')
}
