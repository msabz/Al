package com.example.equationsolver.core

class EquationParser {
    fun parseLinearEquation(equation: String): LinearEquation {
        val normalized = equation.replace(" ", "").lowercase()
        require(normalized.count { it == '=' } == 1) { "المعادلة يجب أن تحتوي على علامة '=' واحدة" }
        val parts = normalized.split('=')
        val left = parseExpression(parts[0])
        val right = parseExpression(parts[1])
        return LinearEquation(
            a = left.a - right.a,
            b = left.b - right.b,
            c = right.c - left.c
        )
    }

    private fun parseExpression(expr: String): LinearEquation {
        require(expr.isNotEmpty()) { "طرف المعادلة فارغ" }
        val normalized = expr.replace("*", "")
        val terms = mutableListOf<String>()
        var start = 0
        for (i in 1 until normalized.length) {
            if (normalized[i] == '+' || normalized[i] == '-') {
                terms += normalized.substring(start, i)
                start = i
            }
        }
        terms += normalized.substring(start)

        var a = 0.0
        var b = 0.0
        var c = 0.0
        for (raw in terms) {
            if (raw.isEmpty() || raw == "+" || raw == "-") throw IllegalArgumentException("حد غير صالح: $raw")
            when {
                raw.endsWith("x") -> a += parseCoefficient(raw.dropLast(1))
                raw.endsWith("y") -> b += parseCoefficient(raw.dropLast(1))
                else -> c += raw.toDoubleOrNull() ?: throw IllegalArgumentException("حد غير معروف: $raw")
            }
        }
        return LinearEquation(a, b, c)
    }

    private fun parseCoefficient(value: String): Double = when (value) {
        "", "+" -> 1.0
        "-" -> -1.0
        else -> value.toDoubleOrNull() ?: throw IllegalArgumentException("معامل غير صالح: $value")
    }
}
