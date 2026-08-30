package com.example.equationsolver.core

import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.log10
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.math.tan

/** Small local expression evaluator used only as a teacher/verifier, never as the neural prediction path. */
object MathExpressionEvaluator {
    fun sides(equation: String, x: Double = 0.0, y: Double = 0.0): Pair<Double, Double> {
        val normalized = normalize(equation)
        require(normalized.count { it == '=' } == 1) { "يجب أن تحتوي المعادلة على علامة '=' واحدة" }
        val split = normalized.indexOf('=')
        val left = Parser(normalized.substring(0, split), x, y).parse()
        val right = Parser(normalized.substring(split + 1), x, y).parse()
        return left to right
    }

    fun residual(equation: String, x: Double = 0.0, y: Double = 0.0): Double {
        val (left, right) = sides(equation, x, y)
        return left - right
    }

    private fun normalize(value: String): String = ArabicEquationNormalizer.normalize(value)
        .lowercase()
        .replace("π", "pi")
        .replace("√", "sqrt")
        .replace("²", "^2")
        .replace("³", "^3")

    private class Parser(private val source: String, private val x: Double, private val y: Double) {
        private var pos = 0

        fun parse(): Double {
            require(source.isNotEmpty()) { "تعبير فارغ" }
            val value = parseExpression()
            require(pos == source.length) { "رمز غير معروف قرب: ${source.substring(pos)}" }
            require(value.isFinite()) { "نتيجة غير عددية" }
            return value
        }

        private fun parseExpression(): Double {
            var value = parseTerm()
            while (pos < source.length) {
                when (source[pos]) {
                    '+' -> { pos++; value += parseTerm() }
                    '-' -> { pos++; value -= parseTerm() }
                    else -> return value
                }
            }
            return value
        }

        private fun parseTerm(): Double {
            var value = parsePower()
            while (pos < source.length) {
                when (source[pos]) {
                    '*' -> { pos++; value *= parsePower() }
                    '/' -> {
                        pos++
                        val denominator = parsePower()
                        require(abs(denominator) > 1e-14) { "قسمة على صفر" }
                        value /= denominator
                    }
                    else -> {
                        if (startsPrimary(source[pos])) value *= parsePower() else return value
                    }
                }
            }
            return value
        }

        private fun parsePower(): Double {
            var value = parseUnary()
            if (pos < source.length && source[pos] == '^') {
                pos++
                value = value.pow(parsePower())
            }
            return value
        }

        private fun parseUnary(): Double {
            if (pos < source.length && source[pos] == '+') { pos++; return parseUnary() }
            if (pos < source.length && source[pos] == '-') { pos++; return -parseUnary() }
            return parsePrimary()
        }

        private fun parsePrimary(): Double {
            require(pos < source.length) { "تعبير ناقص" }
            val ch = source[pos]
            if (ch == '(') {
                pos++
                val value = parseExpression()
                require(pos < source.length && source[pos] == ')') { "قوس إغلاق مفقود" }
                pos++
                return value
            }
            if (ch.isDigit() || ch == '.') return parseNumber()
            if (ch.isLetter()) return parseIdentifier()
            throw IllegalArgumentException("رمز غير صالح: $ch")
        }

        private fun parseNumber(): Double {
            val start = pos
            var dotSeen = false
            while (pos < source.length) {
                val ch = source[pos]
                if (ch.isDigit()) pos++
                else if (ch == '.' && !dotSeen) { dotSeen = true; pos++ }
                else break
            }
            return source.substring(start, pos).toDoubleOrNull()
                ?: throw IllegalArgumentException("رقم غير صالح")
        }

        private fun parseIdentifier(): Double {
            val start = pos
            while (pos < source.length && source[pos].isLetter()) pos++
            val name = source.substring(start, pos)
            return when (name) {
                "x" -> x
                "y" -> y
                "pi" -> Math.PI
                "e" -> Math.E
                "sin", "cos", "tan", "sqrt", "log", "ln", "exp", "abs" -> {
                    require(pos < source.length && source[pos] == '(') { "الدالة $name تحتاج أقواسًا" }
                    pos++
                    val arg = parseExpression()
                    require(pos < source.length && source[pos] == ')') { "قوس إغلاق مفقود للدالة $name" }
                    pos++
                    when (name) {
                        "sin" -> sin(arg)
                        "cos" -> cos(arg)
                        "tan" -> tan(arg)
                        "sqrt" -> { require(arg >= 0.0) { "جذر قيمة سالبة" }; sqrt(arg) }
                        "log" -> { require(arg > 0.0) { "لوغاريتم قيمة غير موجبة" }; log10(arg) }
                        "ln" -> { require(arg > 0.0) { "لوغاريتم قيمة غير موجبة" }; ln(arg) }
                        "exp" -> exp(arg)
                        else -> abs(arg)
                    }
                }
                else -> throw IllegalArgumentException("اسم غير معروف: $name")
            }
        }

        private fun startsPrimary(ch: Char): Boolean = ch == '(' || ch == '.' || ch.isDigit() || ch.isLetter()
    }
}
