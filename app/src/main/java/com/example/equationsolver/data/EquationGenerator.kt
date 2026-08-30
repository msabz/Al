package com.example.equationsolver.data

import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sin
import kotlin.random.Random
import java.util.Locale

data class GeneratedExample(
    val equation: String,
    val x: Double,
    val y: Double,
    val family: String = "unknown"
)

/** Infinite synthetic curriculum for algebraic and analytic equations (no geometry). */
object EquationGenerator {
    fun generate(): GeneratedExample = when (Random.nextInt(100)) {
        in 0..19 -> generateLinear()
        in 20..34 -> generateQuadratic()
        in 35..44 -> generateCubic()
        in 45..54 -> generateRational()
        in 55..64 -> generateRadical()
        in 65..74 -> generateExponential()
        in 75..84 -> generateLogarithmic()
        in 85..89 -> generateAbsolute()
        in 90..94 -> generateTrigonometric()
        else -> generateSystem()
    }

    private fun generateLinear(): GeneratedExample {
        val solveY = Random.nextInt(5) == 0
        val a = nonZeroInt(-12, 13)
        val value = Random.nextInt(-50, 51).toDouble() / Random.nextInt(1, 5)
        val b = Random.nextInt(-30, 31)
        val c = a * value + b
        val variable = if (solveY) 'y' else 'x'
        return GeneratedExample(
            "${a}${variable}${signed(b.toDouble())}=${fmt(c)}",
            x = if (solveY) 0.0 else value,
            y = if (solveY) value else 0.0,
            family = "linear"
        )
    }

    private fun generateQuadratic(): GeneratedExample {
        val r1 = Random.nextInt(-15, 16).toDouble()
        val r2 = Random.nextInt(-15, 16).toDouble()
        val b = -(r1 + r2)
        val c = r1 * r2
        val target = canonicalRoot(listOf(r1, r2))
        return GeneratedExample("x^2${signed(b)}x${signed(c)}=0", target, 0.0, "quadratic")
    }

    private fun generateCubic(): GeneratedExample {
        val r1 = Random.nextInt(-8, 9).toDouble()
        val r2 = Random.nextInt(-8, 9).toDouble()
        val r3 = Random.nextInt(-8, 9).toDouble()
        val a2 = -(r1 + r2 + r3)
        val a1 = r1 * r2 + r1 * r3 + r2 * r3
        val a0 = -r1 * r2 * r3
        val target = canonicalRoot(listOf(r1, r2, r3))
        return GeneratedExample("x^3${signed(a2)}x^2${signed(a1)}x${signed(a0)}=0", target, 0.0, "cubic")
    }

    private fun generateRational(): GeneratedExample {
        while (true) {
            val root = Random.nextInt(-15, 16).toDouble()
            val a = nonZeroInt(-8, 9)
            val c = nonZeroInt(-6, 7)
            val d = Random.nextInt(-12, 13)
            val k = nonZeroInt(-5, 6)
            if (abs(c * root + d) < 1e-9) continue
            val b = k * (c * root + d) - a * root
            return GeneratedExample("(${a}x${signed(b)})/(${c}x${signed(d.toDouble())})=$k", root, 0.0, "rational")
        }
    }

    private fun generateRadical(): GeneratedExample {
        val root = Random.nextInt(-20, 21).toDouble()
        val a = Random.nextInt(1, 7)
        val result = Random.nextInt(1, 13).toDouble()
        val b = result * result - a * root
        return GeneratedExample("sqrt(${a}x${signed(b)})=${fmt(result)}", root, 0.0, "radical")
    }

    private fun generateExponential(): GeneratedExample {
        val root = Random.nextInt(-30, 31).toDouble() / 10.0
        val a = nonZeroInt(-3, 4)
        val b = Random.nextInt(-2, 3)
        val rhs = exp(a * root + b)
        return GeneratedExample("exp(${a}x${signed(b.toDouble())})=${fmt(rhs, 8)}", root, 0.0, "exponential")
    }

    private fun generateLogarithmic(): GeneratedExample {
        val root = Random.nextInt(-20, 21).toDouble() / 2.0
        val a = Random.nextInt(1, 6)
        val inner = Random.nextInt(1, 13).toDouble()
        val b = inner - a * root
        val rhs = ln(inner)
        return GeneratedExample("ln(${a}x${signed(b)})=${fmt(rhs, 8)}", root, 0.0, "logarithmic")
    }

    private fun generateAbsolute(): GeneratedExample {
        val root = Random.nextInt(-30, 31).toDouble() / 2.0
        val a = nonZeroInt(-8, 9)
        val b = -a * root
        return GeneratedExample("abs(${a}x${signed(b)})=0", root, 0.0, "absolute")
    }

    private fun generateTrigonometric(): GeneratedExample {
        val root = Random.nextInt(-14, 15).toDouble() / 10.0
        val rhs = sin(root)
        return GeneratedExample("sin(x)=${fmt(rhs, 8)}", root, 0.0, "trigonometric")
    }

    private fun generateSystem(): GeneratedExample {
        while (true) {
            val a1 = Random.nextInt(-9, 10)
            val b1 = Random.nextInt(-9, 10)
            val a2 = Random.nextInt(-9, 10)
            val b2 = Random.nextInt(-9, 10)
            val det = a1 * b2 - a2 * b1
            if (det == 0 || (a1 == 0 && b1 == 0) || (a2 == 0 && b2 == 0)) continue
            val x = Random.nextInt(-12, 13).toDouble()
            val y = Random.nextInt(-12, 13).toDouble()
            val c1 = a1 * x + b1 * y
            val c2 = a2 * x + b2 * y
            return GeneratedExample(
                "${term(a1, 'x')}${term(b1, 'y', true)}=${fmt(c1)};${term(a2, 'x')}${term(b2, 'y', true)}=${fmt(c2)}",
                x, y, "linear-system"
            )
        }
    }

    private fun canonicalRoot(values: List<Double>): Double = values.distinct().minWithOrNull(
        compareBy<Double> { abs(it) }.thenBy { it }
    ) ?: 0.0

    private fun nonZeroInt(from: Int, until: Int): Int {
        var value: Int
        do value = Random.nextInt(from, until) while (value == 0)
        return value
    }

    private fun term(coef: Int, variable: Char, appendSign: Boolean = false): String {
        val sign = if (coef < 0) "-" else if (appendSign) "+" else ""
        val magnitude = abs(coef)
        val number = if (magnitude == 1) "" else magnitude.toString()
        return "$sign$number$variable"
    }

    private fun signed(value: Double): String = if (value >= 0) "+${fmt(value)}" else fmt(value)

    private fun fmt(value: Double, digits: Int = 6): String {
        if (abs(value) < 1e-12) return "0"
        if (abs(value - value.toLong()) < 1e-10) return value.toLong().toString()
        return String.format(Locale.US, "%.${digits}f", value).trimEnd('0').trimEnd('.')
    }
}
