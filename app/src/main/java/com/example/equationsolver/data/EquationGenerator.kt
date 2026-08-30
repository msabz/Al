package com.example.equationsolver.data

import java.util.Locale
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.log10
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.tan
import kotlin.random.Random

data class GeneratedExample(
    val equation: String,
    val x: Double,
    val y: Double,
    val family: String = "unknown",
    val xActive: Boolean = equation.lowercase().contains('x'),
    val yActive: Boolean = equation.lowercase().contains('y')
)

/** Infinite synthetic curriculum for mathematical equations only (no geometry). */
object EquationGenerator {
    fun generate(random: Random = Random.Default): GeneratedExample = when (random.nextInt(100)) {
        in 0..17 -> generateLinear(random)
        in 18..29 -> generateQuadratic(random)
        in 30..37 -> generateCubic(random)
        in 38..42 -> generateQuartic(random)
        in 43..46 -> generateQuintic(random)
        in 47..55 -> generateRational(random)
        in 56..64 -> generateRadical(random)
        in 65..72 -> generateExponential(random)
        in 73..80 -> generateLogarithmic(random)
        in 81..84 -> generateAbsolute(random)
        in 85..91 -> generateTrigonometric(random)
        else -> generateSystem(random)
    }

    private fun generateLinear(random: Random): GeneratedExample {
        val solveY = random.nextInt(3) == 0
        val a = nonZeroInt(random, -12, 13)
        val value = random.nextInt(-50, 51).toDouble() / random.nextInt(1, 5)
        val b = random.nextInt(-30, 31)
        val variable = if (solveY) 'y' else 'x'
        val equation = when (random.nextInt(3)) {
            0 -> "${a}${variable}${signed(b.toDouble())}=${fmt(a * value + b)}"
            1 -> {
                var d: Int
                do d = random.nextInt(-10, 11) while (d == a)
                val rightConstant = a * value + b - d * value
                "${a}${variable}${signed(b.toDouble())}=${d}${variable}${signed(rightConstant)}"
            }
            else -> {
                val shift = random.nextInt(-12, 13).toDouble()
                "${a}*(${variable}${signed(shift)})=${fmt(a * (value + shift))}"
            }
        }
        return GeneratedExample(
            equation,
            x = if (solveY) 0.0 else value,
            y = if (solveY) value else 0.0,
            family = "linear"
        )
    }

    private fun generateQuadratic(random: Random): GeneratedExample {
        val roots = List(2) { random.nextInt(-15, 16).toDouble() }
        val b = -(roots[0] + roots[1])
        val c = roots[0] * roots[1]
        val equation = if (random.nextBoolean()) {
            "x^2${signed(b)}x${signed(c)}=0"
        } else {
            "${factor(roots[0])}*${factor(roots[1])}=0"
        }
        return GeneratedExample(equation, canonicalRoot(roots), 0.0, "quadratic")
    }

    private fun generateCubic(random: Random): GeneratedExample {
        val r = List(3) { random.nextInt(-8, 9).toDouble() }
        val a2 = -(r[0] + r[1] + r[2])
        val a1 = r[0] * r[1] + r[0] * r[2] + r[1] * r[2]
        val a0 = -r[0] * r[1] * r[2]
        val equation = if (random.nextBoolean()) {
            "x^3${signed(a2)}x^2${signed(a1)}x${signed(a0)}=0"
        } else r.joinToString("*") { factor(it) } + "=0"
        return GeneratedExample(equation, canonicalRoot(r), 0.0, "cubic")
    }

    private fun generateQuartic(random: Random): GeneratedExample {
        val r = List(4) { random.nextInt(-6, 7).toDouble() }
        val s1 = r.sum()
        var s2 = 0.0
        var s3 = 0.0
        for (i in 0..3) for (j in i + 1..3) s2 += r[i] * r[j]
        for (i in 0..3) for (j in i + 1..3) for (k in j + 1..3) s3 += r[i] * r[j] * r[k]
        val s4 = r.reduce { acc, value -> acc * value }
        val equation = if (random.nextBoolean()) {
            "x^4${signed(-s1)}x^3${signed(s2)}x^2${signed(-s3)}x${signed(s4)}=0"
        } else r.joinToString("*") { factor(it) } + "=0"
        return GeneratedExample(equation, canonicalRoot(r), 0.0, "quartic")
    }

    private fun generateQuintic(random: Random): GeneratedExample {
        val roots = List(5) { random.nextInt(-5, 6).toDouble() }
        return GeneratedExample(
            roots.joinToString("*") { factor(it) } + "=0",
            canonicalRoot(roots),
            0.0,
            "quintic"
        )
    }

    private fun generateRational(random: Random): GeneratedExample {
        while (true) {
            val root = random.nextInt(-15, 16).toDouble()
            val a = nonZeroInt(random, -8, 9)
            val c = nonZeroInt(random, -6, 7)
            val d = random.nextInt(-12, 13)
            val k = nonZeroInt(random, -5, 6)
            // If a == k*c, the equation collapses into an identity (or contradiction),
            // making a single numeric target ambiguous. Never feed such labels to the NN.
            if (a == k * c) continue
            if (abs(c * root + d) < 1e-9) continue
            val b = k * (c * root + d) - a * root
            return GeneratedExample("(${a}x${signed(b)})/(${c}x${signed(d.toDouble())})=$k", root, 0.0, "rational")
        }
    }

    private fun generateRadical(random: Random): GeneratedExample {
        val root = random.nextInt(-20, 21).toDouble()
        val a = random.nextInt(1, 7)
        val result = random.nextInt(1, 13).toDouble()
        val b = result * result - a * root
        return GeneratedExample("sqrt(${a}x${signed(b)})=${fmt(result)}", root, 0.0, "radical")
    }

    private fun generateExponential(random: Random): GeneratedExample {
        val root = random.nextInt(-30, 31).toDouble() / 10.0
        return if (random.nextBoolean()) {
            val a = nonZeroInt(random, -3, 4)
            val b = random.nextInt(-2, 3)
            val rhs = exp(a * root + b)
            GeneratedExample("exp(${a}x${signed(b.toDouble())})=${fmt(rhs, 8)}", root, 0.0, "exponential-exp")
        } else {
            val base = random.nextInt(2, 6).toDouble()
            val rhs = base.pow(root)
            GeneratedExample("${fmt(base)}^x=${fmt(rhs, 8)}", root, 0.0, "exponential-power")
        }
    }

    private fun generateLogarithmic(random: Random): GeneratedExample {
        val root = random.nextInt(-20, 21).toDouble() / 2.0
        val a = random.nextInt(1, 6)
        val inner = random.nextInt(1, 13).toDouble()
        val b = inner - a * root
        return if (random.nextBoolean()) {
            GeneratedExample("ln(${a}x${signed(b)})=${fmt(ln(inner), 8)}", root, 0.0, "logarithmic-ln")
        } else {
            GeneratedExample("log(${a}x${signed(b)})=${fmt(log10(inner), 8)}", root, 0.0, "logarithmic-log10")
        }
    }

    private fun generateAbsolute(random: Random): GeneratedExample {
        val root = random.nextInt(-30, 31).toDouble() / 2.0
        val a = nonZeroInt(random, -8, 9)
        val b = -a * root
        return GeneratedExample("abs(${a}x${signed(b)})=0", root, 0.0, "absolute")
    }

    private fun generateTrigonometric(random: Random): GeneratedExample {
        return when (random.nextInt(3)) {
            0 -> {
                val root = random.nextInt(-14, 15).toDouble() / 10.0
                GeneratedExample("sin(x)=${fmt(sin(root), 8)}", root, 0.0, "trigonometric-sin")
            }
            1 -> {
                val root = random.nextInt(-12, 13).toDouble() / 10.0
                GeneratedExample("tan(x)=${fmt(tan(root), 8)}", root, 0.0, "trigonometric-tan")
            }
            else -> {
                val magnitude = random.nextInt(0, 15).toDouble() / 10.0
                val canonical = -magnitude
                GeneratedExample("cos(x)=${fmt(cos(magnitude), 8)}", canonical, 0.0, "trigonometric-cos")
            }
        }
    }

    private fun generateSystem(random: Random): GeneratedExample {
        while (true) {
            val a1 = random.nextInt(-9, 10)
            val b1 = random.nextInt(-9, 10)
            val a2 = random.nextInt(-9, 10)
            val b2 = random.nextInt(-9, 10)
            val det = a1 * b2 - a2 * b1
            if (det == 0 || (a1 == 0 && b1 == 0) || (a2 == 0 && b2 == 0)) continue
            val x = random.nextInt(-12, 13).toDouble()
            val y = random.nextInt(-12, 13).toDouble()
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

    private fun nonZeroInt(random: Random, from: Int, until: Int): Int {
        var value: Int
        do value = random.nextInt(from, until) while (value == 0)
        return value
    }

    private fun factor(root: Double): String = "(x${signed(-root)})"

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
