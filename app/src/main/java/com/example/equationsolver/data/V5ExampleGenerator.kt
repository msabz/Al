package com.example.equationsolver.data

import com.example.equationsolver.ai.EquationFamily
import com.example.equationsolver.ai.SolutionState
import com.example.equationsolver.ai.V5Target
import java.util.Locale
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.log10
import kotlin.math.pow
import kotlin.math.tan
import kotlin.random.Random

data class V5GeneratedExample(
    val equation: String,
    val equivalentEquation: String,
    val target: V5Target,
    val familyName: String
)

/** Programmatic supervision for v5. All finite targets are known before formatting the equation. */
object V5ExampleGenerator {
    fun generate(random: Random = Random.Default, maxAbs: Int = 100): V5GeneratedExample {
        return when (random.nextInt(100)) {
            in 0..17 -> linear(random, maxAbs)
            in 18..42 -> polynomial(random, maxAbs)
            in 43..66 -> analytic(random, maxAbs)
            in 67..88 -> system(random, maxAbs)
            else -> specialState(random)
        }
    }

    private fun linear(r: Random, maxAbs: Int): V5GeneratedExample {
        val variable = if (r.nextInt(4) == 0) 'y' else 'x'
        val root = r.nextInt(-maxAbs.coerceAtMost(300), maxAbs.coerceAtMost(300) + 1).toDouble() / r.nextInt(1, 5)
        val a = nonZero(r, -12, 13)
        val b = r.nextInt(-40, 41)
        val c = a * root + b
        val eq = when (r.nextInt(3)) {
            0 -> "${a}${variable}${signed(b.toDouble())}=${fmt(c)}"
            1 -> {
                var d: Int
                do d = r.nextInt(-10, 11) while (d == a)
                val rc = c - d * root
                "${a}${variable}${signed(b.toDouble())}=${d}${variable}${signed(rc)}"
            }
            else -> {
                val shift = r.nextInt(-10, 11).toDouble()
                "${a}*(${variable}${signed(shift)})=${fmt(a * (root + shift))}"
            }
        }
        return finiteRoot(eq, doubleArrayOf(root), EquationFamily.LINEAR, "linear")
    }

    private fun polynomial(r: Random, maxAbs: Int): V5GeneratedExample {
        val degree = r.nextInt(2, 6)
        val radius = minOf(12, maxOf(3, maxAbs / 8))
        val roots = MutableList(degree) { r.nextInt(-radius, radius + 1).toDouble() }
        // Repeated roots are allowed in the polynomial but the prediction target is a set.
        val equation = roots.joinToString("*") { factor(it) } + "=0"
        val unique = roots.distinct().sorted().toDoubleArray()
        return finiteRoot(equation, unique, EquationFamily.POLYNOMIAL, "polynomial-$degree")
    }

    private fun analytic(r: Random, maxAbs: Int): V5GeneratedExample {
        val root = r.nextInt(-minOf(maxAbs, 40), minOf(maxAbs, 40) + 1).toDouble() / 4.0
        return when (r.nextInt(7)) {
            0 -> {
                val a = nonZero(r, -7, 8)
                val c = nonZero(r, -6, 7)
                val d = r.nextInt(-10, 11)
                val k = nonZero(r, -5, 6)
                if (a == k * c || abs(c * root + d) < 1e-6) return analytic(r, maxAbs)
                val b = k * (c * root + d) - a * root
                finiteRoot("(${a}x${signed(b)})/(${c}x${signed(d.toDouble())})=$k", doubleArrayOf(root), EquationFamily.ANALYTIC, "rational")
            }
            1 -> {
                val a = r.nextInt(1, 7)
                val result = r.nextInt(1, 12).toDouble()
                val b = result * result - a * root
                finiteRoot("sqrt(${a}x${signed(b)})=${fmt(result)}", doubleArrayOf(root), EquationFamily.ANALYTIC, "radical")
            }
            2 -> {
                val a = nonZero(r, -3, 4)
                val b = r.nextInt(-2, 3)
                val rhs = exp(a * root + b)
                finiteRoot("exp(${a}x${signed(b.toDouble())})=${fmt(rhs, 8)}", doubleArrayOf(root), EquationFamily.ANALYTIC, "exp")
            }
            3 -> {
                val base = r.nextInt(2, 7).toDouble()
                val rhs = base.pow(root)
                finiteRoot("${fmt(base)}^x=${fmt(rhs, 8)}", doubleArrayOf(root), EquationFamily.ANALYTIC, "power")
            }
            4 -> {
                val a = r.nextInt(1, 6)
                val inner = r.nextInt(1, 15).toDouble()
                val b = inner - a * root
                finiteRoot("ln(${a}x${signed(b)})=${fmt(ln(inner), 8)}", doubleArrayOf(root), EquationFamily.ANALYTIC, "ln")
            }
            5 -> {
                val a = r.nextInt(1, 6)
                val inner = r.nextInt(1, 15).toDouble()
                val b = inner - a * root
                finiteRoot("log(${a}x${signed(b)})=${fmt(log10(inner), 8)}", doubleArrayOf(root), EquationFamily.ANALYTIC, "log10")
            }
            else -> {
                // Principal-branch supervision for transcendental equations.
                val bounded = root.coerceIn(-1.2, 1.2)
                finiteRoot("tan(x)=${fmt(tan(bounded), 8)}", doubleArrayOf(bounded), EquationFamily.ANALYTIC, "tan-principal")
            }
        }
    }

    private fun system(r: Random, maxAbs: Int): V5GeneratedExample {
        while (true) {
            val a1 = r.nextInt(-9, 10); val b1 = r.nextInt(-9, 10)
            val a2 = r.nextInt(-9, 10); val b2 = r.nextInt(-9, 10)
            val det = a1 * b2 - a2 * b1
            if (det == 0 || (a1 == 0 && b1 == 0) || (a2 == 0 && b2 == 0)) continue
            val bound = minOf(maxAbs, 25)
            val x = r.nextInt(-bound, bound + 1).toDouble()
            val y = r.nextInt(-bound, bound + 1).toDouble()
            val c1 = a1 * x + b1 * y
            val c2 = a2 * x + b2 * y
            val e1 = "${term(a1, 'x')}${term(b1, 'y', true)}=${fmt(c1)}"
            val e2 = "${term(a2, 'x')}${term(b2, 'y', true)}=${fmt(c2)}"
            val eq = "$e1;$e2"
            val alt = "${swapSides(e2)};${swapSides(e1)}"
            return V5GeneratedExample(eq, alt, V5Target(EquationFamily.SYSTEM, SolutionState.FINITE, systemValues = doubleArrayOf(x, y)), "linear-system")
        }
    }

    private fun specialState(r: Random): V5GeneratedExample {
        return when (r.nextInt(4)) {
            0 -> V5GeneratedExample("0*x=1", "1=0*x", V5Target(EquationFamily.LINEAR, SolutionState.NO_SOLUTION), "no-solution")
            1 -> V5GeneratedExample("0*x=0", "0=0*x", V5Target(EquationFamily.LINEAR, SolutionState.INFINITE), "infinite")
            2 -> {
                val eq = "x+y=2;2x+2y=4"
                V5GeneratedExample(eq, "2x+2y=4;x+y=2", V5Target(EquationFamily.SYSTEM, SolutionState.INFINITE), "system-infinite")
            }
            else -> {
                val eq = "x*y=1"
                V5GeneratedExample(eq, "1=x*y", V5Target(EquationFamily.LINEAR, SolutionState.UNSUPPORTED), "unsupported-nonlinear-multivariable")
            }
        }
    }

    private fun finiteRoot(eq: String, roots: DoubleArray, family: EquationFamily, name: String): V5GeneratedExample =
        V5GeneratedExample(eq, swapSides(eq), V5Target(family, SolutionState.FINITE, roots = roots), name)

    private fun swapSides(eq: String): String {
        val i = eq.indexOf('=')
        return if (i < 0) eq else eq.substring(i + 1) + "=" + eq.substring(0, i)
    }

    private fun factor(root: Double): String = "(x${signed(-root)})"
    private fun nonZero(r: Random, from: Int, until: Int): Int { var v: Int; do v = r.nextInt(from, until) while (v == 0); return v }
    private fun term(coef: Int, variable: Char, appendSign: Boolean = false): String {
        if (coef == 0) return if (appendSign) "+0$variable" else "0$variable"
        val sign = if (coef < 0) "-" else if (appendSign) "+" else ""
        val magnitude = abs(coef)
        return sign + (if (magnitude == 1) "" else magnitude.toString()) + variable
    }
    private fun signed(value: Double): String = if (value >= 0) "+${fmt(value)}" else fmt(value)
    private fun fmt(value: Double, digits: Int = 6): String {
        if (abs(value) < 1e-12) return "0"
        if (abs(value - value.toLong()) < 1e-10) return value.toLong().toString()
        return String.format(Locale.US, "%.${digits}f", value).trimEnd('0').trimEnd('.')
    }
}
