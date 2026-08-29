package com.example.equationsolver.data

import kotlin.random.Random

data class GeneratedExample(val equation: String, val x: Double, val y: Double)

object EquationGenerator {
    fun generate(): GeneratedExample {
        return if (Random.nextBoolean()) generateSingle() else generateSystem()
    }

    private fun generateSingle(): GeneratedExample {
        var a: Int
        do { a = Random.nextInt(1, 11) } while (a == 0)
        val x = Random.nextInt(-20, 21).toDouble() / Random.nextInt(1, 5)
        val b = Random.nextInt(-20, 21)
        val c = a * x + b
        return "${a}x${if (b >= 0) "+$b" else b}=${fmt(c)}".let { GeneratedExample(it, x, 0.0) }
    }

    private fun generateSystem(): GeneratedExample {
        while (true) {
            val a1 = Random.nextInt(-9, 10)
            val b1 = Random.nextInt(-9, 10)
            val a2 = Random.nextInt(-9, 10)
            val b2 = Random.nextInt(-9, 10)
            val det = a1 * b2 - a2 * b1
            if (det == 0 || (a1 == 0 && b1 == 0) || (a2 == 0 && b2 == 0)) continue
            val x = Random.nextInt(-10, 11).toDouble()
            val y = Random.nextInt(-10, 11).toDouble()
            val c1 = a1 * x + b1 * y
            val c2 = a2 * x + b2 * y
            val eq1 = "${term(a1, 'x')}${term(b1, 'y', true)}=${fmt(c1)}"
            val eq2 = "${term(a2, 'x')}${term(b2, 'y', true)}=${fmt(c2)}"
            return GeneratedExample("$eq1;$eq2", x, y)
        }
    }

    private fun term(coef: Int, variable: Char, appendSign: Boolean = false): String {
        val sign = if (coef < 0) "-" else if (appendSign) "+" else ""
        val abs = kotlin.math.abs(coef)
        val number = if (abs == 1) "" else abs.toString()
        return "$sign$number$variable"
    }

    private fun fmt(value: Double): String = if (value % 1.0 == 0.0) value.toInt().toString() else "%.4f".format(value)
}
