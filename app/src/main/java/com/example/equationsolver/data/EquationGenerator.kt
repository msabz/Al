package com.example.equationsolver.data

import kotlin.math.abs
import kotlin.random.Random

data class GeneratedExample(val equation: String, val x: Double, val y: Double)

object EquationGenerator {
    fun generate(): GeneratedExample = when (Random.nextInt(100)) {
        in 0..49 -> generateSingleLinear()
        in 50..74 -> generateQuadratic()
        else -> generateSystem()
    }

    private fun generateSingleLinear(): GeneratedExample {
        val a = Random.nextInt(1, 11)
        val x = Random.nextInt(-40, 41).toDouble() / Random.nextInt(1, 5)
        val b = Random.nextInt(-20, 21)
        val c = a * x + b
        return GeneratedExample("${a}x${if (b >= 0) "+$b" else b}=${fmt(c)}", x, 0.0)
    }

    private fun generateQuadratic(): GeneratedExample {
        val r1 = Random.nextInt(-12, 13).toDouble()
        val r2 = Random.nextInt(-12, 13).toDouble()
        val a = Random.nextInt(1, 6).toDouble()
        val b = -a * (r1 + r2)
        val c = a * r1 * r2
        val equation = "${fmt(a)}x^2${signed(b)}x${signed(c)}=0"
        return GeneratedExample(equation, r1, 0.0)
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
        val magnitude = abs(coef)
        val number = if (magnitude == 1) "" else magnitude.toString()
        return "$sign$number$variable"
    }

    private fun signed(value: Double): String = if (value >= 0) "+${fmt(value)}" else fmt(value)

    private fun fmt(value: Double): String =
        if (abs(value - value.toInt()) < 1e-10) value.toInt().toString()
        else "%.4f".format(value)
}
