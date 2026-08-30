package com.example.equationsolver.ai

import com.example.equationsolver.core.ArabicEquationNormalizer
import kotlin.math.abs
import kotlin.math.ln
import kotlin.math.sign

/** Deterministic RPN encoder shared by training and inference. */
object StructuralMathEncoder {
    object Kind {
        const val PAD = 0
        const val NUMBER = 1
        const val X = 2
        const val Y = 3
        const val PI = 4
        const val E = 5
        const val ADD = 6
        const val SUB = 7
        const val MUL = 8
        const val DIV = 9
        const val POW = 10
        const val NEG = 11
        const val SIN = 12
        const val COS = 13
        const val TAN = 14
        const val SQRT = 15
        const val LOG = 16
        const val LN = 17
        const val EXP = 18
        const val ABS = 19
        const val EQ = 20
        const val SEP = 21
    }

    data class Encoding(
        val source: String,
        val kinds: IntArray,
        val numeric: FloatArray,
        val depth: FloatArray,
        val nodeCount: Int,
        val truncated: Boolean,
        val family: EquationFamily
    )

    private data class Lex(val text: String, val type: Type)
    private enum class Type { NUMBER, NAME, OP, LPAREN, RPAREN }
    private data class DegreeValue(val degree: Int, val constant: Double? = null)

    private val functions = setOf("sin", "cos", "tan", "sqrt", "log", "ln", "exp", "abs")

    fun encode(raw: String): Encoding {
        val source = normalize(raw)
        require(source.isNotBlank()) { "المعادلة فارغة" }
        val equations = source.split(';').map { it.trim() }.filter { it.isNotEmpty() }
        require(equations.isNotEmpty()) { "لا توجد معادلة" }
        require(equations.size <= 2) { "v5 يدعم معادلة واحدة أو نظامًا من معادلتين" }

        val nodes = ArrayList<Pair<Int, Double>>(V5ModelSpec.MAX_NODES)
        equations.forEachIndexed { index, equation ->
            val equalAt = topLevelEquals(equation)
            require(equalAt >= 0) { "كل معادلة يجب أن تحتوي =" }
            nodes += toRpn(equation.substring(0, equalAt))
            nodes += toRpn(equation.substring(equalAt + 1))
            nodes += Kind.EQ to 0.0
            if (index > 0) nodes += Kind.SEP to 0.0
        }

        val kinds = IntArray(V5ModelSpec.MAX_NODES)
        val numeric = FloatArray(V5ModelSpec.MAX_NODES)
        val depth = FloatArray(V5ModelSpec.MAX_NODES)
        var stackDepth = 0
        val count = minOf(nodes.size, V5ModelSpec.MAX_NODES)
        for (i in 0 until count) {
            val (kind, value) = nodes[i]
            kinds[i] = kind
            numeric[i] = if (kind == Kind.NUMBER) normalizeNumber(value) else 0f
            stackDepth = when (kind) {
                Kind.NUMBER, Kind.X, Kind.Y, Kind.PI, Kind.E -> stackDepth + 1
                Kind.NEG, Kind.SIN, Kind.COS, Kind.TAN, Kind.SQRT, Kind.LOG, Kind.LN, Kind.EXP, Kind.ABS -> maxOf(1, stackDepth)
                Kind.ADD, Kind.SUB, Kind.MUL, Kind.DIV, Kind.POW, Kind.EQ, Kind.SEP -> maxOf(1, stackDepth - 1)
                else -> stackDepth
            }
            depth[i] = stackDepth.coerceIn(0, 12) / 12.0f
        }
        return Encoding(source, kinds, numeric, depth, count, nodes.size > V5ModelSpec.MAX_NODES, classify(source))
    }

    fun classify(raw: String): EquationFamily {
        val s = normalize(raw).lowercase()
        if (s.contains(';')) return EquationFamily.SYSTEM
        if (functions.any { s.contains("$it(") }) return EquationFamily.ANALYTIC
        if (hasVariableInDenominator(s)) return EquationFamily.ANALYTIC
        val equalAt = topLevelEquals(s)
        if (equalAt < 0) return EquationFamily.LINEAR
        val leftDegree = runCatching { polynomialDegree(toRpn(s.substring(0, equalAt))) }.getOrDefault(1)
        val rightDegree = runCatching { polynomialDegree(toRpn(s.substring(equalAt + 1))) }.getOrDefault(1)
        return if (maxOf(leftDegree, rightDegree) <= 1) EquationFamily.LINEAR else EquationFamily.POLYNOMIAL
    }

    /** Structural degree catches factored forms such as (x-2)*(x+3), not only x^2 syntax. */
    private fun polynomialDegree(nodes: List<Pair<Int, Double>>): Int {
        val stack = ArrayDeque<DegreeValue>()
        fun pop(): DegreeValue = if (stack.isEmpty()) DegreeValue(0) else stack.removeLast()
        for ((kind, rawValue) in nodes) {
            when (kind) {
                Kind.NUMBER -> stack.addLast(DegreeValue(0, rawValue))
                Kind.PI, Kind.E -> stack.addLast(DegreeValue(0))
                Kind.X, Kind.Y -> stack.addLast(DegreeValue(1))
                Kind.NEG -> {
                    val a = pop(); stack.addLast(DegreeValue(a.degree, a.constant?.let { -it }))
                }
                Kind.ADD, Kind.SUB -> {
                    val b = pop(); val a = pop()
                    val constant = if (a.constant != null && b.constant != null) {
                        if (kind == Kind.ADD) a.constant + b.constant else a.constant - b.constant
                    } else null
                    stack.addLast(DegreeValue(maxOf(a.degree, b.degree), constant))
                }
                Kind.MUL -> {
                    val b = pop(); val a = pop()
                    stack.addLast(DegreeValue(a.degree + b.degree, if (a.constant != null && b.constant != null) a.constant * b.constant else null))
                }
                Kind.DIV -> {
                    val b = pop(); val a = pop()
                    if (b.degree != 0) return 99
                    stack.addLast(DegreeValue(a.degree, if (a.constant != null && b.constant != null && abs(b.constant) > 1e-12) a.constant / b.constant else null))
                }
                Kind.POW -> {
                    val exponent = pop(); val base = pop()
                    val n = exponent.constant?.toInt()
                    if (n == null || n < 0 || abs(exponent.constant - n) > 1e-9) return 99
                    stack.addLast(DegreeValue(base.degree * n, if (base.constant != null) base.constant.powInt(n) else null))
                }
                else -> return 99
            }
        }
        return stack.lastOrNull()?.degree ?: 0
    }

    private fun Double.powInt(n: Int): Double {
        var result = 1.0
        repeat(n) { result *= this }
        return result
    }

    private fun normalize(raw: String): String = ArabicEquationNormalizer.normalize(raw)
        .lowercase().replace("**", "^").replace("²", "^2").replace("³", "^3")
        .replace("π", "pi").replace("√", "sqrt").replace("−", "-")
        .replace("×", "*").replace("÷", "/").replace(" ", "")

    private fun topLevelEquals(s: String): Int {
        var depth = 0
        for (i in s.indices) when (s[i]) {
            '(' -> depth++
            ')' -> depth--
            '=' -> if (depth == 0) return i
        }
        return -1
    }

    private fun hasVariableInDenominator(s: String): Boolean {
        var i = 0
        while (i < s.length) {
            if (s[i] == '/') {
                var j = i + 1
                var depth = 0
                while (j < s.length) {
                    if (s[j] == '(') depth++
                    if (s[j] == ')') { if (depth == 0) break; depth-- }
                    if (depth == 0 && s[j] in charArrayOf('+', '-', '=', ';')) break
                    if (s[j] == 'x' || s[j] == 'y') return true
                    j++
                }
            }
            i++
        }
        return false
    }

    private fun toRpn(expression: String): List<Pair<Int, Double>> {
        val tokens = addImplicitMultiplication(lex(expression))
        val output = ArrayList<Pair<Int, Double>>()
        val operators = ArrayDeque<Lex>()
        var previous: Lex? = null
        for (token in tokens) {
            when (token.type) {
                Type.NUMBER -> output += Kind.NUMBER to (token.text.toDoubleOrNull() ?: throw IllegalArgumentException("رقم غير صالح: ${token.text}"))
                Type.NAME -> if (token.text in functions) operators.addLast(token) else output += nameKind(token.text) to 0.0
                Type.LPAREN -> operators.addLast(token)
                Type.RPAREN -> {
                    while (operators.isNotEmpty() && operators.last().type != Type.LPAREN) popOperator(operators.removeLast(), output)
                    require(operators.isNotEmpty() && operators.last().type == Type.LPAREN) { "أقواس غير متوازنة" }
                    operators.removeLast()
                    if (operators.isNotEmpty() && operators.last().type == Type.NAME && operators.last().text in functions) popOperator(operators.removeLast(), output)
                }
                Type.OP -> {
                    val unaryMinus = token.text == "-" && (previous == null || previous?.type == Type.OP || previous?.type == Type.LPAREN)
                    val effective = if (unaryMinus) Lex("neg", Type.OP) else token
                    while (operators.isNotEmpty() && shouldPop(operators.last(), effective)) popOperator(operators.removeLast(), output)
                    operators.addLast(effective)
                }
            }
            previous = token
        }
        while (operators.isNotEmpty()) {
            val op = operators.removeLast()
            require(op.type != Type.LPAREN && op.type != Type.RPAREN) { "أقواس غير متوازنة" }
            popOperator(op, output)
        }
        require(output.isNotEmpty()) { "طرف معادلة فارغ" }
        return output
    }

    private fun lex(s: String): List<Lex> {
        val out = ArrayList<Lex>()
        var i = 0
        while (i < s.length) {
            val ch = s[i]
            when {
                ch.isDigit() || ch == '.' -> {
                    val start = i; i++
                    while (i < s.length && (s[i].isDigit() || s[i] == '.')) i++
                    out += Lex(s.substring(start, i), Type.NUMBER); continue
                }
                ch.isLetter() -> {
                    val start = i; i++
                    while (i < s.length && s[i].isLetter()) i++
                    out += Lex(s.substring(start, i), Type.NAME); continue
                }
                ch == '(' -> out += Lex("(", Type.LPAREN)
                ch == ')' -> out += Lex(")", Type.RPAREN)
                ch in charArrayOf('+', '-', '*', '/', '^') -> out += Lex(ch.toString(), Type.OP)
                else -> throw IllegalArgumentException("رمز غير مدعوم: $ch")
            }
            i++
        }
        return out
    }

    private fun addImplicitMultiplication(input: List<Lex>): List<Lex> {
        if (input.size < 2) return input
        val out = ArrayList<Lex>()
        for (i in input.indices) {
            val current = input[i]
            if (i > 0) {
                val prev = input[i - 1]
                val leftValue = prev.type == Type.NUMBER || (prev.type == Type.NAME && prev.text !in functions) || prev.type == Type.RPAREN
                val rightValue = current.type == Type.NUMBER || current.type == Type.LPAREN || current.type == Type.NAME
                val functionCall = prev.type == Type.NAME && prev.text in functions && current.type == Type.LPAREN
                if (leftValue && rightValue && !functionCall) out += Lex("*", Type.OP)
            }
            out += current
        }
        return out
    }

    private fun nameKind(name: String): Int = when (name) {
        "x" -> Kind.X
        "y" -> Kind.Y
        "pi" -> Kind.PI
        "e" -> Kind.E
        else -> throw IllegalArgumentException("متغير غير مدعوم: $name")
    }

    private fun shouldPop(top: Lex, current: Lex): Boolean {
        if (top.type == Type.LPAREN) return false
        if (top.type == Type.NAME && top.text in functions) return true
        if (top.type != Type.OP) return false
        val topP = precedence(top.text); val curP = precedence(current.text)
        val rightAssociative = current.text == "^" || current.text == "neg"
        return topP > curP || (topP == curP && !rightAssociative)
    }

    private fun precedence(op: String): Int = when (op) { "+", "-" -> 1; "*", "/" -> 2; "neg" -> 3; "^" -> 4; else -> 5 }

    private fun popOperator(op: Lex, output: MutableList<Pair<Int, Double>>) {
        val kind = when (op.text) {
            "+" -> Kind.ADD; "-" -> Kind.SUB; "*" -> Kind.MUL; "/" -> Kind.DIV; "^" -> Kind.POW; "neg" -> Kind.NEG
            "sin" -> Kind.SIN; "cos" -> Kind.COS; "tan" -> Kind.TAN; "sqrt" -> Kind.SQRT
            "log" -> Kind.LOG; "ln" -> Kind.LN; "exp" -> Kind.EXP; "abs" -> Kind.ABS
            else -> throw IllegalArgumentException("مؤثر غير معروف: ${op.text}")
        }
        output += kind to 0.0
    }

    private fun normalizeNumber(value: Double): Float {
        if (!value.isFinite() || abs(value) < 1e-15) return 0f
        return (sign(value) * ln(1.0 + abs(value)) / 8.0).coerceIn(-2.5, 2.5).toFloat()
    }
}
