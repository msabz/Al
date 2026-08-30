package com.example.equationsolver.ai

import com.example.equationsolver.core.ArabicEquationNormalizer
import kotlin.math.abs
import kotlin.math.ln
import kotlin.math.sign

/**
 * Deterministic structural encoder used by BOTH training and inference.
 * Every numeric literal is a single RPN node; numbers are not split into digits.
 */
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
            val left = equation.substring(0, equalAt)
            val right = equation.substring(equalAt + 1)
            nodes += toRpn(left)
            nodes += toRpn(right)
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
        return Encoding(
            source = source,
            kinds = kinds,
            numeric = numeric,
            depth = depth,
            nodeCount = count,
            truncated = nodes.size > V5ModelSpec.MAX_NODES,
            family = classify(source)
        )
    }

    fun classify(raw: String): EquationFamily {
        val s = normalize(raw).lowercase()
        if (s.contains(';')) return EquationFamily.SYSTEM
        if (functions.any { s.contains("$it(") }) return EquationFamily.ANALYTIC
        if (hasVariableInDenominator(s)) return EquationFamily.ANALYTIC

        val explicitDegree = Regex("[xy]\\^(\\d+)").findAll(s)
            .mapNotNull { it.groupValues.getOrNull(1)?.toIntOrNull() }
            .maxOrNull() ?: 1
        if (explicitDegree >= 2) return EquationFamily.POLYNOMIAL

        // Detect factored/implicit polynomial forms even when no x^n token is written.
        // Examples: (x-2)*(x+3)=0, x*x-1=0, 2x*(x+1)=4.
        val factorProduct = Regex("\\([^)]*[xy][^)]*\\)\\*\\([^)]*[xy][^)]*\\)").containsMatchIn(s)
        val repeatedVariableProduct = Regex("[xy]\\*+[xy]").containsMatchIn(s)
        val variableTimesFactor = Regex("[xy]\\*\\([^)]*[xy][^)]*\\)|\\([^)]*[xy][^)]*\\)\\*[xy]").containsMatchIn(s)
        if (factorProduct || repeatedVariableProduct || variableTimesFactor) return EquationFamily.POLYNOMIAL

        // A product of more than one variable-bearing top-level factor is also nonlinear.
        val variableFactors = Regex("\\([^)]*[xy][^)]*\\)").findAll(s).count()
        if (variableFactors >= 2 && s.contains('*')) return EquationFamily.POLYNOMIAL
        return EquationFamily.LINEAR
    }

    private fun normalize(raw: String): String = ArabicEquationNormalizer.normalize(raw)
        .lowercase()
        .replace("**", "^")
        .replace("²", "^2")
        .replace("³", "^3")
        .replace("π", "pi")
        .replace("√", "sqrt")
        .replace("−", "-")
        .replace("×", "*")
        .replace("÷", "/")
        .replace(" ", "")

    private fun topLevelEquals(s: String): Int {
        var depth = 0
        for (i in s.indices) {
            when (s[i]) {
                '(' -> depth++
                ')' -> depth--
                '=' -> if (depth == 0) return i
            }
        }
        return -1
    }

    private fun hasVariableInDenominator(s: String): Boolean {
        val slash = s.indexOf('/')
        if (slash < 0) return false
        val after = s.substring(slash + 1)
        return after.contains('x') || after.contains('y')
    }

    private fun toRpn(expression: String): List<Pair<Int, Double>> {
        val rawTokens = lex(expression)
        val tokens = addImplicitMultiplication(rawTokens)
        val output = ArrayList<Pair<Int, Double>>()
        val operators = ArrayDeque<Lex>()
        var previous: Lex? = null

        for (token in tokens) {
            when (token.type) {
                Type.NUMBER -> output += Kind.NUMBER to (token.text.toDoubleOrNull()
                    ?: throw IllegalArgumentException("رقم غير صالح: ${token.text}"))
                Type.NAME -> {
                    if (token.text in functions) operators.addLast(token)
                    else output += nameKind(token.text) to 0.0
                }
                Type.LPAREN -> operators.addLast(token)
                Type.RPAREN -> {
                    while (operators.isNotEmpty() && operators.last().type != Type.LPAREN) popOperator(operators.removeLast(), output)
                    require(operators.isNotEmpty() && operators.last().type == Type.LPAREN) { "أقواس غير متوازنة" }
                    operators.removeLast()
                    if (operators.isNotEmpty() && operators.last().type == Type.NAME && operators.last().text in functions) {
                        popOperator(operators.removeLast(), output)
                    }
                }
                Type.OP -> {
                    val unaryMinus = token.text == "-" && (previous == null || previous!!.type == Type.OP || previous!!.type == Type.LPAREN)
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
                    val start = i
                    i++
                    while (i < s.length && (s[i].isDigit() || s[i] == '.')) i++
                    out += Lex(s.substring(start, i), Type.NUMBER)
                    continue
                }
                ch.isLetter() -> {
                    val start = i
                    i++
                    while (i < s.length && s[i].isLetter()) i++
                    out += Lex(s.substring(start, i), Type.NAME)
                    continue
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
        val topP = precedence(top.text)
        val curP = precedence(current.text)
        val rightAssociative = current.text == "^" || current.text == "neg"
        return topP > curP || (topP == curP && !rightAssociative)
    }

    private fun precedence(op: String): Int = when (op) {
        "+", "-" -> 1
        "*", "/" -> 2
        "neg" -> 3
        "^" -> 4
        else -> 5
    }

    private fun popOperator(op: Lex, output: MutableList<Pair<Int, Double>>) {
        val kind = when (op.text) {
            "+" -> Kind.ADD
            "-" -> Kind.SUB
            "*" -> Kind.MUL
            "/" -> Kind.DIV
            "^" -> Kind.POW
            "neg" -> Kind.NEG
            "sin" -> Kind.SIN
            "cos" -> Kind.COS
            "tan" -> Kind.TAN
            "sqrt" -> Kind.SQRT
            "log" -> Kind.LOG
            "ln" -> Kind.LN
            "exp" -> Kind.EXP
            "abs" -> Kind.ABS
            else -> throw IllegalArgumentException("مؤثر غير معروف: ${op.text}")
        }
        output += kind to 0.0
    }

    private fun normalizeNumber(value: Double): Float {
        if (!value.isFinite() || abs(value) < 1e-15) return 0f
        val compressed = sign(value) * ln(1.0 + abs(value)) / 8.0
        return compressed.coerceIn(-2.5, 2.5).toFloat()
    }
}
