package com.example.equationsolver.ai

import com.example.equationsolver.core.ArabicEquationNormalizer
import kotlin.math.*

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

        val family = classify(source)
        canonicalNumericEncoding(source, family)?.let { return it }

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
            family = family
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


    private const val CANONICAL_EPS = 1e-10
    private data class Affine(val x: Double, val y: Double, val c: Double)

    /**
     * Convert the algebraic families used by the DeepMind training contract into
     * fixed numeric coefficients. This removes spelling/order/side shortcuts.
     * LINEAR/POLYNOMIAL: [c0,c1,c2,c3,c4,c5] for p(x)=0.
     * SYSTEM: [a1,b1,c1,a2,b2,c2] for a*x+b*y=c, with canonical row order.
     */
    private fun canonicalNumericEncoding(source: String, family: EquationFamily): Encoding? = when (family) {
        EquationFamily.LINEAR, EquationFamily.POLYNOMIAL -> canonicalPolynomialEncoding(source, family)
        EquationFamily.SYSTEM -> canonicalSystemEncoding(source)
        EquationFamily.ANALYTIC -> null
    }

    private fun coefficientEncoding(source: String, family: EquationFamily, values: DoubleArray): Encoding {
        require(values.size == V5ModelSpec.CANONICAL_COEFF_SLOTS)
        val kinds = IntArray(V5ModelSpec.MAX_NODES)
        val numeric = FloatArray(V5ModelSpec.MAX_NODES)
        val depth = FloatArray(V5ModelSpec.MAX_NODES)
        for (i in values.indices) {
            kinds[i] = Kind.NUMBER
            numeric[i] = values[i].toFloat()
            // Reuse the third scalar feature as a strong coefficient-slot identity.
            depth[i] = (i + 1).toFloat() / values.size.toFloat()
        }
        return Encoding(source, kinds, numeric, depth, values.size, false, family)
    }

    private fun canonicalPolynomialEncoding(source: String, family: EquationFamily): Encoding? {
        if (';' in source) return null
        val at = topLevelEquals(source)
        if (at < 0) return null
        val hasX = source.contains('x')
        val hasY = source.contains('y')
        if (hasX && hasY) return null
        val variableKind = if (hasY) Kind.Y else Kind.X
        val left = polynomialOf(toRpn(source.substring(0, at)), variableKind) ?: return null
        val right = polynomialOf(toRpn(source.substring(at + 1)), variableKind) ?: return null
        val coeff = DoubleArray(V5ModelSpec.CANONICAL_COEFF_SLOTS) { left[it] - right[it] }
        val degree = polynomialDegree(coeff)
        if (family == EquationFamily.LINEAR && degree > 1) return null
        if (family == EquationFamily.POLYNOMIAL && degree < 2) return null
        canonicalizePolynomial(coeff)
        return coefficientEncoding(source, family, coeff)
    }

    private fun canonicalSystemEncoding(source: String): Encoding? {
        val equations = source.split(';').filter { it.isNotBlank() }
        if (equations.size != 2) return null
        val rows = equations.map { canonicalAffineRow(it) ?: return null }.toMutableList()
        rows.sortWith(Comparator { a, b ->
            var result = 0
            for (i in 0..2) {
                result = a[i].compareTo(b[i])
                if (result != 0) break
            }
            result
        })
        val values = doubleArrayOf(
            rows[0][0], rows[0][1], rows[0][2],
            rows[1][0], rows[1][1], rows[1][2]
        )
        return coefficientEncoding(source, EquationFamily.SYSTEM, values)
    }

    private fun canonicalAffineRow(equation: String): DoubleArray? {
        val at = topLevelEquals(equation)
        if (at < 0) return null
        val left = affineOf(toRpn(equation.substring(0, at))) ?: return null
        val right = affineOf(toRpn(equation.substring(at + 1))) ?: return null
        var a = left.x - right.x
        var b = left.y - right.y
        var c = -(left.c - right.c)
        val scale = maxOf(abs(a), abs(b), abs(c))
        if (scale <= CANONICAL_EPS) return doubleArrayOf(0.0, 0.0, 0.0)
        a /= scale; b /= scale; c /= scale
        val first = listOf(a, b, c).firstOrNull { abs(it) > CANONICAL_EPS } ?: 0.0
        if (first < 0.0) { a = -a; b = -b; c = -c }
        return doubleArrayOf(cleanZero(a), cleanZero(b), cleanZero(c))
    }

    private fun affineOf(nodes: List<Pair<Int, Double>>): Affine? {
        val stack = ArrayDeque<Affine>()
        fun pop(): Affine? = if (stack.isEmpty()) null else stack.removeLast()
        for ((kind, value) in nodes) {
            when (kind) {
                Kind.NUMBER -> stack.addLast(Affine(0.0, 0.0, value))
                Kind.X -> stack.addLast(Affine(1.0, 0.0, 0.0))
                Kind.Y -> stack.addLast(Affine(0.0, 1.0, 0.0))
                Kind.PI -> stack.addLast(Affine(0.0, 0.0, Math.PI))
                Kind.E -> stack.addLast(Affine(0.0, 0.0, Math.E))
                Kind.NEG -> { val a = pop() ?: return null; stack.addLast(Affine(-a.x, -a.y, -a.c)) }
                Kind.ADD, Kind.SUB -> {
                    val b = pop() ?: return null; val a = pop() ?: return null
                    val s = if (kind == Kind.ADD) 1.0 else -1.0
                    stack.addLast(Affine(a.x + s*b.x, a.y + s*b.y, a.c + s*b.c))
                }
                Kind.MUL -> {
                    val b = pop() ?: return null; val a = pop() ?: return null
                    val av = abs(a.x) > CANONICAL_EPS || abs(a.y) > CANONICAL_EPS
                    val bv = abs(b.x) > CANONICAL_EPS || abs(b.y) > CANONICAL_EPS
                    if (av && bv) return null
                    stack.addLast(if (av) Affine(a.x*b.c, a.y*b.c, a.c*b.c) else Affine(b.x*a.c, b.y*a.c, b.c*a.c))
                }
                Kind.DIV -> {
                    val b = pop() ?: return null; val a = pop() ?: return null
                    if (abs(b.x) > CANONICAL_EPS || abs(b.y) > CANONICAL_EPS || abs(b.c) <= CANONICAL_EPS) return null
                    stack.addLast(Affine(a.x/b.c, a.y/b.c, a.c/b.c))
                }
                Kind.POW -> {
                    val exponent = pop() ?: return null; val base = pop() ?: return null
                    if (abs(exponent.x) > CANONICAL_EPS || abs(exponent.y) > CANONICAL_EPS) return null
                    val e = exponent.c.roundToInt()
                    if (abs(exponent.c - e) > 1e-9) return null
                    val baseHasVar = abs(base.x) > CANONICAL_EPS || abs(base.y) > CANONICAL_EPS
                    when {
                        e == 0 -> stack.addLast(Affine(0.0, 0.0, 1.0))
                        e == 1 -> stack.addLast(base)
                        !baseHasVar -> stack.addLast(Affine(0.0, 0.0, base.c.pow(e.toDouble())))
                        else -> return null
                    }
                }
                Kind.SIN, Kind.COS, Kind.TAN, Kind.SQRT, Kind.LOG, Kind.LN, Kind.EXP, Kind.ABS -> {
                    val a = pop() ?: return null
                    if (abs(a.x) > CANONICAL_EPS || abs(a.y) > CANONICAL_EPS) return null
                    val v = constantFunction(kind, a.c) ?: return null
                    stack.addLast(Affine(0.0, 0.0, v))
                }
                else -> return null
            }
        }
        return if (stack.size == 1) stack.removeLast() else null
    }

    private fun polynomialOf(nodes: List<Pair<Int, Double>>, variableKind: Int): DoubleArray? {
        val stack = ArrayDeque<DoubleArray>()
        fun pop(): DoubleArray? = if (stack.isEmpty()) null else stack.removeLast()
        for ((kind, value) in nodes) {
            when (kind) {
                Kind.NUMBER -> stack.addLast(constantPolynomial(value))
                variableKind -> { val p = constantPolynomial(0.0); p[1] = 1.0; stack.addLast(p) }
                Kind.X, Kind.Y -> return null
                Kind.PI -> stack.addLast(constantPolynomial(Math.PI))
                Kind.E -> stack.addLast(constantPolynomial(Math.E))
                Kind.NEG -> { val a = pop() ?: return null; stack.addLast(DoubleArray(a.size) { -a[it] }) }
                Kind.ADD, Kind.SUB -> {
                    val b = pop() ?: return null; val a = pop() ?: return null
                    val s = if (kind == Kind.ADD) 1.0 else -1.0
                    stack.addLast(DoubleArray(a.size) { a[it] + s*b[it] })
                }
                Kind.MUL -> {
                    val b = pop() ?: return null; val a = pop() ?: return null
                    stack.addLast(polynomialMultiply(a, b) ?: return null)
                }
                Kind.DIV -> {
                    val b = pop() ?: return null; val a = pop() ?: return null
                    if (polynomialDegree(b) > 0 || abs(b[0]) <= CANONICAL_EPS) return null
                    stack.addLast(DoubleArray(a.size) { a[it] / b[0] })
                }
                Kind.POW -> {
                    val exponent = pop() ?: return null; val base = pop() ?: return null
                    if (polynomialDegree(exponent) > 0) return null
                    val e = exponent[0].roundToInt()
                    if (abs(exponent[0] - e) > 1e-9 || e !in 0..5) return null
                    stack.addLast(polynomialPower(base, e) ?: return null)
                }
                Kind.SIN, Kind.COS, Kind.TAN, Kind.SQRT, Kind.LOG, Kind.LN, Kind.EXP, Kind.ABS -> {
                    val a = pop() ?: return null
                    if (polynomialDegree(a) > 0) return null
                    stack.addLast(constantPolynomial(constantFunction(kind, a[0]) ?: return null))
                }
                else -> return null
            }
        }
        return if (stack.size == 1) stack.removeLast() else null
    }

    private fun constantPolynomial(v: Double): DoubleArray = DoubleArray(V5ModelSpec.CANONICAL_COEFF_SLOTS).also { it[0] = v }

    private fun polynomialMultiply(a: DoubleArray, b: DoubleArray): DoubleArray? {
        val out = DoubleArray(V5ModelSpec.CANONICAL_COEFF_SLOTS)
        for (i in a.indices) for (j in b.indices) {
            val term = a[i] * b[j]
            if (abs(term) <= CANONICAL_EPS) continue
            if (i + j >= out.size) return null
            out[i + j] += term
        }
        return out
    }

    private fun polynomialPower(base: DoubleArray, exponent: Int): DoubleArray? {
        var out = constantPolynomial(1.0)
        repeat(exponent) { out = polynomialMultiply(out, base) ?: return null }
        return out
    }

    private fun polynomialDegree(p: DoubleArray): Int = (p.lastIndex downTo 0).firstOrNull { abs(p[it]) > CANONICAL_EPS } ?: 0

    private fun canonicalizePolynomial(p: DoubleArray) {
        val scale = p.maxOf { abs(it) }
        if (scale <= CANONICAL_EPS) { p.fill(0.0); return }
        for (i in p.indices) p[i] /= scale
        val degree = polynomialDegree(p)
        if (p[degree] < 0.0) for (i in p.indices) p[i] = -p[i]
        for (i in p.indices) p[i] = cleanZero(p[i])
    }

    private fun constantFunction(kind: Int, v: Double): Double? = runCatching {
        when (kind) {
            Kind.SIN -> sin(v)
            Kind.COS -> cos(v)
            Kind.TAN -> tan(v)
            Kind.SQRT -> sqrt(v)
            Kind.LOG -> log10(v)
            Kind.LN -> ln(v)
            Kind.EXP -> exp(v)
            Kind.ABS -> abs(v)
            else -> return null
        }
    }.getOrNull()?.takeIf { it.isFinite() }

    private fun cleanZero(v: Double): Double = if (abs(v) < 1e-12) 0.0 else v

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
