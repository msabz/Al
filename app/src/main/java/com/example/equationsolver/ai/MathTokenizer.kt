package com.example.equationsolver.ai

import com.example.equationsolver.core.ArabicEquationNormalizer

/** Lightweight positional tokenizer for mathematical equations. */
object MathTokenizer {
    const val MAX_TOKENS = 72

    data class Encoding(
        val tokens: IntArray,
        val tokenCount: Int,
        val unknownCount: Int,
        val truncated: Boolean
    )

    private val vocabulary = listOf(
        "<pad>", "<unk>",
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        ".", "x", "y", "+", "-", "*", "/", "^", "=", ";", "(", ")",
        "sin", "cos", "tan", "sqrt", "log", "ln", "exp", "abs", "pi", "e"
    )
    private val ids = vocabulary.withIndex().associate { it.value to it.index }
    private val namedTokens = listOf("sqrt", "sin", "cos", "tan", "log", "exp", "abs", "pi", "ln")

    val vocabSize: Int get() = vocabulary.size

    fun tokenize(input: String): IntArray = encode(input).tokens

    fun encode(input: String): Encoding {
        val normalized = ArabicEquationNormalizer.normalize(input)
            .lowercase()
            .replace("π", "pi")
            .replace("√", "sqrt")
            .replace("²", "^2")
            .replace("³", "^3")

        val result = IntArray(MAX_TOKENS)
        var out = 0
        var tokenCount = 0
        var unknownCount = 0
        var truncated = false
        var i = 0
        while (i < normalized.length) {
            val ch = normalized[i]
            if (ch.isWhitespace()) {
                i++
                continue
            }

            val named = namedTokens.firstOrNull { token -> normalized.startsWith(token, i) }
            if (named != null) {
                tokenCount++
                if (out < MAX_TOKENS) result[out++] = ids.getValue(named) else truncated = true
                i += named.length
                continue
            }

            val token = when {
                ch.isDigit() -> ch.toString()
                ch == '.' -> "."
                ch == 'x' -> "x"
                ch == 'y' -> "y"
                ch == 'e' -> "e"
                ch in charArrayOf('+', '-', '*', '/', '^', '=', ';', '(', ')') -> ch.toString()
                else -> null
            }
            val id = ids[token] ?: 1
            tokenCount++
            if (id == 1) unknownCount++
            if (out < MAX_TOKENS) result[out++] = id else truncated = true
            i++
        }
        return Encoding(result, tokenCount, unknownCount, truncated)
    }
}
