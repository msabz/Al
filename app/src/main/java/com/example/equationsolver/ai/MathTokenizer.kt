package com.example.equationsolver.ai

import com.example.equationsolver.core.ArabicEquationNormalizer

/** Lightweight positional tokenizer for mathematical equations. */
object MathTokenizer {
    const val MAX_TOKENS = 72

    private val vocabulary = listOf(
        "<pad>", "<unk>",
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        ".", "x", "y", "+", "-", "*", "/", "^", "=", ";", "(", ")",
        "sin", "cos", "tan", "sqrt", "log", "ln", "exp", "abs", "pi", "e"
    )
    private val ids = vocabulary.withIndex().associate { it.value to it.index }
    private val namedTokens = listOf("sqrt", "sin", "cos", "tan", "log", "exp", "abs", "pi", "ln")

    val vocabSize: Int get() = vocabulary.size

    fun tokenize(input: String): IntArray {
        val normalized = ArabicEquationNormalizer.normalize(input)
            .lowercase()
            .replace("π", "pi")
            .replace("√", "sqrt")
            .replace("²", "^2")
            .replace("³", "^3")

        val result = IntArray(MAX_TOKENS)
        var out = 0
        var i = 0
        while (i < normalized.length && out < MAX_TOKENS) {
            val ch = normalized[i]
            if (ch.isWhitespace()) {
                i++
                continue
            }

            val named = namedTokens.firstOrNull { token -> normalized.startsWith(token, i) }
            if (named != null) {
                result[out++] = ids.getValue(named)
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
            result[out++] = ids[token] ?: 1
            i++
        }
        return result
    }
}
