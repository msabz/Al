package com.example.equationsolver.data

/** Generates only examples that pass the independent verifier. */
object VerifiedEquationSuggester {
    fun next(): GeneratedExample {
        repeat(100) {
            val sample = EquationGenerator.generate()
            if (GeneratedEquationValidator.isValid(sample)) return sample
        }
        error("تعذر توليد معادلة صحيحة الآن")
    }
}
