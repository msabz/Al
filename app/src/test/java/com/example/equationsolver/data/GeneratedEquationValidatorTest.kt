package com.example.equationsolver.data

import org.junit.Assert.assertTrue
import org.junit.Test

class GeneratedEquationValidatorTest {
    @Test fun generatedCurriculumIsMathematicallyValid() {
        repeat(300) {
            val sample = EquationGenerator.generate()
            assertTrue("Invalid ${sample.family}: ${sample.equation} -> (${sample.x}, ${sample.y})", GeneratedEquationValidator.isValid(sample))
        }
    }
}
