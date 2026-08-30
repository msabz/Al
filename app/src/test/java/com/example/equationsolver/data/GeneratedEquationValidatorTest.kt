package com.example.equationsolver.data

import org.junit.Assert.assertTrue
import org.junit.Assert.assertEquals
import org.junit.Test
import kotlin.random.Random

class GeneratedEquationValidatorTest {
    @Test fun generatedCurriculumIsMathematicallyValid() {
        val random = Random(20260830)
        repeat(1_000) {
            val sample = EquationGenerator.generate(random)
            assertTrue("Invalid ${sample.family}: ${sample.equation} -> (${sample.x}, ${sample.y})", GeneratedEquationValidator.isValid(sample))
        }
    }

    @Test fun seededCurriculumIsReproducibleAndCoversMajorFamilies() {
        val first = Random(0x0A16)
        val second = Random(0x0A16)
        val firstSequence = List(1_000) { EquationGenerator.generate(first) }
        val secondSequence = List(1_000) { EquationGenerator.generate(second) }

        assertEquals(firstSequence, secondSequence)
        val families = firstSequence.map { it.family }.toSet()
        assertTrue(families.containsAll(setOf(
            "linear", "quadratic", "cubic", "quartic", "quintic", "rational",
            "radical", "absolute", "linear-system"
        )))
        assertTrue(families.any { it.startsWith("exponential") })
        assertTrue(families.any { it.startsWith("logarithmic") })
        assertTrue(families.any { it.startsWith("trigonometric") })
    }
}
