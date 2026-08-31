package com.example.equationsolver.ai

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.BufferedInputStream
import java.io.DataInputStream
import java.io.File

/**
 * Colab injects default_model.mai5 + v5_interop_expected.tsv before Gradle runs.
 * Normal repository CI has no embedded trained model yet, so the test is a no-op there.
 */
class EmbeddedMai5InteropTest {
    @Test
    fun embeddedColabModelMatchesPythonReferenceWhenPresent() {
        val assets = locateAssetsDir()
        val modelFile = File(assets, "default_model.mai5")
        val expectedFile = File(assets, "v5_interop_expected.tsv")
        if (!modelFile.isFile && !expectedFile.isFile) return

        assertTrue("default_model.mai5 missing", modelFile.isFile)
        assertTrue("v5_interop_expected.tsv missing", expectedFile.isFile)

        val network = NeuralNetwork()
        DataInputStream(BufferedInputStream(modelFile.inputStream())).use(network::loadState)

        var checked = 0
        expectedFile.forEachLine { raw ->
            if (raw.isBlank() || raw.startsWith("#")) return@forEachLine
            val parts = raw.split('\t')
            require(parts.size == 6) { "Bad interop row: $raw" }
            val equation = parts[0]
            val family = parts[1].toInt()
            val state = parts[2].toInt()
            val expectedSlots = csv(parts[3])
            val expectedPresence = csv(parts[4])
            val expectedStateProbs = csv(parts[5])

            val prediction = network.predict(StructuralMathEncoder.encode(equation))
            assertEquals("family for $equation", family, prediction.family.id)
            assertEquals("state for $equation", state, prediction.state.id)
            assertEquals(V5ModelSpec.ROOT_SLOTS, expectedSlots.size)
            assertEquals(V5ModelSpec.ROOT_SLOTS, expectedPresence.size)
            assertEquals(V5ModelSpec.STATE_COUNT, expectedStateProbs.size)

            for (i in expectedSlots.indices) {
                assertEquals("slot $i for $equation", expectedSlots[i], prediction.slotValues[i], 0.02)
                assertEquals("presence $i for $equation", expectedPresence[i], prediction.presenceProbabilities[i], 2e-4)
            }
            for (i in expectedStateProbs.indices) {
                assertEquals("state probability $i for $equation", expectedStateProbs[i], prediction.stateProbabilities[i], 2e-4)
            }
            checked++
        }
        assertTrue("Interop sidecar must contain test equations", checked >= 4)
    }

    private fun locateAssetsDir(): File {
        val cwd = File(System.getProperty("user.dir"))
        val candidates = listOf(
            File(cwd, "src/main/assets"),
            File(cwd, "app/src/main/assets"),
            File(cwd.parentFile ?: cwd, "app/src/main/assets")
        )
        return candidates.firstOrNull { it.exists() } ?: candidates.first()
    }

    private fun csv(text: String): DoubleArray =
        if (text.isBlank()) doubleArrayOf() else text.split(',').map { it.toDouble() }.toDoubleArray()
}
