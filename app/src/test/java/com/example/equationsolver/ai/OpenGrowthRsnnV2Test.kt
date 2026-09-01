package com.example.equationsolver.ai

import com.example.equationsolver.data.DeepMindSample
import org.junit.Assert.*
import org.junit.Test
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.DataInputStream
import java.io.DataOutputStream

class OpenGrowthRsnnV2Test {
    @Test fun defaultP30_int8_train_and_checkpoint_roundtrip() {
        val cfg = ModelConfig(); val m = OpenGrowthRsnnV2(cfg)
        assertEquals(18816, m.activeWeights())
        val p = m.predictInt8(floatArrayOf(1f,0f,.25f,0f,1f,-.5f)); assertEquals(2, p.size); assertTrue(p.all { it.isFinite() })
        val t = m.trainBatch(listOf(DeepMindSample(floatArrayOf(1f,0f,.25f,0f,1f,-.5f), floatArrayOf(25f,-50f)))); assertTrue(t.loss.isFinite()); assertEquals(1L, t.step)
        val bytes = ByteArrayOutputStream(); DataOutputStream(bytes).use { m.save(it) }
        val restored = DataInputStream(ByteArrayInputStream(bytes.toByteArray())).use { OpenGrowthRsnnV2.load(it, cfg) }
        assertEquals(m.activeWeights(), restored.activeWeights()); assertEquals(m.step, restored.step)
    }
}
