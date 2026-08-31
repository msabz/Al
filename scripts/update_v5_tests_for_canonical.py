#!/usr/bin/env python3
from pathlib import Path

path = Path("app/src/test/java/com/example/equationsolver/ai/NeuralNetworkTest.kt")
text = path.read_text()

old = '''    @Test fun structuralEncoderTreatsWholeNumberAsOneNode() {
        val e = StructuralMathEncoder.encode("12.5x+4=29")
        assertEquals(EquationFamily.LINEAR, e.family)
        assertEquals(3, e.kinds.take(e.nodeCount).count { it == StructuralMathEncoder.Kind.NUMBER })
        assertTrue(!e.truncated)
    }
'''
new = '''    @Test fun linearEncoderUsesCanonicalCoefficientSlots() {
        val e = StructuralMathEncoder.encode("12.5x+4=29")
        assertEquals(EquationFamily.LINEAR, e.family)
        assertEquals(V5ModelSpec.CANONICAL_COEFF_SLOTS, e.nodeCount)
        assertTrue((0 until V5ModelSpec.CANONICAL_COEFF_SLOTS).all {
            e.kinds[it] == StructuralMathEncoder.Kind.NUMBER
        })
        assertTrue(!e.truncated)
    }
'''
if text.count(old) != 1:
    raise SystemExit(f"old structural encoder test target count={text.count(old)}")
text = text.replace(old, new, 1)

old_count = '        assertEquals(300_984, n.parameterCount())'
new_count = '        assertEquals(167_800, n.parameterCount())'
if text.count(old_count) != 1:
    raise SystemExit(f"old parameter-count assertion target count={text.count(old_count)}")
text = text.replace(old_count, new_count, 1)

path.write_text(text)
print("CANONICAL_TEST_CONTRACT_UPDATED")
