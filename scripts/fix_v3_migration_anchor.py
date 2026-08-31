#!/usr/bin/env python3
from pathlib import Path
p = Path(__file__).with_name('apply_v5_v3_arithmetic_residual.py')
text = p.read_text()
old = '''needle = "        val stateStart = V5ModelSpec.ROOT_SLOTS * 2\\n"\ninsert = """'''
new = '''needle = (\n    "        val stateStart = V5ModelSpec.ROOT_SLOTS * 2\\n"\n    "        val stateLogits = FloatArray(V5ModelSpec.STATE_COUNT) { out[stateStart + it] }\\n"\n)\ninsert = """'''
if text.count(old) != 1:
    raise RuntimeError(f'old anchor declaration count={text.count(old)}')
text = text.replace(old, new, 1)
old2 = '''if text.count(needle) != 1:\n    raise RuntimeError("NeuralNetwork.kt: stateStart target count != 1")\np.write_text(text.replace(needle, insert + needle, 1))'''
new2 = '''if text.count(needle) != 1:\n    raise RuntimeError("NeuralNetwork.kt: supervised state block target count != 1")\np.write_text(text.replace(needle, insert + needle, 1))'''
if text.count(old2) != 1:
    raise RuntimeError(f'old anchor use count={text.count(old2)}')
text = text.replace(old2, new2, 1)
p.write_text(text)
print('V3_MIGRATION_ANCHOR_FIXED')
