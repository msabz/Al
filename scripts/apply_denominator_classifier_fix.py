#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path, old, new):
    p = ROOT / path
    text = p.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one target, found {count}: {old[:120]!r}")
    p.write_text(text.replace(old, new, 1))
    print("patched", path)

# Kotlin: inspect only the immediate denominator factor after each slash.
replace_once(
    "app/src/main/java/com/example/equationsolver/ai/StructuralMathEncoder.kt",
    '''    private fun hasVariableInDenominator(s: String): Boolean {
        val slash = s.indexOf('/')
        if (slash < 0) return false
        val after = s.substring(slash + 1)
        return after.contains('x') || after.contains('y')
    }
''',
    '''    private fun hasVariableInDenominator(s: String): Boolean {
        var slash = s.indexOf('/')
        while (slash >= 0) {
            var i = slash + 1
            while (i < s.length && (s[i] == '+' || s[i] == '-')) i++
            if (i >= s.length) return false
            if (s[i] == '(') {
                var depth = 0
                val start = i
                var end = -1
                while (i < s.length) {
                    if (s[i] == '(') depth++
                    else if (s[i] == ')') {
                        depth--
                        if (depth == 0) { end = i; break }
                    }
                    i++
                }
                if (end < 0) return true
                val factor = s.substring(start + 1, end)
                if (factor.contains('x') || factor.contains('y')) return true
            } else if (s[i] == 'x' || s[i] == 'y') {
                return true
            }
            slash = s.indexOf('/', slash + 1)
        }
        return false
    }
'''
)

# Python mirror: same immediate-factor rule. A numeric denominator such as /3
# must not be treated as variable-bearing merely because x/y occurs later.
replace_once(
    "colab/train_v5_deepmind.py",
    '''    slash=s.find("/")
    if slash>=0 and ("x" in s[slash+1:] or "y" in s[slash+1:]): return ANALYTIC
''',
    '''    def variable_denominator(text):
        slash=text.find("/")
        while slash>=0:
            i=slash+1
            while i<len(text) and text[i] in "+-": i+=1
            if i>=len(text): return False
            if text[i]=="(":
                depth=0; start=i; end=-1
                while i<len(text):
                    if text[i]=="(": depth+=1
                    elif text[i]==")":
                        depth-=1
                        if depth==0: end=i; break
                    i+=1
                if end<0: return True
                factor=text[start+1:end]
                if "x" in factor or "y" in factor: return True
            elif text[i] in "xy":
                return True
            slash=text.find("/",slash+1)
        return False
    if variable_denominator(s): return ANALYTIC
'''
)

# Add regression tests directly to the canonical encoder test produced by v3 migration.
p = ROOT / "app/src/test/java/com/example/equationsolver/ai/CanonicalNumericEncoderTest.kt"
text = p.read_text()
anchor = '''    @Test fun polynomialFactoredAndExpandedAreCanonicalAndRootScaled() {
'''
insert = '''    @Test fun coefficientFractionsDoNotRoutePolynomialToAnalytic() {
        val e = encoding("54*x^5-15556*x^4/3+55154*x^3-153764*x^2/3+1232*x=0")
        assertEquals(EquationFamily.POLYNOMIAL, e.family)
        assertEquals(V5ModelSpec.POLYNOMIAL_FEATURE_SLOTS, e.nodeCount)
        assertEquals(EquationFamily.ANALYTIC, StructuralMathEncoder.classify("1/x=2"))
        assertEquals(EquationFamily.ANALYTIC, StructuralMathEncoder.classify("1/(x+1)=2"))
    }

'''
if text.count(anchor) != 1:
    raise RuntimeError("CanonicalNumericEncoderTest polynomial anchor missing")
p.write_text(text.replace(anchor, insert + anchor, 1))
print("patched classifier regression tests")

print("DENOMINATOR_CLASSIFIER_FIX_OK")
