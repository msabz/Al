package com.example.equationsolver.ai

import kotlin.math.abs

enum class EquationFamily(val id: Int) {
    LINEAR(0),
    POLYNOMIAL(1),
    ANALYTIC(2),
    SYSTEM(3);

    companion object {
        fun fromId(id: Int): EquationFamily = values().firstOrNull { it.id == id } ?: LINEAR
    }
}

enum class SolutionState(val id: Int) {
    FINITE(0),
    NO_SOLUTION(1),
    INFINITE(2),
    UNSUPPORTED(3);

    companion object {
        fun fromId(id: Int): SolutionState = values().firstOrNull { it.id == id } ?: UNSUPPORTED
    }
}

data class V5Target(
    val family: EquationFamily,
    val state: SolutionState,
    val roots: DoubleArray = doubleArrayOf(),
    val systemValues: DoubleArray = doubleArrayOf()
) {
    init {
        require(roots.size <= V5ModelSpec.ROOT_SLOTS) { "عدد الجذور يتجاوز سعة النموذج" }
        require(roots.all { it.isFinite() }) { "الجذور يجب أن تكون أرقامًا محدودة" }
        require(systemValues.all { it.isFinite() }) { "قيم النظام يجب أن تكون محدودة" }
    }

    fun canonicalRoots(): DoubleArray = roots.distinctBy { kotlin.math.round(it * 1e8) }
        .sortedWith(compareBy<Double> { abs(it) }.thenBy { it })
        .take(V5ModelSpec.ROOT_SLOTS)
        .toDoubleArray()
}

data class V5TrainItem(
    val input: StructuralMathEncoder.Encoding,
    val target: V5Target,
    val equivalent: StructuralMathEncoder.Encoding? = null
)

data class V5Prediction(
    val family: EquationFamily,
    val state: SolutionState,
    val stateProbabilities: DoubleArray,
    val slotValues: DoubleArray,
    val presenceProbabilities: DoubleArray
) {
    val roots: DoubleArray
        get() = slotValues.indices
            .filter { presenceProbabilities.getOrElse(it) { 0.0 } >= V5ModelSpec.PRESENCE_THRESHOLD }
            .map { slotValues[it] }
            .sorted()
            .toDoubleArray()

    val x: Double? get() = if (family == EquationFamily.SYSTEM && presenceProbabilities.getOrElse(0) { 0.0 } >= V5ModelSpec.PRESENCE_THRESHOLD) slotValues[0] else null
    val y: Double? get() = if (family == EquationFamily.SYSTEM && presenceProbabilities.getOrElse(1) { 0.0 } >= V5ModelSpec.PRESENCE_THRESHOLD) slotValues[1] else null
}

object V5ModelSpec {
    const val FILE_MAGIC = 0x4D414935 // "MAI5"
    const val FILE_VERSION = 3

    const val MAX_NODES = 80
    const val TOKEN_VOCAB = 22
    const val EMBEDDING_SIZE = 16
    const val EXTRA_FEATURES = 3 // numeric value, position, stack depth
    const val NODE_FEATURES = EMBEDDING_SIZE + EXTRA_FEATURES
    const val INPUT_SIZE = MAX_NODES * NODE_FEATURES

    const val SHARED_1 = 96
    const val SHARED_2 = 64
    const val HEAD_HIDDEN = 48
    const val HEAD_COUNT = 4

    const val ROOT_SLOTS = 5
    const val STATE_COUNT = 4
    const val HEAD_OUTPUT = ROOT_SLOTS + ROOT_SLOTS + STATE_COUNT

    const val ROOT_SCALE = 100.0
    const val PRESENCE_THRESHOLD = 0.50
    const val MAX_GRADIENT_NORM = 5.0
    const val WEIGHT_DECAY = 1e-5
    const val CANONICAL_COEFF_SLOTS = 6
    const val POLYNOMIAL_FEATURE_SLOTS = 7 // six q(z)=P(ROOT_SCALE*z) coefficients + degree/5
    const val SYSTEM_FEATURE_SLOTS = 9     // two scaled rows + normalized Cramer invariants (det,nx,ny)
    const val POLYNOMIAL_RESIDUAL_WEIGHT = 0.15
}
