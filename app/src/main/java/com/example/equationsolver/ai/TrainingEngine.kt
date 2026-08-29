package com.example.equationsolver.ai

import com.example.equationsolver.core.EquationFeatures
import com.example.equationsolver.core.UniversalEquationSolver
import com.example.equationsolver.data.EquationGenerator
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.isActive

object TrainingEngine {
    const val BATCH_SIZE = 64
    const val EPOCHS = 3
    const val VALIDATION_RATIO = 0.10

    @Volatile var lastValidationMse: Double = 0.0
        private set

    fun trainRandom(samples: Int, progress: (Int) -> Unit = {}) {
        trainRandomInternal(samples, progress, {})
    }

    suspend fun trainContinuous(
        learningRate: Double = 0.001,
        progress: (samples: Long, batches: Long, epoch: Long, loss: Double, validationMse: Double) -> Unit = { _, _, _, _, _ -> }
    ) {
        var samples = 0L
        var batches = 0L
        var epoch = 0L
        val inputs = ArrayList<DoubleArray>(BATCH_SIZE)
        val targets = ArrayList<DoubleArray>(BATCH_SIZE)

        while (currentCoroutineContext().isActive) {
            val sample = EquationGenerator.generate()
            val result = UniversalEquationSolver.solve(sample.equation)
            val target = solutionVector(result) ?: continue
            inputs += EquationFeatures.fromInput(sample.equation).values
            targets += target
            samples++

            if (inputs.size == BATCH_SIZE) {
                val loss = ModelManager.nn.trainBatch(inputs.toTypedArray(), targets.toTypedArray(), learningRate)
                batches++
                if (batches % 5L == 0L) {
                    epoch = samples / 1000L
                    lastValidationMse = validationOnFreshRandomSet(128)
                    progress(samples, batches, epoch, loss, lastValidationMse)
                }
                inputs.clear()
                targets.clear()
            }
        }
    }

    private fun trainRandomInternal(samples: Int, progress: (Int) -> Unit, lossProgress: (Double) -> Unit) {
        val inputs = ArrayList<DoubleArray>(BATCH_SIZE)
        val targets = ArrayList<DoubleArray>(BATCH_SIZE)
        for (i in 1..samples) {
            val sample = EquationGenerator.generate()
            val result = UniversalEquationSolver.solve(sample.equation)
            val target = solutionVector(result) ?: continue
            inputs += EquationFeatures.fromInput(sample.equation).values
            targets += target
            if (inputs.size == BATCH_SIZE || i == samples) {
                val loss = ModelManager.nn.trainBatch(inputs.toTypedArray(), targets.toTypedArray(), 0.001)
                lossProgress(loss)
                inputs.clear()
                targets.clear()
            }
            if (i % 500 == 0) progress(i)
        }
    }

    fun trainFile(examples: List<Pair<String, DoubleArray>>, progress: (Int) -> Unit = {}) {
        if (examples.isEmpty()) return
        val indices = IntArray(examples.size) { it }
        indices.shuffle()
        val validationCount = maxOf(1, (examples.size * VALIDATION_RATIO).toInt())
        val trainCount = examples.size - validationCount
        repeat(EPOCHS) { epoch ->
            val inputs = ArrayList<DoubleArray>(BATCH_SIZE)
            val targets = ArrayList<DoubleArray>(BATCH_SIZE)
            for (position in 0 until trainCount) {
                val pair = examples[indices[position]]
                val result = UniversalEquationSolver.solve(pair.first)
                val target = solutionVector(result) ?: suppliedTarget(pair.second)
                inputs += EquationFeatures.fromInput(pair.first).values
                targets += target
                if (inputs.size == BATCH_SIZE || position == trainCount - 1) {
                    ModelManager.nn.trainBatch(inputs.toTypedArray(), targets.toTypedArray(), 0.001)
                    inputs.clear(); targets.clear()
                }
                if ((position + 1) % 1000 == 0) progress(epoch * trainCount + position + 1)
            }
            lastValidationMse = validationMse(examples, indices, trainCount)
        }
    }

    private fun validationOnFreshRandomSet(size: Int): Double {
        val inputs = ArrayList<DoubleArray>(size)
        val targets = ArrayList<DoubleArray>(size)
        repeat(size) {
            val sample = EquationGenerator.generate()
            val target = solutionVector(UniversalEquationSolver.solve(sample.equation)) ?: return@repeat
            inputs += EquationFeatures.fromInput(sample.equation).values
            targets += target
        }
        return ModelManager.nn.meanSquaredError(inputs, targets)
    }

    private fun validationMse(examples: List<Pair<String, DoubleArray>>, indices: IntArray, start: Int): Double {
        var total = 0.0; var count = 0
        for (position in start until indices.size) {
            val pair = examples[indices[position]]
            val result = UniversalEquationSolver.solve(pair.first)
            val target = solutionVector(result) ?: suppliedTarget(pair.second)
            val prediction = ModelManager.nn.predict(EquationFeatures.fromInput(pair.first).values)
            for (j in prediction.indices) { val error = prediction[j] - target[j]; total += error * error }
            count++
        }
        return if (count == 0) 0.0 else total / (count * 2.0)
    }

    private fun suppliedTarget(values: DoubleArray) = doubleArrayOf(values.getOrElse(0) { 0.0 } / 100.0, values.getOrElse(1) { 0.0 } / 100.0)

    private fun solutionVector(result: UniversalEquationSolver.Result): DoubleArray? {
        val x = result.x ?: return null
        return doubleArrayOf(x / 100.0, (result.y ?: 0.0) / 100.0)
    }
}
