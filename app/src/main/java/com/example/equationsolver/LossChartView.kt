package com.example.equationsolver

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Path
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min

class LossChartView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null
) : View(context, attrs) {
    private data class Point(val training: Double, val validation: Double)

    private val values = ArrayDeque<Point>()
    private val trainPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = ContextCompat.getColor(context, R.color.mint)
        strokeWidth = 4f
        style = Paint.Style.STROKE
    }
    private val validationPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = ContextCompat.getColor(context, R.color.sky)
        strokeWidth = 4f
        style = Paint.Style.STROKE
    }
    private val gridPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = ContextCompat.getColor(context, R.color.border)
        strokeWidth = 1f
    }
    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = ContextCompat.getColor(context, R.color.text_secondary)
        textSize = 27f
        style = Paint.Style.FILL
    }

    fun addLoss(loss: Double) {
        addMetrics(loss, Double.NaN)
    }

    fun addMetrics(trainingLoss: Double, validationLoss: Double) {
        if (!trainingLoss.isFinite()) return
        values.addLast(Point(trainingLoss.coerceAtLeast(0.0), validationLoss.takeIf { it.isFinite() } ?: Double.NaN))
        while (values.size > 120) values.removeFirst()
        postInvalidate()
    }

    fun clearData() { values.clear(); invalidate() }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        canvas.drawText("Train MSE", 18f, 34f, textPaint)
        textPaint.color = ContextCompat.getColor(context, R.color.sky)
        canvas.drawText("Holdout", 178f, 34f, textPaint)
        textPaint.color = ContextCompat.getColor(context, R.color.text_secondary)
        if (values.size < 2) return

        val left = 20f
        val top = 54f
        val right = width - 20f
        val bottom = height - 24f
        repeat(4) { row ->
            val y = top + (bottom - top) * row / 3f
            canvas.drawLine(left, y, right, y, gridPaint)
        }

        val logValues = values.flatMap { point ->
            buildList {
                if (point.training.isFinite()) add(logValue(point.training))
                if (point.validation.isFinite()) add(logValue(point.validation))
            }
        }
        val maxValue = max(logValues.maxOrNull() ?: 0.0, -20.0)
        val minValue = min(logValues.minOrNull() ?: -1.0, maxValue)
        val range = max(maxValue - minValue, 0.25)

        drawSeries(canvas, values.map { it.training }, trainPaint, left, top, right, bottom, minValue, range)
        drawSeries(canvas, values.map { it.validation }, validationPaint, left, top, right, bottom, minValue, range)
    }

    private fun drawSeries(
        canvas: Canvas,
        series: List<Double>,
        paint: Paint,
        left: Float,
        top: Float,
        right: Float,
        bottom: Float,
        minValue: Double,
        range: Double
    ) {
        val path = Path()
        var drawing = false
        series.forEachIndexed { index, value ->
            if (!value.isFinite()) {
                drawing = false
                return@forEachIndexed
            }
            val x = left + (right - left) * index.toFloat() / (values.size - 1).toFloat()
            val normalized = (logValue(value) - minValue) / range
            val y = bottom - (bottom - top) * normalized.toFloat()
            if (!drawing) path.moveTo(x, y) else path.lineTo(x, y)
            drawing = true
        }
        canvas.drawPath(path, paint)
    }

    private fun logValue(value: Double): Double = ln(value.coerceAtLeast(1e-10))
}
