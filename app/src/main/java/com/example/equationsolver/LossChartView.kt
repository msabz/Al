package com.example.equationsolver

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Path
import android.util.AttributeSet
import android.view.View
import kotlin.math.max
import kotlin.math.min
import kotlin.math.ln

class LossChartView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null
) : View(context, attrs) {
    private val values = ArrayDeque<Double>()
    private val linePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { strokeWidth = 5f; style = Paint.Style.STROKE }
    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { textSize = 30f; style = Paint.Style.FILL }

    fun addLoss(loss: Double) {
        if (!loss.isFinite()) return
        values.addLast(loss.coerceAtLeast(0.0))
        while (values.size > 120) values.removeFirst()
        postInvalidate()
    }

    fun clearData() { values.clear(); invalidate() }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        canvas.drawText("Loss", 16f, 34f, textPaint)
        if (values.size < 2) return
        val left = 20f; val top = 50f; val right = width - 20f; val bottom = height - 20f
        val maxValue = max(values.maxOrNull() ?: 1.0, 1e-9)
        val minValue = min(values.minOrNull() ?: 0.0, maxValue)
        val range = max(maxValue - minValue, maxValue * 0.02)
        val path = Path()
        values.forEachIndexed { index, value ->
            val x = left + (right - left) * index.toFloat() / (values.size - 1).toFloat()
            val normalized = (value - minValue) / range
            val y = bottom - (bottom - top) * normalized.toFloat()
            if (index == 0) path.moveTo(x, y) else path.lineTo(x, y)
        }
        canvas.drawPath(path, linePaint)
        val last = values.last()
        canvas.drawText("%.6f".format(last), 16f, height - 2f, textPaint)
    }
}
