package com.example.equationsolver

import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.example.equationsolver.ai.ModelManager
import com.example.equationsolver.core.ExactSolver
import com.example.equationsolver.core.SolutionResult
import java.util.Locale
import kotlin.math.abs

class TestActivity : AppCompatActivity() {
    private val solver = ExactSolver()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_test)

        // Defensive initialization in case this activity is launched from a non-standard entry point.
        ModelManager.init(applicationContext)

        val input = findViewById<EditText>(R.id.editEquation)
        val exactText = findViewById<TextView>(R.id.textResult)
        val aiText = findViewById<TextView>(R.id.textAi)
        val solveButton = findViewById<Button>(R.id.btnSolve)
        val clearButton = findViewById<Button>(R.id.btnClear)

        input.isEnabled = true
        input.isFocusable = true
        input.isFocusableInTouchMode = true
        input.isCursorVisible = true

        clearButton.setOnClickListener {
            input.text.clear()
            exactText.text = ""
            aiText.text = ""
            input.requestFocus()
        }

        solveButton.setOnClickListener {
            val equation = input.text.toString().trim()
            if (equation.isEmpty()) {
                input.error = "أدخل معادلة أولاً"
                input.requestFocus()
                Toast.makeText(this, "أدخل معادلة للحل", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            try {
                when (val result = solver.solve(equation)) {
                    is SolutionResult.SingleVariable -> {
                        exactText.text = "الحل الدقيق:\nx = ${fmt(result.x)}"
                    }
                    is SolutionResult.TwoVariables -> {
                        exactText.text = "الحل الدقيق:\nx = ${fmt(result.x)}\ny = ${fmt(result.y)}"
                    }
                    SolutionResult.NoSolution -> {
                        exactText.text = "الحل الدقيق:\nلا يوجد حل."
                    }
                    SolutionResult.InfiniteSolutions -> {
                        exactText.text = "الحل الدقيق:\nعدد لا نهائي من الحلول."
                    }
                    is SolutionResult.Error -> {
                        exactText.text = "خطأ في المعادلة:\n${result.message}"
                    }
                }

                try {
                    val prediction = ModelManager.predict(equation)
                    val px = prediction.getOrNull(0)?.times(100.0) ?: 0.0
                    val py = prediction.getOrNull(1)?.times(100.0) ?: 0.0
                    aiText.text = if (abs(py) < 1e-9) {
                        "تقدير AI:\nx = ${fmt(px)}"
                    } else {
                        "تقدير AI:\nx = ${fmt(px)}\ny = ${fmt(py)}"
                    }
                } catch (e: Exception) {
                    aiText.text = "تقدير AI غير متاح:\n${e.message ?: "خطأ غير معروف"}"
                }
            } catch (e: Exception) {
                exactText.text = "تعذر حل المعادلة:\n${e.message ?: "خطأ غير معروف"}"
            }
        }
    }

    private fun fmt(value: Double): String =
        if (abs(value) < 1e-9) "0" else String.format(Locale.US, "%.6f", value)
}
