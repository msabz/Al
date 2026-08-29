package com.example.equationsolver

import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.example.equationsolver.ai.ModelManager
import com.example.equationsolver.core.UniversalEquationSolver
import java.util.Locale
import kotlin.math.abs

class TestActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_test)

        ModelManager.init(applicationContext)
        val input = findViewById<EditText>(R.id.editEquation)
        val typeText = findViewById<TextView>(R.id.textType)
        val exactText = findViewById<TextView>(R.id.textResult)
        val stepsText = findViewById<TextView>(R.id.textSteps)
        val aiText = findViewById<TextView>(R.id.textAi)
        val solveButton = findViewById<Button>(R.id.btnSolve)
        val clearButton = findViewById<Button>(R.id.btnClear)

        input.isEnabled = true
        input.isFocusable = true
        input.isFocusableInTouchMode = true
        input.isCursorVisible = true

        clearButton.setOnClickListener {
            input.text.clear()
            typeText.text = ""
            exactText.text = ""
            stepsText.text = ""
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
                typeText.text = "نوع المسألة: ${UniversalEquationSolver.equationType(equation)}"
                val result = UniversalEquationSolver.solve(equation)
                exactText.text = "الحل الدقيق:\n${result.summary}"
                stepsText.text = "خطوات الحل:\n" + result.steps.mapIndexed { i, step -> "${i + 1}. $step" }.joinToString("\n")

                try {
                    val prediction = ModelManager.predict(equation)
                    val px = prediction.getOrNull(0)?.times(100.0)
                    val py = prediction.getOrNull(1)?.times(100.0)
                    aiText.text = when {
                        px == null -> "تقدير AI غير متاح"
                        py != null && abs(py) >= 1e-9 -> "تقدير AI:\nx ≈ ${fmt(px)}\ny ≈ ${fmt(py)}"
                        else -> "تقدير AI:\nx ≈ ${fmt(px)}"
                    }
                } catch (e: Exception) {
                    aiText.text = "تقدير AI غير متاح:\n${e.message ?: "خطأ غير معروف"}"
                }
            } catch (e: Exception) {
                typeText.text = ""
                exactText.text = "تعذر حل المعادلة:\n${e.message ?: "خطأ غير معروف"}"
                stepsText.text = ""
            }
        }
    }

    private fun fmt(value: Double): String =
        if (abs(value) < 1e-9) "0" else String.format(Locale.US, "%.6f", value).trimEnd('0').trimEnd('.')
}
