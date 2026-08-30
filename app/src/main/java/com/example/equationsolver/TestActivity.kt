package com.example.equationsolver

import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.example.equationsolver.ai.ModelManager
import com.example.equationsolver.core.ArabicEquationNormalizer
import com.example.equationsolver.core.MathTeacher
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
            val raw = input.text.toString().trim()
            if (raw.isEmpty()) {
                input.error = "أدخل معادلة أولاً"
                input.requestFocus()
                Toast.makeText(this, "أدخل معادلة للحل", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            val equation = ArabicEquationNormalizer.normalize(raw)
            try {
                val answer = MathTeacher.solve(equation)
                typeText.text = "نوع المسألة: ${answer.type}"
                exactText.text = "الجواب الصحيح:\n${answer.summary}"
                stepsText.text = if (answer.steps.isEmpty()) "" else "طريقة التحقق:\n" +
                    answer.steps.mapIndexed { i, step -> "${i + 1}. $step" }.joinToString("\n")

                val prediction = ModelManager.predict(equation)
                val px = prediction.getOrElse(0) { 0.0 } * 100.0
                val py = prediction.getOrElse(1) { 0.0 } * 100.0
                aiText.text = when {
                    answer.x != null && answer.y != null -> {
                        val error = abs(px - answer.x) + abs(py - answer.y)
                        "جواب النموذج:\nx ≈ ${fmt(px)}\ny ≈ ${fmt(py)}\nمجموع الخطأ = ${fmt(error)}"
                    }
                    answer.x != null -> {
                        val error = abs(px - answer.x)
                        "جواب النموذج:\nx ≈ ${fmt(px)}\nالخطأ عن الجذر المرجعي = ${fmt(error)}"
                    }
                    answer.y != null -> {
                        val error = abs(py - answer.y)
                        "جواب النموذج:\ny ≈ ${fmt(py)}\nالخطأ = ${fmt(error)}"
                    }
                    else -> "جواب النموذج الخام:\nx ≈ ${fmt(px)}\ny ≈ ${fmt(py)}\nلا يوجد حل عددي مرجعي للمقارنة بهذه الحالة."
                }
            } catch (e: Exception) {
                typeText.text = ""
                exactText.text = "تعذر التحقق من المعادلة:\n${e.message ?: "خطأ غير معروف"}"
                stepsText.text = ""
                aiText.text = ""
            }
        }
    }

    private fun fmt(value: Double): String =
        if (abs(value) < 1e-9) "0" else String.format(Locale.US, "%.8f", value).trimEnd('0').trimEnd('.')
}
