package com.example.equationsolver

import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.example.equationsolver.ai.ModelManager
import com.example.equationsolver.ai.MathTokenizer
import com.example.equationsolver.core.ArabicEquationNormalizer
import com.example.equationsolver.core.MathTeacher
import java.util.Locale
import kotlin.math.abs

class TestActivity : AppCompatActivity() {
    @Volatile
    private var requestId = 0

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

        findViewById<Button>(R.id.btnBackTest).setOnClickListener { finish() }
        findViewById<Button>(R.id.btnExampleLinear).setOnClickListener { input.setText("7x-4=3x+20") }
        findViewById<Button>(R.id.btnExamplePolynomial).setOnClickListener { input.setText("(x-2)*(x+3)=0") }
        findViewById<Button>(R.id.btnExampleFunction).setOnClickListener { input.setText("ln(2x+1)=1.60943791") }
        findViewById<Button>(R.id.btnExampleSystem).setOnClickListener { input.setText("2x+3y=5;x-y=1") }

        findViewById<TextView>(R.id.textTestTrainingState).text = if (ModelManager.isTrainingEnabled(this)) {
            "● التدريب مستمر بالخلفية. الاختبار يقرأ الأوزان الحالية دون إيقافه."
        } else {
            "جواب النموذج يأتي من الأوزان فقط؛ الجواب المرجعي منفصل للمقارنة."
        }

        findViewById<Button>(R.id.btnClear).setOnClickListener {
            requestId++
            input.text.clear()
            typeText.text = ""
            exactText.text = "—"
            stepsText.text = ""
            aiText.text = "لم يُجرَ اختبار بعد"
            solveButton.isEnabled = true
            input.requestFocus()
        }

        solveButton.setOnClickListener {
            val raw = input.text.toString().trim()
            if (raw.isEmpty()) {
                input.error = "أدخل معادلة أولاً"
                Toast.makeText(this, "أدخل معادلة للاختبار", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            val equation = ArabicEquationNormalizer.normalize(raw)
            val encoding = MathTokenizer.encode(equation)
            if (encoding.truncated || encoding.unknownCount > 0) {
                input.error = if (encoding.truncated) "المعادلة تتجاوز ${MathTokenizer.MAX_TOKENS} token" else "توجد رموز غير مدعومة"
                return@setOnClickListener
            }
            val currentRequest = ++requestId
            solveButton.isEnabled = false
            aiText.text = "النموذج ينفّذ Forward pass..."
            exactText.text = "المصحح يحسب الجواب المرجعي..."
            stepsText.text = ""
            typeText.text = ""

            Thread {
                try {
                    // This runs first and calls only tokenizer + neural weights.
                    val prediction = ModelManager.predictValues(equation)
                    // The teacher is invoked separately and only for the comparison card.
                    val answer = MathTeacher.solve(equation)
                    runOnUiThread {
                        if (currentRequest != requestId || isFinishing || isDestroyed) return@runOnUiThread
                        renderResult(prediction, answer, typeText, exactText, stepsText, aiText)
                        solveButton.isEnabled = true
                    }
                } catch (e: Exception) {
                    runOnUiThread {
                        if (currentRequest != requestId) return@runOnUiThread
                        typeText.text = "تعذر تحليل الإدخال"
                        exactText.text = e.message ?: "خطأ غير معروف"
                        stepsText.text = ""
                        aiText.text = "لم يكتمل الاختبار"
                        solveButton.isEnabled = true
                    }
                }
            }.start()
        }
    }

    override fun onDestroy() {
        requestId++
        super.onDestroy()
    }

    private fun renderResult(
        prediction: DoubleArray,
        answer: MathTeacher.Answer,
        typeText: TextView,
        exactText: TextView,
        stepsText: TextView,
        aiText: TextView
    ) {
        val px = prediction.getOrElse(0) { 0.0 }
        val py = prediction.getOrElse(1) { 0.0 }
        val optimizerStep = ModelManager.modelInfo(this).optimizerStep
        typeText.text = "${answer.type} • أوزان Adam عند الخطوة %,d".format(Locale.US, optimizerStep)
        exactText.text = answer.summary
        stepsText.text = if (answer.steps.isEmpty()) "" else answer.steps
            .mapIndexed { index, step -> "${index + 1}. $step" }
            .joinToString("\n")

        aiText.text = when {
            answer.x != null && answer.y != null -> {
                val xError = abs(px - answer.x)
                val yError = abs(py - answer.y)
                "x ≈ ${fmt(px)}\ny ≈ ${fmt(py)}\n|خطأ x| = ${fmt(xError)} • |خطأ y| = ${fmt(yError)}"
            }
            answer.x != null -> "x ≈ ${fmt(px)}\nالخطأ المطلق عن الجذر الرئيسي = ${fmt(abs(px - answer.x))}"
            answer.y != null -> "y ≈ ${fmt(py)}\nالخطأ المطلق = ${fmt(abs(py - answer.y))}"
            else -> "x ≈ ${fmt(px)}\ny ≈ ${fmt(py)}\nلا توجد قيمة مرجعية واحدة صالحة لحساب الخطأ."
        }
    }

    private fun fmt(value: Double): String =
        if (abs(value) < 1e-9) "0" else String.format(Locale.US, "%.6f", value).trimEnd('0').trimEnd('.')
}
