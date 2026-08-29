package com.example.equationsolver

import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.example.equationsolver.ai.ModelManager
import com.example.equationsolver.core.ArabicEquationNormalizer
import com.example.equationsolver.core.UniversalEquationSolver
import java.util.Locale
import kotlin.math.abs

class ReinforcementActivity : AppCompatActivity() {
    private var currentEquation: String? = null
    private var currentRawEquation: String? = null
    private lateinit var resultText: TextView
    private lateinit var correctionText: TextView
    private lateinit var feedback: LinearLayout

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_reinforcement)
        ModelManager.init(applicationContext)

        val edit = findViewById<EditText>(R.id.editInteractiveEq)
        val solve = findViewById<Button>(R.id.btnSolveInteractive)
        feedback = findViewById(R.id.layoutFeedback)
        resultText = findViewById(R.id.textInteractiveResult)
        correctionText = findViewById(R.id.textCorrection)

        solve.setOnClickListener {
            val raw = edit.text.toString().trim()
            if (raw.isEmpty()) {
                edit.error = "أدخل معادلة أولاً"
                return@setOnClickListener
            }
            val equation = ArabicEquationNormalizer.normalize(raw)
            currentRawEquation = raw
            currentEquation = equation
            showEvaluation(equation)
            feedback.visibility = View.VISIBLE
        }

        findViewById<Button>(R.id.btnReward).setOnClickListener {
            val equation = currentEquation ?: return@setOnClickListener
            try {
                val exact = UniversalEquationSolver.solve(equation)
                ModelManager.trainOnSolution(equation, toSolutionResult(exact), repeats = 5, learningRate = 0.0005)
                ModelManager.save(this)
                correctionText.text = "تم تعزيز النموذج بالحل الصحيح للمسألة."
                feedback.visibility = View.GONE
                Toast.makeText(this, "تم التعزيز وحفظ النموذج", Toast.LENGTH_SHORT).show()
            } catch (e: Exception) {
                correctionText.text = "فشل التعزيز: ${e.message ?: "خطأ غير معروف"}"
            }
        }

        findViewById<Button>(R.id.btnPunish).setOnClickListener {
            val equation = currentEquation ?: return@setOnClickListener
            try {
                val exact = UniversalEquationSolver.solve(equation)
                ModelManager.trainOnSolution(equation, toSolutionResult(exact), repeats = 40, learningRate = 0.002)
                ModelManager.save(this)
                correctionText.text = "تم تصحيح النموذج باستخدام الحل الدقيق وإعادة تدريبه على المسألة."
                showEvaluation(equation)
                feedback.visibility = View.GONE
                Toast.makeText(this, "تم التصحيح وحفظ النموذج", Toast.LENGTH_SHORT).show()
            } catch (e: Exception) {
                correctionText.text = "فشل التصحيح: ${e.message ?: "خطأ غير معروف"}"
            }
        }
    }

    private fun showEvaluation(equation: String) {
        try {
            val prediction = ModelManager.predict(equation)
            val px = prediction.getOrNull(0)?.times(100.0) ?: 0.0
            val py = prediction.getOrNull(1)?.times(100.0) ?: 0.0
            val exact = UniversalEquationSolver.solve(equation)
            resultText.text = when {
                exact.x != null && exact.y == null -> {
                    val error = abs(px - exact.x)
                    "AI: x ≈ ${fmt(px)}\nExact: x = ${fmt(exact.x)}\nالخطأ = ${fmt(error)}"
                }
                exact.x == null && exact.y != null -> {
                    val error = abs(py - exact.y)
                    "AI: y ≈ ${fmt(py)}\nExact: y = ${fmt(exact.y)}\nالخطأ = ${fmt(error)}"
                }
                exact.x != null && exact.y != null -> {
                    val error = abs(px - exact.x) + abs(py - exact.y)
                    "AI: x ≈ ${fmt(px)}, y ≈ ${fmt(py)}\nExact: x = ${fmt(exact.x)}, y = ${fmt(exact.y)}\nمجموع الخطأ = ${fmt(error)}"
                }
                else -> "الحالة: ${exact.summary}"
            }
        } catch (e: Exception) {
            resultText.text = "خطأ: ${e.message ?: "خطأ غير معروف"}"
        }
    }

    private fun toSolutionResult(result: UniversalEquationSolver.Result): com.example.equationsolver.core.SolutionResult {
        return when {
            result.x != null && result.y != null -> com.example.equationsolver.core.SolutionResult.TwoVariables(result.x, result.y)
            result.x != null -> com.example.equationsolver.core.SolutionResult.SingleVariable(result.x)
            result.y != null -> com.example.equationsolver.core.SolutionResult.SingleVariable(result.y)
            else -> com.example.equationsolver.core.SolutionResult.Error(result.summary)
        }
    }

    private fun fmt(value: Double): String =
        if (abs(value) < 1e-9) "0" else String.format(Locale.US, "%.6f", value).trimEnd('0').trimEnd('.')
}
