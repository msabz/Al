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
import com.example.equationsolver.data.EquationGenerator
import com.example.equationsolver.data.GeneratedEquationValidator
import java.util.Locale
import kotlin.math.abs

class ReinforcementActivity : AppCompatActivity() {
    private var currentEquation: String? = null
    private lateinit var edit: EditText
    private lateinit var resultText: TextView
    private lateinit var correctionText: TextView
    private lateinit var feedback: LinearLayout

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_reinforcement)
        ModelManager.init(applicationContext)
        edit = findViewById(R.id.editInteractiveEq)
        resultText = findViewById(R.id.textInteractiveResult)
        correctionText = findViewById(R.id.textCorrection)
        feedback = findViewById(R.id.layoutFeedback)
        findViewById<Button>(R.id.btnSolveInteractive).setOnClickListener {
            val raw = edit.text.toString().trim()
            if (raw.isEmpty()) { edit.error = "أدخل معادلة أولاً"; return@setOnClickListener }
            val equation = ArabicEquationNormalizer.normalize(raw)
            currentEquation = equation
            showEvaluation(equation); feedback.visibility = View.VISIBLE
        }
        findViewById<Button>(R.id.btnReward).setOnClickListener { reinforceCurrent(5, 0.0005) }
        findViewById<Button>(R.id.btnPunish).setOnClickListener { reinforceCurrent(40, 0.002) }
        suggestNextEquation()
    }

    private fun reinforceCurrent(repeats: Int, learningRate: Double) {
        val equation = currentEquation ?: return
        try {
            val exact = UniversalEquationSolver.solve(equation)
            if (exact.x == null && exact.y == null) error("لا يوجد حل عددي يمكن تدريبه")
            ModelManager.trainOnSolution(equation, toSolutionResult(exact), repeats, learningRate)
            ModelManager.save(this)
            correctionText.text = "تم تحديث النموذج بالحل الصحيح."
            Toast.makeText(this, "تم التعزيز وحفظ النموذج", Toast.LENGTH_SHORT).show()
            suggestNextEquation()
        } catch (e: Exception) { correctionText.text = "فشل التحديث: ${e.message ?: "خطأ غير معروف"}" }
    }

    private fun suggestNextEquation() {
        var candidate: com.example.equationsolver.data.GeneratedExample
        do { candidate = EquationGenerator.generate() } while (!GeneratedEquationValidator.isValid(candidate))
        currentEquation = candidate.equation
        edit.setText(candidate.equation)
        edit.setSelection(edit.text.length)
        showEvaluation(candidate.equation)
        feedback.visibility = View.VISIBLE
        correctionText.text = "معادلة صحيحة مقترحة تلقائيًا. اضغط تقييم ثم اختر التعزيز أو التصحيح."
    }

    private fun showEvaluation(equation: String) {
        try {
            val prediction = ModelManager.predict(equation)
            val px = prediction.getOrNull(0)?.times(100.0) ?: 0.0
            val py = prediction.getOrNull(1)?.times(100.0) ?: 0.0
            val exact = UniversalEquationSolver.solve(equation)
            resultText.text = when {
                exact.x != null && exact.y == null -> "AI: x ≈ ${fmt(px)}\nExact: x = ${fmt(exact.x)}\nالخطأ = ${fmt(abs(px - exact.x))}"
                exact.x != null && exact.y != null -> "AI: x ≈ ${fmt(px)}, y ≈ ${fmt(py)}\nExact: x = ${fmt(exact.x)}, y = ${fmt(exact.y)}\nمجموع الخطأ = ${fmt(abs(px - exact.x) + abs(py - exact.y))}"
                else -> "الحالة: ${exact.summary}"
            }
        } catch (e: Exception) { resultText.text = "خطأ: ${e.message ?: "خطأ غير معروف"}" }
    }

    private fun toSolutionResult(result: UniversalEquationSolver.Result): com.example.equationsolver.core.SolutionResult = when {
        result.x != null && result.y != null -> com.example.equationsolver.core.SolutionResult.TwoVariables(result.x, result.y)
        result.x != null -> com.example.equationsolver.core.SolutionResult.SingleVariable(result.x)
        result.y != null -> com.example.equationsolver.core.SolutionResult.SingleVariable(result.y)
        else -> com.example.equationsolver.core.SolutionResult.Error(result.summary)
    }

    private fun fmt(value: Double): String = if (abs(value) < 1e-9) "0" else String.format(Locale.US, "%.6f", value).trimEnd('0').trimEnd('.')
}
