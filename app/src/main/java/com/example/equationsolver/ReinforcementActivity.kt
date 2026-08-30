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
import com.example.equationsolver.core.MathTeacher
import com.example.equationsolver.data.VerifiedEquationSuggester
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
            showEvaluation(equation)
            feedback.visibility = View.VISIBLE
        }
        findViewById<Button>(R.id.btnReward).setOnClickListener { reinforceCurrent(6, 0.00035) }
        findViewById<Button>(R.id.btnPunish).setOnClickListener { reinforceCurrent(30, 0.0012) }
        suggestNextEquation()
    }

    private fun reinforceCurrent(repeats: Int, learningRate: Double) {
        val equation = currentEquation ?: return
        try {
            val answer = MathTeacher.solve(equation)
            if (answer.x == null && answer.y == null) error("لا يوجد حل عددي مرجعي يمكن التدريب عليه")
            ModelManager.trainWithTarget(
                equation,
                x = answer.x ?: 0.0,
                y = answer.y ?: 0.0,
                repeats = repeats,
                learningRate = learningRate
            )
            val state = com.example.equationsolver.ai.TrainingEngine.snapshot()
            ModelManager.save(this, state.samples, state.batches, state.bestValidationMse, state.loss)
            correctionText.text = "تم تحديث النموذج بالجواب المرجعي وحفظ الـCheckpoint."
            Toast.makeText(this, "تم التعزيز وحفظ النموذج", Toast.LENGTH_SHORT).show()
            suggestNextEquation()
        } catch (e: Exception) {
            correctionText.text = "فشل التحديث: ${e.message ?: "خطأ غير معروف"}"
        }
    }

    private fun suggestNextEquation() {
        try {
            val candidate = VerifiedEquationSuggester.next()
            currentEquation = candidate.equation
            edit.setText(candidate.equation)
            edit.setSelection(edit.text.length)
            showEvaluation(candidate.equation)
            feedback.visibility = View.VISIBLE
            correctionText.text = "معادلة ${candidate.family} صحيحة مقترحة تلقائيًا. قيّم جواب النموذج ثم عزّزه إذا أردت."
        } catch (e: Exception) {
            correctionText.text = e.message ?: "تعذر اقتراح معادلة"
        }
    }

    private fun showEvaluation(equation: String) {
        try {
            val prediction = ModelManager.predict(equation)
            val px = prediction.getOrElse(0) { 0.0 } * 100.0
            val py = prediction.getOrElse(1) { 0.0 } * 100.0
            val answer = MathTeacher.solve(equation)
            resultText.text = when {
                answer.x != null && answer.y != null ->
                    "AI: x ≈ ${fmt(px)}, y ≈ ${fmt(py)}\nCorrect: x = ${fmt(answer.x)}, y = ${fmt(answer.y)}\nمجموع الخطأ = ${fmt(abs(px - answer.x) + abs(py - answer.y))}"
                answer.x != null ->
                    "AI: x ≈ ${fmt(px)}\nCorrect: ${answer.summary}\nالخطأ عن الجذر المرجعي = ${fmt(abs(px - answer.x))}"
                answer.y != null ->
                    "AI: y ≈ ${fmt(py)}\nCorrect: ${answer.summary}\nالخطأ = ${fmt(abs(py - answer.y))}"
                else -> "Correct: ${answer.summary}\nAI raw: x≈${fmt(px)}, y≈${fmt(py)}"
            }
        } catch (e: Exception) {
            resultText.text = "خطأ: ${e.message ?: "خطأ غير معروف"}"
        }
    }

    private fun fmt(value: Double): String =
        if (abs(value) < 1e-9) "0" else String.format(Locale.US, "%.8f", value).trimEnd('0').trimEnd('.')
}
