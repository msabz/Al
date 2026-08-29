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
import com.example.equationsolver.core.ExactSolver
import com.example.equationsolver.core.SolutionResult
import kotlin.math.abs

class ReinforcementActivity : AppCompatActivity() {
    private val solver = ExactSolver()
    private var currentEquation: String? = null
    private lateinit var resultText: TextView
    private lateinit var correctionText: TextView
    private lateinit var feedback: LinearLayout

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_reinforcement)
        val edit = findViewById<EditText>(R.id.editInteractiveEq)
        val solve = findViewById<Button>(R.id.btnSolveInteractive)
        feedback = findViewById(R.id.layoutFeedback)
        resultText = findViewById(R.id.textInteractiveResult)
        correctionText = findViewById(R.id.textCorrection)

        solve.setOnClickListener {
            val equation = edit.text.toString().trim()
            if (equation.isEmpty()) return@setOnClickListener
            currentEquation = equation
            showEvaluation(equation)
            feedback.visibility = View.VISIBLE
        }

        findViewById<Button>(R.id.btnReward).setOnClickListener {
            val equation = currentEquation ?: return@setOnClickListener
            val exact = solver.solve(equation)
            ModelManager.trainOnSolution(equation, exact, repeats = 3, learningRate = 0.0005)
            ModelManager.save(this)
            Toast.makeText(this, "تم تعزيز النتيجة اعتمادًا على الحل الرمزي.", Toast.LENGTH_SHORT).show()
            feedback.visibility = View.GONE
        }

        findViewById<Button>(R.id.btnPunish).setOnClickListener {
            val equation = currentEquation ?: return@setOnClickListener
            val exact = solver.solve(equation)
            ModelManager.trainOnSolution(equation, exact, repeats = 25, learningRate = 0.003)
            ModelManager.save(this)
            correctionText.text = "تم التصحيح تلقائيًا باستخدام ExactSolver كـ Ground Truth."
            showEvaluation(equation)
            feedback.visibility = View.GONE
        }
    }

    private fun showEvaluation(equation: String) {
        try {
            val prediction = ModelManager.predict(equation)
            val px = prediction[0] * 100.0
            val py = prediction[1] * 100.0
            when (val exact = solver.solve(equation)) {
                is SolutionResult.SingleVariable -> {
                    val error = abs(px - exact.x)
                    resultText.text = "AI: x = ${fmt(px)}\nExact: x = ${fmt(exact.x)}\nالخطأ = ${fmt(error)}"
                }
                is SolutionResult.TwoVariables -> {
                    val error = abs(px - exact.x) + abs(py - exact.y)
                    resultText.text = "AI: x=${fmt(px)}, y=${fmt(py)}\nExact: x=${fmt(exact.x)}, y=${fmt(exact.y)}\nمجموع الخطأ = ${fmt(error)}"
                }
                SolutionResult.NoSolution -> resultText.text = "الحالة: لا يوجد حل."
                SolutionResult.InfiniteSolutions -> resultText.text = "الحالة: عدد لا نهائي من الحلول."
                is SolutionResult.Error -> resultText.text = "خطأ: ${exact.message}"
            }
        } catch (e: Exception) {
            resultText.text = "خطأ: ${e.message}"
        }
    }

    private fun fmt(value: Double) = "%.6f".format(value)
}
