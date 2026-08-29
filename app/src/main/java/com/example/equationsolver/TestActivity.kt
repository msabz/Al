package com.example.equationsolver

import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.example.equationsolver.ai.ModelManager
import com.example.equationsolver.core.ExactSolver
import com.example.equationsolver.core.SolutionResult
import kotlin.math.abs

class TestActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_test)
        val input = findViewById<EditText>(R.id.editEquation)
        val exactText = findViewById<TextView>(R.id.textResult)
        val aiText = findViewById<TextView>(R.id.textAi)
        val solver = ExactSolver()
        findViewById<Button>(R.id.btnSolve).setOnClickListener {
            val equation = input.text.toString().trim()
            if (equation.isEmpty()) return@setOnClickListener
            when (val result = solver.solve(equation)) {
                is SolutionResult.SingleVariable -> exactText.text = "الحل الدقيق: x = ${fmt(result.x)}"
                is SolutionResult.TwoVariables -> exactText.text = "الحل الدقيق: x = ${fmt(result.x)}\ny = ${fmt(result.y)}"
                SolutionResult.NoSolution -> exactText.text = "لا يوجد حل."
                SolutionResult.InfiniteSolutions -> exactText.text = "عدد لا نهائي من الحلول."
                is SolutionResult.Error -> exactText.text = "خطأ: ${result.message}"
            }
            try {
                val p = ModelManager.predict(equation)
                aiText.text = "تقدير AI: x = ${fmt(p[0] * 100.0)}, y = ${fmt(p[1] * 100.0)}"
            } catch (e: Exception) {
                aiText.text = "تعذر حساب تقدير AI: ${e.message}"
            }
        }
    }

    private fun fmt(v: Double): String = if (abs(v) < 1e-9) "0" else "%.6f".format(v)
}
