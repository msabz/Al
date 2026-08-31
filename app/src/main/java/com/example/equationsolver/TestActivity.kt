package com.example.equationsolver

import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.example.equationsolver.ai.EquationFamily
import com.example.equationsolver.ai.ModelManager
import com.example.equationsolver.ai.SolutionState
import com.example.equationsolver.ai.StructuralMathEncoder
import com.example.equationsolver.ai.V5Prediction
import com.example.equationsolver.core.ArabicEquationNormalizer
import com.example.equationsolver.core.MathTeacher
import java.util.Locale
import kotlin.math.abs

class TestActivity : AppCompatActivity() {
    @Volatile private var requestId = 0

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
        findViewById<TextView>(R.id.textTestTrainingState).text = if (ModelManager.isTrainingEnabled(this))
            "● التدريب v5 مستمر بالخلفية. الاختبار يقرأ الأوزان الحالية فقط."
        else "جواب النموذج يأتي من v5 فقط؛ الجواب المرجعي منفصل للمقارنة."

        findViewById<Button>(R.id.btnClear).setOnClickListener {
            requestId++; input.text.clear(); typeText.text=""; exactText.text="—"; stepsText.text=""; aiText.text="لم يُجرَ اختبار بعد"; solveButton.isEnabled=true
        }

        solveButton.setOnClickListener {
            val raw = input.text.toString().trim()
            if (raw.isEmpty()) { input.error="أدخل معادلة أولاً"; return@setOnClickListener }
            val equation = ArabicEquationNormalizer.normalize(raw)
            val encoding = runCatching { StructuralMathEncoder.encode(equation) }.getOrElse {
                input.error = it.message ?: "إدخال غير مدعوم"; return@setOnClickListener
            }
            if (encoding.truncated) { input.error="المعادلة تتجاوز حد ${com.example.equationsolver.ai.V5ModelSpec.MAX_NODES} عقدة RPN"; return@setOnClickListener }
            val current = ++requestId
            solveButton.isEnabled=false; aiText.text="v5 ينفّذ Forward pass..."; exactText.text="المصحح يحسب الجواب المرجعي..."
            Thread {
                val result = runCatching { ModelManager.predictStructured(equation) to MathTeacher.solve(equation) }
                runOnUiThread {
                    if (current != requestId || isFinishing || isDestroyed) return@runOnUiThread
                    if (result.isSuccess) {
                        val (prediction, answer) = result.getOrThrow()
                        render(prediction, answer, typeText, exactText, stepsText, aiText)
                    } else Toast.makeText(this, result.exceptionOrNull()?.message ?: "فشل الاختبار", Toast.LENGTH_LONG).show()
                    solveButton.isEnabled=true
                }
            }.start()
        }
    }

    override fun onDestroy() { requestId++; super.onDestroy() }

    private fun render(p: V5Prediction, a: MathTeacher.Answer, type: TextView, exact: TextView, steps: TextView, ai: TextView) {
        type.text = "${familyName(p.family)} • ${stateName(p.state)} • Adam %,d".format(Locale.US, ModelManager.modelInfo(this).optimizerStep)
        exact.text = a.summary
        steps.text = a.steps.mapIndexed { i, s -> "${i+1}. $s" }.joinToString("\n")
        val stateProb = p.stateProbabilities.getOrElse(p.state.id) { 0.0 }
        val body = when {
            p.state != SolutionState.FINITE -> "حالة النموذج: ${stateName(p.state)}\nالثقة ≈ ${fmt(stateProb*100)}%"
            p.family == EquationFamily.SYSTEM -> {
                val x=p.x ?: p.slotValues[0]; val y=p.y ?: p.slotValues[1]
                buildString {
                    append("x ≈ ${fmt(x)}\ny ≈ ${fmt(y)}")
                    if (a.x != null) append("\n|خطأ x| = ${fmt(abs(x-a.x))}")
                    if (a.y != null) append(" • |خطأ y| = ${fmt(abs(y-a.y))}")
                    append("\nPresence: ${fmt(p.presenceProbabilities[0]*100)}%, ${fmt(p.presenceProbabilities[1]*100)}%")
                }
            }
            else -> {
                val roots = p.roots
                val shown = if (roots.isEmpty()) "لا توجد خانة تجاوزت Presence 50%" else roots.joinToString(", ") { fmt(it) }
                val refRoots = a.roots.ifEmpty { a.x?.let { listOf(it) } ?: emptyList() }
                val mae = if (roots.isNotEmpty() && refRoots.isNotEmpty()) nearestSetMae(roots, refRoots.toDoubleArray()) else Double.NaN
                buildString {
                    append("جذور النموذج: {$shown}\n")
                    append("Presence: "+p.presenceProbabilities.joinToString(", ") { "${fmt(it*100)}%" })
                    if (mae.isFinite()) append("\nمتوسط خطأ المطابقة ≈ ${fmt(mae)}")
                    append("\nثقة حالة الحل ≈ ${fmt(stateProb*100)}%")
                }
            }
        }
        ai.text = body
    }

    private fun nearestSetMae(pred: DoubleArray, expected: DoubleArray): Double {
        val used=BooleanArray(pred.size); var total=0.0; var count=0
        for (v in expected) { var best=-1; var err=Double.POSITIVE_INFINITY; for (i in pred.indices) if(!used[i]) { val e=abs(pred[i]-v); if(e<err){err=e;best=i} }; if(best>=0){used[best]=true;total+=err;count++} }
        return if(count==0) Double.NaN else total/count
    }
    private fun familyName(f: EquationFamily)=when(f){ EquationFamily.LINEAR->"خطية"; EquationFamily.POLYNOMIAL->"كثيرات حدود"; EquationFamily.ANALYTIC->"دوال/تحليلية"; EquationFamily.SYSTEM->"نظام x,y" }
    private fun stateName(s: SolutionState)=when(s){ SolutionState.FINITE->"حل محدود"; SolutionState.NO_SOLUTION->"لا حل"; SolutionState.INFINITE->"حلول لا نهائية"; SolutionState.UNSUPPORTED->"غير مدعوم" }
    private fun fmt(v: Double): String = if(abs(v)<1e-9) "0" else String.format(Locale.US,"%.6f",v).trimEnd('0').trimEnd('.')
}
