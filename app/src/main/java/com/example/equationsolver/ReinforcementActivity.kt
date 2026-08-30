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
import com.example.equationsolver.ai.MathTokenizer
import com.example.equationsolver.ai.TrainingEngine
import com.example.equationsolver.core.ArabicEquationNormalizer
import com.example.equationsolver.core.MathTeacher
import com.example.equationsolver.data.VerifiedEquationSuggester
import java.util.Locale
import kotlin.math.abs

class ReinforcementActivity : AppCompatActivity() {
    private var currentEquation: String? = null
    private var currentAnswer: MathTeacher.Answer? = null
    private var operationId = 0
    private var busy = false

    private lateinit var edit: EditText
    private lateinit var resultText: TextView
    private lateinit var correctionText: TextView
    private lateinit var feedback: LinearLayout
    private lateinit var evaluateButton: Button
    private lateinit var nextButton: Button
    private lateinit var rewardButton: Button
    private lateinit var correctionButton: Button

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_reinforcement)
        ModelManager.init(applicationContext)
        bindViews()

        findViewById<Button>(R.id.btnBackReinforcement).setOnClickListener { finish() }
        evaluateButton.setOnClickListener {
            val raw = edit.text.toString().trim()
            if (raw.isEmpty()) {
                edit.error = "أدخل معادلة أولاً"
                return@setOnClickListener
            }
            evaluateEquation(ArabicEquationNormalizer.normalize(raw))
        }
        nextButton.setOnClickListener { suggestNextEquation() }
        rewardButton.setOnClickListener { reinforceCurrent(repeats = 4, learningRate = 0.0002, label = "تعزيز خفيف") }
        correctionButton.setOnClickListener { reinforceCurrent(repeats = 18, learningRate = 0.00045, label = "تصحيح قوي") }
        suggestNextEquation()
    }

    override fun onDestroy() {
        operationId++
        super.onDestroy()
    }

    private fun bindViews() {
        edit = findViewById(R.id.editInteractiveEq)
        resultText = findViewById(R.id.textInteractiveResult)
        correctionText = findViewById(R.id.textCorrection)
        feedback = findViewById(R.id.layoutFeedback)
        evaluateButton = findViewById(R.id.btnSolveInteractive)
        nextButton = findViewById(R.id.btnNextEquation)
        rewardButton = findViewById(R.id.btnReward)
        correctionButton = findViewById(R.id.btnPunish)
    }

    private fun suggestNextEquation(preserveCorrection: Boolean = false) {
        if (busy) return
        try {
            val candidate = generateSequence { VerifiedEquationSuggester.next() }
                .first { !TrainingEngine.isReservedForValidation(it.equation) }
            edit.setText(candidate.equation)
            edit.setSelection(edit.text.length)
            if (!preserveCorrection) correctionText.text = "مثال ${familyLabel(candidate.family)} اجتاز التحقق بالتعويض قبل عرضه."
            evaluateEquation(candidate.equation)
        } catch (e: Exception) {
            correctionText.text = e.message ?: "تعذر اقتراح معادلة"
        }
    }

    private fun evaluateEquation(equation: String) {
        if (busy) return
        val encoding = MathTokenizer.encode(equation)
        if (encoding.truncated || encoding.unknownCount > 0) {
            resultText.text = if (encoding.truncated) {
                "المعادلة تتجاوز حد النموذج (${MathTokenizer.MAX_TOKENS} token)."
            } else {
                "المعادلة تحتوي رموزًا غير موجودة في قاموس النموذج."
            }
            feedback.visibility = View.GONE
            return
        }
        currentEquation = equation
        currentAnswer = null
        val id = ++operationId
        setBusy(true)
        resultText.text = "جارٍ تنفيذ Forward pass وحساب المرجع المنفصل..."
        Thread {
            try {
                val prediction = ModelManager.predictValues(equation)
                val answer = MathTeacher.solve(equation)
                runOnUiThread {
                    if (id != operationId || isFinishing || isDestroyed) return@runOnUiThread
                    currentAnswer = answer
                    resultText.text = evaluationText(prediction, answer)
                    feedback.visibility = if (answer.x != null || answer.y != null) View.VISIBLE else View.GONE
                    setBusy(false)
                }
            } catch (e: Exception) {
                runOnUiThread {
                    if (id != operationId) return@runOnUiThread
                    resultText.text = "فشل التقييم: ${e.message ?: "خطأ غير معروف"}"
                    feedback.visibility = View.GONE
                    setBusy(false)
                }
            }
        }.start()
    }

    private fun reinforceCurrent(repeats: Int, learningRate: Double, label: String) {
        if (busy) return
        val equation = currentEquation ?: return
        val answer = currentAnswer ?: return
        if (answer.x == null && answer.y == null) return
        if (TrainingEngine.isReservedForValidation(equation)) {
            correctionText.text = "هذه المعادلة محجوزة لقياس الجودة ولم تدخل التدريب؛ اختر مثالًا آخر كي يبقى القياس نزيهًا."
            return
        }

        val id = ++operationId
        setBusy(true)
        correctionText.text = "$label: يجري الآن Backpropagation وتحديث Adam..."
        Thread {
            try {
                val delta = ModelManager.trainWithTarget(
                    input = equation,
                    x = answer.x ?: 0.0,
                    y = answer.y ?: 0.0,
                    repeats = repeats,
                    learningRate = learningRate
                )
                ModelManager.save(this)
                val before = activeError(delta.before, answer)
                val after = activeError(delta.after, answer)
                val improvement = if (before > 1e-12) ((before - after) / before * 100.0) else 0.0
                runOnUiThread {
                    if (id != operationId || isFinishing || isDestroyed) return@runOnUiThread
                    correctionText.text =
                        "$label اكتمل وحُفظ.\nالخطأ قبل التحديث: ${fmt(before)}\nالخطأ بعد التحديث: ${fmt(after)}\nالتغيّر: ${fmt(improvement)}% • خطوات Adam: ${delta.optimizerSteps}"
                    resultText.text = evaluationText(delta.after, answer)
                    setBusy(false)
                    Toast.makeText(this, "تم تحديث الأوزان وحفظها", Toast.LENGTH_SHORT).show()
                    resultText.postDelayed({ suggestNextEquation(preserveCorrection = true) }, 900L)
                }
            } catch (e: Exception) {
                runOnUiThread {
                    if (id != operationId) return@runOnUiThread
                    correctionText.text = "فشل التحديث: ${e.message ?: "خطأ غير معروف"}"
                    setBusy(false)
                }
            }
        }.start()
    }

    private fun evaluationText(prediction: DoubleArray, answer: MathTeacher.Answer): String {
        val px = prediction.getOrElse(0) { 0.0 }
        val py = prediction.getOrElse(1) { 0.0 }
        return when {
            answer.x != null && answer.y != null ->
                "النموذج: x≈${fmt(px)}، y≈${fmt(py)}\nالمرجع: x=${fmt(answer.x)}، y=${fmt(answer.y)}\nمتوسط الخطأ = ${fmt(activeError(prediction, answer))}"
            answer.x != null ->
                "النموذج: x≈${fmt(px)}\nالمرجع الرئيسي: ${answer.summary}\nالخطأ المطلق = ${fmt(abs(px - answer.x))}"
            answer.y != null ->
                "النموذج: y≈${fmt(py)}\nالمرجع: ${answer.summary}\nالخطأ المطلق = ${fmt(abs(py - answer.y))}"
            else -> "المرجع: ${answer.summary}\nخرج النموذج الخام: x≈${fmt(px)}، y≈${fmt(py)}"
        }
    }

    private fun activeError(prediction: DoubleArray, answer: MathTeacher.Answer): Double {
        var total = 0.0
        var count = 0
        answer.x?.let { total += abs(prediction.getOrElse(0) { 0.0 } - it); count++ }
        answer.y?.let { total += abs(prediction.getOrElse(1) { 0.0 } - it); count++ }
        return if (count == 0) Double.NaN else total / count
    }

    private fun setBusy(value: Boolean) {
        busy = value
        evaluateButton.isEnabled = !value
        nextButton.isEnabled = !value
        rewardButton.isEnabled = !value
        correctionButton.isEnabled = !value
        edit.isEnabled = !value
    }

    private fun familyLabel(value: String): String = when {
        value == "linear-system" -> "لنظام خطي"
        value == "quadratic" -> "تربيعي"
        value == "cubic" -> "تكعيبي"
        value == "quartic" -> "من الدرجة الرابعة"
        value == "quintic" -> "من الدرجة الخامسة"
        value.startsWith("trigonometric") -> "مثلثي"
        value.startsWith("logarithmic") -> "لوغاريتمي"
        value.startsWith("exponential") -> "أسي"
        else -> value
    }

    private fun fmt(value: Double): String =
        if (!value.isFinite()) "—" else if (abs(value) < 1e-9) "0" else String.format(Locale.US, "%.6f", value).trimEnd('0').trimEnd('.')
}
