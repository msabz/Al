package com.example.equationsolver

import android.content.Intent
import android.os.Build
import android.os.Bundle
import android.text.InputType
import android.view.Gravity
import android.view.ViewGroup
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.Spinner
import android.widget.ArrayAdapter
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import com.example.equationsolver.ai.ModelManager
import com.example.equationsolver.ai.TrainingService
import com.example.equationsolver.ai.V5Settings
import java.util.Locale

class SettingsActivity : AppCompatActivity() {
    private lateinit var power: Spinner
    private lateinit var learningRate: EditText
    private lateinit var consistency: EditText
    private lateinit var range: EditText
    private lateinit var validateEvery: EditText
    private lateinit var checkpoint: EditText
    private lateinit var linear: CheckBox
    private lateinit var polynomial: CheckBox
    private lateinit var analytic: CheckBox
    private lateinit var system: CheckBox
    private lateinit var modelInfo: TextView

    private val importLauncher = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri == null) return@registerForActivityResult
        stopTrainingForModelSwap()
        Thread {
            val result = runCatching {
                contentResolver.openInputStream(uri)?.use { ModelManager.importWeights(applicationContext, it, resetTrainingStats = true) }
                    ?: error("تعذر فتح الملف")
            }
            runOnUiThread {
                if (result.isSuccess) {
                    Toast.makeText(this, "تم استيراد أوزان MAI5 بنجاح", Toast.LENGTH_LONG).show()
                    renderModelInfo()
                } else Toast.makeText(this, "فشل الاستيراد: ${result.exceptionOrNull()?.message}", Toast.LENGTH_LONG).show()
            }
        }.start()
    }

    private val exportLauncher = registerForActivityResult(ActivityResultContracts.CreateDocument("application/octet-stream")) { uri ->
        if (uri == null) return@registerForActivityResult
        Thread {
            val result = runCatching {
                contentResolver.openOutputStream(uri, "w")?.use { ModelManager.exportWeights(it) } ?: error("تعذر إنشاء الملف")
            }
            runOnUiThread {
                Toast.makeText(this, if (result.isSuccess) "تم تصدير النموذج" else "فشل التصدير: ${result.exceptionOrNull()?.message}", Toast.LENGTH_LONG).show()
            }
        }.start()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        ModelManager.init(applicationContext)
        setContentView(buildUi())
        loadSettings()
        renderModelInfo()
    }

    private fun buildUi(): ScrollView {
        val scroll = ScrollView(this)
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(20), dp(24), dp(20), dp(40))
        }
        scroll.addView(root, ViewGroup.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))

        root += title("إعدادات Math AI v5", 26f)
        root += text("هذه الخيارات تطبّق من الدفعة التالية ولا تغيّر بنية الـCheckpoint. تغيير الطبقات أو عدد الخانات غير متاح عمدًا حتى تبقى أوزان Colab والهاتف متوافقة.")

        power = Spinner(this).also {
            it.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, listOf("اقتصادي", "متوازن", "سريع"))
        }
        root += label("نمط الطاقة")
        root += power

        learningRate = numberField("0.0006")
        consistency = numberField("0.05")
        range = integerField("100")
        validateEvery = integerField("40")
        checkpoint = integerField("5")
        root += fieldBlock("Learning rate", learningRate)
        root += fieldBlock("وزن Consistency loss (0..1)", consistency)
        root += fieldBlock("أقصى مدى تدريب للأرقام (10..120)", range)
        root += fieldBlock("التحقق كل N دفعة", validateEvery)
        root += fieldBlock("حفظ Checkpoint كل N دقيقة", checkpoint)

        root += label("عائلات التدريب")
        linear = check("خطية")
        polynomial = check("كثيرات الحدود حتى 5 جذور")
        analytic = check("كسرية / جذرية / أسية / لوغاريتمية / مثلثية")
        system = check("أنظمة x,y")
        root += linear; root += polynomial; root += analytic; root += system

        root += button("حفظ الإعدادات") { saveSettings() }
        root += title("أوزان النموذج", 20f)
        modelInfo = text("")
        root += modelInfo
        root += button("استيراد أوزان Colab (.mai5)") { importLauncher.launch(arrayOf("application/octet-stream", "application/*")) }
        root += button("تصدير النموذج الحالي (.mai5)") { exportLauncher.launch("math_ai_v5.mai5") }
        root += button("تصفير النموذج") { confirmReset() }
        root += button("رجوع") { finish() }
        return scroll
    }

    private fun loadSettings() {
        val s = V5Settings.read(this)
        power.setSelection(when (s.powerMode) { V5Settings.PowerMode.ECO -> 0; V5Settings.PowerMode.BALANCED -> 1; V5Settings.PowerMode.FAST -> 2 })
        learningRate.setText(String.format(Locale.US, "%.6f", s.learningRate).trimEnd('0'))
        consistency.setText(String.format(Locale.US, "%.3f", s.consistencyWeight).trimEnd('0').trimEnd('.'))
        range.setText(s.maxAbsTrainingValue.toString())
        validateEvery.setText(s.validateEveryBatches.toString())
        checkpoint.setText(s.checkpointMinutes.toString())
        linear.isChecked = s.enableLinear
        polynomial.isChecked = s.enablePolynomial
        analytic.isChecked = s.enableAnalytic
        system.isChecked = s.enableSystem
    }

    private fun saveSettings() {
        val current = V5Settings.read(this)
        val next = V5Settings.Snapshot(
            powerMode = when (power.selectedItemPosition) { 0 -> V5Settings.PowerMode.ECO; 2 -> V5Settings.PowerMode.FAST; else -> V5Settings.PowerMode.BALANCED },
            learningRate = learningRate.text.toString().toDoubleOrNull() ?: current.learningRate,
            consistencyWeight = consistency.text.toString().toDoubleOrNull() ?: current.consistencyWeight,
            maxAbsTrainingValue = range.text.toString().toIntOrNull() ?: current.maxAbsTrainingValue,
            validateEveryBatches = validateEvery.text.toString().toIntOrNull() ?: current.validateEveryBatches,
            checkpointMinutes = checkpoint.text.toString().toIntOrNull() ?: current.checkpointMinutes,
            enableLinear = linear.isChecked,
            enablePolynomial = polynomial.isChecked,
            enableAnalytic = analytic.isChecked,
            enableSystem = system.isChecked
        )
        if (!next.enableLinear && !next.enablePolynomial && !next.enableAnalytic && !next.enableSystem) {
            Toast.makeText(this, "فعّل عائلة تدريب واحدة على الأقل", Toast.LENGTH_SHORT).show(); return
        }
        V5Settings.write(this, next)
        loadSettings()
        Toast.makeText(this, "تم حفظ الإعدادات وستطبق من الدفعة التالية", Toast.LENGTH_LONG).show()
    }

    private fun confirmReset() {
        AlertDialog.Builder(this)
            .setTitle("تصفير النموذج؟")
            .setMessage("سيتم حذف الأوزان الحالية وحالة Adam والبدء بنموذج جديد. صدّر نسخة MAI5 أولاً إذا كنت تحتاجها.")
            .setNegativeButton("إلغاء", null)
            .setPositiveButton("تصفير") { _, _ ->
                stopTrainingForModelSwap()
                ModelManager.resetModel(applicationContext)
                renderModelInfo()
                Toast.makeText(this, "تم إنشاء نموذج v5 جديد", Toast.LENGTH_LONG).show()
            }.show()
    }

    private fun stopTrainingForModelSwap() {
        ModelManager.setTrainingEnabled(applicationContext, false)
        runCatching { startService(Intent(this, TrainingService::class.java).setAction(TrainingService.ACTION_STOP)) }
    }

    private fun renderModelInfo() {
        val info = ModelManager.modelInfo(this)
        modelInfo.text = "%,d معامل • Adam step %,d\nCheckpoint: %.2f MB%s".format(
            Locale.US, info.parameterCount, info.optimizerStep, info.checkpointBytes / 1_048_576.0,
            if (info.importedAt > 0) "\nآخر استيراد: ${java.text.DateFormat.getDateTimeInstance().format(info.importedAt)}" else ""
        )
    }

    private fun title(value: String, size: Float) = TextView(this).apply { text = value; textSize = size; gravity = Gravity.START; setPadding(0, dp(14), 0, dp(8)) }
    private fun label(value: String) = TextView(this).apply { text = value; textSize = 15f; setPadding(0, dp(12), 0, dp(4)) }
    private fun text(value: String) = TextView(this).apply { text = value; textSize = 14f; setPadding(0, dp(4), 0, dp(8)) }
    private fun check(value: String) = CheckBox(this).apply { text = value }
    private fun numberField(hintValue: String) = EditText(this).apply { hint = hintValue; inputType = InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_FLAG_DECIMAL }
    private fun integerField(hintValue: String) = EditText(this).apply { hint = hintValue; inputType = InputType.TYPE_CLASS_NUMBER }
    private fun fieldBlock(name: String, field: EditText) = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; addView(label(name)); addView(field) }
    private fun button(label: String, click: () -> Unit) = Button(this).apply { text = label; setOnClickListener { click() }; setPadding(dp(8), dp(8), dp(8), dp(8)) }
    private fun dp(value: Int): Int = (value * resources.displayMetrics.density).toInt()

    private operator fun LinearLayout.plusAssign(view: android.view.View) { addView(view, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply { topMargin = dp(4) }) }
}
