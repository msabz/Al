package com.example.equationsolver

import android.content.Intent
import android.os.Build
import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.example.equationsolver.ai.ModelManager
import com.example.equationsolver.ai.TrainingEngine
import com.example.equationsolver.ai.TrainingService
import java.util.Locale
import kotlin.math.max
import kotlin.math.sqrt

class MainActivity : AppCompatActivity() {
    private var resumeError: String? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        findViewById<Button>(R.id.btnGoTest).setOnClickListener { startActivity(Intent(this, TestActivity::class.java)) }
        findViewById<Button>(R.id.btnGoTrain).setOnClickListener { startActivity(Intent(this, TrainingActivity::class.java)) }
        findViewById<Button>(R.id.btnGoReinforcement).setOnClickListener { startActivity(Intent(this, ReinforcementActivity::class.java)) }
        findViewById<Button>(R.id.btnGoSettings).setOnClickListener { startActivity(Intent(this, SettingsActivity::class.java)) }
    }

    override fun onResume() {
        super.onResume()
        ModelManager.init(applicationContext)
        resumeTrainingIfRequested()
        renderDashboard()
    }

    private fun resumeTrainingIfRequested() {
        resumeError = null
        if (!ModelManager.isTrainingEnabled(this)) return
        val intent = Intent(this, TrainingService::class.java).setAction(TrainingService.ACTION_START)
        resumeError = try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) ContextCompat.startForegroundService(this, intent) else startService(intent)
            null
        } catch (e: Exception) { e.message ?: "رفض Android تشغيل خدمة التدريب" }
    }

    private fun renderDashboard() {
        val snapshot = TrainingEngine.snapshot()
        val samples = max(ModelManager.trainingSamples(this), snapshot.samples)
        val storedMse = ModelManager.lastValidationMse(this)
        val validation = snapshot.validation
        val mse = if (validation.normalizedMse.isFinite()) validation.normalizedMse else storedMse
        val accuracy = if (validation.withinOneUnitRatio.isFinite()) validation.withinOneUnitRatio else ModelManager.lastValidationAccuracy(this)
        val info = ModelManager.modelInfo(this)

        findViewById<TextView>(R.id.textMainSamples).text = "%,d".format(Locale.US, samples)
        findViewById<TextView>(R.id.textMainRmse).text = if (mse.isFinite()) "%.2f".format(Locale.US, sqrt(mse) * TrainingEngine.OUTPUT_SCALE) else "—"
        findViewById<TextView>(R.id.textMainAccuracy).text = if (accuracy.isFinite()) "%.0f%%".format(Locale.US, accuracy * 100.0) else "—"
        findViewById<TextView>(R.id.textTrainingState).text = when {
            resumeError != null -> "تعذر استئناف الخدمة: $resumeError"
            snapshot.paused -> "متوقف مؤقتًا لحماية الهاتف: ${snapshot.reason}"
            TrainingEngine.isExternalFileSessionActive() -> "● تدريب ملف خارجي يعمل الآن"
            ModelManager.isTrainingEnabled(this) -> "● تدريب v5 البنيوي يعمل في الخلفية"
            samples > 0 -> "نموذج v5 محفوظ وجاهز للمتابعة"
            info.bootstrappedFromAsset -> "نموذج Colab المدمج جاهز — يمكنك الاختبار أو متابعة التدريب"
            else -> "نموذج v5 جديد — ابدأ التدريب أو استورد MAI5 من Colab"
        }
        val checkpoint = if (info.checkpointBytes > 0L) "%.1f MB".format(Locale.US, info.checkpointBytes / 1_048_576.0) else "غير محفوظ بعد"
        val stateAcc = if (validation.stateAccuracy.isFinite()) " • دقة حالة الحل %.0f%%".format(Locale.US, validation.stateAccuracy * 100.0) else ""
        val source = when {
            info.importedAt > 0L -> " • MAI5 مستورد يدويًا"
            info.bootstrappedFromAsset -> " • أوزان Colab مدمجة بالـAPK"
            else -> ""
        }
        findViewById<TextView>(R.id.textModelDetails).text =
            "%,d معامل • %,d خطوة Adam%s\nCheckpoint: %s%s".format(Locale.US, info.parameterCount, info.optimizerStep, stateAcc, checkpoint, source)
    }
}
