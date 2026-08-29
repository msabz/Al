package com.example.equationsolver

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.net.Uri
import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.example.equationsolver.ai.ModelManager
import com.example.equationsolver.ai.TrainingEngine
import com.example.equationsolver.ai.TrainingService
import kotlinx.coroutines.*
import java.io.BufferedReader
import java.io.InputStreamReader

class TrainingActivity : AppCompatActivity() {
    private lateinit var status: TextView
    private lateinit var startButton: Button
    private lateinit var stopButton: Button
    private lateinit var fileButton: Button
    private lateinit var lossChart: LossChartView
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private val picker = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? -> uri?.let(::readTrainingFile) }
    private val receiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            if (intent.action != TrainingService.ACTION_PROGRESS) return
            val samples = intent.getLongExtra(TrainingService.EXTRA_SAMPLES, 0)
            val batches = intent.getLongExtra(TrainingService.EXTRA_BATCHES, 0)
            val epoch = intent.getLongExtra(TrainingService.EXTRA_EPOCH, 0)
            val loss = intent.getDoubleExtra(TrainingService.EXTRA_LOSS, 0.0)
            val validation = intent.getDoubleExtra(TrainingService.EXTRA_VALIDATION, 0.0)
            lossChart.addLoss(loss)
            status.text = "التدريب مستمر بالخلفية\nالمعادلات: %,d\nالدفعات: %,d\nEpoch: %,d\nLoss: %.8f\nValidation Loss: %.8f".format(samples, batches, epoch, loss, validation)
            setBusy(true)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_training)
        status = findViewById(R.id.textStatus); startButton = findViewById(R.id.btnTrainRandom)
        stopButton = findViewById(R.id.btnStopTraining); fileButton = findViewById(R.id.btnLoadFile); lossChart = findViewById(R.id.lossChart)
        ModelManager.init(applicationContext)
        startButton.setOnClickListener { startContinuousTraining() }
        stopButton.setOnClickListener { stopContinuousTraining() }
        fileButton.setOnClickListener { picker.launch("text/plain") }
        showCheckpointInfo()
    }

    override fun onStart() {
        super.onStart()
        ContextCompat.registerReceiver(this, receiver, IntentFilter(TrainingService.ACTION_PROGRESS), ContextCompat.RECEIVER_NOT_EXPORTED)
        setBusy(false)
    }

    override fun onStop() { unregisterReceiver(receiver); super.onStop() }

    private fun showCheckpointInfo() {
        val samples = ModelManager.trainingSamples(this); val best = ModelManager.bestValidationMse(this)
        if (samples > 0) status.text = "نموذج محفوظ\nالمعادلات المدربة: %,d\nأفضل Validation Loss: %.8f\nيمكنك متابعة التدريب من نفس النقطة.".format(samples, best)
    }

    private fun startContinuousTraining() {
        startService(Intent(this, TrainingService::class.java).setAction(TrainingService.ACTION_START))
        setBusy(true); status.text = "بدأ التدريب بالخلفية... يمكنك العودة لقائمة الاختبار، وسيستمر التدريب."
    }

    private fun stopContinuousTraining() {
        startService(Intent(this, TrainingService::class.java).setAction(TrainingService.ACTION_STOP))
        setBusy(false); status.text = "تم إيقاف التدريب وحفظ النموذج."
    }

    private fun readTrainingFile(uri: Uri) {
        setBusy(true); status.text = "جاري قراءة الملف..."
        scope.launch {
            try {
                val examples = mutableListOf<Pair<String, DoubleArray>>()
                contentResolver.openInputStream(uri)?.use { stream -> BufferedReader(InputStreamReader(stream)).useLines { lines ->
                    lines.forEach { line ->
                        if (line.trimStart().startsWith("#")) return@forEach
                        val parts = line.split('|', limit = 2); if (parts.size != 2) return@forEach
                        val equation = parts[0].trim(); val values = parts[1].split(',').mapNotNull { it.trim().toDoubleOrNull() }
                        if (equation.isNotEmpty() && values.isNotEmpty() && values.size <= 2) examples += equation to doubleArrayOf(values[0], values.getOrElse(1) { 0.0 })
                    }
                } }
                if (examples.isEmpty()) error("الملف فارغ أو الصيغة غير صحيحة")
                withContext(Dispatchers.Main) { status.text = "تمت قراءة ${examples.size} مثال. بدء التدريب..." }
                TrainingEngine.trainFile(examples) { n -> scope.launch(Dispatchers.Main) { status.text = "تقدم التدريب: $n مثال معالجة..." } }
                ModelManager.save(this@TrainingActivity)
                withContext(Dispatchers.Main) { status.text = "اكتمل التدريب: ${examples.size} مثال\nValidation MSE: %.6f\nتم حفظ النموذج.".format(TrainingEngine.lastValidationMse); setBusy(false) }
            } catch (e: Exception) { withContext(Dispatchers.Main) { status.text = "خطأ: ${e.message}"; setBusy(false) } }
        }
    }

    private fun setBusy(busy: Boolean) { startButton.isEnabled = !busy; stopButton.isEnabled = busy; fileButton.isEnabled = !busy }
    override fun onDestroy() { scope.cancel(); super.onDestroy() }
}
