package com.example.equationsolver

import android.Manifest
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.content.ContextCompat.registerReceiver
import com.example.equationsolver.ai.ModelManager
import com.example.equationsolver.ai.TrainingEngine
import com.example.equationsolver.ai.TrainingService
import com.example.equationsolver.core.MathTeacher
import java.util.Locale

class TrainingActivity : AppCompatActivity() {
    private lateinit var status: TextView
    private lateinit var startButton: Button
    private lateinit var stopButton: Button
    private lateinit var fileButton: Button
    private lateinit var lossChart: LossChartView
    @Volatile private var fileTraining = false

    private val filePicker = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) startFileTraining(uri)
    }

    private val receiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            when (intent.action) {
                TrainingService.ACTION_PROGRESS -> {
                    val samples = intent.getLongExtra(TrainingService.EXTRA_SAMPLES, 0L)
                    val batches = intent.getLongExtra(TrainingService.EXTRA_BATCHES, 0L)
                    val epoch = intent.getLongExtra(TrainingService.EXTRA_EPOCH, 0L)
                    val loss = intent.getDoubleExtra(TrainingService.EXTRA_LOSS, Double.NaN)
                    val validation = intent.getDoubleExtra(TrainingService.EXTRA_VALIDATION, Double.NaN)
                    if (loss.isFinite()) lossChart.addLoss(loss)
                    status.text = "التدريب مستمر\nالمعادلات: %,d\nالدفعات: %,d\nEpoch: %,d\nLoss: %.8f\nValidation Loss: %.8f".format(
                        Locale.US, samples, batches, epoch, loss, validation
                    )
                }
                TrainingService.ACTION_PAUSED -> status.text =
                    "التدريب متوقف مؤقتًا\nالسبب: ${intent.getStringExtra(TrainingService.EXTRA_REASON) ?: "حماية الجهاز"}\nسيستأنف تلقائيًا عندما تصبح الظروف مناسبة."
            }
            refreshButtons()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_training)
        status = findViewById(R.id.textStatus)
        startButton = findViewById(R.id.btnTrainRandom)
        stopButton = findViewById(R.id.btnStopTraining)
        fileButton = findViewById(R.id.btnLoadFile)
        lossChart = findViewById(R.id.lossChart)
        ModelManager.init(applicationContext)

        if (Build.VERSION.SDK_INT >= 33 && ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.POST_NOTIFICATIONS), 7001)
        }
        startButton.setOnClickListener { startTrainingService() }
        stopButton.setOnClickListener { stopTrainingService() }
        fileButton.setOnClickListener {
            if (ModelManager.isTrainingEnabled(this)) {
                Toast.makeText(this, "أوقف التدريب المستمر أولًا قبل تدريب ملف خارجي", Toast.LENGTH_SHORT).show()
            } else filePicker.launch(arrayOf("text/plain", "text/*", "*/*"))
        }
        showCheckpointInfo()
        refreshButtons()
    }

    override fun onStart() {
        super.onStart()
        val filter = IntentFilter().apply {
            addAction(TrainingService.ACTION_PROGRESS)
            addAction(TrainingService.ACTION_PAUSED)
        }
        registerReceiver(this, receiver, filter, ContextCompat.RECEIVER_NOT_EXPORTED)
        refreshButtons()
    }

    override fun onStop() {
        unregisterReceiver(receiver)
        super.onStop()
    }

    private fun startTrainingService() {
        if (fileTraining) return
        val intent = Intent(this, TrainingService::class.java).setAction(TrainingService.ACTION_START)
        if (Build.VERSION.SDK_INT >= 26) startForegroundService(intent) else startService(intent)
        status.text = "تم تشغيل التدريب المستمر في الخلفية. يمكنك مغادرة التطبيق أو إطفاء الشاشة."
        refreshButtons()
    }

    private fun stopTrainingService() {
        startService(Intent(this, TrainingService::class.java).setAction(TrainingService.ACTION_STOP))
        status.text = "جاري إيقاف التدريب وحفظ آخر Checkpoint كامل..."
        refreshButtons()
    }

    private fun startFileTraining(uri: android.net.Uri) {
        if (fileTraining) return
        fileTraining = true
        refreshButtons()
        val baseSamples = ModelManager.trainingSamples(this)
        val baseBatches = ModelManager.trainingBatches(this)

        Thread {
            var processed = 0L
            var valid = 0L
            val chunk = ArrayList<Pair<String, DoubleArray>>(2000)
            try {
                contentResolver.openInputStream(uri)?.bufferedReader()?.useLines { lines ->
                    lines.forEach { raw ->
                        val line = raw.trim()
                        if (line.isEmpty() || line.startsWith("#")) return@forEach
                        processed++
                        parseTrainingLine(line)?.let { example ->
                            chunk += example
                            valid++
                        }
                        if (chunk.size >= 2000) {
                            TrainingEngine.trainFile(chunk) { }
                            chunk.clear()
                            runOnUiThread { status.text = "تدريب الملف...\nتمت قراءة %,d سطر\nأمثلة صالحة: %,d".format(Locale.US, processed, valid) }
                        }
                    }
                } ?: error("تعذر فتح الملف")
                if (chunk.isNotEmpty()) TrainingEngine.trainFile(chunk) { }

                val newSamples = baseSamples + valid
                val newBatches = baseBatches + valid / TrainingEngine.BATCH_SIZE
                ModelManager.save(this, newSamples, newBatches, TrainingEngine.lastValidationMse, Double.NaN)
                runOnUiThread {
                    status.text = "اكتمل تدريب الملف\nالأسطر: %,d\nالأمثلة المستخدمة: %,d\nتم حفظ النموذج.".format(Locale.US, processed, valid)
                }
            } catch (e: Exception) {
                runOnUiThread { status.text = "فشل تدريب الملف:\n${e.message ?: "خطأ غير معروف"}" }
            } finally {
                fileTraining = false
                runOnUiThread { refreshButtons() }
            }
        }.start()
    }

    private fun parseTrainingLine(line: String): Pair<String, DoubleArray>? {
        return try {
            val parts = line.split('|', limit = 2)
            val equation = parts[0].trim()
            if (equation.isEmpty()) return null
            val values = if (parts.size == 2 && parts[1].isNotBlank()) {
                val nums = parts[1].split(',').mapNotNull { it.trim().toDoubleOrNull() }
                if (nums.isEmpty()) return null
                doubleArrayOf(nums[0], nums.getOrElse(1) { 0.0 })
            } else {
                val answer = MathTeacher.solve(equation)
                if (answer.x == null && answer.y == null) return null
                doubleArrayOf(answer.x ?: 0.0, answer.y ?: 0.0)
            }
            equation to values
        } catch (_: Exception) { null }
    }

    private fun showCheckpointInfo() {
        val samples = ModelManager.trainingSamples(this)
        val best = ModelManager.bestValidationMse(this)
        val last = ModelManager.lastLoss(this)
        if (samples > 0) {
            status.text = "نموذج محفوظ\nالمعادلات المدربة: %,d\nآخر Loss: %s\nأفضل Validation Loss: %s\nيمكن متابعة التدريب من نفس النقطة مع حالة Adam محفوظة.".format(
                Locale.US,
                samples,
                if (last.isFinite()) "%.8f".format(Locale.US, last) else "غير متاح",
                if (best.isFinite()) "%.8f".format(Locale.US, best) else "غير متاح"
            )
        }
    }

    private fun refreshButtons() {
        val enabled = ModelManager.isTrainingEnabled(this)
        startButton.isEnabled = !enabled && !fileTraining
        stopButton.isEnabled = enabled
        fileButton.isEnabled = !enabled && !fileTraining
    }
}
