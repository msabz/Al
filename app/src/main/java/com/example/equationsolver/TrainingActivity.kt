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
import com.example.equationsolver.ai.ModelManager
import com.example.equationsolver.ai.MathTokenizer
import com.example.equationsolver.ai.TrainingEngine
import com.example.equationsolver.ai.TrainingService
import com.example.equationsolver.core.MathTeacher
import com.example.equationsolver.data.GeneratedEquationValidator
import com.example.equationsolver.data.GeneratedExample
import java.util.Locale
import kotlin.math.ceil
import kotlin.math.min
import kotlin.math.sqrt

class TrainingActivity : AppCompatActivity() {
    private lateinit var status: TextView
    private lateinit var stateText: TextView
    private lateinit var startButton: Button
    private lateinit var stopButton: Button
    private lateinit var fileButton: Button
    private lateinit var lossChart: LossChartView
    private lateinit var samplesText: TextView
    private lateinit var batchesText: TextView
    private lateinit var lossText: TextView
    private lateinit var rmseText: TextView
    private lateinit var accuracyText: TextView
    private lateinit var gradientText: TextView
    private lateinit var familyText: TextView
    private lateinit var equationText: TextView

    @Volatile
    private var fileTraining = false

    private val filePicker = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) startFileTraining(uri)
    }

    private val receiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            when (intent.action) {
                TrainingService.ACTION_PROGRESS -> renderBroadcast(intent)
                TrainingService.ACTION_PAUSED -> {
                    val reason = intent.getStringExtra(TrainingService.EXTRA_REASON) ?: "حماية الجهاز"
                    stateText.text = "● متوقف مؤقتًا"
                    status.text = "$reason\nسيستأنف تلقائيًا عندما تصبح الظروف مناسبة."
                }
                TrainingService.ACTION_ERROR -> {
                    stateText.text = "حدث خطأ وسيعيد المحاولة"
                    status.text = intent.getStringExtra(TrainingService.EXTRA_REASON) ?: "خطأ تدريب غير معروف"
                }
                TrainingService.ACTION_STOPPED -> {
                    renderStoredState()
                    stateText.text = "تم الإيقاف والحفظ"
                    status.text = "أُغلقت خدمة التدريب بعد حفظ الأوزان وحالة Adam."
                }
            }
            refreshButtons()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_training)
        bindViews()
        ModelManager.init(applicationContext)

        if (Build.VERSION.SDK_INT >= 33 &&
            ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED
        ) requestPermissions(arrayOf(Manifest.permission.POST_NOTIFICATIONS), 7001)

        findViewById<Button>(R.id.btnBackTraining).setOnClickListener { finish() }
        startButton.setOnClickListener { startTrainingService() }
        stopButton.setOnClickListener { stopTrainingService() }
        fileButton.setOnClickListener {
            if (ModelManager.isTrainingEnabled(this)) {
                Toast.makeText(this, "أوقف التدريب المستمر أولًا", Toast.LENGTH_SHORT).show()
            } else filePicker.launch(arrayOf("text/plain", "text/*", "*/*"))
        }
        renderStoredState()
        renderSnapshot(TrainingEngine.snapshot())
        refreshButtons()
    }

    override fun onStart() {
        super.onStart()
        val filter = IntentFilter().apply {
            addAction(TrainingService.ACTION_PROGRESS)
            addAction(TrainingService.ACTION_PAUSED)
            addAction(TrainingService.ACTION_ERROR)
            addAction(TrainingService.ACTION_STOPPED)
        }
        ContextCompat.registerReceiver(this, receiver, filter, ContextCompat.RECEIVER_NOT_EXPORTED)
        renderSnapshot(TrainingEngine.snapshot())
        refreshButtons()
    }

    override fun onStop() {
        unregisterReceiver(receiver)
        super.onStop()
    }

    private fun bindViews() {
        status = findViewById(R.id.textStatus)
        stateText = findViewById(R.id.textTrainingState)
        startButton = findViewById(R.id.btnTrainRandom)
        stopButton = findViewById(R.id.btnStopTraining)
        fileButton = findViewById(R.id.btnLoadFile)
        lossChart = findViewById(R.id.lossChart)
        samplesText = findViewById(R.id.textMetricSamples)
        batchesText = findViewById(R.id.textMetricBatches)
        lossText = findViewById(R.id.textMetricLoss)
        rmseText = findViewById(R.id.textMetricRmse)
        accuracyText = findViewById(R.id.textMetricAccuracy)
        gradientText = findViewById(R.id.textMetricGradient)
        familyText = findViewById(R.id.textLiveFamily)
        equationText = findViewById(R.id.textLiveEquation)
    }

    private fun startTrainingService() {
        if (fileTraining || TrainingEngine.isExternalFileSessionActive()) {
            Toast.makeText(this, "تدريب الملف ما زال يعمل", Toast.LENGTH_SHORT).show()
            return
        }
        val intent = Intent(this, TrainingService::class.java).setAction(TrainingService.ACTION_START)
        try {
            ModelManager.setTrainingEnabled(this, true)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) ContextCompat.startForegroundService(this, intent)
            else startService(intent)
            stateText.text = "● جارٍ تشغيل التدريب"
            status.text = "يستمر التدريب عند مغادرة الشاشة أو إطفائها. الإغلاق القسري من إعدادات Android يوقفه."
        } catch (e: Exception) {
            ModelManager.setTrainingEnabled(this, false)
            stateText.text = "تعذر بدء التدريب"
            status.text = e.message ?: "رفض Android تشغيل الخدمة"
        }
        refreshButtons()
    }

    private fun stopTrainingService() {
        if (TrainingEngine.isExternalFileSessionActive()) {
            TrainingEngine.requestExternalFileSessionStop()
            stateText.text = "جارٍ إيقاف تدريب الملف..."
            status.text = "سيتوقف بعد الدفعة الحالية ثم يحفظ ما اكتمل منها."
            refreshButtons()
            return
        }
        ModelManager.setTrainingEnabled(this, false)
        startService(Intent(this, TrainingService::class.java).setAction(TrainingService.ACTION_STOP))
        stateText.text = "جارٍ الإيقاف والحفظ..."
        status.text = "لن تُغلق الخدمة قبل كتابة الأوزان وحالة Adam إلى الـCheckpoint."
        refreshButtons()
    }

    private fun renderBroadcast(intent: Intent) {
        val samples = intent.getLongExtra(TrainingService.EXTRA_SAMPLES, 0L)
        val batches = intent.getLongExtra(TrainingService.EXTRA_BATCHES, 0L)
        val round = intent.getLongExtra(TrainingService.EXTRA_EPOCH, 0L)
        val loss = intent.getDoubleExtra(TrainingService.EXTRA_LOSS, Double.NaN)
        val validationMse = intent.getDoubleExtra(TrainingService.EXTRA_VALIDATION, Double.NaN)
        val rmse = intent.getDoubleExtra(TrainingService.EXTRA_VALIDATION_RMSE, Double.NaN)
        val accuracy = intent.getDoubleExtra(TrainingService.EXTRA_ACCURACY, Double.NaN)
        val gradient = intent.getDoubleExtra(TrainingService.EXTRA_GRADIENT_NORM, Double.NaN)
        val equation = intent.getStringExtra(TrainingService.EXTRA_EQUATION).orEmpty()
        val family = intent.getStringExtra(TrainingService.EXTRA_FAMILY).orEmpty()

        stateText.text = "● التدريب يعمل في الخلفية"
        samplesText.text = number(samples)
        batchesText.text = number(batches)
        lossText.text = metric(loss, 6)
        rmseText.text = metric(rmse, 2)
        accuracyText.text = if (accuracy.isFinite()) "%.1f%%".format(Locale.US, accuracy * 100.0) else "—"
        gradientText.text = metric(gradient, 3)
        if (equation.isNotBlank()) equationText.text = equation
        if (family.isNotBlank()) familyText.text = "آخر عائلة: ${familyLabel(family)}"
        status.text = "دورة منهجية: %,d • التحقق على 160 مثالًا ثابتًا لم تدخل التدريب.".format(Locale.US, round)
        lossChart.addMetrics(loss, validationMse)
    }

    private fun renderSnapshot(snapshot: TrainingEngine.Snapshot) {
        if (snapshot.samples <= 0L && snapshot.batches <= 0L) return
        samplesText.text = number(snapshot.samples)
        batchesText.text = number(snapshot.batches)
        lossText.text = metric(snapshot.loss, 6)
        rmseText.text = metric(snapshot.validation.rmse, 2)
        accuracyText.text = if (snapshot.validation.withinOneUnitRatio.isFinite()) {
            "%.1f%%".format(Locale.US, snapshot.validation.withinOneUnitRatio * 100.0)
        } else "—"
        gradientText.text = metric(snapshot.gradientNorm, 3)
        if (snapshot.lastEquation.isNotBlank()) equationText.text = snapshot.lastEquation
        if (snapshot.lastFamily.isNotBlank()) familyText.text = "آخر عائلة: ${familyLabel(snapshot.lastFamily)}"
        if (snapshot.loss.isFinite()) lossChart.addMetrics(snapshot.loss, snapshot.validation.normalizedMse)
        if (snapshot.paused) {
            stateText.text = "● متوقف مؤقتًا"
            status.text = snapshot.reason
        }
    }

    private fun renderStoredState() {
        val samples = ModelManager.trainingSamples(this)
        val batches = ModelManager.trainingBatches(this)
        val lastLoss = ModelManager.lastLoss(this)
        val validationMse = ModelManager.lastValidationMse(this)
        val accuracy = ModelManager.lastValidationAccuracy(this)
        samplesText.text = number(samples)
        batchesText.text = number(batches)
        lossText.text = metric(lastLoss, 6)
        rmseText.text = if (validationMse.isFinite()) metric(sqrt(validationMse) * TrainingEngine.OUTPUT_SCALE, 2) else "—"
        accuracyText.text = if (accuracy.isFinite()) "%.1f%%".format(Locale.US, accuracy * 100.0) else "—"
        stateText.text = when {
            TrainingEngine.isExternalFileSessionActive() -> "● تدريب ملف خارجي يعمل"
            ModelManager.isTrainingEnabled(this) -> "● التدريب مفعّل"
            samples > 0 -> "Checkpoint محفوظ"
            else -> "التدريب متوقف"
        }
        status.text = when {
            TrainingEngine.isExternalFileSessionActive() -> "يمكنك مغادرة هذه الشاشة؛ يبقى خيط التدريب داخل عملية التطبيق ما دامت العملية حية."
            samples > 0 -> "استئناف حقيقي من نفس الأوزان ولحظات Adam؛ لا يبدأ النموذج من الصفر."
            else -> status.text
        }
    }

    private fun startFileTraining(uri: android.net.Uri) {
        if (fileTraining || !TrainingEngine.beginExternalFileSession()) {
            Toast.makeText(this, "يوجد تدريب ملف يعمل بالفعل", Toast.LENGTH_SHORT).show()
            return
        }
        fileTraining = true
        refreshButtons()
        val baseSamples = ModelManager.trainingSamples(this)
        val baseBatches = ModelManager.trainingBatches(this)

        Thread {
            android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_BACKGROUND)
            var processed = 0L
            var valid = 0L
            var trainedValid = 0L
            var trainedBatches = 0L
            var lastCheckpointAt = android.os.SystemClock.elapsedRealtime()
            val chunk = ArrayList<Pair<String, DoubleArray>>(2_000)
            fun saveFileCheckpoint() {
                val validation = TrainingEngine.lastValidation
                val previousBest = ModelManager.bestValidationMse(this@TrainingActivity)
                val best = if (validation.normalizedMse.isFinite()) min(previousBest, validation.normalizedMse) else previousBest
                ModelManager.save(
                    context = this@TrainingActivity,
                    samples = baseSamples + trainedValid,
                    batches = baseBatches + trainedBatches,
                    bestValidationMse = best,
                    lastValidationMse = validation.normalizedMse,
                    validationAccuracy = validation.withinOneUnitRatio
                )
            }
            try {
                contentResolver.openInputStream(uri)?.bufferedReader()?.useLines { lines ->
                    lines.forEach { raw ->
                        if (TrainingEngine.isExternalFileStopRequested()) throw InterruptedException("أوقف المستخدم تدريب الملف")
                        val line = raw.trim()
                        if (line.isEmpty() || line.startsWith("#")) return@forEach
                        processed++
                        parseTrainingLine(line)?.let { example ->
                            chunk += example
                            valid++
                        }
                        if (chunk.size >= 2_000) {
                            val chunkSize = chunk.size
                            TrainingEngine.trainFile(chunk)
                            if (TrainingEngine.isExternalFileStopRequested()) throw InterruptedException("أوقف المستخدم تدريب الملف")
                            trainedValid += chunkSize
                            trainedBatches += ceil(chunkSize.toDouble() / TrainingEngine.BATCH_SIZE).toLong() * TrainingEngine.EPOCHS
                            chunk.clear()
                            val now = android.os.SystemClock.elapsedRealtime()
                            if (now - lastCheckpointAt >= 5L * 60L * 1_000L) {
                                saveFileCheckpoint()
                                lastCheckpointAt = now
                            }
                            runOnUiThread { status.text = "تدريب الملف... قرأت ${number(processed)} سطر، وقبلت ${number(valid)}." }
                        }
                    }
                } ?: error("تعذر فتح الملف")
                if (chunk.isNotEmpty()) {
                    val chunkSize = chunk.size
                    TrainingEngine.trainFile(chunk)
                    if (!TrainingEngine.isExternalFileStopRequested()) {
                        trainedValid += chunkSize
                        trainedBatches += ceil(chunkSize.toDouble() / TrainingEngine.BATCH_SIZE).toLong() * TrainingEngine.EPOCHS
                    }
                }
                if (TrainingEngine.isExternalFileStopRequested()) throw InterruptedException("أوقف المستخدم تدريب الملف")

                saveFileCheckpoint()
                runOnUiThread {
                    renderStoredState()
                    status.text = "اكتمل تدريب الملف: ${number(trainedValid)} مدرّب، ${number(processed - valid)} مرفوض. تم حفظ النموذج."
                }
            } catch (e: Exception) {
                try { saveFileCheckpoint() } catch (_: Exception) { }
                runOnUiThread {
                    status.text = if (e is InterruptedException) "تم إيقاف تدريب الملف وحفظ الأوزان الحالية (${number(trainedValid)} مثال مكتمل)."
                    else "فشل تدريب الملف: ${e.message ?: "خطأ غير معروف"}"
                }
            } finally {
                fileTraining = false
                TrainingEngine.endExternalFileSession()
                runOnUiThread { refreshButtons() }
            }
        }.start()
    }

    private fun parseTrainingLine(line: String): Pair<String, DoubleArray>? {
        return try {
            val parts = line.split('|', limit = 2)
            val equation = parts[0].trim()
            if (equation.isEmpty() || equation.length > 512) return null
            val encoding = MathTokenizer.encode(equation)
            if (encoding.truncated || encoding.unknownCount > 0) return null
            val values = if (parts.size == 2 && parts[1].isNotBlank()) {
                val nums = parts[1].split(',').mapNotNull { it.trim().toDoubleOrNull() }
                if (nums.isEmpty()) return null
                doubleArrayOf(nums[0], nums.getOrElse(1) { 0.0 })
            } else {
                val answer = MathTeacher.solve(equation)
                if (answer.x == null && answer.y == null) return null
                doubleArrayOf(answer.x ?: 0.0, answer.y ?: 0.0)
            }
            val candidate = GeneratedExample(equation, values[0], values[1], "file")
            if (TrainingEngine.isReservedForValidation(equation) || !GeneratedEquationValidator.isValid(candidate)) null else equation to values
        } catch (_: Exception) {
            null
        }
    }

    private fun refreshButtons() {
        val enabled = ModelManager.isTrainingEnabled(this)
        val fileActive = fileTraining || TrainingEngine.isExternalFileSessionActive()
        startButton.isEnabled = !enabled && !fileActive
        stopButton.isEnabled = enabled || fileActive
        fileButton.isEnabled = !enabled && !fileActive
    }

    private fun number(value: Long): String = "%,d".format(Locale.US, value)
    private fun metric(value: Double, digits: Int): String = if (value.isFinite()) "%.${digits}f".format(Locale.US, value) else "—"

    private fun familyLabel(value: String): String = when {
        value == "linear" -> "خطية متنوعة"
        value == "linear-system" -> "نظام خطي 2×2"
        value == "quadratic" -> "تربيعية"
        value == "cubic" -> "تكعيبية"
        value == "quartic" -> "درجة رابعة"
        value == "quintic" -> "درجة خامسة"
        value == "rational" -> "كسرية"
        value == "radical" -> "جذرية"
        value == "absolute" -> "قيمة مطلقة"
        value.startsWith("exponential") -> "أسية"
        value.startsWith("logarithmic") -> "لوغاريتمية"
        value.startsWith("trigonometric") -> "مثلثية"
        else -> value
    }
}
