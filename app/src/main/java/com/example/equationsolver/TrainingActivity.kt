package com.example.equationsolver

import android.net.Uri
import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import com.example.equationsolver.ai.ModelManager
import com.example.equationsolver.ai.TrainingEngine
import kotlinx.coroutines.*
import java.io.BufferedReader
import java.io.InputStreamReader

class TrainingActivity : AppCompatActivity() {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private lateinit var status: TextView
    private lateinit var startButton: Button
    private lateinit var stopButton: Button
    private lateinit var fileButton: Button
    private lateinit var lossChart: LossChartView
    private var trainingJob: Job? = null
    private val picker = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? -> uri?.let(::readTrainingFile) }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_training)
        status = findViewById(R.id.textStatus)
        startButton = findViewById(R.id.btnTrainRandom)
        stopButton = findViewById(R.id.btnStopTraining)
        fileButton = findViewById(R.id.btnLoadFile)
        lossChart = findViewById(R.id.lossChart)
        ModelManager.init(applicationContext)

        startButton.setOnClickListener { startContinuousTraining() }
        stopButton.setOnClickListener { stopContinuousTraining() }
        fileButton.setOnClickListener { if (trainingJob?.isActive != true) picker.launch("text/plain") }
    }

    private fun startContinuousTraining() {
        if (trainingJob?.isActive == true) return
        ModelManager.init(applicationContext)
        lossChart.clearData()
        setBusy(true)
        status.text = "جاري بدء التدريب..."
        trainingJob = scope.launch {
            try {
                TrainingEngine.trainContinuous(learningRate = 0.001) { samples, batches, epoch, loss, validation ->
                    launch(Dispatchers.Main) {
                        lossChart.addLoss(loss)
                        status.text = "التدريب مستمر\n" +
                            "المعادلات: %,d\nالدفعات: %,d\nEpoch: %,d\nLoss: %.8f\nValidation Loss: %.8f".format(samples, batches, epoch, loss, validation)
                    }
                }
            } catch (_: CancellationException) {
            } catch (e: Exception) {
                withContext(Dispatchers.Main) { status.text = "خطأ: ${e.message}" }
            } finally {
                ModelManager.save(this@TrainingActivity)
                withContext(Dispatchers.Main) { setBusy(false); status.text = status.text.toString() + "\nتم إيقاف التدريب وحفظ النموذج." }
            }
        }
    }

    private fun stopContinuousTraining() {
        trainingJob?.cancel()
        trainingJob = null
    }

    private fun readTrainingFile(uri: Uri) {
        setBusy(true)
        status.text = "جاري قراءة الملف..."
        scope.launch {
            try {
                val examples = mutableListOf<Pair<String, DoubleArray>>()
                contentResolver.openInputStream(uri)?.use { stream ->
                    BufferedReader(InputStreamReader(stream)).useLines { lines ->
                        lines.forEach { line ->
                            if (line.trimStart().startsWith("#")) return@forEach
                            val parts = line.split('|', limit = 2)
                            if (parts.size != 2) return@forEach
                            val equation = parts[0].trim()
                            val values = parts[1].split(',').mapNotNull { it.trim().toDoubleOrNull() }
                            if (equation.isNotEmpty() && values.isNotEmpty() && values.size <= 2) examples += equation to doubleArrayOf(values[0], values.getOrElse(1) { 0.0 })
                        }
                    }
                }
                if (examples.isEmpty()) error("الملف فارغ أو الصيغة غير صحيحة")
                withContext(Dispatchers.Main) { status.text = "تمت قراءة ${examples.size} مثال. بدء التدريب..." }
                TrainingEngine.trainFile(examples) { n -> launch(Dispatchers.Main) { status.text = "تقدم التدريب: $n مثال معالجة..." } }
                ModelManager.save(this@TrainingActivity)
                withContext(Dispatchers.Main) { status.text = "اكتمل التدريب: ${examples.size} مثال\nValidation MSE: %.6f\nتم حفظ النموذج.".format(TrainingEngine.lastValidationMse); setBusy(false) }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) { status.text = "خطأ: ${e.message}"; setBusy(false) }
            }
        }
    }

    private fun setBusy(busy: Boolean) {
        startButton.isEnabled = !busy
        stopButton.isEnabled = busy
        fileButton.isEnabled = !busy
    }

    override fun onDestroy() {
        trainingJob?.cancel()
        scope.cancel()
        super.onDestroy()
    }
}
