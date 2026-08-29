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
    private lateinit var randomButton: Button
    private lateinit var fileButton: Button
    private val picker = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? -> uri?.let(::readTrainingFile) }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_training)
        status = findViewById(R.id.textStatus)
        randomButton = findViewById(R.id.btnTrainRandom)
        fileButton = findViewById(R.id.btnLoadFile)
        randomButton.setOnClickListener {
            setBusy(true)
            scope.launch {
                TrainingEngine.trainRandom(10_000) { n -> launch(Dispatchers.Main) { status.text = "تم تدريب $n مثال..." } }
                ModelManager.save(this@TrainingActivity)
                withContext(Dispatchers.Main) { status.text = "اكتمل التدريب وحُفظ النموذج."; setBusy(false) }
            }
        }
        fileButton.setOnClickListener { picker.launch("text/plain") }
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
                            if (equation.isNotEmpty() && values.isNotEmpty() && values.size <= 2) {
                                examples += equation to doubleArrayOf(values[0], values.getOrElse(1) { 0.0 })
                            }
                        }
                    }
                }
                if (examples.isEmpty()) error("الملف فارغ أو الصيغة غير صحيحة")
                TrainingEngine.trainFile(examples) { n -> launch(Dispatchers.Main) { status.text = "تمت معالجة $n مثال..." } }
                ModelManager.save(this@TrainingActivity)
                withContext(Dispatchers.Main) { status.text = "تم تدريب ${examples.size} مثال وحفظ النموذج."; setBusy(false) }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) { status.text = "خطأ: ${e.message}"; setBusy(false) }
            }
        }
    }

    private fun setBusy(busy: Boolean) {
        randomButton.isEnabled = !busy
        fileButton.isEnabled = !busy
    }

    override fun onDestroy() {
        scope.cancel()
        super.onDestroy()
    }
}
