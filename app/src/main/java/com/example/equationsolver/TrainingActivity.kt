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
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.content.ContextCompat.registerReceiver
import com.example.equationsolver.ai.ModelManager
import com.example.equationsolver.ai.TrainingService
import java.util.Locale

class TrainingActivity : AppCompatActivity() {
    private lateinit var status: TextView
    private lateinit var startButton: Button
    private lateinit var stopButton: Button
    private lateinit var fileButton: Button
    private lateinit var lossChart: LossChartView
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
                    status.text = "التدريب مستمر\nالمعادلات: %,d\nالدفعات: %,d\nEpoch: %,d\nLoss: %.8f\nValidation Loss: %.8f".format(Locale.US, samples, batches, epoch, loss, validation)
                }
                TrainingService.ACTION_PAUSED -> status.text = "التدريب متوقف مؤقتًا\nالسبب: ${intent.getStringExtra(TrainingService.EXTRA_REASON) ?: "حماية الجهاز"}\nسيستأنف تلقائيًا عندما تصبح الظروف مناسبة."
            }
            refreshButtons()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_training)
        status = findViewById(R.id.textStatus); startButton = findViewById(R.id.btnTrainRandom); stopButton = findViewById(R.id.btnStopTraining); fileButton = findViewById(R.id.btnLoadFile); lossChart = findViewById(R.id.lossChart)
        ModelManager.init(applicationContext)
        if (Build.VERSION.SDK_INT >= 33 && ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) requestPermissions(arrayOf(Manifest.permission.POST_NOTIFICATIONS), 7001)
        startButton.setOnClickListener { startTrainingService() }
        stopButton.setOnClickListener { stopTrainingService() }
        fileButton.isEnabled = true
        showCheckpointInfo(); refreshButtons()
    }

    override fun onStart() { super.onStart(); val filter = IntentFilter().apply { addAction(TrainingService.ACTION_PROGRESS); addAction(TrainingService.ACTION_PAUSED) }; registerReceiver(this, receiver, filter, ContextCompat.RECEIVER_NOT_EXPORTED); refreshButtons() }
    override fun onStop() { unregisterReceiver(receiver); super.onStop() }

    private fun startTrainingService() {
        val intent = Intent(this, TrainingService::class.java).setAction(TrainingService.ACTION_START)
        if (Build.VERSION.SDK_INT >= 26) startForegroundService(intent) else startService(intent)
        status.text = "تم تشغيل التدريب في الخلفية. يمكنك مغادرة التطبيق وستستمر الخدمة."
        refreshButtons()
    }

    private fun stopTrainingService() {
        startService(Intent(this, TrainingService::class.java).setAction(TrainingService.ACTION_STOP))
        status.text = "جاري إيقاف التدريب وحفظ آخر Checkpoint..."
        refreshButtons()
    }

    private fun showCheckpointInfo() {
        val samples = ModelManager.trainingSamples(this); val best = ModelManager.bestValidationMse(this); val last = ModelManager.lastLoss(this)
        if (samples > 0) status.text = "نموذج محفوظ\nالمعادلات المدربة: %,d\nآخر Loss: %.8f\nأفضل Validation Loss: %.8f\nيمكن متابعة التدريب من نفس النقطة.".format(Locale.US, samples, last, best)
    }

    private fun refreshButtons() {
        val enabled = ModelManager.isTrainingEnabled(this)
        startButton.isEnabled = !enabled
        stopButton.isEnabled = enabled
    }

    override fun onDestroy() { super.onDestroy() }
}
