package com.example.equationsolver.ai

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.os.Build
import android.os.IBinder
import androidx.core.app.NotificationCompat
import com.example.equationsolver.R
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch

class TrainingService : Service() {
    companion object {
        const val ACTION_START = "com.example.equationsolver.START_TRAINING"
        const val ACTION_STOP = "com.example.equationsolver.STOP_TRAINING"
        const val ACTION_PROGRESS = "com.example.equationsolver.TRAINING_PROGRESS"
        const val EXTRA_SAMPLES = "samples"
        const val EXTRA_BATCHES = "batches"
        const val EXTRA_EPOCH = "epoch"
        const val EXTRA_LOSS = "loss"
        const val EXTRA_VALIDATION = "validation"
        private const val CHANNEL_ID = "continuous_training"
        private const val NOTIFICATION_ID = 2201
    }

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private var job: Job? = null

    override fun onCreate() {
        super.onCreate()
        ModelManager.init(applicationContext)
        createChannel()
        startForeground(NOTIFICATION_ID, notification("تدريب النموذج مستمر"))
        startTrainingIfNeeded()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP) stopTrainingAndSelf()
        else startTrainingIfNeeded()
        return START_STICKY
    }

    private fun startTrainingIfNeeded() {
        if (job?.isActive == true) return
        job = scope.launch {
            TrainingEngine.trainContinuous(applicationContext, learningRate = 0.001) { samples, batches, epoch, loss, validation ->
                val update = Intent(ACTION_PROGRESS).setPackage(packageName)
                    .putExtra(EXTRA_SAMPLES, samples).putExtra(EXTRA_BATCHES, batches)
                    .putExtra(EXTRA_EPOCH, epoch).putExtra(EXTRA_LOSS, loss)
                    .putExtra(EXTRA_VALIDATION, validation)
                sendBroadcast(update)
                if (batches % 20L == 0L) updateNotification("تدريب: %,d معادلة | Loss %.6f".format(samples, loss))
            }
        }
    }

    private fun stopTrainingAndSelf() {
        job?.cancel(); job = null
        ModelManager.save(applicationContext, ModelManager.trainingSamples(applicationContext), ModelManager.trainingBatches(applicationContext), ModelManager.bestValidationMse(applicationContext))
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    private fun createChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val manager = getSystemService(NotificationManager::class.java)
            manager.createNotificationChannel(NotificationChannel(CHANNEL_ID, "تدريب النموذج", NotificationManager.IMPORTANCE_LOW))
        }
    }

    private fun notification(text: String): Notification = NotificationCompat.Builder(this, CHANNEL_ID)
        .setSmallIcon(android.R.drawable.stat_sys_download)
        .setContentTitle("Equation Solver AI")
        .setContentText(text)
        .setOngoing(true)
        .build()

    private fun updateNotification(text: String) = getSystemService(NotificationManager::class.java).notify(NOTIFICATION_ID, notification(text))
    override fun onBind(intent: Intent?): IBinder? = null
    override fun onDestroy() { job?.cancel(); scope.cancel(); super.onDestroy() }
}
