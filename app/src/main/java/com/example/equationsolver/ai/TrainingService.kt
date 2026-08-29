package com.example.equationsolver.ai

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import androidx.core.app.NotificationCompat
import com.example.equationsolver.MainActivity
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
        const val ACTION_PAUSED = "com.example.equationsolver.TRAINING_PAUSED"
        const val EXTRA_SAMPLES = "samples"
        const val EXTRA_BATCHES = "batches"
        const val EXTRA_EPOCH = "epoch"
        const val EXTRA_LOSS = "loss"
        const val EXTRA_VALIDATION = "validation"
        const val EXTRA_REASON = "reason"
        private const val CHANNEL_ID = "continuous_training"
        private const val NOTIFICATION_ID = 2201
    }

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private var job: Job? = null
    private var wakeLock: PowerManager.WakeLock? = null

    override fun onCreate() {
        super.onCreate()
        ModelManager.init(applicationContext)
        createChannel()
        startForeground(NOTIFICATION_ID, notification("التدريب مستمر"))
        acquireWakeLock()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        return when (intent?.action) {
            ACTION_STOP -> { stopTrainingAndSelf(); START_NOT_STICKY }
            ACTION_START, null -> {
                ModelManager.setTrainingEnabled(applicationContext, true)
                startTrainingIfNeeded()
                START_STICKY
            }
            else -> START_STICKY
        }
    }

    private fun startTrainingIfNeeded() {
        if (job?.isActive == true) return
        job = scope.launch {
            try {
                TrainingEngine.trainContinuous(applicationContext, learningRate = 0.001) { samples, batches, epoch, loss, validation, paused, reason ->
                    if (paused) {
                        updateNotification("التدريب متوقف مؤقتًا: $reason")
                        sendBroadcast(Intent(ACTION_PAUSED).setPackage(packageName).putExtra(EXTRA_REASON, reason))
                    } else {
                        updateNotification("تدريب: %,d معادلة | Loss %.6f".format(samples, loss))
                        sendBroadcast(Intent(ACTION_PROGRESS).setPackage(packageName)
                            .putExtra(EXTRA_SAMPLES, samples).putExtra(EXTRA_BATCHES, batches)
                            .putExtra(EXTRA_EPOCH, epoch).putExtra(EXTRA_LOSS, loss)
                            .putExtra(EXTRA_VALIDATION, validation))
                    }
                }
            } catch (_: Exception) {
                // START_STICKY lets Android recreate the service after process loss.
            }
        }
    }

    private fun stopTrainingAndSelf() {
        ModelManager.setTrainingEnabled(applicationContext, false)
        job?.cancel(); job = null
        ModelManager.save(applicationContext, ModelManager.trainingSamples(applicationContext), ModelManager.trainingBatches(applicationContext), ModelManager.bestValidationMse(applicationContext), ModelManager.lastLoss(applicationContext))
        releaseWakeLock()
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    private fun acquireWakeLock() {
        val pm = getSystemService(POWER_SERVICE) as PowerManager
        wakeLock = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "EquationSolverAI:Training").apply {
            setReferenceCounted(false)
            acquire()
        }
    }

    private fun releaseWakeLock() {
        wakeLock?.takeIf { it.isHeld }?.release()
        wakeLock = null
    }

    private fun createChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            getSystemService(NotificationManager::class.java).createNotificationChannel(
                NotificationChannel(CHANNEL_ID, "تدريب النموذج", NotificationManager.IMPORTANCE_LOW)
            )
        }
    }

    private fun notification(text: String): Notification {
        val stopIntent = Intent(this, TrainingStopReceiver::class.java).setAction(ACTION_STOP)
        val flags = PendingIntent.FLAG_UPDATE_CURRENT or if (Build.VERSION.SDK_INT >= 23) PendingIntent.FLAG_IMMUTABLE else 0
        val stopPending = PendingIntent.getBroadcast(this, 2202, stopIntent, flags)
        val openPending = PendingIntent.getActivity(this, 2203, Intent(this, MainActivity::class.java), flags)
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.stat_sys_download)
            .setContentTitle("Equation Solver AI")
            .setContentText(text)
            .setContentIntent(openPending)
            .addAction(android.R.drawable.ic_media_pause, "إيقاف", stopPending)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .build()
    }

    private fun updateNotification(text: String) = getSystemService(NotificationManager::class.java).notify(NOTIFICATION_ID, notification(text))
    override fun onBind(intent: Intent?): IBinder? = null
    override fun onDestroy() { job?.cancel(); releaseWakeLock(); scope.cancel(); super.onDestroy() }
}
