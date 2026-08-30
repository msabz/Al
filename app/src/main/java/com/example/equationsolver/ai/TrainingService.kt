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
import android.os.Process
import androidx.core.app.NotificationCompat
import com.example.equationsolver.MainActivity
import com.example.equationsolver.R
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

class TrainingService : Service() {
    companion object {
        const val ACTION_START = "com.example.equationsolver.START_TRAINING"
        const val ACTION_STOP = "com.example.equationsolver.STOP_TRAINING"
        const val ACTION_PROGRESS = "com.example.equationsolver.TRAINING_PROGRESS"
        const val ACTION_PAUSED = "com.example.equationsolver.TRAINING_PAUSED"
        const val ACTION_ERROR = "com.example.equationsolver.TRAINING_ERROR"
        const val ACTION_STOPPED = "com.example.equationsolver.TRAINING_STOPPED"
        const val EXTRA_SAMPLES = "samples"
        const val EXTRA_BATCHES = "batches"
        const val EXTRA_EPOCH = "epoch"
        const val EXTRA_LOSS = "loss"
        const val EXTRA_VALIDATION = "validation"
        const val EXTRA_VALIDATION_RMSE = "validation_rmse"
        const val EXTRA_VALIDATION_MAE = "validation_mae"
        const val EXTRA_ACCURACY = "accuracy"
        const val EXTRA_EQUATION = "equation"
        const val EXTRA_FAMILY = "family"
        const val EXTRA_GRADIENT_NORM = "gradient_norm"
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
        startForeground(NOTIFICATION_ID, notification("التدريب جاهز"))
        acquireWakeLock()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        return when (intent?.action) {
            ACTION_STOP -> {
                stopTrainingAndSelf()
                START_NOT_STICKY
            }
            ACTION_START -> {
                ModelManager.setTrainingEnabled(applicationContext, true)
                startTrainingIfNeeded()
                START_STICKY
            }
            null -> {
                if (ModelManager.isTrainingEnabled(applicationContext)) {
                    startTrainingIfNeeded()
                    START_STICKY
                } else {
                    releaseWakeLock()
                    stopForeground(STOP_FOREGROUND_REMOVE)
                    stopSelf()
                    START_NOT_STICKY
                }
            }
            else -> START_STICKY
        }
    }

    private fun startTrainingIfNeeded() {
        if (job?.isActive == true) return
        job = scope.launch {
            Process.setThreadPriority(Process.THREAD_PRIORITY_BACKGROUND)
            while (isActive && ModelManager.isTrainingEnabled(applicationContext)) {
                try {
                    TrainingEngine.trainContinuous(applicationContext, learningRate = 0.0007) { state ->
                        if (state.paused) {
                            updateNotification("متوقف مؤقتًا: ${state.reason}")
                            sendBroadcast(Intent(ACTION_PAUSED).setPackage(packageName).putExtra(EXTRA_REASON, state.reason))
                        } else {
                            val metric = if (state.validation.rmse.isFinite()) {
                                "RMSE %.2f".format(state.validation.rmse)
                            } else if (state.loss.isFinite()) {
                                "Loss %.6f".format(state.loss)
                            } else "تهيئة أول دفعة"
                            updateNotification("%,d معادلة | %s".format(state.samples, metric))
                            sendBroadcast(
                                Intent(ACTION_PROGRESS).setPackage(packageName)
                                    .putExtra(EXTRA_SAMPLES, state.samples)
                                    .putExtra(EXTRA_BATCHES, state.batches)
                                    .putExtra(EXTRA_EPOCH, state.curriculumRound)
                                    .putExtra(EXTRA_LOSS, state.loss)
                                    .putExtra(EXTRA_VALIDATION, state.validation.normalizedMse)
                                    .putExtra(EXTRA_VALIDATION_RMSE, state.validation.rmse)
                                    .putExtra(EXTRA_VALIDATION_MAE, state.validation.meanAbsoluteError)
                                    .putExtra(EXTRA_ACCURACY, state.validation.withinOneUnitRatio)
                                    .putExtra(EXTRA_EQUATION, state.lastEquation)
                                    .putExtra(EXTRA_FAMILY, state.lastFamily)
                                    .putExtra(EXTRA_GRADIENT_NORM, state.gradientNorm)
                            )
                        }
                    }
                } catch (e: CancellationException) {
                    throw e
                } catch (e: Exception) {
                    updateNotification("خطأ تدريب؛ ستتم إعادة المحاولة تلقائيًا")
                    sendBroadcast(
                        Intent(ACTION_ERROR).setPackage(packageName)
                            .putExtra(EXTRA_REASON, e.message ?: "خطأ تدريب غير معروف")
                    )
                    delay(5000L)
                }
            }
        }
    }

    private fun stopTrainingAndSelf() {
        ModelManager.setTrainingEnabled(applicationContext, false)
        val activeJob = job
        job = null
        scope.launch {
            activeJob?.cancelAndJoin()
            sendBroadcast(Intent(ACTION_STOPPED).setPackage(packageName))
            releaseWakeLock()
            stopForeground(STOP_FOREGROUND_REMOVE)
            stopSelf()
        }
    }

    private fun acquireWakeLock() {
        if (wakeLock?.isHeld == true) return
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
            .setContentTitle(getString(R.string.app_name))
            .setContentText(text)
            .setContentIntent(openPending)
            .addAction(android.R.drawable.ic_media_pause, "إيقاف وحفظ", stopPending)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .build()
    }

    private fun updateNotification(text: String) =
        getSystemService(NotificationManager::class.java).notify(NOTIFICATION_ID, notification(text))

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        job?.cancel()
        releaseWakeLock()
        scope.cancel()
        super.onDestroy()
    }
}
