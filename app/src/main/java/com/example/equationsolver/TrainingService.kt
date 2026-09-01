package com.example.equationsolver

import android.app.*
import android.content.Intent
import android.net.Uri
import android.os.*
import androidx.core.app.NotificationCompat
import com.example.equationsolver.ai.ModelStore
import com.example.equationsolver.data.DeepMindDataset
import com.example.equationsolver.data.DeepMindSample
import java.util.concurrent.atomic.AtomicBoolean

class TrainingService:Service(){
    companion object{const val ACTION_START="rsnn.START";const val ACTION_STOP="rsnn.STOP";const val ACTION_PROGRESS="rsnn.PROGRESS";const val EXTRA_STATUS="status";const val EXTRA_LOSS="loss";const val EXTRA_SAMPLES="samples";const val EXTRA_BATCHES="batches";private const val CH="rsnn_training";private const val ID=4101}
    private val stop=AtomicBoolean(false);private var thread:Thread?=null;private var wake:PowerManager.WakeLock?=null
    override fun onCreate(){super.onCreate();ModelStore.init(this);createChannel();startForeground(ID,note("جاهز"));wake=(getSystemService(POWER_SERVICE)as PowerManager).newWakeLock(PowerManager.PARTIAL_WAKE_LOCK,"RSNN:training").apply{setReferenceCounted(false);acquire()}}
    override fun onStartCommand(i:Intent?,flags:Int,startId:Int):Int{when(i?.action){ACTION_STOP->stopNow();ACTION_START,null->startIfNeeded()};return START_STICKY}
    private fun startIfNeeded(){if(thread?.isAlive==true)return;val train=ModelStore.trainUri()?:run{update("اختر ملف DeepMind train أولاً");ModelStore.setTrainingEnabled(false);return};stop.set(false);ModelStore.setTrainingEnabled(true);thread=Thread{runLoop(Uri.parse(train))}.apply{name="RSNN-DeepMind-Training";priority=Thread.NORM_PRIORITY-1;start()}}
    private fun runLoop(uri:Uri){var samples=0L;var batches=0L;var lastSave=SystemClock.elapsedRealtime();val batch=ArrayList<DeepMindSample>();val structure=ArrayList<DeepMindSample>()
        try{while(!stop.get()&&ModelStore.trainingEnabled()){
            for(s in DeepMindDataset.open(contentResolver,uri)){if(stop.get())break;while(!stop.get()&&pauseForDevice())SystemClock.sleep(3000);if(stop.get())break;batch+=s;if(structure.size<32)structure+=s;samples++
                if(batch.size>=ModelStore.config.batchSize){val m=ModelStore.model.trainBatch(batch);batch.clear();batches++;if(batches%ModelStore.config.structureEveryBatches==0L&&structure.isNotEmpty())ModelStore.model.structuralStep(structure);if(batches%5L==0L){val msg="%,d samples • loss %.6f • active %,d".format(samples,m.loss,m.activeWeights);broadcast(msg,m.loss,samples,batches);update(msg)}
                val now=SystemClock.elapsedRealtime();if(now-lastSave>=ModelStore.config.checkpointMinutes*60_000L){ModelStore.saveCheckpoint();lastSave=now}}
            }
        }}catch(e:Exception){if(!stop.get())broadcast("خطأ تدريب: ${e.message}",Float.NaN,samples,batches)}finally{runCatching{ModelStore.saveCheckpoint()};ModelStore.setTrainingEnabled(false);stopForeground(STOP_FOREGROUND_REMOVE);stopSelf()}}
    private fun pauseForDevice():Boolean{val bm=getSystemService(BATTERY_SERVICE)as android.os.BatteryManager;val level=bm.getIntProperty(android.os.BatteryManager.BATTERY_PROPERTY_CAPACITY);if(level in 0 until ModelStore.config.minBatteryPercent){update("متوقف: البطارية $level%");return true};if(Build.VERSION.SDK_INT>=29){val pm=getSystemService(POWER_SERVICE)as PowerManager;if(pm.currentThermalStatus>=PowerManager.THERMAL_STATUS_MODERATE){update("متوقف مؤقتاً بسبب الحرارة");return true}};return false}
    private fun stopNow(){stop.set(true);ModelStore.setTrainingEnabled(false);thread?.interrupt()}
    private fun broadcast(status:String,loss:Float,samples:Long,batches:Long){sendBroadcast(Intent(ACTION_PROGRESS).setPackage(packageName).putExtra(EXTRA_STATUS,status).putExtra(EXTRA_LOSS,loss).putExtra(EXTRA_SAMPLES,samples).putExtra(EXTRA_BATCHES,batches))}
    private fun createChannel(){if(Build.VERSION.SDK_INT>=26)(getSystemService(NOTIFICATION_SERVICE)as NotificationManager).createNotificationChannel(NotificationChannel(CH,"RSNN training",NotificationManager.IMPORTANCE_LOW))}
    private fun note(t:String)=NotificationCompat.Builder(this,CH).setSmallIcon(android.R.drawable.stat_sys_download).setContentTitle("RSNN Lab V2").setContentText(t).setOngoing(true).setContentIntent(PendingIntent.getActivity(this,1,Intent(this,TrainingActivity::class.java),PendingIntent.FLAG_UPDATE_CURRENT or if(Build.VERSION.SDK_INT>=23)PendingIntent.FLAG_IMMUTABLE else 0)).build()
    private fun update(t:String)=(getSystemService(NOTIFICATION_SERVICE)as NotificationManager).notify(ID,note(t))
    override fun onBind(i:Intent?)=null
    override fun onDestroy(){stop.set(true);wake?.takeIf{it.isHeld}?.release();super.onDestroy()}
}
