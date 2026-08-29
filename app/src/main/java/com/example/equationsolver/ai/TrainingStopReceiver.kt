package com.example.equationsolver.ai

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent

class TrainingStopReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent?) {
        if (intent?.action != TrainingService.ACTION_STOP) return
        ModelManager.setTrainingEnabled(context, false)
        context.stopService(Intent(context, TrainingService::class.java))
    }
}
