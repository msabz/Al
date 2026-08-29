package com.example.equationsolver.ai

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent

class TrainingStopReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent?) {
        if (intent?.action != TrainingService.ACTION_STOP) return
        context.startService(Intent(context, TrainingService::class.java).setAction(TrainingService.ACTION_STOP))
    }
}
