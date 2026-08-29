package com.example.equationsolver.ai

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import androidx.core.content.ContextCompat

class TrainingBootReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent?) {
        if (intent?.action != Intent.ACTION_BOOT_COMPLETED && intent?.action != Intent.ACTION_LOCKED_BOOT_COMPLETED) return
        if (!ModelManager.isTrainingEnabled(context)) return
        ContextCompat.startForegroundService(context, Intent(context, TrainingService::class.java).setAction(TrainingService.ACTION_START))
    }
}
