package com.example.equationsolver

import android.app.Application
import com.example.equationsolver.ai.ModelManager

class EquationSolverApp : Application() {
    override fun onCreate() {
        super.onCreate()
        ModelManager.init(this)
    }
}
