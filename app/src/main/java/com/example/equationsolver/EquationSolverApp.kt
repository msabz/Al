package com.example.equationsolver

import android.app.Application
import com.example.equationsolver.ai.ModelStore

class EquationSolverApp : Application() {
    override fun onCreate() { super.onCreate(); ModelStore.init(this) }
}
