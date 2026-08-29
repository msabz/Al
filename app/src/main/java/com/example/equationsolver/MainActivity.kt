package com.example.equationsolver

import android.content.Intent
import android.os.Bundle
import android.widget.Button
import androidx.appcompat.app.AppCompatActivity

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        findViewById<Button>(R.id.btnGoTest).setOnClickListener { startActivity(Intent(this, TestActivity::class.java)) }
        findViewById<Button>(R.id.btnGoTrain).setOnClickListener { startActivity(Intent(this, TrainingActivity::class.java)) }
        findViewById<Button>(R.id.btnGoReinforcement).setOnClickListener { startActivity(Intent(this, ReinforcementActivity::class.java)) }
    }
}
