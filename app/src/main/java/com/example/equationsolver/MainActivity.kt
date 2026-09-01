package com.example.equationsolver

import android.content.Intent
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import com.example.equationsolver.ai.ModelStore
import com.example.equationsolver.ui.*
import java.util.Locale

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) { super.onCreate(savedInstanceState); ModelStore.init(this); render() }
    override fun onResume(){super.onResume();render()}
    private fun render(){
        val root=screen("RSNN Lab V2","Open-Growth • Full INT8 inference • DeepMind-only training")
        val c=root.card();val m=ModelStore.model;val cfg=ModelStore.config;val core=m.coreSnapshot()
        c.label("${cfg.hiddenDim} عصبون • T=${cfg.timeSteps} • INT8", ACCENT,17f)
        c.label("Active: %,d / %,d   •   Protected: %,d".format(Locale.US,core.active,core.active+core.dormant,core.protected))
        c.label("Phase: ${core.phase}   •   Step: ${core.step}   •   Grad: %.4f".format(Locale.US,core.gradientNorm))
        c.label(if(ModelStore.trainingEnabled())"● التدريب يعمل في الخلفية" else "التدريب متوقف")
        root.button("تدريب DeepMind",true){startActivity(Intent(this,TrainingActivity::class.java))}
        root.button("اختبار النموذج"){startActivity(Intent(this,TestActivity::class.java))}
        root.button("قلب النموذج — Live Core"){startActivity(Intent(this,CoreActivity::class.java))}
        root.button("إعدادات النموذج — Expert"){startActivity(Intent(this,SettingsActivity::class.java))}
        root.button("Model Vault — Google Drive"){startActivity(Intent(this,VaultActivity::class.java))}
        root.button("استيراد أوزان / Checkpoint"){startActivity(Intent(this,WeightsActivity::class.java))}
        val d=root.card();d.label("مصدر التدريب",SKY,16f);d.label("لا يوجد مولّد بيانات داخلي. التطبيق يقبل فقط ملفات algebra.linear_2d المصدّرة من Google DeepMind mathematics_dataset عند commit المثبّت.")
    }
}
