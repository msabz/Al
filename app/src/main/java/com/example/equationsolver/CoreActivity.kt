package com.example.equationsolver

import android.os.Bundle
import android.os.Handler
import android.os.Looper
import androidx.appcompat.app.AppCompatActivity
import com.example.equationsolver.ai.ModelStore
import com.example.equationsolver.ui.*
import java.util.Locale

class CoreActivity:AppCompatActivity(){
    private val h=Handler(Looper.getMainLooper());private var frozen=false;private lateinit var view:CoreVisualizerView;private lateinit var stats:android.widget.TextView
    private val tick=object:Runnable{override fun run(){if(!frozen){val s=ModelStore.model.coreSnapshot();view.update(s);stats.text="Phase ${s.phase} • cycle ${s.structuralCycle}\nActive %,d • Dormant %,d • Protected %,d\nFiring %.3f • |W|mean %.5f • |W|max %.5f\nINT8 saturation %.3f%% • grad %.4f • step %,d\nLast growth +%,d • prune -%,d".format(Locale.US,s.active,s.dormant,s.protected,s.firingRate,s.weightMeanAbs,s.weightMaxAbs,s.int8Saturation*100,s.gradientNorm,s.step,s.lastGrown,s.lastPruned)};h.postDelayed(this,ModelStore.config.coreRefreshMs.toLong())}}
    override fun onCreate(b:Bundle?){super.onCreate(b);val r=screen("Model Core — Live","عرض مباشر لنشاط العصبونات، membrane، spikes، النمو والحذف والحماية.");view=CoreVisualizerView(this);r.addView(view,android.widget.LinearLayout.LayoutParams(android.view.ViewGroup.LayoutParams.MATCH_PARENT,view.dp(360)).apply{setMargins(0,view.dp(12),0,0)});val c=r.card();stats=c.label("...",TEXT,14f);r.button("Freeze / Resume"){frozen=!frozen;toast(if(frozen)"تم تجميد العرض فقط" else "عاد العرض المباشر")};r.button("تشغيل خطوة اختبار INT8"){ModelStore.model.predictInt8(floatArrayOf(1f,0f,0f,0f,1f,0f))};r.button("رجوع"){finish()}}
    override fun onStart(){super.onStart();h.post(tick)}
    override fun onStop(){h.removeCallbacks(tick);super.onStop()}
}
