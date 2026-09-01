package com.example.equationsolver

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.view.View
import com.example.equationsolver.ai.OpenGrowthRsnnV2
import com.example.equationsolver.ui.*
import kotlin.math.ceil
import kotlin.math.sqrt

class CoreVisualizerView(context:Context):View(context){
    private val p=Paint(Paint.ANTI_ALIAS_FLAG);private var snap:OpenGrowthRsnnV2.CoreSnapshot?=null
    fun update(s:OpenGrowthRsnnV2.CoreSnapshot){snap=s;invalidate()}
    override fun onMeasure(w:Int,h:Int){setMeasuredDimension(MeasureSpec.getSize(w),(resources.displayMetrics.density*360).toInt())}
    override fun onDraw(c:Canvas){super.onDraw(c);c.drawColor(SURFACE);val s=snap?:return;val n=s.spikes.size;if(n==0)return;val cols=ceil(sqrt(n.toDouble())).toInt();val rows=ceil(n/cols.toDouble()).toInt();val dx=width.toFloat()/(cols+1);val dy=height.toFloat()/(rows+1);val maxMem=s.membrane.maxOfOrNull{kotlin.math.abs(it)}?.coerceAtLeast(.001f)?:1f
        for(i in 0 until n){val x=(i%cols+1)*dx;val y=(i/cols+1)*dy;val a=(kotlin.math.abs(s.membrane[i])/maxMem).coerceIn(0f,1f);p.color=when{ s.spikes[i].toInt()!=0->ACCENT; a>.66f->AMBER; else->SKY};p.alpha=(90+165*a).toInt();c.drawCircle(x,y,4f+8f*a,p)};p.alpha=255}
}
