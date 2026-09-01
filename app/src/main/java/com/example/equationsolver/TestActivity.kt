package com.example.equationsolver

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import com.example.equationsolver.ai.ModelStore
import com.example.equationsolver.ui.*
import java.util.Locale

class TestActivity:AppCompatActivity(){
    override fun onCreate(savedInstanceState:Bundle?){super.onCreate(savedInstanceState);val r=screen("اختبار Full INT8","أدخل معاملات النظام بعد التطبيع: a,b,c,d,e,f");val c=r.card();val e=c.edit("المعاملات الستة","1,0,0,0,1,0",false);val out=c.label("النتيجة ستظهر هنا",ACCENT,18f);r.button("تشغيل INT8",true){val v=e.text.toString().split(',').mapNotNull{it.trim().toFloatOrNull()};if(v.size!=6){out.text="يلزم 6 أرقام"}else{runCatching{ModelStore.model.predictInt8(v.toFloatArray())}.onSuccess{out.text="x = %.6f\ny = %.6f".format(Locale.US,it[0],it[1])}.onFailure{out.text=it.message}}};r.button("رجوع"){finish()}}
}
