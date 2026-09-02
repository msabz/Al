package com.example.equationsolver

import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.example.equationsolver.ai.ModelManager
import com.example.equationsolver.core.ArabicEquationNormalizer
import com.example.equationsolver.core.MathTeacher
import java.util.Locale
import kotlin.math.abs

class TestActivity : AppCompatActivity() {
    @Volatile private var requestId = 0

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_test)
        ModelManager.init(applicationContext)
        val input=findViewById<EditText>(R.id.editEquation)
        val typeText=findViewById<TextView>(R.id.textType)
        val exactText=findViewById<TextView>(R.id.textResult)
        val stepsText=findViewById<TextView>(R.id.textSteps)
        val aiText=findViewById<TextView>(R.id.textAi)
        val solveButton=findViewById<Button>(R.id.btnSolve)
        findViewById<Button>(R.id.btnBackTest).setOnClickListener{finish()}
        findViewById<Button>(R.id.btnExampleLinear).setOnClickListener{input.setText("2x+y=7;x-y=2")}
        findViewById<Button>(R.id.btnExamplePolynomial).setOnClickListener{input.setText("x+y=5;x-y=1")}
        findViewById<Button>(R.id.btnExampleFunction).setOnClickListener{input.setText("x+2y=8;3x-y=3")}
        findViewById<Button>(R.id.btnExampleSystem).setOnClickListener{input.setText("4x+3y=18;x+y=5")}
        findViewById<TextView>(R.id.textTestTrainingState).text = if(ModelManager.isTrainingEnabled(this)) "● التدريب مستمر بالخلفية على نفس الـCheckpoint المدرب." else "Open-Growth RSNN محمّل من التدريب السابق؛ الاختبار لا يبدأ نموذجًا جديدًا."
        findViewById<Button>(R.id.btnClear).setOnClickListener{ requestId++;input.text.clear();typeText.text="";exactText.text="—";stepsText.text="";aiText.text="لم يُجرَ اختبار بعد";solveButton.isEnabled=true;input.requestFocus() }
        solveButton.setOnClickListener{
            val raw=input.text.toString().trim(); if(raw.isEmpty()){input.error="أدخل معادلتين";Toast.makeText(this,"مثال: 2x+y=7;x-y=2",Toast.LENGTH_SHORT).show();return@setOnClickListener}
            val equation=ArabicEquationNormalizer.normalize(raw); val id=++requestId; solveButton.isEnabled=false; aiText.text="النموذج يحسب..."
            Thread{
                try{
                    val prediction=ModelManager.predictValues(equation); val answer=MathTeacher.solve(equation)
                    runOnUiThread{if(id!=requestId||isFinishing||isDestroyed)return@runOnUiThread;renderResult(prediction,answer,typeText,exactText,stepsText,aiText);solveButton.isEnabled=true}
                }catch(e:Exception){runOnUiThread{if(id!=requestId)return@runOnUiThread;typeText.text="هذا النموذج مخصص لنظام خطي 2×2";exactText.text=e.message?:"خطأ";stepsText.text="أدخل مثل: 2x+y=7;x-y=2";aiText.text="لم يكتمل الاختبار";solveButton.isEnabled=true}}
            }.start()
        }
    }
    override fun onDestroy(){requestId++;super.onDestroy()}
    private fun renderResult(prediction:DoubleArray,answer:MathTeacher.Answer,typeText:TextView,exactText:TextView,stepsText:TextView,aiText:TextView){
        val px=prediction.getOrElse(0){0.0};val py=prediction.getOrElse(1){0.0};val info=ModelManager.modelInfo(this)
        typeText.text="Open-Growth RSNN • Adam step %,d".format(Locale.US,info.optimizerStep)
        exactText.text=answer.summary;stepsText.text=if(answer.steps.isEmpty())"" else answer.steps.mapIndexed{i,s->"${i+1}. $s"}.joinToString("\n")
        aiText.text=if(answer.x!=null&&answer.y!=null)"x ≈ ${fmt(px)}\ny ≈ ${fmt(py)}\n|خطأ x| = ${fmt(abs(px-answer.x))} • |خطأ y| = ${fmt(abs(py-answer.y))}" else "x ≈ ${fmt(px)}\ny ≈ ${fmt(py)}"
    }
    private fun fmt(v:Double)=if(abs(v)<1e-9)"0" else String.format(Locale.US,"%.6f",v).trimEnd('0').trimEnd('.')
}
