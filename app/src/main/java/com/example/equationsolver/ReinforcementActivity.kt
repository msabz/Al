package com.example.equationsolver

import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.example.equationsolver.ai.ModelManager
import com.example.equationsolver.ai.TrainingEngine
import com.example.equationsolver.core.ArabicEquationNormalizer
import com.example.equationsolver.core.MathTeacher
import java.util.Locale
import kotlin.math.abs

class ReinforcementActivity : AppCompatActivity() {
    private var currentEquation:String?=null; private var currentAnswer:MathTeacher.Answer?=null; private var operationId=0; private var busy=false
    private lateinit var edit:EditText;private lateinit var resultText:TextView;private lateinit var correctionText:TextView;private lateinit var feedback:LinearLayout
    private lateinit var evaluateButton:Button;private lateinit var nextButton:Button;private lateinit var rewardButton:Button;private lateinit var correctionButton:Button
    override fun onCreate(savedInstanceState:Bundle?){super.onCreate(savedInstanceState);setContentView(R.layout.activity_reinforcement);ModelManager.init(applicationContext);bindViews();findViewById<Button>(R.id.btnBackReinforcement).setOnClickListener{finish()};evaluateButton.setOnClickListener{val raw=edit.text.toString().trim();if(raw.isEmpty()){edit.error="أدخل نظامًا خطيًا"}else evaluateEquation(ArabicEquationNormalizer.normalize(raw))};nextButton.setOnClickListener{suggestNextEquation()};rewardButton.setOnClickListener{reinforceCurrent(2,1e-5,"تعزيز")};correctionButton.setOnClickListener{reinforceCurrent(8,2e-5,"تصحيح")};suggestNextEquation()}
    private fun bindViews(){edit=findViewById(R.id.editInteractiveEq);resultText=findViewById(R.id.textInteractiveResult);correctionText=findViewById(R.id.textCorrection);feedback=findViewById(R.id.layoutFeedback);evaluateButton=findViewById(R.id.btnSolveInteractive);nextButton=findViewById(R.id.btnNextEquation);rewardButton=findViewById(R.id.btnReward);correctionButton=findViewById(R.id.btnPunish)}
    private fun suggestNextEquation(){if(busy)return;val s=TrainingEngine.suggestEquation();edit.setText(s.equation);edit.setSelection(edit.text.length);correctionText.text="مثال نظام خطي جديد";evaluateEquation(s.equation)}
    private fun evaluateEquation(eq:String){if(busy)return;currentEquation=eq;currentAnswer=null;val id=++operationId;setBusy(true);Thread{try{val p=ModelManager.predictValues(eq);val a=MathTeacher.solve(eq);runOnUiThread{if(id!=operationId)return@runOnUiThread;currentAnswer=a;resultText.text=evaluationText(p,a);feedback.visibility=if(a.x!=null&&a.y!=null)View.VISIBLE else View.GONE;setBusy(false)}}catch(e:Exception){runOnUiThread{resultText.text="فشل: ${e.message}";feedback.visibility=View.GONE;setBusy(false)}}}.start()}
    private fun reinforceCurrent(repeats:Int,learningRate:Double,label:String){if(busy)return;val eq=currentEquation?:return;val a=currentAnswer?:return;if(a.x==null||a.y==null)return;val id=++operationId;setBusy(true);Thread{try{val d=ModelManager.trainWithTarget(eq,a.x,a.y,repeats,learningRate);ModelManager.save(this);runOnUiThread{if(id!=operationId)return@runOnUiThread;correctionText.text="$label اكتمل وحُفظ من نفس الـCheckpoint.";resultText.text=evaluationText(d.after,a);setBusy(false);Toast.makeText(this,"تم تحديث نفس النموذج",Toast.LENGTH_SHORT).show()}}catch(e:Exception){runOnUiThread{correctionText.text="فشل: ${e.message}";setBusy(false)}}}.start()}
    private fun evaluationText(p:DoubleArray,a:MathTeacher.Answer):String{val x=p.getOrElse(0){0.0};val y=p.getOrElse(1){0.0};return if(a.x!=null&&a.y!=null)"النموذج: x≈${fmt(x)}، y≈${fmt(y)}\nالمرجع: x=${fmt(a.x)}، y=${fmt(a.y)}\nمتوسط الخطأ=${fmt((abs(x-a.x)+abs(y-a.y))/2)}" else "x≈${fmt(x)}، y≈${fmt(y)}"}
    private fun setBusy(v:Boolean){busy=v;evaluateButton.isEnabled=!v;nextButton.isEnabled=!v;rewardButton.isEnabled=!v;correctionButton.isEnabled=!v;edit.isEnabled=!v}
    private fun fmt(v:Double)=if(!v.isFinite())"—" else String.format(Locale.US,"%.6f",v).trimEnd('0').trimEnd('.')
}
