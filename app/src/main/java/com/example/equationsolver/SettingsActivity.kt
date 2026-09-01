package com.example.equationsolver

import android.app.AlertDialog
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import com.example.equationsolver.ai.ModelConfig
import com.example.equationsolver.ai.ModelStore
import com.example.equationsolver.ui.*

class SettingsActivity:AppCompatActivity(){
    override fun onCreate(savedInstanceState:Bundle?){super.onCreate(savedInstanceState);render()}
    private fun render(){val old=ModelStore.config;val r=screen("Model Control — Expert","كل الإعدادات الفعالة للنموذج والتدريب قابلة للتعديل. تغيير hiddenDim ينشئ نموذجاً جديداً.");val c=r.card();
        val fields=linkedMapOf<String,android.widget.EditText>()
        fun f(k:String,l:String,v:Any){fields[k]=c.edit(l,v.toString())}
        f("hiddenDim","Hidden neurons",old.hiddenDim);f("timeSteps","Time steps T",old.timeSteps);f("decay","LIF decay β",old.decay);f("threshold","Spike threshold",old.threshold);f("initialSparsity","Initial sparsity",old.initialSparsity)
        f("learningRate","Learning rate",old.learningRate);f("weightDecay","Weight decay",old.weightDecay);f("adamBeta1","Adam β1",old.adamBeta1);f("adamBeta2","Adam β2",old.adamBeta2);f("adamEps","Adam epsilon",old.adamEps);f("gradientClip","Gradient clip",old.gradientClip)
        f("utilityBeta","Utility EMA β",old.utilityBeta);f("importantFraction","Important fraction",old.importantFraction);f("protectFraction","Rolling protected fraction",old.protectFraction);f("pruneFraction","Prune fraction",old.pruneFraction);f("noveltyLimit","Novelty threshold",old.noveltyLimit);f("stableCycles","Stable cycles",old.stableCycles);f("regrowInitScale","Regrow init scale",old.regrowInitScale);f("structureEveryBatches","Structure every batches",old.structureEveryBatches)
        f("batchSize","Batch size",old.batchSize);f("inputClip","INT8 input clip",old.inputClip);f("membraneClip","INT8 membrane range",old.membraneClip);f("checkpointMinutes","Checkpoint minutes",old.checkpointMinutes);f("minBatteryPercent","Minimum battery %",old.minBatteryPercent);f("coreRefreshMs","Core refresh ms",old.coreRefreshMs)
        r.button("حفظ الإعدادات",true){runCatching{ModelConfig(
            hiddenDim=i(fields,"hiddenDim"),timeSteps=i(fields,"timeSteps"),decay=x(fields,"decay"),threshold=x(fields,"threshold"),initialSparsity=x(fields,"initialSparsity"),learningRate=x(fields,"learningRate"),weightDecay=x(fields,"weightDecay"),adamBeta1=x(fields,"adamBeta1"),adamBeta2=x(fields,"adamBeta2"),adamEps=x(fields,"adamEps"),gradientClip=x(fields,"gradientClip"),utilityBeta=x(fields,"utilityBeta"),importantFraction=x(fields,"importantFraction"),protectFraction=x(fields,"protectFraction"),pruneFraction=x(fields,"pruneFraction"),noveltyLimit=x(fields,"noveltyLimit"),stableCycles=i(fields,"stableCycles"),regrowInitScale=x(fields,"regrowInitScale"),structureEveryBatches=i(fields,"structureEveryBatches"),batchSize=i(fields,"batchSize"),inputClip=x(fields,"inputClip"),membraneClip=x(fields,"membraneClip"),checkpointMinutes=i(fields,"checkpointMinutes"),minBatteryPercent=i(fields,"minBatteryPercent"),coreRefreshMs=i(fields,"coreRefreshMs")
        ).normalized()}.onSuccess{cfg->if(!old.architectureCompatible(cfg)){AlertDialog.Builder(this).setTitle("تغيير البنية").setMessage("تغيير hiddenDim لا يمكنه استخدام الأوزان الحالية. سيتم إنشاء نموذج جديد؛ احفظ الحالي في Model Vault أولاً إذا أردت الاحتفاظ به.").setPositiveButton("إنشاء جديد"){_,_->ModelStore.applyConfig(cfg,true);toast("تم إنشاء نموذج جديد")}.setNegativeButton("إلغاء",null).show()}else{ModelStore.applyConfig(cfg,false);toast("تم تطبيق الإعدادات")}}.onFailure{toast("قيمة غير صالحة: ${it.message}")}}
        r.button("استعادة الإعدادات الافتراضية"){ModelStore.applyConfig(ModelConfig(),true);toast("تمت الاستعادة");recreate()};r.button("رجوع"){finish()}
    }
    private fun i(m:Map<String,android.widget.EditText>,k:String)=m[k]!!.text.toString().toInt()
    private fun x(m:Map<String,android.widget.EditText>,k:String)=m[k]!!.text.toString().toFloat()
}
