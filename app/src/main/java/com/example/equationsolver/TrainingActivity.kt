package com.example.equationsolver

import android.Manifest
import android.content.*
import android.os.Build
import android.os.Bundle
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.example.equationsolver.ai.ModelStore
import com.example.equationsolver.data.DeepMindDataset
import com.example.equationsolver.ui.*

class TrainingActivity:AppCompatActivity(){
    private lateinit var status:android.widget.TextView
    private var choosingTrain=true
    private val picker=registerForActivityResult(ActivityResultContracts.OpenDocument()){uri->if(uri!=null){runCatching{contentResolver.takePersistableUriPermission(uri,Intent.FLAG_GRANT_READ_URI_PERMISSION);DeepMindDataset.inspect(contentResolver,uri)}.onSuccess{m->if(choosingTrain)ModelStore.setTrainingUris(uri.toString(),ModelStore.validationUri())else ModelStore.setTrainingUris(ModelStore.trainUri(),uri.toString());status.text="DeepMind OK • split=${m.split} • commit=${m.commit.take(8)}"}.onFailure{status.text="رفض الملف: ${it.message}"}}}
    private val recv=object:BroadcastReceiver(){override fun onReceive(c:Context?,i:Intent?){status.text=i?.getStringExtra(TrainingService.EXTRA_STATUS)?:"تحديث"}}
    override fun onCreate(b:Bundle?){super.onCreate(b);if(Build.VERSION.SDK_INT>=33&&checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS)!=android.content.pm.PackageManager.PERMISSION_GRANTED)requestPermissions(arrayOf(Manifest.permission.POST_NOTIFICATIONS),9);val r=screen("تدريب DeepMind","هذه النسخة تبدأ من checkpoint Open-Growth المدرب مسبقاً، ولا تنشئ نموذجاً عشوائياً جديداً.");val c=r.card();val pretrained="النموذج المدرب محمّل • step %,d • active %,d".format(ModelStore.model.step,ModelStore.model.activeWeights());status=c.label(if(ModelStore.trainUri()!=null)"$pretrained\nملف التدريب محفوظ" else "$pretrained\nاختر ملف DeepMind train",ACCENT,16f);r.button("اختيار DeepMind TRAIN",true){choosingTrain=true;picker.launch(arrayOf("text/*","application/octet-stream","*/*"))};r.button("اختيار DeepMind VALIDATION"){choosingTrain=false;picker.launch(arrayOf("text/*","application/octet-stream","*/*"))};r.button("بدء التدريب المستمر",true){if(ModelStore.trainUri()==null){toast("اختر train أولاً")}else{val i=Intent(this,TrainingService::class.java).setAction(TrainingService.ACTION_START);if(Build.VERSION.SDK_INT>=26)ContextCompat.startForegroundService(this,i)else startService(i);status.text="بدأ استكمال تدريب النموذج المدرب في الخلفية"}};r.button("إيقاف وحفظ"){startService(Intent(this,TrainingService::class.java).setAction(TrainingService.ACTION_STOP));status.text="جارٍ الإيقاف والحفظ"};r.button("تقييم validation"){val v=ModelStore.validationUri();if(v==null)toast("اختر validation")else Thread{val samples=DeepMindDataset.open(contentResolver,android.net.Uri.parse(v)).take(256).toList();val m=ModelStore.model.evaluate(samples);runOnUiThread{status.text="INT8 validation: MAE %.4f • RMSE %.4f • ≤1 %.1f%%".format(m.mae,m.rmse,m.strictWithinOne*100)}}.start()};r.button("رجوع"){finish()}}
    override fun onStart(){super.onStart();ContextCompat.registerReceiver(this,recv,IntentFilter(TrainingService.ACTION_PROGRESS),ContextCompat.RECEIVER_NOT_EXPORTED)}
    override fun onStop(){runCatching{unregisterReceiver(recv)};super.onStop()}
}
