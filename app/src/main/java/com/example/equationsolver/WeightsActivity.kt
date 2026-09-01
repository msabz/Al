package com.example.equationsolver

import android.os.Bundle
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import com.example.equationsolver.ai.ModelStore
import com.example.equationsolver.ui.*

class WeightsActivity:AppCompatActivity(){
    private var status:android.widget.TextView?=null
    private val picker=registerForActivityResult(ActivityResultContracts.OpenDocument()){uri->if(uri!=null){runCatching{contentResolver.takePersistableUriPermission(uri,android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION);ModelStore.importExternal(this,uri)}.onSuccess{status?.text="تم التحميل: $it"}.onFailure{status?.text="رفض الملف: ${it.message}"}}}
    override fun onCreate(savedInstanceState:Bundle?){super.onCreate(savedInstanceState);val r=screen("استيراد نموذج أو أوزان","الامتداد غير مهم؛ يتم فحص المحتوى والبنية قبل الاستبدال.");val c=r.card();status=c.label("لم يتم اختيار ملف");r.button("اختيار أي ملف من الهاتف",true){picker.launch(arrayOf("*/*"))};r.button("رجوع"){finish()}}
}
