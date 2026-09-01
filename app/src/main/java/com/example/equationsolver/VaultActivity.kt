package com.example.equationsolver

import android.app.AlertDialog
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.documentfile.provider.DocumentFile
import com.example.equationsolver.ai.ModelStore
import com.example.equationsolver.ui.*
import java.io.FileInputStream
import java.text.SimpleDateFormat
import java.util.*

class VaultActivity:AppCompatActivity(){
    private lateinit var status:android.widget.TextView
    private val folderPicker=registerForActivityResult(ActivityResultContracts.OpenDocumentTree()){uri->if(uri!=null){contentResolver.takePersistableUriPermission(uri,Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_GRANT_WRITE_URI_PERMISSION);ModelStore.setDriveTree(uri.toString());status.text="تم ربط المجلد. إذا اخترته من Google Drive فالحفظ سيكون مباشرة على Drive."}}
    override fun onCreate(b:Bundle?){super.onCreate(b);val r=screen("Model Vault — Google Drive","اختر مجلدًا داخل Google Drive من منتقي الملفات؛ التطبيق يحصل فقط على صلاحية هذا المجلد.");val c=r.card();status=c.label(if(ModelStore.driveTree()!=null)"مجلد Drive مرتبط" else "لم يتم ربط مجلد",ACCENT,16f);r.button("اختيار مجلد Google Drive",true){folderPicker.launch(null)};r.button("نسخ Checkpoint إلى Drive",true){backup()};r.button("استعادة Checkpoint من Drive"){restoreDialog()};r.button("رجوع"){finish()}}
    private fun root():DocumentFile{val u=ModelStore.driveTree()?:error("اربط مجلد Google Drive أولاً");return DocumentFile.fromTreeUri(this,Uri.parse(u))?:error("تعذر فتح المجلد")}
    private fun backup(){runCatching{val src=ModelStore.saveCheckpoint();val name="rsnn_v2_${SimpleDateFormat("yyyyMMdd_HHmmss",Locale.US).format(Date())}.chk";val dst=root().createFile("application/octet-stream",name)?:error("تعذر إنشاء الملف");contentResolver.openOutputStream(dst.uri,"w")!!.use{o->FileInputStream(src).use{it.copyTo(o)}};status.text="تم الحفظ: $name\nSHA-256 ${ModelStore.sha256().take(16)}…"}.onFailure{status.text="فشل الحفظ: ${it.message}"}}
    private fun restoreDialog(){runCatching{root().listFiles().filter{it.isFile}.sortedByDescending{it.lastModified()}}.onSuccess{files->if(files.isEmpty()){toast("لا توجد نماذج");return@onSuccess};val names=files.map{it.name?:"model"}.toTypedArray();AlertDialog.Builder(this).setTitle("اختر نسخة").setItems(names){_,which->runCatching{val msg=ModelStore.importExternal(this,files[which].uri);status.text="تمت الاستعادة: $msg"}.onFailure{status.text="فشلت الاستعادة: ${it.message}"}}.show()}.onFailure{status.text="تعذر قراءة المجلد: ${it.message}"}}
}
