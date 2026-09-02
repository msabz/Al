package com.example.equationsolver.ai

import android.content.Context
import android.net.Uri
import java.io.*
import java.security.MessageDigest

object ModelStore {
    private const val PREFS = "rsnn_v2"
    private const val KEY_CONFIG = "config"
    private const val MODEL_MAGIC = 0x4D4F4456
    private const val WEIGHTS_MAGIC = 0x57544733
    private const val WRAPPER_VERSION = 1
    // This build is intentionally pinned to the already-trained stage-1 model.
    // A new filename prevents an older installed random checkpoint from taking precedence.
    private const val FILE_NAME = "open_growth_stage1_pretrained_resume.chk"
    private const val BUNDLED_ASSET = "pretrained_open_growth_stage1.chk"

    @Volatile lateinit var model: OpenGrowthRsnnV2; private set
    @Volatile var config: ModelConfig = ModelConfig(); private set
    private lateinit var app: Context

    @Synchronized fun init(context: Context) {
        if (::app.isInitialized) return
        app = context.applicationContext
        val checkpoint = File(app.filesDir, FILE_NAME)
        if (!checkpoint.exists()) installBundledCheckpoint(checkpoint)
        // Important: never silently create a fresh random model in this build.
        model = loadCheckpoint(checkpoint)
        persist()
    }

    private fun installBundledCheckpoint(file: File) {
        val tmp = File(file.parentFile, file.name + ".tmp")
        app.assets.open(BUNDLED_ASSET).use { src ->
            BufferedOutputStream(FileOutputStream(tmp)).use { dst -> src.copyTo(dst) }
        }
        if (file.exists()) file.delete()
        require(tmp.renameTo(file)) { "تعذر تثبيت النموذج المدرب داخل التطبيق" }
    }

    @Synchronized fun applyConfig(newConfig: ModelConfig, rebuildIfNeeded: Boolean) {
        val c = newConfig.normalized()
        if (!config.architectureCompatible(c)) {
            require(rebuildIfNeeded) { "تغيير hiddenDim يتطلب نموذجًا جديدًا" }
            config = c; model = OpenGrowthRsnnV2(c)
        } else { config = c; model.updateRuntimeConfig(c) }
        app.getSharedPreferences(PREFS,0).edit().putString(KEY_CONFIG,c.toJson()).apply(); saveCheckpoint()
    }

    @Synchronized fun saveCheckpoint(file: File = File(app.filesDir, FILE_NAME)): File {
        val tmp = File(file.parentFile, file.name+".tmp")
        DataOutputStream(BufferedOutputStream(FileOutputStream(tmp))).use { out -> out.writeInt(MODEL_MAGIC);out.writeInt(WRAPPER_VERSION);out.writeUTF(config.toJson());model.save(out) }
        if(file.exists())file.delete();require(tmp.renameTo(file)){"تعذر تثبيت checkpoint"};return file
    }

    @Synchronized fun exportWeights(file: File): File {
        DataOutputStream(BufferedOutputStream(FileOutputStream(file))).use{out->out.writeInt(WEIGHTS_MAGIC);out.writeInt(WRAPPER_VERSION);out.writeUTF(config.toJson());model.exportWeights(out)};return file
    }

    @Synchronized fun importExternal(context: Context, uri: Uri): String {
        val resolver=context.contentResolver
        resolver.openInputStream(uri)?.use { raw -> DataInputStream(BufferedInputStream(raw)).use { input ->
            val magic=input.readInt();val version=input.readInt();require(version==WRAPPER_VERSION){"نسخة الملف غير مدعومة"};val incoming=ModelConfig.fromJson(input.readUTF())
            when(magic){
                MODEL_MAGIC->{val loaded=OpenGrowthRsnnV2.load(input,incoming);config=incoming;model=loaded;persist();saveCheckpoint();return "Full checkpoint • ${incoming.hiddenDim} hidden • ${model.activeWeights()} active"}
                WEIGHTS_MAGIC->{if(!config.architectureCompatible(incoming)){config=incoming;model=OpenGrowthRsnnV2(incoming)}else{config=incoming;model.updateRuntimeConfig(incoming)};model.importWeights(input);persist();saveCheckpoint();return "Weights only • ${incoming.hiddenDim} hidden • ${model.activeWeights()} active"}
                else->error("المحتوى ليس ملف أوزان/Checkpoint متوافق، بغض النظر عن الامتداد")
            }
        }} ?: error("تعذر فتح ملف الأوزان")
    }

    private fun persist(){app.getSharedPreferences(PREFS,0).edit().putString(KEY_CONFIG,config.toJson()).apply()}
    private fun loadCheckpoint(file: File): OpenGrowthRsnnV2 {
        DataInputStream(BufferedInputStream(FileInputStream(file))).use{input->require(input.readInt()==MODEL_MAGIC);require(input.readInt()==WRAPPER_VERSION);val c=ModelConfig.fromJson(input.readUTF());config=c;return OpenGrowthRsnnV2.load(input,c)}
    }
    fun checkpointFile(): File = File(app.filesDir, FILE_NAME)
    fun sha256(file: File = checkpointFile()): String { val d=MessageDigest.getInstance("SHA-256");FileInputStream(file).use{f->val b=ByteArray(8192);while(true){val n=f.read(b);if(n<=0)break;d.update(b,0,n)}};return d.digest().joinToString(""){"%02x".format(it)}}
    fun setTrainingUris(train: String?, validation: String?){app.getSharedPreferences(PREFS,0).edit().putString("train_uri",train).putString("val_uri",validation).apply()}
    fun trainUri(): String?=app.getSharedPreferences(PREFS,0).getString("train_uri",null)
    fun validationUri(): String?=app.getSharedPreferences(PREFS,0).getString("val_uri",null)
    fun setDriveTree(uri:String?){app.getSharedPreferences(PREFS,0).edit().putString("drive_tree",uri).apply()}
    fun driveTree():String?=app.getSharedPreferences(PREFS,0).getString("drive_tree",null)
    fun setTrainingEnabled(v:Boolean){app.getSharedPreferences(PREFS,0).edit().putBoolean("training_enabled",v).apply()}
    fun trainingEnabled():Boolean=app.getSharedPreferences(PREFS,0).getBoolean("training_enabled",false)
}
