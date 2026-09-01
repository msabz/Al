package com.example.equationsolver.ai

import org.json.JSONObject

data class ModelConfig(
    val hiddenDim: Int = 160, val timeSteps: Int = 25, val decay: Float = 0.88f, val threshold: Float = 1.0f,
    val initialSparsity: Float = 0.30f, val learningRate: Float = 0.003f, val weightDecay: Float = 1e-4f,
    val adamBeta1: Float = 0.9f, val adamBeta2: Float = 0.999f, val adamEps: Float = 1e-8f, val gradientClip: Float = 5f,
    val utilityBeta: Float = 0.95f, val importantFraction: Float = 0.20f, val protectFraction: Float = 0.02f,
    val pruneFraction: Float = 0.02f, val noveltyLimit: Float = 0.01f, val stableCycles: Int = 3,
    val regrowInitScale: Float = 0.01f, val structureEveryBatches: Int = 200, val batchSize: Int = 16,
    val inputClip: Float = 1f, val membraneClip: Float = 8f, val checkpointMinutes: Int = 5,
    val minBatteryPercent: Int = 20, val coreRefreshMs: Int = 350
) {
    fun normalized() = copy(
        hiddenDim=hiddenDim.coerceIn(16,512), timeSteps=timeSteps.coerceIn(1,128), decay=decay.coerceIn(0f,.9999f),
        threshold=threshold.coerceIn(.05f,20f), initialSparsity=initialSparsity.coerceIn(0f,.95f), learningRate=learningRate.coerceIn(1e-6f,.1f),
        weightDecay=weightDecay.coerceIn(0f,.1f), adamBeta1=adamBeta1.coerceIn(0f,.9999f), adamBeta2=adamBeta2.coerceIn(0f,.99999f),
        adamEps=adamEps.coerceIn(1e-12f,1e-2f), gradientClip=gradientClip.coerceIn(.01f,100f), utilityBeta=utilityBeta.coerceIn(0f,.9999f),
        importantFraction=importantFraction.coerceIn(.001f,1f), protectFraction=protectFraction.coerceIn(0f,.5f), pruneFraction=pruneFraction.coerceIn(0f,.5f),
        noveltyLimit=noveltyLimit.coerceIn(.00001f,.5f), stableCycles=stableCycles.coerceIn(1,100), regrowInitScale=regrowInitScale.coerceIn(1e-5f,1f),
        structureEveryBatches=structureEveryBatches.coerceIn(1,100000), batchSize=batchSize.coerceIn(1,256), inputClip=inputClip.coerceIn(.01f,100f),
        membraneClip=membraneClip.coerceIn(.1f,100f), checkpointMinutes=checkpointMinutes.coerceIn(1,1440), minBatteryPercent=minBatteryPercent.coerceIn(1,90),
        coreRefreshMs=coreRefreshMs.coerceIn(100,5000)
    )
    fun architectureCompatible(other: ModelConfig)=hiddenDim==other.hiddenDim
    fun toJson()=JSONObject().apply{
        put("hiddenDim",hiddenDim);put("timeSteps",timeSteps);put("decay",decay.toDouble());put("threshold",threshold.toDouble());put("initialSparsity",initialSparsity.toDouble());put("learningRate",learningRate.toDouble());put("weightDecay",weightDecay.toDouble());put("adamBeta1",adamBeta1.toDouble());put("adamBeta2",adamBeta2.toDouble());put("adamEps",adamEps.toDouble());put("gradientClip",gradientClip.toDouble());put("utilityBeta",utilityBeta.toDouble());put("importantFraction",importantFraction.toDouble());put("protectFraction",protectFraction.toDouble());put("pruneFraction",pruneFraction.toDouble());put("noveltyLimit",noveltyLimit.toDouble());put("stableCycles",stableCycles);put("regrowInitScale",regrowInitScale.toDouble());put("structureEveryBatches",structureEveryBatches);put("batchSize",batchSize);put("inputClip",inputClip.toDouble());put("membraneClip",membraneClip.toDouble());put("checkpointMinutes",checkpointMinutes);put("minBatteryPercent",minBatteryPercent);put("coreRefreshMs",coreRefreshMs)
    }.toString()
    companion object { fun fromJson(text:String):ModelConfig { val j=JSONObject(text); return ModelConfig(
        j.optInt("hiddenDim",160),j.optInt("timeSteps",25),j.optDouble("decay",.88).toFloat(),j.optDouble("threshold",1.0).toFloat(),j.optDouble("initialSparsity",.30).toFloat(),j.optDouble("learningRate",.003).toFloat(),j.optDouble("weightDecay",1e-4).toFloat(),j.optDouble("adamBeta1",.9).toFloat(),j.optDouble("adamBeta2",.999).toFloat(),j.optDouble("adamEps",1e-8).toFloat(),j.optDouble("gradientClip",5.0).toFloat(),j.optDouble("utilityBeta",.95).toFloat(),j.optDouble("importantFraction",.20).toFloat(),j.optDouble("protectFraction",.02).toFloat(),j.optDouble("pruneFraction",.02).toFloat(),j.optDouble("noveltyLimit",.01).toFloat(),j.optInt("stableCycles",3),j.optDouble("regrowInitScale",.01).toFloat(),j.optInt("structureEveryBatches",200),j.optInt("batchSize",16),j.optDouble("inputClip",1.0).toFloat(),j.optDouble("membraneClip",8.0).toFloat(),j.optInt("checkpointMinutes",5),j.optInt("minBatteryPercent",20),j.optInt("coreRefreshMs",350)
    ).normalized() } }
}
