package com.example.equationsolver.ai

import com.example.equationsolver.data.DeepMindSample
import java.io.DataInputStream
import java.io.DataOutputStream
import java.util.Random
import kotlin.math.*

class OpenGrowthRsnnV2(var config: ModelConfig, seed: Long = 11L, initialize: Boolean = true) {
    companion object {
        const val IN_DIM = 6
        const val OUT_DIM = 2
        const val TARGET_SCALE = 100f
        const val MAGIC = 0x52534E33
        const val VERSION = 3
        private const val FIXED_SHIFT = 20
        fun load(input: DataInputStream, config: ModelConfig): OpenGrowthRsnnV2 {
            require(input.readInt() == MAGIC) { "RSNN checkpoint غير متوافق" }
            require(input.readInt() == VERSION) { "نسخة checkpoint غير مدعومة" }
            val m = OpenGrowthRsnnV2(config, initialize = false)
            m.step = input.readLong().coerceAtLeast(0L)
            m.phase = input.readUTF(); m.structuralCycle = input.readInt(); m.growthStreak = input.readInt()
            m.selectionStreak = input.readInt(); m.selectionCycles = input.readInt(); m.topologyStable = input.readBoolean()
            m.readState(input, m.inState); m.readState(input, m.recState); m.readState(input, m.outState)
            input.readFully(m.prevImportant); m.invalidateRuntime(); return m
        }
    }
    data class TrainMetrics(val loss: Float, val gradientNorm: Float, val activeWeights: Int, val step: Long)
    data class EvalMetrics(val mae: Float, val rmse: Float, val strictWithinOne: Float, val count: Int)
    data class StructuralEvent(val cycle: Int, val phase: String, val grown: Int, val pruned: Int, val active: Int, val protected: Int, val novelty: Float, val stable: Boolean)
    data class CoreSnapshot(
        val membrane: FloatArray, val spikes: ByteArray, val firingRate: Float,
        val active: Int, val dormant: Int, val protected: Int, val phase: String,
        val structuralCycle: Int, val lastGrown: Int, val lastPruned: Int,
        val weightMeanAbs: Float, val weightMaxAbs: Float, val int8Saturation: Float,
        val gradientNorm: Float, val step: Long
    )
    private data class MatrixState(val w: FloatArray, val mask: ByteArray, val m: FloatArray, val v: FloatArray, val utility: FloatArray, val appearance: IntArray, val protected: ByteArray, val cooldown: ByteArray)
    private data class Effective(val wi: FloatArray, val wr: FloatArray, val wo: FloatArray)
    private data class Grad(val gi: FloatArray, val gr: FloatArray, val go: FloatArray)
    private data class Quant(val q: ByteArray, val scale: Float, val sat: Int)
    private data class QuantCache(val wi: Quant, val wr: Quant, val wo: Quant)
    private val h get() = config.hiddenDim
    private val totalWeights get() = IN_DIM * h + h * h + OUT_DIM * h
    private val rng = Random(seed)
    private val inState = state(h * IN_DIM)
    private val recState = state(h * h)
    private val outState = state(OUT_DIM * h)
    private val prevImportant = ByteArray(totalWeights)
    @Volatile private var lastCore = CoreSnapshot(FloatArray(h), ByteArray(h), 0f, 0, totalWeights, 0, "growth", 0, 0, 0, 0f, 0f, 0f, 0f, 0)
    @Volatile private var lastEvent = StructuralEvent(0, "growth", 0, 0, 0, 0, 1f, false)
    @Volatile var lastGradientNorm: Float = 0f; private set
    var phase = "growth"; private set
    var structuralCycle = 0; private set
    var growthStreak = 0; private set
    var selectionStreak = 0; private set
    var selectionCycles = 0; private set
    var topologyStable = false; private set
    var step = 0L; private set
    @Transient private var qCache: QuantCache? = null
    init {
        config = config.normalized()
        if (initialize) {
            initWeights(inState, 1f / sqrt(IN_DIM.toFloat())); initWeights(recState, 1f / sqrt(h.toFloat())); initWeights(outState, 1f / sqrt(h.toFloat()))
            pruneInitial(inState); pruneInitial(recState); pruneInitial(outState); updateCore(FloatArray(h), ByteArray(h), 0f)
        }
    }
    private fun state(n: Int) = MatrixState(FloatArray(n), ByteArray(n) { 1 }, FloatArray(n), FloatArray(n), FloatArray(n), IntArray(n), ByteArray(n), ByteArray(n))
    private fun initWeights(s: MatrixState, std: Float) { for (i in s.w.indices) s.w[i] = (rng.nextGaussian() * std).toFloat() }
    private fun pruneInitial(s: MatrixState) {
        val count = (s.w.size * config.initialSparsity).roundToInt().coerceIn(0, s.w.size)
        val ids = s.w.indices.sortedBy { abs(s.w[it]) }
        for (n in 0 until count) { val i = ids[n]; s.mask[i] = 0; s.w[i] = 0f }
    }
    @Synchronized fun activeWeights(): Int = active(inState) + active(recState) + active(outState)
    @Synchronized fun protectedWeights(): Int = protected(inState) + protected(recState) + protected(outState)
    fun coreSnapshot(): CoreSnapshot = lastCore.copy(membrane = lastCore.membrane.copyOf(), spikes = lastCore.spikes.copyOf())
    fun lastStructuralEvent(): StructuralEvent = lastEvent
    private fun active(s: MatrixState) = s.mask.count { it.toInt() != 0 }
    private fun protected(s: MatrixState) = s.protected.count { it.toInt() != 0 }
    @Synchronized fun predictInt8(features: FloatArray): FloatArray {
        require(features.size == IN_DIM)
        val q = qCache ?: buildQuantCache().also { qCache = it }
        val input = ByteArray(IN_DIM) { i -> (features[i].coerceIn(-config.inputClip, config.inputClip) / config.inputClip * 127f).roundToInt().coerceIn(-127, 127).toByte() }
        val mem = ByteArray(h); val spikes = ByteArray(h); val next = ByteArray(h); val syn = IntArray(h); val out = IntArray(OUT_DIM)
        val memScale = config.membraneClip / 127f
        val inMul = fixed((config.inputClip / 127f) * q.wi.scale / memScale)
        val recMul = fixed(q.wr.scale / memScale); val decayMul = fixed(config.decay); val thresholdQ = (config.threshold / memScale).roundToInt()
        for (j in 0 until h) { var a = 0; val b = j * IN_DIM; for (i in 0 until IN_DIM) a += q.wi.q[b+i].toInt() * input[i].toInt(); syn[j] = a }
        var spikeCount = 0
        repeat(config.timeSteps) {
            for (j in 0 until h) {
                var r = 0; val b = j*h; for (k in 0 until h) r += q.wr.q[b+k].toInt() * spikes[k].toInt()
                val u = ((mem[j].toInt().toLong()*decayMul) shr FIXED_SHIFT).toInt() + ((syn[j].toLong()*inMul) shr FIXED_SHIFT).toInt() + ((r.toLong()*recMul) shr FIXED_SHIFT).toInt()
                val s = if (u >= thresholdQ) 1 else 0; next[j] = s.toByte(); spikeCount += s; mem[j] = (u - s*thresholdQ).coerceIn(-127,127).toByte()
            }
            for (o in 0 until OUT_DIM) { var a=0; val b=o*h; for(j in 0 until h) a += q.wo.q[b+j].toInt()*next[j].toInt(); out[o]+=a }
            System.arraycopy(next,0,spikes,0,h)
        }
        val factor = q.wo.scale/config.timeSteps*TARGET_SCALE
        val membrane = FloatArray(h) { mem[it].toInt()*memScale }
        val rate = spikeCount.toFloat()/(h*config.timeSteps).coerceAtLeast(1)
        updateCore(membrane, spikes.copyOf(), rate)
        return floatArrayOf(out[0]*factor, out[1]*factor)
    }
    @Synchronized fun trainBatch(samples: List<DeepMindSample>): TrainMetrics {
        require(samples.isNotEmpty())
        val ew = effective(); val g = Grad(FloatArray(h*IN_DIM), FloatArray(h*h), FloatArray(OUT_DIM*h)); var loss=0f
        for (s in samples) loss += gradientOne(s, ew, g)
        val inv=1f/samples.size; scale(g,inv); maskGrad(g)
        val norm=norm(g); lastGradientNorm=norm; if(norm>config.gradientClip) scale(g, config.gradientClip/norm)
        utility(inState,g.gi); utility(recState,g.gr); utility(outState,g.go); step++
        adam(inState,g.gi); adam(recState,g.gr); adam(outState,g.go); invalidateRuntime()
        val c=lastCore; lastCore=c.copy(gradientNorm=norm,step=step,active=activeWeights(),dormant=totalWeights-activeWeights(),protected=protectedWeights())
        return TrainMetrics(loss*inv,norm,activeWeights(),step)
    }
    private fun gradientOne(sample: DeepMindSample, ew: Effective, g: Grad): Float {
        val u=Array(config.timeSteps){FloatArray(h)}; val spikes=Array(config.timeSteps){FloatArray(h)}
        val input=FloatArray(IN_DIM){ fqInput(sample.features[it]) }; val syn=FloatArray(h)
        for(j in 0 until h){ var s=0f; val b=j*IN_DIM; for(i in 0 until IN_DIM)s+=ew.wi[b+i]*input[i]; syn[j]=s }
        var mem=FloatArray(h); var prev=FloatArray(h); val out=FloatArray(OUT_DIM)
        for(t in 0 until config.timeSteps){ val nm=FloatArray(h); for(j in 0 until h){ var r=0f; val b=j*h; for(k in 0 until h)r+=ew.wr[b+k]*prev[k]; val pre=config.decay*mem[j]+syn[j]+r; u[t][j]=pre; val sp=if(pre>=config.threshold)1f else 0f; spikes[t][j]=sp; nm[j]=fqMem(pre-sp*config.threshold) }
            for(o in 0 until OUT_DIM){ val b=o*h; for(j in 0 until h)out[o]+=ew.wo[b+j]*spikes[t][j] }; mem=nm; prev=spikes[t] }
        val go=FloatArray(OUT_DIM); var loss=0f
        for(o in 0 until OUT_DIM){ val e=out[o]/config.timeSteps-sample.targets[o]/TARGET_SCALE; val a=abs(e); loss+=if(a<1f).5f*e*e else a-.5f; go[o]=(if(a<1f)e else sign(e))/OUT_DIM/config.timeSteps }
        loss/=OUT_DIM
        for(t in 0 until config.timeSteps)for(o in 0 until OUT_DIM){val b=o*h;for(j in 0 until h)g.go[b+j]+=go[o]*spikes[t][j]}
        var nextU=FloatArray(h)
        for(t in config.timeSteps-1 downTo 0){ val gu=FloatArray(h)
            for(k in 0 until h){ var direct=0f; for(o in 0 until OUT_DIM)direct+=ew.wo[o*h+k]*go[o]; for(j in 0 until h)direct+=ew.wr[j*h+k]*nextU[j]
                val z=u[t][k]-config.threshold; val sur=1f/((1f+2f*abs(z))*(1f+2f*abs(z))); gu[k]=config.decay*nextU[k]*(1f-config.threshold*sur)+direct*sur
                val ib=k*IN_DIM; for(i in 0 until IN_DIM)g.gi[ib+i]+=gu[k]*input[i]
                if(t>0){ val rb=k*h; for(j in 0 until h)g.gr[rb+j]+=gu[k]*spikes[t-1][j] }
            }; nextU=gu }
        updateCore(mem, ByteArray(h){ if(prev[it]>0f)1 else 0 }, prev.sum()/h.coerceAtLeast(1)); return loss
    }
    @Synchronized fun structuralStep(bank: List<DeepMindSample>): StructuralEvent {
        if(bank.isEmpty()||topologyStable) return lastEvent
        structuralCycle++
        val shadow=averageGrad(bank,true); val grown=grow(shadow)
        if(phase=="growth"){
            val novelty=grown.toFloat()/totalWeights; growthStreak=if(novelty<config.noveltyLimit)growthStreak+1 else 0; if(growthStreak>=config.stableCycles)phase="selection"
            lastEvent=StructuralEvent(structuralCycle,phase,grown,0,activeWeights(),protectedWeights(),novelty,false); refreshEvent(); return lastEvent
        }
        selectionCycles++
        val ng=averageGrad(bank,false); val ci=contribution(inState,ng.gi); val cr=contribution(recState,ng.gr); val co=contribution(outState,ng.go)
        appearance(inState,ci); appearance(recState,cr); appearance(outState,co)
        val scores=FloatArray(totalWeights); combined(inState,ci,scores,0); combined(recState,cr,scores,inState.w.size); combined(outState,co,scores,inState.w.size+recState.w.size)
        val active=globalActive(); val k=max(1,(active.size*config.importantFraction).roundToInt()); val important=active.sortedByDescending{scores[it]}.take(k)
        var fresh=0; for(id in important) if(prevImportant[id].toInt()==0)fresh++; val novelty=fresh.toFloat()/k; prevImportant.fill(0); for(id in important)prevImportant[id]=1
        selectionStreak=if(novelty<config.noveltyLimit)selectionStreak+1 else 0
        if(selectionStreak>=config.stableCycles){phase="final";topologyStable=true;lastEvent=StructuralEvent(structuralCycle,phase,grown,0,activeWeights(),protectedWeights(),novelty,true);refreshEvent();return lastEvent}
        clearProtected(); val np=max(1,(active.size*config.protectFraction).roundToInt()).coerceAtMost(active.size); for(id in active.sortedByDescending{scores[it]}.take(np))setProtected(id,true)
        val removable=active.filter{!isProtected(it)}; val nr=max(1,(active.size*config.pruneFraction).roundToInt()).coerceAtMost(removable.size); val bottom=removable.sortedBy{scores[it]}.take(nr); for(id in bottom)prune(id)
        invalidateRuntime(); lastEvent=StructuralEvent(structuralCycle,phase,grown,bottom.size,activeWeights(),protectedWeights(),novelty,false);refreshEvent();return lastEvent
    }
    @Synchronized fun evaluate(samples: List<DeepMindSample>): EvalMetrics {
        if(samples.isEmpty())return EvalMetrics(Float.NaN,Float.NaN,Float.NaN,0);var a=0.0;var sq=0.0;var strict=0
        for(s in samples){val p=predictInt8(s.features);var mx=0f;for(o in 0 until OUT_DIM){val e=abs(p[o]-s.targets[o]);a+=e;sq+=e*e;mx=max(mx,e)};if(mx<=1f)strict++}
        val n=samples.size*OUT_DIM;return EvalMetrics((a/n).toFloat(),sqrt(sq/n).toFloat(),strict.toFloat()/samples.size,samples.size)
    }
    fun updateRuntimeConfig(newConfig: ModelConfig) { require(config.architectureCompatible(newConfig)); config=newConfig.normalized(); invalidateRuntime() }
    @Synchronized fun save(out: DataOutputStream) {
        out.writeInt(MAGIC);out.writeInt(VERSION);out.writeLong(step);out.writeUTF(phase);out.writeInt(structuralCycle);out.writeInt(growthStreak);out.writeInt(selectionStreak);out.writeInt(selectionCycles);out.writeBoolean(topologyStable)
        writeState(out,inState);writeState(out,recState);writeState(out,outState);out.write(prevImportant)
    }
    @Synchronized fun exportWeights(out: DataOutputStream) { out.writeInt(totalWeights); for(s in arrayOf(inState,recState,outState)){out.writeInt(s.w.size);for(i in s.w.indices)out.writeFloat(if(s.mask[i].toInt()!=0)s.w[i] else 0f)} }
    @Synchronized fun importWeights(input: DataInputStream) {
        require(input.readInt()==totalWeights){"عدد الأوزان لا يطابق البنية الحالية"};for(s in arrayOf(inState,recState,outState)){require(input.readInt()==s.w.size);for(i in s.w.indices){val v=input.readFloat();s.w[i]=v;s.mask[i]=if(v==0f)0 else 1;s.m[i]=0f;s.v[i]=0f;s.utility[i]=0f;s.appearance[i]=0;s.protected[i]=0;s.cooldown[i]=0}}
        step=0;phase="growth";topologyStable=false;structuralCycle=0;growthStreak=0;selectionStreak=0;selectionCycles=0;prevImportant.fill(0);invalidateRuntime();updateCore(FloatArray(h),ByteArray(h),0f)
    }
    private fun effective()=Effective(fqWeights(inState),fqWeights(recState),fqWeights(outState))
    private fun fqWeights(s:MatrixState):FloatArray{val sc=maxAbs(s).coerceAtLeast(1e-8f)/127f;return FloatArray(s.w.size){i->if(s.mask[i].toInt()==0)0f else (s.w[i]/sc).roundToInt().coerceIn(-127,127)*sc}}
    private fun maxAbs(s:MatrixState):Float{var m=0f;for(i in s.w.indices)if(s.mask[i].toInt()!=0)m=max(m,abs(s.w[i]));return m}
    private fun fqInput(v:Float)=((v.coerceIn(-config.inputClip,config.inputClip)/config.inputClip*127f).roundToInt().coerceIn(-127,127)/127f)*config.inputClip
    private fun fqMem(v:Float):Float{val sc=config.membraneClip/127f;return (v/sc).roundToInt().coerceIn(-127,127)*sc}
    private fun scale(g:Grad,f:Float){for(i in g.gi.indices)g.gi[i]*=f;for(i in g.gr.indices)g.gr[i]*=f;for(i in g.go.indices)g.go[i]*=f}
    private fun maskGrad(g:Grad){mask(inState,g.gi);mask(recState,g.gr);mask(outState,g.go)}
    private fun mask(s:MatrixState,a:FloatArray){for(i in a.indices)if(s.mask[i].toInt()==0)a[i]=0f}
    private fun norm(g:Grad):Float{var s=0.0;for(v in g.gi)s+=v*v;for(v in g.gr)s+=v*v;for(v in g.go)s+=v*v;return sqrt(s).toFloat()}
    private fun utility(s:MatrixState,g:FloatArray){for(i in g.indices)s.utility[i]=if(s.mask[i].toInt()!=0)config.utilityBeta*s.utility[i]+(1-config.utilityBeta)*abs(s.w[i]*g[i])else 0f}
    private fun adam(s:MatrixState,g:FloatArray){val c1=1.0-config.adamBeta1.toDouble().pow(step.toDouble());val c2=1.0-config.adamBeta2.toDouble().pow(step.toDouble());for(i in s.w.indices){if(s.mask[i].toInt()==0){s.w[i]=0f;s.m[i]=0f;s.v[i]=0f;continue};s.m[i]=config.adamBeta1*s.m[i]+(1-config.adamBeta1)*g[i];s.v[i]=config.adamBeta2*s.v[i]+(1-config.adamBeta2)*g[i]*g[i];val mh=(s.m[i]/c1).toFloat();val vh=(s.v[i]/c2).toFloat();s.w[i]-=config.learningRate*config.weightDecay*s.w[i];s.w[i]-=config.learningRate*mh/(sqrt(vh)+config.adamEps)}}
    private fun averageGrad(bank:List<DeepMindSample>,shadow:Boolean):Grad{val ew=if(shadow)Effective(rawShadow(inState),rawShadow(recState),rawShadow(outState))else effective();val g=Grad(FloatArray(h*IN_DIM),FloatArray(h*h),FloatArray(OUT_DIM*h));val work=bank.take(32);for(s in work)gradientOne(s,ew,g);scale(g,1f/work.size.coerceAtLeast(1));return g}
    private fun rawShadow(s:MatrixState)=FloatArray(s.w.size){s.w[it]}
    private fun grow(g:Grad)=growOne(inState,g.gi,0)+growOne(recState,g.gr,1)+growOne(outState,g.go,2)
    private fun growOne(s:MatrixState,g:FloatArray,mid:Int):Int{for(i in s.cooldown.indices)if(s.cooldown[i]>0)s.cooldown[i]=(s.cooldown[i]-1).toByte();val scores=s.w.indices.filter{s.mask[it].toInt()!=0}.map{abs(g[it])};val thr=if(scores.isEmpty())0f else quantile(scores,.25f)*.5f;var sq=0.0;var n=0;for(i in s.w.indices)if(s.mask[i].toInt()!=0){sq+=s.w[i]*s.w[i];n++};val init=max((if(n>0)sqrt(sq/n).toFloat()else .01f)*config.regrowInitScale,1e-5f);val r=Random(11L+structuralCycle*100003L+mid*1009L);var count=0;for(i in s.w.indices)if(s.mask[i].toInt()==0&&s.cooldown[i]<=0&&abs(g[i])>=thr){s.w[i]=(if(r.nextBoolean())1f else -1f)*(.5f+r.nextFloat())*init;s.mask[i]=1;s.m[i]=0f;s.v[i]=0f;count++};return count}
    private fun contribution(s:MatrixState,g:FloatArray)=FloatArray(s.w.size){i->if(s.mask[i].toInt()!=0)abs(s.w[i]*g[i])else 0f}
    private fun appearance(s:MatrixState,c:FloatArray){val ids=s.w.indices.filter{s.mask[it].toInt()!=0};if(ids.isEmpty())return;val k=max(1,(ids.size*config.importantFraction).roundToInt());for(id in ids.sortedByDescending{c[it]}.take(k))s.appearance[id]++}
    private fun combined(s:MatrixState,c:FloatArray,out:FloatArray,offset:Int){val vals=s.w.indices.filter{s.mask[it].toInt()!=0}.map{c[it]};val med=if(vals.isEmpty())1e-12f else quantile(vals,.5f).coerceAtLeast(1e-12f);val den=max(selectionCycles+1,1).toFloat();for(i in s.w.indices)if(s.mask[i].toInt()!=0){val cn=c[i]/(c[i]+med);val p=s.appearance[i]/den;out[offset+i]=sqrt((p+1e-6f)*(cn+1e-6f))}}
    private fun quantile(v:List<Float>,q:Float):Float{if(v.isEmpty())return 0f;val s=v.sorted();return s[(q.coerceIn(0f,1f)*(s.size-1)).roundToInt()]}
    private fun globalActive():List<Int>{val o=ArrayList<Int>();var off=0;for(s in arrayOf(inState,recState,outState)){for(i in s.w.indices)if(s.mask[i].toInt()!=0)o+=off+i;off+=s.w.size};return o}
    private fun locate(id:Int):Pair<MatrixState,Int> = when { id<inState.w.size->inState to id; id<inState.w.size+recState.w.size->recState to(id-inState.w.size); else->outState to(id-inState.w.size-recState.w.size)}
    private fun clearProtected(){inState.protected.fill(0);recState.protected.fill(0);outState.protected.fill(0)}
    private fun setProtected(id:Int,v:Boolean){val(s,i)=locate(id);s.protected[i]=if(v)1 else 0}
    private fun isProtected(id:Int):Boolean{val(s,i)=locate(id);return s.protected[i].toInt()!=0}
    private fun prune(id:Int){val(s,i)=locate(id);s.mask[i]=0;s.w[i]=0f;s.m[i]=0f;s.v[i]=0f;s.utility[i]=0f;s.protected[i]=0;s.cooldown[i]=1}
    private fun buildQuantCache()=QuantCache(quant(inState),quant(recState),quant(outState))
    private fun quant(s:MatrixState):Quant{val scale=maxAbs(s).coerceAtLeast(1e-8f)/127f;val q=ByteArray(s.w.size);var sat=0;for(i in s.w.indices){if(s.mask[i].toInt()==0)q[i]=0 else{val raw=(s.w[i]/scale).roundToInt();if(abs(raw)>=127)sat++;q[i]=raw.coerceIn(-127,127).toByte()}};return Quant(q,scale,sat)}
    private fun fixed(v:Float)=(v*(1 shl FIXED_SHIFT)).roundToInt().coerceAtLeast(0)
    private fun invalidateRuntime(){qCache=null}
    private fun refreshEvent(){val c=lastCore;lastCore=c.copy(active=activeWeights(),dormant=totalWeights-activeWeights(),protected=protectedWeights(),phase=phase,structuralCycle=structuralCycle,lastGrown=lastEvent.grown,lastPruned=lastEvent.pruned)}
    private fun updateCore(mem:FloatArray,sp:ByteArray,rate:Float){var sum=0f;var maxW=0f;var n=0;for(s in arrayOf(inState,recState,outState))for(i in s.w.indices)if(s.mask[i].toInt()!=0){val a=abs(s.w[i]);sum+=a;maxW=max(maxW,a);n++};val qc=qCache?:buildQuantCache();val sat=(qc.wi.sat+qc.wr.sat+qc.wo.sat).toFloat()/max(totalWeights,1);lastCore=CoreSnapshot(mem.copyOf(),sp.copyOf(),rate,activeWeights(),totalWeights-activeWeights(),protectedWeights(),phase,structuralCycle,lastEvent.grown,lastEvent.pruned,if(n>0)sum/n else 0f,maxW,sat,lastGradientNorm,step)}
    private fun writeState(out:DataOutputStream,s:MatrixState){out.writeInt(s.w.size);for(x in s.w)out.writeFloat(x);out.write(s.mask);for(x in s.m)out.writeFloat(x);for(x in s.v)out.writeFloat(x);for(x in s.utility)out.writeFloat(x);for(x in s.appearance)out.writeInt(x);out.write(s.protected);out.write(s.cooldown)}
    private fun readState(input:DataInputStream,s:MatrixState){require(input.readInt()==s.w.size);for(i in s.w.indices)s.w[i]=input.readFloat();input.readFully(s.mask);for(i in s.m.indices)s.m[i]=input.readFloat();for(i in s.v.indices)s.v[i]=input.readFloat();for(i in s.utility.indices)s.utility[i]=input.readFloat();for(i in s.appearance.indices)s.appearance[i]=input.readInt();input.readFully(s.protected);input.readFully(s.cooldown)}
}
