package com.example.equationsolver.data

import android.content.ContentResolver
import android.net.Uri
import java.io.BufferedReader
import java.io.InputStreamReader

data class DeepMindSample(val features: FloatArray, val targets: FloatArray)
data class DeepMindMeta(val split: String, val rows: Long, val commit: String)

object DeepMindDataset {
    const val PINNED_COMMIT = "427f45075f84b8b9774950196ad63867ca20ffb3"
    private const val MAGIC = "# DEEPMIND_MATHEMATICS_DATASET"
    private const val MODULE = "algebra.linear_2d"
    fun open(resolver: ContentResolver, uri: Uri): Sequence<DeepMindSample> = sequence {
        resolver.openInputStream(uri)?.use { input ->
            BufferedReader(InputStreamReader(input)).use { reader ->
                validateHeader(reader)
                while (true) { val line = reader.readLine() ?: break; val s = parseRow(line) ?: continue; yield(s) }
            }
        } ?: error("تعذر فتح ملف DeepMind")
    }
    fun inspect(resolver: ContentResolver, uri: Uri, maxRows: Int = 2000): DeepMindMeta {
        var split = "unknown"; var rows = 0L
        resolver.openInputStream(uri)?.use { input -> BufferedReader(InputStreamReader(input)).use { reader ->
            val h = readHeader(reader); require(h["commit"] == PINNED_COMMIT) { "إصدار DeepMind غير مطابق" }; require(h["module"] == MODULE) { "الملف ليس algebra.linear_2d" }
            split = h["split"] ?: "unknown"; while (rows < maxRows && reader.readLine() != null) rows++
        }} ?: error("تعذر فتح الملف")
        return DeepMindMeta(split, rows, PINNED_COMMIT)
    }
    private fun validateHeader(reader: BufferedReader) {
        val h = readHeader(reader); require(h["commit"] == PINNED_COMMIT) { "رفض الملف: commit DeepMind غير مطابق" }; require(h["module"] == MODULE) { "رفض الملف: المطلوب algebra.linear_2d" }; require(h["format"] == "a,b,c,d,e,f|x,y") { "صيغة ملف DeepMind غير مدعومة" }
    }
    private fun readHeader(reader: BufferedReader): Map<String, String> {
        val first = reader.readLine()?.trim() ?: error("ملف فارغ"); require(first == MAGIC) { "الملف ليس DeepMind export موثوق" }; val out = linkedMapOf<String, String>()
        while (true) { reader.mark(8192); val line = reader.readLine() ?: break; if (!line.startsWith("#")) { reader.reset(); break }; val p = line.removePrefix("#").trim().split("=", limit = 2); if (p.size == 2) out[p[0].trim().lowercase()] = p[1].trim() }
        return out
    }
    private fun parseRow(raw: String): DeepMindSample? {
        val line = raw.trim(); if (line.isEmpty() || line.startsWith("#")) return null; val parts = line.split('|', limit = 2); if (parts.size != 2) return null
        val x = parts[0].split(',').mapNotNull { it.trim().toFloatOrNull() }; val y = parts[1].split(',').mapNotNull { it.trim().toFloatOrNull() }
        if (x.size != 6 || y.size != 2 || x.any { !it.isFinite() } || y.any { !it.isFinite() }) return null
        return DeepMindSample(x.toFloatArray(), y.toFloatArray())
    }
}
