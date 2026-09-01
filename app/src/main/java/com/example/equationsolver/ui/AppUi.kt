package com.example.equationsolver.ui

import android.app.Activity
import android.content.Context
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.*

val BG = Color.rgb(7, 17, 31)
val SURFACE = Color.rgb(14, 27, 46)
val BORDER = Color.rgb(36, 58, 85)
val TEXT = Color.rgb(244, 247, 251)
val MUTED = Color.rgb(156, 176, 201)
val ACCENT = Color.rgb(88, 225, 193)
val SKY = Color.rgb(120, 169, 255)
val AMBER = Color.rgb(244, 201, 107)
val RED = Color.rgb(255, 124, 139)

fun Context.dp(v: Int) = (v * resources.displayMetrics.density).toInt()
fun View.dp(v: Int) = context.dp(v)

fun Activity.screen(title: String, subtitle: String? = null): LinearLayout {
    window.statusBarColor = BG; window.navigationBarColor = BG
    val scroll = ScrollView(this).apply { setBackgroundColor(BG); isFillViewport = true }
    val root = LinearLayout(this).apply {
        orientation = LinearLayout.VERTICAL; setPadding(dp(18), dp(24), dp(18), dp(32))
    }
    root.addView(TextView(this).apply {
        text = title; setTextColor(TEXT); textSize = 29f; setTypeface(typeface, Typeface.BOLD); gravity = Gravity.END
    })
    if (!subtitle.isNullOrBlank()) root.addView(TextView(this).apply {
        text = subtitle; setTextColor(MUTED); textSize = 14f; gravity = Gravity.END; setPadding(0, dp(6), 0, dp(12))
    })
    scroll.addView(root, ViewGroup.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))
    setContentView(scroll)
    return root
}

fun LinearLayout.card(): LinearLayout {
    val c = LinearLayout(context).apply {
        orientation = LinearLayout.VERTICAL; setPadding(dp(16), dp(16), dp(16), dp(16))
        background = rounded(SURFACE, 20, BORDER, 1)
    }
    addView(c, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply { setMargins(0, dp(10), 0, 0) })
    return c
}

fun LinearLayout.label(text: String, color: Int = MUTED, size: Float = 13f): TextView = TextView(context).also {
    it.text = text; it.setTextColor(color); it.textSize = size; it.gravity = Gravity.END
    addView(it, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply { setMargins(0, dp(4), 0, dp(4)) })
}

fun LinearLayout.button(text: String, accent: Boolean = false, onClick: () -> Unit): Button = Button(context).also {
    it.text = text; it.isAllCaps = false; it.textSize = 15f; it.setTypeface(it.typeface, Typeface.BOLD)
    it.setTextColor(if (accent) Color.rgb(6, 19, 26) else TEXT)
    it.background = rounded(if (accent) ACCENT else Color.rgb(20, 36, 58), 16, if (accent) ACCENT else BORDER, 1)
    it.setOnClickListener { onClick() }
    addView(it, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(54)).apply { setMargins(0, dp(8), 0, 0) })
}

fun LinearLayout.edit(label: String, value: String, numeric: Boolean = true): EditText {
    label(label)
    return EditText(context).also {
        it.setText(value); it.setTextColor(TEXT); it.setHintTextColor(MUTED); it.textSize = 16f
        it.gravity = Gravity.END; it.setPadding(dp(12), dp(10), dp(12), dp(10)); it.background = rounded(Color.rgb(10, 22, 39), 12, BORDER, 1)
        if (numeric) it.inputType = android.text.InputType.TYPE_CLASS_NUMBER or android.text.InputType.TYPE_NUMBER_FLAG_DECIMAL or android.text.InputType.TYPE_NUMBER_FLAG_SIGNED
        addView(it, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))
    }
}

fun Activity.toast(msg: String) = Toast.makeText(this, msg, Toast.LENGTH_SHORT).show()
private fun rounded(fill: Int, radiusDp: Int, stroke: Int, strokeDp: Int) = GradientDrawable().apply {
    shape = GradientDrawable.RECTANGLE; setColor(fill); cornerRadius = radiusDp * 2f; setStroke(strokeDp, stroke)
}
