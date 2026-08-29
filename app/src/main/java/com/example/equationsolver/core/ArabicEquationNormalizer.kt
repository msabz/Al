package com.example.equationsolver.core

object ArabicEquationNormalizer {
    private val numberWords = mapOf(
        "أحد عشر" to "11", "اثنا عشر" to "12",
        "صفر" to "0", "واحد" to "1", "اثنان" to "2", "اثنين" to "2", "إثنان" to "2", "إثنين" to "2",
        "ثلاثة" to "3", "ثلاث" to "3", "أربعة" to "4", "اربعة" to "4", "أربعه" to "4",
        "خمسة" to "5", "خمس" to "5", "ستة" to "6", "ست" to "6", "سبعة" to "7", "سبع" to "7",
        "ثمانية" to "8", "ثماني" to "8", "تسعة" to "9", "تسع" to "9", "عشرة" to "10", "عشر" to "10"
    )

    fun normalize(input: String): String {
        var s = input.trim()
            .replace('−', '-')
            .replace('×', '*')
            .replace('÷', '/')
            .replace("يساوي", "=")
            .replace("مساوي", "=")
            .replace("يساوى", "=")

        for ((word, value) in numberWords.entries.sortedByDescending { it.key.length }) {
            s = s.replace(word, value)
        }

        s = s.replace(" زائد ", "+")
            .replace(" ناقص ", "-")
            .replace(" ضرب ", "*")
            .replace(" في ", "*")
            .replace(" مقسوم على ", "/")
            .replace(" تقسيم ", "/")
            .replace("اكس", "x")
            .replace("إكس", "x")
            .replace("واي", "y")
            .replace("واى", "y")

        s = s.replace(Regex("\\s+"), "")
        return s
    }
}
