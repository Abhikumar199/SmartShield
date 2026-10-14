package com.example.smartshield

import android.content.Context
import org.json.JSONObject

object VocabHelper {

    private var vocab: Map<String, Int>? = null
    private const val VOCAB_SIZE = 900

    fun loadVocab(context: Context) {

        if (vocab != null) return

        val json = context.assets.open("vocab.json")
            .bufferedReader()
            .use { it.readText() }

        val jsonObject = JSONObject(json)

        vocab = jsonObject.keys().asSequence().associateWith {
            jsonObject.getInt(it)
        }
    }

    fun textToVector(context: Context, text: String): FloatArray {

        loadVocab(context)

        val vector = FloatArray(VOCAB_SIZE)

        val words = text.lowercase().split(" ")

        for (word in words) {
            val index = vocab!![word]
            if (index != null && index < VOCAB_SIZE) {
                vector[index] += 1f
            }
        }

        return vector
    }
}