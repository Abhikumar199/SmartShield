package com.example.smartshield

import android.content.Context
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor

object SpamDetector {

    private var module: Module? = null

    fun detect(context: Context, text: String): Boolean {

        if (module == null) {
            module = Module.load(Utils.assetFilePath(context, "spam_model_mobile.pt"))
        }

        val vector = VocabHelper.textToVector(context, text)

        val inputTensor = Tensor.fromBlob(vector, longArrayOf(1, vector.size.toLong()))

        val output = module!!.forward(IValue.from(inputTensor)).toTensor()

        val score = output.dataAsFloatArray[0]

        return score > 0.5
    }
}