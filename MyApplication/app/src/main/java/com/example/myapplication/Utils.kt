package com.example.smartshield

import android.content.Context
import java.io.File
import java.io.FileOutputStream

object Utils {

    fun assetFilePath(context: Context, assetName: String): String {

        val file = File(context.filesDir, assetName)

        if (file.exists()) return file.absolutePath

        context.assets.open(assetName).use { input ->
            FileOutputStream(file).use { output ->
                input.copyTo(output)
            }
        }

        return file.absolutePath
    }
}