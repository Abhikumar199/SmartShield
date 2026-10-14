package com.example.smartshield

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.Alignment
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material.icons.filled.CheckCircle
import org.json.JSONObject
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import java.io.File
import java.io.FileOutputStream

class MainActivity : ComponentActivity() {

    private lateinit var module: Module

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        requestSmsPermission()

        try {
            module = Module.load(assetFilePath("spam_model_mobile.pt"))
        } catch (e: Exception) {
            e.printStackTrace()
        }

        setContent {
            SpamDetectorUI(module)
        }
    }

    private fun requestSmsPermission() {

        if (ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.RECEIVE_SMS
            ) != PackageManager.PERMISSION_GRANTED
        ) {

            ActivityCompat.requestPermissions(
                this,
                arrayOf(
                    Manifest.permission.RECEIVE_SMS,
                    Manifest.permission.READ_SMS
                ),
                1
            )
        }
    }

    private fun assetFilePath(assetName: String): String {

        val file = File(filesDir, assetName)

        if (file.exists() && file.length() > 0)
            return file.absolutePath

        assets.open(assetName).use { input ->
            FileOutputStream(file).use { output ->
                input.copyTo(output)
            }
        }

        return file.absolutePath
    }
}

/* ----------- Phishing URL Detection ----------- */

fun detectPhishingUrl(text: String): Boolean {

    val suspiciousDomains = listOf(
        ".xyz", ".top", ".click", ".gq", ".work", ".com"
    )

    val shorteners = listOf(
        "bit.ly", "tinyurl.com", "t.co", "goo.gl"
    )

    val words = text.lowercase().split(" ")

    for (word in words) {

        if (word.contains("http") || word.contains("www")) {

            for (domain in suspiciousDomains)
                if (word.contains(domain)) return true

            for (short in shorteners)
                if (word.contains(short)) return true
        }
    }

    return false
}

/* ----------- Fraud Rules ----------- */

val fraudKeywords = listOf(
    "otp","verify","verification","suspended","account blocked",
    "urgent","claim reward","lottery","winner","won rs",
    "lucky draw","pay now","electricity bill","update kyc",
    "bank alert","secure account","login now"
)

fun keywordFraud(text: String): Boolean {

    val msg = text.lowercase()

    for (word in fraudKeywords) {
        if (msg.contains(word)) return true
    }

    return false
}

fun extractUrls(text: String): List<String> {

    val regex = Regex("(https?://\\S+|www\\.\\S+)")
    return regex.findAll(text).map { it.value }.toList()
}

fun fraudScore(text: String, modelPrediction: Int): Int {

    var score = 0

    val urls = extractUrls(text)

    if (keywordFraud(text)) score += 4
    if (urls.isNotEmpty()) score += 2

    if (modelPrediction == 1) score += 3

    return score
}

/* ----------- UI ----------- */

@Composable
fun SpamDetectorUI(module: Module) {

    val context = LocalContext.current

    var text by remember { mutableStateOf("") }
    var result by remember { mutableStateOf("") }
    var confidenceValue by remember { mutableIntStateOf(0) }

    val vocab = remember {

        val inputStream = context.assets.open("vocab.json")
        val jsonString = inputStream.bufferedReader().use { it.readText() }
        JSONObject(jsonString)
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(20.dp)
    ) {

        Text(
            text = "SmartShield",
            style = MaterialTheme.typography.headlineLarge
        )

        Text(
            text = "AI SMS Fraud Protection",
            style = MaterialTheme.typography.bodyLarge
        )

        Spacer(modifier = Modifier.height(20.dp))

        Card(modifier = Modifier.fillMaxWidth()) {

            Column(modifier = Modifier.padding(16.dp)) {

                Text("Protection Status")

                Spacer(modifier = Modifier.height(6.dp))

                Text("🟢 Active Protection")
            }
        }

        Spacer(modifier = Modifier.height(20.dp))

        Text("Scan SMS Message")

        Spacer(modifier = Modifier.height(10.dp))

        TextField(
            value = text,
            onValueChange = { text = it },
            label = { Text("Paste SMS message here") },
            modifier = Modifier.fillMaxWidth()
        )

        Spacer(modifier = Modifier.height(15.dp))

        Button(
            onClick = {

                val lowerText = text.lowercase()

                if (detectPhishingUrl(lowerText)) {

                    result = "Phishing link detected"
                    confidenceValue = 100
                    return@Button
                }

                val words = lowerText.split(" ")

                val vector = FloatArray(1200)

                for (word in words) {

                    if (vocab.has(word)) {

                        val index = vocab.getInt(word)

                        if (index < 1200)
                            vector[index] += 1f
                    }
                }

                val tensor = Tensor.fromBlob(vector, longArrayOf(1, 1200))

                val output = module.forward(IValue.from(tensor)).toTensor()

                val scores = output.dataAsFloatArray

                val safeScore = scores[0].toDouble()
                val spamScore = scores[1].toDouble()

                val expSafe = kotlin.math.exp(safeScore)
                val expSpam = kotlin.math.exp(spamScore)

                val sum = expSafe + expSpam

                val probSafe = expSafe / sum
                val probSpam = expSpam / sum

                val modelPrediction =
                    if (probSpam > probSafe) 1 else 0

                val score = fraudScore(lowerText, modelPrediction)

                val finalPrediction =
                    if (score >= 7) 1 else modelPrediction

                confidenceValue =
                    (kotlin.math.max(probSafe, probSpam) * 100).toInt()

                result =
                    if (finalPrediction == 1)
                        "Spam message detected"
                    else
                        "Message appears safe"
            },

            modifier = Modifier.fillMaxWidth()
        ) {
            Text("Analyze Message")
        }

        Spacer(modifier = Modifier.height(20.dp))

        if (result.isNotEmpty()) {

            val isSpam = result.contains("Spam")

            val icon =
                if (isSpam) Icons.Default.Warning
                else Icons.Default.CheckCircle

            val color =
                if (isSpam) Color.Red
                else Color(0xFF2E7D32)

            val riskLevel =
                if (isSpam) "HIGH RISK"
                else "SAFE"

            Card(modifier = Modifier.fillMaxWidth()) {

                Column(
                    modifier = Modifier.padding(16.dp),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {

                    Icon(
                        imageVector = icon,
                        contentDescription = null,
                        tint = color,
                        modifier = Modifier.size(40.dp)
                    )

                    Spacer(modifier = Modifier.height(8.dp))

                    Text(
                        text = "Risk Level: $riskLevel",
                        color = color
                    )

                    Spacer(modifier = Modifier.height(6.dp))

                    Text(result)

                    Spacer(modifier = Modifier.height(6.dp))

                    Text("Model Confidence: $confidenceValue%")
                }
            }
        }
    }
}