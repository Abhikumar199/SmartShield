package com.example.smartshield

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.provider.Telephony
import android.widget.Toast

class SmsReceiver : BroadcastReceiver() {

    override fun onReceive(context: Context, intent: Intent) {

        if (intent.action == Telephony.Sms.Intents.SMS_RECEIVED_ACTION) {

            val messages = Telephony.Sms.Intents.getMessagesFromIntent(intent)

            for (sms in messages) {

                val messageBody = sms.messageBody

                Toast.makeText(
                    context,
                    "SMS received: $messageBody",
                    Toast.LENGTH_LONG
                ).show()

            }
        }
    }
}