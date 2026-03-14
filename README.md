# SmartShield

**On-device SMS fraud detection using Spiking Neural Networks — built for privacy, speed, and real-world Indian scam patterns.**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![snnTorch](https://img.shields.io/badge/snnTorch-neuromorphic-purple.svg)](https://snntorch.readthedocs.io)
[![ONNX](https://img.shields.io/badge/ONNX-Runtime_Mobile-green.svg)](https://onnxruntime.ai)
[![React Native](https://img.shields.io/badge/React_Native-TypeScript-61DAFB.svg)](https://reactnative.dev)

---

## What is this?

SmartShield is something we built because we were genuinely frustrated with how SMS fraud detection works today. Keyword filters are trivially easy to bypass. Cloud-based ML means your messages are being sent to some server somewhere. And most solutions are built with Western spam patterns in mind — they miss UPI scams, fake KYC alerts, OTP theft messages that are extremely common in India.

So we built something different. A fully on-device fraud detection system that uses a **Spiking Neural Network (SNN)** — a type of neuromorphic model inspired by how biological neurons actually fire. It runs entirely on your phone, no internet required, and is trained specifically to catch the kind of fraud Indian users actually face.

**Team:** Abhishek Kumar (Team Lead) · Hitesh Pratap Singh  
*Built for NIT Hackathon*

---

## How it works

When an SMS arrives, it goes through two stages.

First, a rule-based filter quickly checks for obvious red flags — suspicious sender patterns, known OTP-theft phrases, embedded URLs, etc. Most clear-cut spam gets caught here in microseconds.

If the message passes that filter (or looks borderline), it goes into the SNN classifier. We extract 9 features from the message — things like TF-IDF text representation, digit ratio, capital letter ratio, urgency keyword score — and encode them into **spike trains** using latency encoding:

```
tᵢ = Tmax(1 − xᵢ)
```

These spike trains feed into a 1D Convolutional SNN with Leaky Integrate-and-Fire (LIF) neurons. The LIF model only "fires" when input crosses a threshold, which means most neurons stay silent for most inputs. That's what makes it energy efficient compared to a regular neural network.

The model outputs one of four classes: **Phishing, Spam, Suspicious, or Safe** — along with a confidence score and a suggested action (Ignore / Block / Report).

Everything runs on-device via ONNX Runtime Mobile. Your messages never leave your phone.

---

## Project structure

```
SmartShield/
├── code.ipynb            # where all the training happens
├── predict.py            # run inference from the command line
├── rules.py              # the rule-based pre-filter
├── export_onnx.py        # converts trained model to ONNX
├── export_vocab.py       # exports vocab for on-device tokenization
├── snn_spam_model.pth    # saved model weights
├── vectorizer.pkl        # fitted TF-IDF vectorizer
├── scaler.pkl            # feature scaler
├── abhi.csv              # our custom Indian fraud dataset
├── syn.csv               # synthetic phishing samples we generated
└── uci                   # UCI SMS Spam Collection (benchmark data)
```

---

## Dataset

We trained on a mix of two sources.

**80% — UCI SMS Spam Collection.** Standard benchmark, well-labeled, good general coverage of spam vs. ham.

**20% — Custom dataset we built ourselves.** This is the part we're most proud of. We collected and generated samples covering UPI fraud, fake KYC alerts, bank impersonation, delivery scam messages, OTP theft attempts — stuff that's everywhere in India but totally absent from standard datasets. Without this, the model would be blind to a huge category of real-world attacks.

We handled class imbalance using a weighted loss function and kept the train/val split stratified so the class ratio stays consistent across splits.

---

## Training & running locally

**Setup**

```bash
git clone https://github.com/Abhikumar199/SmartShield.git
cd SmartShield

python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate

pip install torch snntorch scikit-learn pandas numpy onnx onnxruntime
```

**Training**

Open `code.ipynb` in Jupyter and run through it. The notebook trains the SNN, evaluates it, and saves the weights to `snn_spam_model.pth`.

```bash
jupyter notebook code.ipynb
```

**Quick inference test**

```bash
python predict.py --message "Dear customer, your KYC is expired. Click here to verify now."
```

Output looks like:
```
Rule-Based Filter: flagged (urgency keyword + URL pattern)
SNN Inference: PHISHING  [confidence: 0.91]
Suggested action: Block & Report
```

**Export for mobile**

```bash
python export_onnx.py      # produces snn_spam_model.onnx
python export_vocab.py     # produces vocab.json for on-device tokenizer
```

---

## Android app

The mobile side is a React Native app (TypeScript) with native Kotlin modules handling the actual ONNX inference and SMS interception.

### What you need

- Android Studio (Hedgehog or later)
- Android SDK API 26+
- Node.js 18+
- JDK 17

### Getting it running

```bash
npm install
npm install onnxruntime-react-native

# drop the model files into Android assets
cp snn_spam_model.onnx android/app/src/main/assets/
cp vocab.json android/app/src/main/assets/

# start metro
npx react-native start

# run on device or emulator
npx react-native run-android
```

### Permissions

Add these to `android/app/src/main/AndroidManifest.xml`:

```xml
<uses-permission android:name="android.permission.RECEIVE_SMS" />
<uses-permission android:name="android.permission.READ_SMS" />
<uses-permission android:name="android.permission.READ_PHONE_STATE" />
```

### How the SMS interception works (Kotlin)

```kotlin
// SmsReceiver.kt
class SmsReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action == Telephony.Sms.Intents.SMS_RECEIVED_ACTION) {
            val messages = Telephony.Sms.Intents.getMessagesFromIntent(intent)
            for (sms in messages) {
                SmartShieldModule.analyzeMessage(sms.messageBody)
            }
        }
    }
}
```

### ONNX inference module (Kotlin)

```kotlin
// SmartShieldModule.kt
class SmartShieldModule(reactContext: ReactApplicationContext) :
    ReactContextBaseJavaModule(reactContext) {

    private lateinit var ortSession: OrtSession
    private val ortEnv = OrtEnvironment.getEnvironment()

    override fun getName() = "SmartShieldModule"

    @ReactMethod
    fun analyzeMessage(message: String, promise: Promise) {
        val features = extractFeatures(message)
        val inputTensor = OnnxTensor.createTensor(ortEnv, features)
        val result = ortSession.run(mapOf("input" to inputTensor))
        val scores = (result[0].value as Array<FloatArray>)[0]
        promise.resolve(buildResultMap(scores))
    }
}
```

The native module bridges to React Native via JSI/TurboModules to keep the overhead minimal.

---

## Performance targets

We're aiming for:

- F1-score above 0.90
- Inference under 100ms per message
- Meaningfully lower energy use than an equivalent ANN — sparse LIF firing is what makes this possible

These are targets, not guarantees. Actual numbers depend on the device. We'll publish a proper comparison against an ANN baseline once the full evaluation is done.

---

## Tech used

**ML:** Python, PyTorch, snnTorch, scikit-learn, ONNX

**Mobile:** React Native (TypeScript), Kotlin, Swift, JSI/TurboModules, ONNX Runtime Mobile

---

## What's left

- [x] SNN training pipeline
- [x] Rule-based pre-filter
- [x] ONNX export
- [ ] React Native + Android integration (in progress)
- [ ] Background service for passive SMS monitoring
- [ ] iOS support
- [ ] ANN baseline comparison write-up
- [ ] Federated learning (longer-term idea)

---

## Contributing

If you find a bug, have fraud message samples to contribute (especially Indian-context ones), or want to help with the Android side — open an issue or send a PR.

---

## Acknowledgements

UCI SMS Spam Collection for the base dataset, the snnTorch team for making SNN research accessible in PyTorch, and ONNX Runtime Mobile for making on-device inference actually practical on real phones.