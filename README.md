# SmartShield

**On-device SMS fraud detection for Android — SNN model running via PyTorch Mobile, planning to migrate to ONNX.**

> ⚠️ Work in progress. Core ML pipeline and Android app are functional, but the project is still under active development.

---

## What is this?

SmartShield is a native Android app that detects SMS fraud entirely on-device — no internet, no cloud, no data leaving your phone. We trained a Spiking Neural Network on a mix of standard spam datasets and custom Indian fraud samples (UPI scams, KYC alerts, OTP theft, fake bank messages), then deployed it on Android using PyTorch Mobile.

The app lets you paste any SMS and get an instant verdict. It also runs a `BroadcastReceiver` in the background to intercept incoming SMS automatically.

**Team:** Abhishek Kumar (Team Lead) · Hitesh Pratap Singh  
*Built for NIT Hackathon*

---

## What's actually built so far

**ML side (Python)**
- SNN model trained with PyTorch + snnTorch
- TF-IDF vectorizer (vocab size 900–1200 tokens)
- Model exported to `.pt` format via `torch.jit.script` / `torch.jit.trace` for mobile

**Android app (Kotlin + Jetpack Compose)**
- Loads `spam_model_mobile.pt` from assets using PyTorch Android (`org.pytorch:pytorch_android:1.13.1`)
- `VocabHelper.kt` — reads `vocab.json` from assets, converts raw SMS text into a float vector
- `SpamDetector.kt` — runs the `.pt` model using PyTorch `Module.forward()`
- `MainActivity.kt` — Compose UI with a text field to paste and analyze messages, shows risk level + model confidence
- `SmsReceiver.kt` — `BroadcastReceiver` that listens for incoming SMS (currently shows a toast, full integration pending)
- `Utils.kt` — copies model from assets to internal storage (required by PyTorch Mobile)
- Rule-based layer in `MainActivity.kt` — keyword fraud check + phishing URL detection on top of the model output, combines into a `fraudScore`

**What the output looks like**
- Model gives two softmax scores (safe / spam)
- Those are combined with the rule-based `fraudScore`
- Final verdict: `HIGH RISK` or `SAFE`, shown with confidence percentage

---

## Project structure

```
SmartShield/
│
├── app/src/main/
│   ├── assets/
│   │   ├── spam_model_mobile.pt     # PyTorch Mobile model (current)
│   │   └── vocab.json               # TF-IDF vocabulary
│   │
│   └── java/com/example/smartshield/
│       ├── MainActivity.kt          # UI + rule-based logic
│       ├── SpamDetector.kt          # PyTorch inference wrapper
│       ├── SmsReceiver.kt           # BroadcastReceiver for incoming SMS
│       ├── VocabHelper.kt           # Text → float vector conversion
│       ├── Utils.kt                 # Asset file helper
│       └── ui/theme/                # Compose theme files
│
├── app/build.gradle.kts             # Dependencies (PyTorch Mobile, Compose, Room)
└── AndroidManifest.xml              # SMS permissions + receiver registration
```

**Python side (separate repo / notebook)**

```
├── code.ipynb            # SNN training
├── predict.py            # CLI inference test
├── rules.py              # rule-based pre-filter
├── export_onnx.py        # ONNX export (planned migration)
├── export_vocab.py       # exports vocab.json for mobile
├── snn_spam_model.pth    # trained weights
├── vectorizer.pkl        # fitted TF-IDF vectorizer
├── scaler.pkl            # feature scaler
├── abhi.csv              # custom Indian fraud dataset
├── syn.csv               # synthetic phishing samples
└── uci                   # UCI SMS Spam Collection
```

---

## Running the Android app

**Requirements**

- Android Studio (Hedgehog or later)
- Android SDK API 24+
- Physical device or emulator (arm64-v8a — the build currently filters for arm64 only)
- JDK 17

**Steps**

```bash
git clone https://github.com/Abhikumar199/SmartShield.git
cd SmartShield
```

Open the project in Android Studio, let Gradle sync, then run on your device. The model and vocab are already bundled in `assets/` so no extra setup needed.

The first time the app launches, `Utils.kt` copies `spam_model_mobile.pt` from assets to internal storage — this is a PyTorch Mobile requirement, it can't load directly from the asset stream.

**Permissions**

Already declared in `AndroidManifest.xml`:

```xml
<uses-permission android:name="android.permission.RECEIVE_SMS" />
<uses-permission android:name="android.permission.READ_SMS" />
```

The app requests `RECEIVE_SMS` and `READ_SMS` at runtime on first launch.

---

## How the inference works

When you tap "Analyze Message", here's what actually happens in the code:

1. Text is lowercased and split into words
2. `VocabHelper` maps each word to its index in `vocab.json` and builds a `FloatArray` of size 1200 (bag-of-words style)
3. That array is passed to PyTorch Mobile as a `Tensor` of shape `[1, 1200]`
4. The model returns two logits (safe, spam), softmax is applied manually
5. In parallel, `keywordFraud()` checks for known fraud keywords and `detectPhishingUrl()` checks for suspicious domains / link shorteners
6. A `fraudScore` combines both signals — model prediction adds 3 points, keyword hit adds 4, URL adds 2
7. If `fraudScore >= 7`, the message is forced to spam regardless of model output

This hybrid approach means the model doesn't have to be perfect on its own — the rules catch obvious stuff, the model handles the edge cases.

---

## Dependencies (Android)

```kotlin
implementation("org.pytorch:pytorch_android:1.13.1")
implementation("org.pytorch:pytorch_android_torchvision:1.13.1")
// Jetpack Compose, Material3, Room (added for future history feature)
```

---

## Planned / in progress

- [ ] **ONNX migration** — `export_onnx.py` is ready on the Python side, Android integration pending. ONNX Runtime Mobile will replace PyTorch Mobile for smaller binary size and better cross-platform support.
- [ ] **SmsReceiver full integration** — currently shows a toast on SMS receive, needs to run `SpamDetector` and trigger a notification
- [ ] **Notification system** — alert the user when a background SMS gets flagged
- [ ] **Message history** — Room DB is already added as a dependency, schema not implemented yet
- [ ] **iOS support** — not started
- [ ] **ANN baseline comparison** — for the paper/report

---

## Training (Python side)

```bash
python -m venv venv
source venv/bin/activate

pip install torch snntorch scikit-learn pandas numpy onnx onnxruntime

# train
jupyter notebook code.ipynb

# test inference locally
python predict.py --message "Your KYC is expired. Update now to avoid account suspension."

# export for mobile (PyTorch)
# handled inside code.ipynb via torch.jit.trace → spam_model_mobile.pt

# export to ONNX (planned)
python export_onnx.py
```

---

## Why SNN?

Standard neural networks compute on every neuron for every input. Leaky Integrate-and-Fire neurons only fire when their membrane potential crosses a threshold — most stay silent. That sparsity translates to fewer multiply-accumulate operations and lower energy use, which matters a lot for something running continuously on a phone battery.

We use latency encoding to convert text features into spike trains: `tᵢ = Tmax(1 − xᵢ)`. Features with higher importance produce earlier spikes. The 1D Conv SNN layers then process these temporal patterns.

---

## Dataset

80% UCI SMS Spam Collection (standard benchmark), 20% custom samples we built — UPI fraud, fake KYC, OTP theft, delivery scams. The Indian-context data was important because those attack patterns are basically absent from every public dataset we found.

Class imbalance handled with weighted loss + stratified splits.