# SmartShield
### Privacy-Preserving SMS Fraud Detection with Spiking Neural Networks

SmartShield is an Android project exploring **on-device SMS spam and fraud detection** using a hybrid of a Spiking Neural Network (SNN) and lightweight rule-based checks. The goal is to analyze suspicious messages locally, helping reduce reliance on remote services and keeping message content on the device during inference.

> **Project status:** Work in progress. Manual message analysis and the model pipeline are present in the repository. Automated handling of received SMS and the planned ONNX migration still need integration and testing.

## Highlights

- **On-device inference:** the current Android implementation uses PyTorch Mobile.
- **SNN-based text classification:** a model trained with PyTorch and snnTorch.
- **Rule-assisted detection:** keyword and suspicious-URL checks complement the model.
- **Android interface:** Kotlin and Jetpack Compose.
- **Indian fraud scenarios:** custom examples include UPI, KYC, OTP, banking, and delivery-related messages.
- **Privacy-oriented design:** inference is intended to run locally rather than send SMS content to a server.

## How it works

The current manual analysis flow combines text classification with simple rules:

1. The user enters an SMS in the app.
2. The text is converted to a fixed-length feature vector using the vocabulary bundled with the app.
3. The vector is passed to the mobile SNN model.
4. Rule-based checks look for suspicious keywords and URLs.
5. The app combines the signals and displays a risk verdict.

The Python experiments use text features and spike-based temporal encoding for SNN inference. The precise preprocessing and feature representation must remain consistent between training and Android deployment.

**Important:** this is a prototype, not a guarantee that a message is safe or fraudulent. Rule thresholds and model confidence are heuristics and should not be treated as financial or security advice.

## Technology stack

| Area | Tools |
|---|---|
| Android app | Kotlin, Android Studio, Jetpack Compose |
| Model development | Python, PyTorch, snnTorch |
| Text features | scikit-learn TF-IDF / vocabulary utilities |
| On-device runtime | PyTorch Mobile (current implementation) |
| Planned runtime migration | ONNX Runtime Mobile |
| Data analysis | pandas, NumPy |

## Repository layout

```text
SmartShield/
├── app/
│   └── src/main/
│       ├── assets/
│       │   ├── spam_model_mobile.pt
│       │   └── vocab.json
│       └── java/com/example/smartshield/
│           └── MainActivity.kt
├── MyApplication/                  # Additional Android Studio project files
├── Code/
│   ├── code.ipynb                  # SNN training experiments
│   ├── predict.py                  # Python inference/testing
│   ├── rules.py                    # Rule-based helpers
│   ├── export_onnx.py              # ONNX export experiment
│   └── export_vocab.py             # Vocabulary export utility
├── SMSSpamCollection                # SMS spam benchmark data
├── abhi.csv                         # Custom examples
├── syn.csv                          # Synthetic examples
├── snn_spam_model.pth               # Trained PyTorch weights
├── scaler.pkl                       # Saved preprocessing object
├── vectorizer.pkl                   # Saved text vectorizer
└── README.md
```

The repository contains more than one Android-project directory. Open the Android project that contains the app you intend to build, and verify its Gradle configuration before running.

## Run the Android app

### Requirements

- Android Studio
- JDK version compatible with the project's Gradle setup (JDK 17 is expected by the current project notes)
- Android SDK installed through Android Studio
- Android device or emulator compatible with the configured build

### Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/Abhikumar199/SmartShield.git
   cd SmartShield
   ```

2. Open the relevant Android project directory in Android Studio.
3. Allow Gradle sync to finish and resolve any SDK or dependency prompts.
4. Run the app on a compatible device or emulator.
5. Test with sample messages before using real messages.

The Android assets currently include `spam_model_mobile.pt` and `vocab.json`. Ensure that the app's preprocessing and model input dimensions match the versions used to train and export the model.

## Run the Python experiments

Use Python 3.10 or 3.11 for the best chance of compatibility with older ML dependencies, and create an isolated environment.

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install torch snntorch scikit-learn pandas numpy jupyter onnx onnxruntime
```

Run the training notebook:

```bash
jupyter notebook Code/code.ipynb
```

Test the Python inference script:

```bash
python Code/predict.py --message "Your KYC is expired. Update now to avoid account suspension."
```

The inference command and required arguments may depend on the current script version. Run `python Code/predict.py --help` if needed. Pickle files such as `vectorizer.pkl` and `scaler.pkl` should only be loaded from trusted sources.

## Dataset and evaluation

The project uses the UCI SMS Spam Collection alongside custom and synthetic examples intended to represent Indian-context scam messages, including UPI, KYC, OTP, banking, and delivery themes.

Before drawing performance conclusions, evaluate on a held-out test set and report metrics such as precision, recall, F1-score, and a confusion matrix. Keep near-duplicate or synthetic examples from leaking across training and test splits. Results from the existing notebook should be reported with the exact split and preprocessing procedure used; this README does not claim an unverified accuracy figure.

## Current status

- [x] SNN training experiments and saved model artifacts are present.
- [x] Android UI and bundled mobile model assets are present.
- [x] Manual SMS analysis with model and rule-based signals is implemented in the current prototype.
- [ ] Complete background SMS processing and user notifications.
- [ ] Validate preprocessing parity between Python and Android.
- [ ] Finish and test ONNX Runtime Mobile integration.
- [ ] Add reproducible evaluation results and model-versus-baseline comparison.
- [ ] Add automated tests and documented APK installation/build instructions.

## Why use an SNN?

Spiking Neural Networks represent neural activity through discrete spikes and can model temporal information. This project explores spike-based text processing as a path toward lightweight, brain-inspired inference on mobile devices. Actual energy efficiency depends on the model, runtime, and hardware; it should be measured rather than inferred from using an SNN alone.

## Team

- **Abhishek Kumar** — Team Lead
- **Hitesh Pratap Singh**

Built for the NIT Hackathon.

## Disclaimer

SmartShield is an experimental project. It may miss fraudulent messages or flag legitimate messages incorrectly. Do not rely on it as the sole basis for clicking links, sharing OTPs, transferring money, or making financial decisions. Never include real OTPs, account details, or other sensitive data in public test examples.

## Contributing

Issues and suggestions are welcome. When contributing, avoid committing private SMS data, credentials, API keys, or personally identifiable information.
