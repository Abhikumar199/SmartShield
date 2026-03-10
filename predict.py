import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib

# Load preprocessing
vectorizer = joblib.load("vectorizer.pkl")
scaler = joblib.load("scaler.pkl")


# Mobile-compatible SNN (no snntorch)
class Net(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(1200, 256)
        self.fc2 = nn.Linear(256, 2)

        self.beta = 0.9
        self.threshold = 1.0
        self.num_steps = 25

    def forward(self, x):

        batch_size = x.size(0)

        mem1 = torch.zeros(batch_size, 256)
        mem2 = torch.zeros(batch_size, 2)

        spk_sum = torch.zeros(batch_size, 2)

        for _ in range(self.num_steps):

            # layer 1
            mem1 = self.beta * mem1 + self.fc1(x)
            spk1 = (mem1 > self.threshold).float()
            mem1 = mem1 * (mem1 <= self.threshold)

            # layer 2
            mem2 = self.beta * mem2 + self.fc2(spk1)
            spk2 = (mem2 > self.threshold).float()
            mem2 = mem2 * (mem2 <= self.threshold)

            spk_sum += spk2

        return spk_sum


# Load trained weights
model = Net()
state = torch.load("snn_spam_model.pth", map_location="cpu")
model.load_state_dict(state, strict=False)
model.eval()

def predict_sms(text):

    text = text.lower()

    # Step 1: fast rule check
    if not fast_filter(text):
        return 0, 1.0   # safe message, skip model

    # Step 2: ML model
    X = vectorizer.transform([text])
    X = scaler.transform(X.toarray())

    X_tensor = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        output = model(X_tensor)

    probs = F.softmax(output, dim=1)

    prediction = torch.argmax(probs, dim=1)
    confidence = probs.max().item()

    return prediction.item(), confidence




# Interactive testing
while True:

    msg = input("\nEnter SMS (type 'quit' to stop): ")

    if msg.lower() == "quit":
        break

    result, confidence = predict_sms(msg)

    if result == 1:
        print(f"🚨 Prediction: SPAM (confidence {confidence:.2f})")
    else:
        print(f"✅ Prediction: NOT SPAM (confidence {confidence:.2f})")


