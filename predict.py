import torch
import joblib
import snntorch as snn
import torch.nn as nn

# Load preprocessing
vectorizer = joblib.load("vectorizer.pkl")
scaler = joblib.load("scaler.pkl")

# Define model architecture
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # match training architecture
        self.fc1 = nn.Linear(1200, 256)
        self.lif1 = snn.Leaky(beta=0.9)

        self.fc2 = nn.Linear(256, 2)
        self.lif2 = snn.Leaky(beta=0.9)

    def forward(self, x):

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spk2_rec = []

        num_steps = 25  # SNN simulation steps

        for step in range(num_steps):

            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spk2_rec.append(spk2)

        # accumulate spikes over time
        return torch.stack(spk2_rec).sum(0)


# Load model
model = Net()
model.load_state_dict(torch.load("snn_spam_model.pth"))
model.eval()


def predict_sms(text):

    text = text.lower()

    X = vectorizer.transform([text])
    X = scaler.transform(X.toarray())

    X_tensor = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        output = model(X_tensor)

    # Convert logits/spikes to probabilities
    import torch.nn.functional as F
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