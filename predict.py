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

        self.fc1 = nn.Linear(800,128)
        self.lif1 = snn.Leaky(beta=0.9)

        self.fc2 = nn.Linear(128,2)
        self.lif2 = snn.Leaky(beta=0.9)

    def forward(self,x):

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spk2_rec = []

        num_steps = 25   # number of SNN time steps

        for step in range(num_steps):

            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spk2_rec.append(spk2)

        return torch.stack(spk2_rec).sum(0)  # accumulate spikes


# Load model
model = Net()
model.load_state_dict(torch.load("snn_spam_model.pth"))
model.eval()


def predict_sms(text):

    text = text.lower()  # important preprocessing

    X = vectorizer.transform([text])
    X = scaler.transform(X.toarray())

    X_tensor = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        output = model(X_tensor)

    prediction = torch.argmax(output, dim=1)

    return prediction.item()


# Interactive testing
while True:
    msg = input("\nEnter SMS (type 'quit' to stop): ")

    if msg.lower() == "quit":
        break

    result = predict_sms(msg)

    if result == 1:
        print("🚨 Prediction: SPAM")
    else:
        print("✅ Prediction: NOT SPAM")