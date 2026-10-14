import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import re
import numpy as np
import difflib
import math



vectorizer = joblib.load("vectorizer.pkl")
scaler = joblib.load("scaler.pkl")



# Spiking Neural Network Model

class Net(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(917, 128)
        self.fc2 = nn.Linear(128, 2)

        self.beta = 0.9
        self.threshold = 1.0
        self.num_steps = 25

    def forward(self, x):

        batch_size = x.size(0)

        mem1 = torch.zeros(batch_size, 128)
        mem2 = torch.zeros(batch_size, 2)

        spk_sum = torch.zeros(batch_size, 2)

        for _ in range(self.num_steps):

            mem1 = self.beta * mem1 + self.fc1(x)
            spk1 = (mem1 > self.threshold).float()
            mem1 = mem1 * (mem1 <= self.threshold)

            mem2 = self.beta * mem2 + self.fc2(spk1)
            spk2 = (mem2 > self.threshold).float()
            mem2 = mem2 * (mem2 <= self.threshold)

            spk_sum += spk2

        return spk_sum



model = Net()
state = torch.load("snn_spam_model.pth", map_location="cpu")
model.load_state_dict(state, strict=False)
model.eval()


known_brands = [
    "amazon","flipkart","sbi","hdfc","icici","paytm",
    "google","apple","netflix"
]


fraud_keywords = [
"otp","verify","verification","suspended","account blocked",
"urgent","claim reward","lottery","winner","won rs",
"lucky draw","pay now","electricity bill","update kyc",
"bank alert","secure account","login now","link","hacked"
]


def keyword_fraud(msg):

    msg = msg.lower()

    for word in fraud_keywords:
        if word in msg:
            return True

    return False


def extract_urls(text):

    url_pattern = r'(https?://\S+|www\.\S+)'
    return re.findall(url_pattern, text)


def extract_domain(url):

    url = url.lower()
    url = url.replace("https://","")
    url = url.replace("http://","")
    url = url.replace("www.","")

    domain = url.split("/")[0]
    domain = domain.split(".")[0]

    return domain


def brand_typo_detect(url):

    domain = extract_domain(url)

    for brand in known_brands:

        similarity = difflib.SequenceMatcher(None, brand, domain).ratio()

        if similarity > 0.7 and domain != brand:
            return True

    return False


def numeric_spoof(url):

    domain = extract_domain(url)

    if any(char.isdigit() for char in domain):

        for brand in known_brands:

            similarity = difflib.SequenceMatcher(None, brand, domain).ratio()

            if similarity > 0.6:
                return True

    return False


suspicious_domains = [
".xyz",".top",".click",".link",".loan",".gq",".cf",".tk"
]


def suspicious_url(url):

    url = url.lower()

    for d in suspicious_domains:
        if d in url:
            return True

    return False


def domain_entropy(domain):

    prob = [float(domain.count(c)) / len(domain) for c in dict.fromkeys(list(domain))]
    entropy = -sum([p * math.log(p) / math.log(2.0) for p in prob])

    return entropy


def suspicious_entropy(url):

    domain = extract_domain(url)

    ent = domain_entropy(domain)

    if ent > 3.5:
        return True

    return False


def fraud_score(msg, model_prediction):

    score = 0

    urls = extract_urls(msg)

    if keyword_fraud(msg):
        score += 4

    if len(urls) > 0:
        score += 2

    for u in urls:

        if brand_typo_detect(u):
            score += 5

        if numeric_spoof(u):
            score += 5

        if suspicious_url(u):
            score += 4

        if suspicious_entropy(u):
            score += 4

    if model_prediction == 1:
        score += 3

    return score




def predict_sms(text):

    text = text.lower()

    # Vectorize
    X = vectorizer.transform([text]).toarray()

    if X.shape[1] < 917:
        pad = np.zeros((1, 917 - X.shape[1]))
        X = np.concatenate([X, pad], axis=1)

    elif X.shape[1] > 917:
        X = X[:, :917]

    X = scaler.transform(X)

    X_tensor = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        output = model(X_tensor)

    probs = F.softmax(output, dim=1)

    prediction = torch.argmax(probs, dim=1).item()
    confidence = probs.max().item()

    # Fraud rule scoring
    score = fraud_score(text, prediction)

    # Final decision
    if score >= 4:
        prediction = 1

    return prediction, confidence, score





if __name__ == "__main__":

    while True:

        msg = input("\nEnter SMS (type 'quit' to stop): ")

        if msg.lower() == "quit":
            break

        result, confidence, score = predict_sms(msg)

        if result == 1:
            print(f"🚨 Prediction: SPAM (confidence {confidence:.2f}) | Fraud Score: {score}")
        else:
            print(f"✅ Prediction: NOT SPAM (confidence {confidence:.2f}) | Fraud Score: {score}")