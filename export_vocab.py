import joblib
import json

# load vectorizer
vectorizer = joblib.load("vectorizer.pkl")

# get vocabulary
vocab = vectorizer.vocabulary_

# convert numpy int64 → normal int
vocab_fixed = {k: int(v) for k, v in vocab.items()}

# save json
with open("vocab.json", "w") as f:
    json.dump(vocab_fixed, f)

print("vocab.json created successfully")