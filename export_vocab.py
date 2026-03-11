import joblib
import json

# Load vectorizer
vectorizer = joblib.load("vectorizer.pkl")

# Convert numpy int64 → normal int
vocab = {word: int(index) for word, index in vectorizer.vocabulary_.items()}

# Save vocab
with open("vocab.json", "w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False, indent=2)

print("✅ vocab.json exported successfully")
print("Vocabulary size:", len(vocab))