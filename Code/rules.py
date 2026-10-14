# rules.py

def fast_filter(text):

    suspicious_words = ["otp","verify","bank","click","urgent","account"]

    text = text.lower()

    if any(word in text for word in suspicious_words):
        return True

    if "http" in text or "www" in text:
        return True

    return False