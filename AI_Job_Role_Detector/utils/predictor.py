import os
import pickle

# Get project root folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Build correct paths
model_path = os.path.join(BASE_DIR, "model", "model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "model", "vectorizer.pkl")

# Load safely
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

model = pickle.load(open(model_path, "rb"))
vectorizer = pickle.load(open(vectorizer_path, "rb"))

def predict_job(text):
    text_vec = vectorizer.transform([text])

    prediction = model.predict(text_vec)[0]
    probabilities = model.predict_proba(text_vec)[0]

    top_indices = probabilities.argsort()[-3:][::-1]
    top_roles = [(model.classes_[i], probabilities[i]) for i in top_indices]

    confidence = max(probabilities)

    return prediction, confidence, top_roles
