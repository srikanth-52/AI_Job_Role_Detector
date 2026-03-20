import pickle

# Load saved model
model = pickle.load(open("model/model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

def predict_job(text):
    text_vec = vectorizer.transform([text])

    prediction = model.predict(text_vec)[0]
    probabilities = model.predict_proba(text_vec)[0]

    # Top 3 roles
    top_indices = probabilities.argsort()[-3:][::-1]
    top_roles = [(model.classes_[i], probabilities[i]) for i in top_indices]

    confidence = max(probabilities)

    return prediction, confidence, top_roles