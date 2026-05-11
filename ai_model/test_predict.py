import joblib

model = joblib.load("essay_error_model.joblib")

sentences = [
    "Technology have changed the world.",
    "People uses smartphones every day.",
    "Reading books improves vocabulary.",
    "My favorite subjects are math science and english.",
    "I recieved a letter yesterday."
]

for s in sentences:
    prediction = model.predict([s])[0]
    print(s, "=>", prediction)