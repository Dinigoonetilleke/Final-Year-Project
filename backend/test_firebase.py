from firebase_config import db

db.collection("test").document("test001").set({
    "message": "Firebase connected successfully"
})

print("Firebase connected successfully!")