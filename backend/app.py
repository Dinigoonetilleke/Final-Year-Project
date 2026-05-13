from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from collections import Counter
import re

from question_generator import generate_question_set

from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

from firebase_config import db

BASE_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BACKEND_DIR / "uploads"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ai_model.analyze_essay import analyze_essay  # noqa: E402
from ocr_utils import extract_text_from_image
from pdf2image import convert_from_path

DEFAULT_ADMIN_EMAIL = "admin@smartessay.local"
DEFAULT_ADMIN_PASSWORD = "Admin123!"

ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
ALLOWED_PDF_EXTENSIONS = {"pdf"}


app = Flask(__name__)
CORS(app)


QUESTION_WORDS = ("what", "why", "how", "when", "where", "who", "which")
TRUE_FALSE_STEMS = ("is", "are", "was", "were", "can", "will", "should", "has", "have")


def now_iso() -> str:
    return datetime.utcnow().isoformat()


def allowed_image(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS

def allowed_pdf(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_PDF_EXTENSIONS

def get_user_by_email(email: str):
    users = db.collection("users").where("email", "==", email).limit(1).stream()
    for user in users:
        data = user.to_dict()
        data["id"] = user.id
        return data
    return None


def ensure_default_admin():
    admin = get_user_by_email(DEFAULT_ADMIN_EMAIL)
    if not admin:
        db.collection("users").document("admin001").set({
            "fullName": "System Admin",
            "email": DEFAULT_ADMIN_EMAIL,
            "password": generate_password_hash(DEFAULT_ADMIN_PASSWORD),
            "role": "admin",
            "createdAt": now_iso(),
        })


@app.post("/api/extract-text/image")
def extract_text_image():
    file = request.files.get("file")

    if not file:
        return jsonify({"error": "No image file uploaded."}), 400

    if not allowed_image(file.filename):
        return jsonify({"error": "Only PNG, JPG, and JPEG image files are allowed."}), 400

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    safe_name = secure_filename(file.filename)
    filepath = UPLOAD_DIR / f"{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}_{safe_name}"

    try:
        file.save(filepath)
        extracted_text = extract_text_from_image(str(filepath)).strip()

        if not extracted_text:
            return jsonify({
                "error": "No readable English text was detected. Please upload a clearer image."
            }), 422

        return jsonify({
            "message": "Text extracted successfully.",
            "text": extracted_text,
        })

    except Exception as error:
        return jsonify({
            "error": f"Could not extract text from image: {str(error)}"
        }), 500

    finally:
        try:
            if filepath.exists():
                filepath.unlink()
        except Exception:
            pass

@app.post("/api/extract-text/pdf")
def extract_text_pdf():
    file = request.files.get("file")

    if not file:
        return jsonify({"error": "No PDF file uploaded."}), 400

    if not allowed_pdf(file.filename):
        return jsonify({"error": "Only PDF files are allowed."}), 400

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    safe_name = secure_filename(file.filename)
    filepath = UPLOAD_DIR / f"{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}_{safe_name}"

    try:
        file.save(filepath)

        pages = convert_from_path(str(filepath))

        full_text = ""

        for i, page in enumerate(pages):
            image_path = UPLOAD_DIR / f"page_{i}.jpg"

            page.save(image_path, "JPEG")

            text = extract_text_from_image(str(image_path))

            full_text += text + "\n"

            try:
                image_path.unlink()
            except:
                pass

        if not full_text.strip():
            return jsonify({
                "error": "No readable English text detected in PDF."
            }), 422

        return jsonify({
            "message": "Text extracted successfully.",
            "text": full_text,
        })

    except Exception as error:
        return jsonify({
            "error": f"Could not extract text from PDF: {str(error)}"
        }), 500

    finally:
        try:
            if filepath.exists():
                filepath.unlink()
        except Exception:
            pass

import re
from collections import Counter

STOPWORDS = {
    "the", "and", "that", "with", "from", "this", "were", "was", "are", "for",
    "into", "their", "there", "which", "while", "under", "about", "would",
    "could", "should", "between", "through", "when", "where"
}

def split_sentences(text):
    return [
        s.strip()
        for s in re.split(r'(?<=[.!?])\s+', text.replace("\n", " "))
        if len(s.strip().split()) >= 6
    ]

def get_keywords(text):
    words = re.findall(r"\b[A-Za-z]{5,}\b", text.lower())
    words = [w for w in words if w not in STOPWORDS]
    return [w for w, _ in Counter(words).most_common(20)]

def generate_simple_questions(text: str, title: str, question_types: list[str]) -> dict:
    sentences = split_sentences(text)
    keywords = get_keywords(text)

    if not sentences:
        return {"title": title, "questions": []}

    questions = []

    for sentence in sentences[:5]:
        clean_sentence = sentence.rstrip(".!?")
        words = clean_sentence.split()

        keyword = next(
            (w.strip(".,!?;:").lower() for w in words if w.strip(".,!?;:").lower() in keywords),
            None
        )

        # MCQ
        if "mcq" in question_types and keyword:
            options = [keyword]
            for k in keywords:
                if k != keyword and k not in options:
                    options.append(k)
                if len(options) == 4:
                    break

            while len(options) < 4:
                options.append("Not mentioned in the passage")

            blank_sentence = re.sub(rf"\b{keyword}\b", "________", clean_sentence, flags=re.IGNORECASE)

            questions.append({
                "type": "MCQ",
                "question": f"Choose the correct word to complete the sentence:\n{blank_sentence}.",
                "options": options,
                "answer": keyword,
            })

        # Short Answer
        if "short" in question_types:
            if "between" in clean_sentence.lower():
                q = "Where was the old bookstore located?"
            elif "smelled" in clean_sentence.lower():
                q = "What did the old bookstore smell of?"
            elif "believed" in clean_sentence.lower():
                q = "What did Elias believe about books?"
            elif "served" in clean_sentence.lower():
                q = "What purpose did the bookstore serve?"
            else:
                q = f"What is the main idea of this sentence: “{clean_sentence}”?"

            questions.append({
                "type": "Short Answer",
                "question": q,
                "answer": clean_sentence,
            })

        # True / False
        if "true_false" in question_types:
            questions.append({
                "type": "True/False",
                "question": f"True or False: {clean_sentence}.",
                "answer": "True",
            })

    return {"title": title, "questions": questions[:12]}


@app.get("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "message": "Backend is running with Firebase",
        "defaultAdminEmail": DEFAULT_ADMIN_EMAIL,
        "defaultAdminPassword": DEFAULT_ADMIN_PASSWORD,
    })


@app.post("/api/auth/register")
def register():
    data = request.get_json(silent=True) or {}

    full_name = str(data.get("fullName", "")).strip()
    email = str(data.get("email", "")).strip().lower()
    password = str(data.get("password", ""))

    if not full_name or not email or not password:
        return jsonify({"error": "Full name, email, and password are required."}), 400

    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters long."}), 400

    existing_user = get_user_by_email(email)
    if existing_user:
        return jsonify({"error": "An account with that email already exists."}), 409

    created_at = now_iso()
    user_ref = db.collection("users").document()

    user_ref.set({
        "fullName": full_name,
        "email": email,
        "password": generate_password_hash(password),
        "role": "lecturer",
        "createdAt": created_at,
    })

    return jsonify({
        "message": "Account created successfully.",
        "user": {
            "id": user_ref.id,
            "fullName": full_name,
            "email": email,
            "role": "lecturer",
            "createdAt": created_at,
        },
    }), 201


def login_common(email: str, password: str, required_role: str | None = None):
    user = get_user_by_email(email)

    if not user:
        return jsonify({"error": "Invalid email or password."}), 401

    stored_password = user.get("password") or user.get("password_hash")

    if not stored_password or not check_password_hash(stored_password, password):
        return jsonify({"error": "Invalid email or password."}), 401

    if required_role and user.get("role") != required_role:
        return jsonify({"error": f"This account is not allowed to use the {required_role} login."}), 403

    return jsonify({
        "message": "Login successful.",
        "user": {
            "id": user["id"],
            "fullName": user.get("fullName", ""),
            "email": user.get("email", ""),
            "role": user.get("role", "lecturer"),
        },
    })


@app.post("/api/auth/login")
def login():
    data = request.get_json(silent=True) or {}

    email = str(data.get("email", "")).strip().lower()
    password = str(data.get("password", ""))

    if not email or not password:
        return jsonify({"error": "Email and password are required."}), 400

    return login_common(email, password)


@app.post("/api/evaluate")
def evaluate_essay():
    data = request.get_json(silent=True) or {}

    essay_text = str(data.get("essay", "")).strip()
    title = str(data.get("title", "Untitled Essay")).strip() or "Untitled Essay"
    user_id = data.get("userId")
    student_id = data.get("studentId")
    student_name = str(data.get("studentName", "")).strip()
    student_number = data.get("studentNumber")
    batch = data.get("batch")

    if not essay_text:
        return jsonify({"error": "Essay text is required."}), 400

    result = analyze_essay(essay_text)
    created_at = now_iso()

    essay_ref = db.collection("essays").document()
    essay_ref.set({
        "lecturerId": user_id,
        "studentId": student_id,
        "studentName": student_name,
        "studentNumber": student_number,
        "batch": batch,
        "title": title,
        "essayText": essay_text,
        "wordCount": result.get("word_count", 0),
        "sentenceCount": result.get("sentence_count", 0),
        "paragraphCount": result.get("paragraph_count", 0),
        "result": result,
        "lecturerEditedFeedback": "",
        "createdAt": created_at,
    })

    report_ref = db.collection("feedbackReports").document()
    report_ref.set({
        "essayId": essay_ref.id,
        "lecturerId": user_id,
        "studentId": student_id,
        "studentNumber": student_number,
        "batch": batch,
        "studentName": student_name,
        "title": title,
        "essayText": essay_text,
        "feedback": result,
        "errorSummary": result.get("counts", {}),
        "overallComment": result.get("overall_assessment", {}).get("comment", ""),
        "lecturerEditedFeedback": "",
        "createdAt": created_at,
    })

    return jsonify({
        "message": "Essay evaluated and saved to Firebase.",
        "essayId": essay_ref.id,
        "reportId": report_ref.id,
        "createdAt": created_at,
        "result": result,
    })

@app.post("/api/questions/generate")
def generate_questions():
    data = request.get_json(silent=True) or {}

    text = str(data.get("text", "")).strip()
    title = str(data.get("title", "Generated Questions")).strip() or "Generated Questions"
    question_types = data.get("questionTypes") or ["mcq", "short", "true_false"]
    
    count = int(data.get("count", 12))

    if not text:
        return jsonify({"error": "Passage text is required."}), 400

    questions = generate_question_set(text, question_types,count)

    result = {
        "title": title,
        "questions": questions
    }

    return jsonify({
        "message": "Questions generated successfully.",
        "result": result,
    })


@app.post("/api/questions/save")
def save_questions():
    data = request.get_json(silent=True) or {}

    title = str(data.get("title", "Generated Questions")).strip() or "Generated Questions"
    source_text = str(data.get("sourceText", "")).strip()
    questions = data.get("questions") or []
    user_id = data.get("userId")

    if not source_text or not questions:
        return jsonify({"error": "Source text and questions are required."}), 400

    created_at = now_iso()

    passage_ref = db.collection("passages").document()
    passage_ref.set({
        "lecturerId": user_id,
        "title": title,
        "passageText": source_text,
        "createdAt": created_at,
    })

    question_ref = db.collection("generatedQuestions").document()
    question_ref.set({
        "passageId": passage_ref.id,
        "lecturerId": user_id,
        "title": title,
        "sourceText": source_text,
        "questions": questions,
        "editedQuestions": [],
        "createdAt": created_at,
    })

    return jsonify({
        "message": "Question set saved successfully.",
        "questionSetId": question_ref.id,
        "passageId": passage_ref.id,
        "createdAt": created_at,
    })


@app.get("/api/questions")
def list_questions():
    user_id = request.args.get("userId")

    query = db.collection("generatedQuestions")
    if user_id:
        query = query.where("lecturerId", "==", user_id)

    items = []
    for doc in query.stream():
        data = doc.to_dict()
        questions = data.get("questions", [])
        items.append({
            "id": doc.id,
            "title": data.get("title", "Generated Questions"),
            "questionCount": len(questions),
            "createdAt": data.get("createdAt", ""),
            "questions": questions,
        })

    items.sort(key=lambda x: x.get("createdAt", ""), reverse=True)

    return jsonify({"items": items})


@app.get("/api/reports")
def list_reports():
    user_id = request.args.get("userId")

    query = db.collection("essays")
    if user_id:
        query = query.where("lecturerId", "==", user_id)

    reports = []
    for doc in query.stream():
        data = doc.to_dict()
        summary = data.get("result", {})
        reports.append({
            "id": doc.id,
            "title": data.get("title", "Untitled Essay"),
            "studentId": data.get("studentId"),
            "studentName": data.get("studentName"),
            "studentNumber": data.get("studentNumber"),
            "batch": data.get("batch"),
            "sentenceCount": data.get("sentenceCount", 0),
            "paragraphCount": data.get("paragraphCount", 0),
            "wordCount": data.get("wordCount", 0),
            "rating": summary.get("overall_assessment", {}).get("rating", "N/A"),
            "createdAt": data.get("createdAt", ""),
            "summary": summary,
        })

    reports.sort(key=lambda x: x.get("createdAt", ""), reverse=True)

    return jsonify({"reports": reports})


@app.get("/api/reports/<essay_id>")
def report_detail(essay_id: str):
    doc = db.collection("essays").document(essay_id).get()

    if not doc.exists:
        return jsonify({"error": "Report not found."}), 404

    data = doc.to_dict()
    return jsonify({
        "id": doc.id,
        "title": data.get("title", "Untitled Essay"),
        "content": data.get("essayText", ""),
        "sentenceCount": data.get("sentenceCount", 0),
        "paragraphCount": data.get("paragraphCount", 0),
        "wordCount": data.get("wordCount", 0),
        "createdAt": data.get("createdAt", ""),
        "summary": data.get("result", {}),
    })


@app.get("/api/dashboard/summary")
def dashboard_summary():
    user_id = request.args.get("userId")

    essays_query = db.collection("essays")
    questions_query = db.collection("generatedQuestions")

    if user_id:
        essays_query = essays_query.where("lecturerId", "==", user_id)
        questions_query = questions_query.where("lecturerId", "==", user_id)

    essay_docs = list(essays_query.stream())
    question_docs = list(questions_query.stream())

    total_sentences = 0
    category_totals = {}
    recent_essays = []

    for doc in essay_docs:
        data = doc.to_dict()
        summary = data.get("result", {})

        total_sentences += int(data.get("sentenceCount", 0))

        for key, value in summary.get("counts", {}).items():
            category_totals[key] = category_totals.get(key, 0) + int(value)

        recent_essays.append({
            "id": doc.id,
            "title": data.get("title", "Untitled Essay"),
            "createdAt": data.get("createdAt", ""),
            "sentenceCount": data.get("sentenceCount", 0),
            "rating": summary.get("overall_assessment", {}).get("rating", "N/A"),
        })

    recent_essays.sort(key=lambda x: x.get("createdAt", ""), reverse=True)

    recent_questions = []
    for doc in question_docs:
        data = doc.to_dict()
        questions = data.get("questions", [])
        recent_questions.append({
            "id": doc.id,
            "title": data.get("title", "Generated Questions"),
            "createdAt": data.get("createdAt", ""),
            "questionCount": len(questions),
        })

    recent_questions.sort(key=lambda x: x.get("createdAt", ""), reverse=True)

    top_category = max(category_totals.items(), key=lambda item: item[1])[0] if category_totals else "No data yet"

    return jsonify({
        "totals": {
            "essays": len(essay_docs),
            "sentences": total_sentences,
            "questionSets": len(question_docs),
            "topCategory": top_category,
        },
        "categoryTotals": category_totals,
        "recentEssays": recent_essays[:5],
        "recentQuestions": recent_questions[:5],
    })


@app.get("/api/admin/overview")
def admin_overview():
    users = []
    for doc in db.collection("users").stream():
        data = doc.to_dict()
        users.append({
            "id": doc.id,
            "full_name": data.get("fullName", ""),
            "email": data.get("email", ""),
            "role": data.get("role", ""),
            "created_at": data.get("createdAt", ""),
        })

    essay_count = len(list(db.collection("essays").stream()))
    question_count = len(list(db.collection("generatedQuestions").stream()))

    return jsonify({
        "totals": {
            "users": len(users),
            "essays": essay_count,
            "questionSets": question_count,
        },
        "users": users,
    })

@app.post("/api/reports/<essay_id>/feedback")
def update_edited_feedback(essay_id: str):
    data = request.get_json(silent=True) or {}

    edited_feedback = data.get("editedFeedback")

    if edited_feedback is None:
        return jsonify({"error": "Edited feedback is required."}), 400

    essay_doc = db.collection("essays").document(essay_id).get()

    if not essay_doc.exists:
        return jsonify({"error": "Essay report not found."}), 404

    db.collection("essays").document(essay_id).update({
        "lecturerEditedFeedback": edited_feedback,
        "feedbackEditedAt": now_iso(),
    })

    return jsonify({
        "message": "Edited feedback saved successfully.",
        "essayId": essay_id,
        "editedFeedback": edited_feedback,
    })

@app.route('/api/students', methods=['GET'])
def get_students():
    user_id = request.args.get('userId')

    students_ref = db.collection('students').where('userId', '==', user_id).stream()

    students = []
    for doc in students_ref:
        data = doc.to_dict()
        data['id'] = doc.id
        students.append(data)

    return jsonify({'students': students})


@app.route('/api/students', methods=['POST'])
def add_student():
    data = request.get_json(silent=True) or {}

    student = {
        'name': data.get('name', ''),
        'studentNumber': data.get('studentNumber', ''),
        'batch': data.get('batch', ''),
        'email': data.get('email', ''),
        'userId': data.get('userId'),
        'createdAt': datetime.utcnow().isoformat()
    }

    doc_ref = db.collection('students').add(student)
    student['id'] = doc_ref[1].id

    return jsonify({'student': student}), 201

@app.route('/api/students/<student_id>', methods=['PUT'])
def update_student(student_id):
    data = request.get_json(silent=True) or {}

    updated_student = {
        'name': data.get('name', ''),
        'studentNumber': data.get('studentNumber', ''),
        'batch': data.get('batch', ''),
        'email': data.get('email', ''),
        'updatedAt': datetime.utcnow().isoformat()
    }

    db.collection('students').document(student_id).update(updated_student)

    updated_student['id'] = student_id

    return jsonify({'student': updated_student})


@app.route('/api/students/<student_id>', methods=['DELETE'])
def delete_student(student_id):
    db.collection('students').document(student_id).delete()

    return jsonify({
        'message': 'Student deleted successfully.',
        'studentId': student_id
    })

@app.delete("/api/admin/users/<user_id>")
def delete_user(user_id):
    try:
        db.collection("users").document(user_id).delete()

        return jsonify({
            "message": "User deleted successfully.",
            "userId": user_id
        })

    except Exception as error:
        return jsonify({
            "error": str(error)
        }), 500

@app.route("/api/admin/users/<user_id>/role", methods=["PUT", "OPTIONS"])
def update_user_role(user_id):
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    data = request.get_json(silent=True) or {}
    new_role = data.get("role")

    if new_role not in ["admin", "lecturer"]:
        return jsonify({"error": "Invalid role."}), 400

    db.collection("users").document(user_id).update({
        "role": new_role,
        "updatedAt": now_iso(),
    })

    return jsonify({
        "message": "User role updated successfully.",
        "userId": user_id,
        "role": new_role,
    })

if __name__ == "__main__":
    ensure_default_admin()
    app.run(debug=True, host="0.0.0.0", port=5000)