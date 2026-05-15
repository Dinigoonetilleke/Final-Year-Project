# Smart English Essay Evaluation System (SEES)

An AI-powered web-based platform designed to support English lecturers in evaluating essays, detecting writing errors, generating structured feedback, and creating comprehension-based questions.

---

## Project Overview

The Smart English Essay Evaluation System (SEES) was developed as a Final Year Computing Project for the BSc (Hons) Technology Management degree program. The system integrates Artificial Intelligence (AI), Natural Language Processing (NLP), Optical Character Recognition (OCR), and cloud database technologies to support lecturers in managing essay evaluations efficiently.

The platform allows lecturers to:

* Upload essays and written materials
* Detect grammar, spelling, punctuation, and sentence structure errors
* Generate structured lecturer-friendly feedback
* Process image, PDF, and DOCX submissions using OCR
* Generate comprehension-based questions
* Store reports and generated questions
* Access lecturer and administrator dashboards

---

# Key Features

## AI-Based Essay Evaluation

* Grammar error detection
* Spelling mistake identification
* Punctuation analysis
* Sentence structure evaluation
* Word usage classification
* Structured feedback generation

---

## OCR and File Processing

* Image upload support
* OCR text extraction using EasyOCR
* PDF processing
* DOCX processing
* AI-assisted text refinement

---

## Question Generation

* Multiple Choice Questions (MCQ)
* True or False Questions
* Short Answer Questions
* AI-assisted comprehension activity generation

---

## Dashboard and Report Management

* Lecturer dashboard
* Admin dashboard
* Report storage and retrieval
* Question storage management
* Firebase cloud integration

---

# Technologies Used

| Technology                | Purpose              |
| ------------------------- | -------------------- |
| React.js                  | Frontend Development |
| Flask                     | Backend Development  |
| Python                    | AI Model Development |
| Firebase Firestore        | Cloud Database       |
| EasyOCR                   | OCR Text Extraction  |
| Tailwind CSS              | UI Styling           |
| Scikit-learn              | Machine Learning     |
| Hugging Face Transformers | NLP Processing       |
| GitHub                    | Version Control      |

---

# System Architecture

The system follows a client-server architecture.

## Frontend

* React.js
* Tailwind CSS
* Responsive user interface

## Backend

* Flask REST API
* AI processing
* OCR integration
* Database communication

## Database

* Firebase Firestore
* Cloud-based storage management

## AI Components

* TF-IDF Vectorization
* Logistic Regression Classification
* NLP-based feedback generation

---

# Installation Guide

## Clone Repository

```bash id="lc1itv"
git clone https://github.com/Dinigoonetilleke/Final-Year-Project.git
```

---

# Frontend Setup

```bash id="w7j16y"
cd frontend
npm install
npm run dev
```

---

# Backend Setup

```bash id="vbbk4r"
cd backend
pip install -r requirements.txt
python app.py
```

---

# AI Model Setup

Ensure the trained AI model files are placed inside the AI model directory before starting the backend server.

Example:

* essay_error_model.joblib
* vectorizer.joblib

---

# OCR Setup

EasyOCR dependencies should be installed before running OCR-related functionalities.

```bash id="sl8y7s"
pip install easyocr
```

---

# Firebase Configuration

Create a Firebase project and configure:

* Firestore Database
* Firebase credentials
* Firebase Admin SDK

Add the Firebase configuration file inside the backend directory before running the application.

---

# User Roles

## Lecturer

* Upload essays
* Generate feedback
* Generate questions
* Store and retrieve reports

## Administrator

* Monitor users
* Manage reports
* Access analytics
* Manage system activities

---

# Project Objectives

* Reduce manual essay evaluation workload
* Improve consistency in feedback generation
* Support lecturers using AI-assisted technologies
* Provide structured educational assessment support
* Integrate OCR and NLP technologies into educational evaluation systems

---

# Current Limitations

* Supports only English language essays
* No plagiarism detection
* OCR accuracy depends on image quality
* Limited contextual understanding
* Mobile application version not implemented

---

# Future Improvements

* Plagiarism detection integration
* Multilingual support
* Advanced handwriting recognition
* Deep learning-based contextual analysis
* Mobile application support
* Advanced educational analytics

---

# Author

**Dinithi Goonetilleke**
BSc (Hons) Technology Management - Plymouth University UK
NSBM Green University

---

# License

This project was developed for academic and educational purposes as part of a Final Year Computing Project.


# Installation Guide
# How to Run the System

The project contains two main parts:

1. Frontend (React)
2. Backend (Flask API)

Both servers must run simultaneously.

---

## Step 1 – Clone the Repository

```bash
git clone https://github.com/Dinigoonetilleke/Final-Year-Project.git
```

---

## Step 2 – Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Frontend will run on:

```plaintext
http://localhost:5173
```

---

## Step 3 – Backend Setup

Open a second terminal:

```bash
cd backend
pip install -r requirements.txt
python app.py
```

Backend will run on:

```plaintext
http://127.0.0.1:5000
```

---

## Step 4 – Open the System

Open the browser and navigate to:

```plaintext
http://localhost:5173
```

---

# Step 5 - Database Setup

This system uses Firebase Firestore as the cloud database.

## Required Firestore Collections

Create the following collections in Firebase Firestore:

- users
- essays
- reports
- generated_questions
- passages
- students

## Admin User

Create one admin user inside the `users` collection with the following fields:

| Field | Value |
|---|---|
| email | admin@smartessay.local |
| fullName | System Admin |
| role | admin |
| password | hashed password from backend registration/login setup |
| createdAt | current timestamp |

Lecturer users can be created through the signup page.

## Firebase Credentials

The Firebase service account key is not included in this repository for security reasons.  
To run the system, create a Firebase project, generate a service account key, and place it inside the backend directory as required by the application configuration.

# Demo Login Credentials

## Administrator Account

Email:

```plaintext
admin@smartessay.local
```

Password:

```plaintext
Admin123!
```

---

## Lecturer Account

Users can create lecturer accounts using the signup page.