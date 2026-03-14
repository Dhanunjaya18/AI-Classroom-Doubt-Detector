# AI Classroom Doubt Detector

## 📌 Overview

AI Classroom Doubt Detector is a machine learning powered web platform that helps teachers understand which topics students struggle with the most.
Students can submit their academic doubts, and the system automatically groups similar doubts using Natural Language Processing (NLP) and clustering algorithms.

This allows teachers to quickly identify **difficult topics, frequently asked questions, and learning gaps in the classroom**.

---

## 🚀 Key Features

### 👨‍🎓 Student Dashboard

* Submit academic doubts
* View similar doubts suggested by the AI model
* Browse recent doubts from other students
* Search doubts by topic or keywords
* Upvote helpful doubts

### 👨‍🏫 Teacher Dashboard

* View all student doubts
* Identify most asked questions
* Detect difficult topics automatically
* Visual analytics dashboard
* Clustered doubts for better topic analysis

---

## 🧠 Machine Learning Pipeline

1. Student submits doubt text
2. NLP preprocessing cleans the text
3. Text is converted into numerical vectors using **TF-IDF**
4. A **K-Means clustering model** groups similar doubts
5. The system identifies common topics and difficult areas

### NLP Preprocessing Steps

* Lowercase conversion
* Punctuation removal
* Stopword removal
* Lemmatization

### ML Techniques Used

* TF-IDF Vectorization
* K-Means Clustering
* Topic keyword extraction

---

## 🏗️ Tech Stack

### Backend

* Python
* Django

### Machine Learning

* Scikit-learn
* spaCy

### Frontend

* HTML5
* CSS3
* Bootstrap
* JavaScript

### Database

* SQLite

### Visualization

* Chart.js

---

## 🎨 UI Design

The platform uses a **clean light-themed academic interface** with:

* Modern dashboard layout
* Soft card-based UI
* Responsive design
* Interactive charts
* Smooth hover animations

---

## 🗂️ Project Structure

```
AI-Classroom-Doubt-Detector
│
├── doubt_detector/        # Django project
├── doubts/                # Django app for doubts
├── ml_models/             # ML training scripts and saved models
├── templates/             # HTML templates
├── static/                # CSS, JS, assets
├── train_model.py         # ML model training
├── manage.py
├── requirements.txt
└── README.md
```

---



## 📊 Example Use Case

Student submits:

```
"I don't understand recursion in data structures"
```

The system may group it with:

* recursion base case doubts
* recursion stack overflow doubts
* recursion vs iteration questions

Teacher dashboard will show **recursion as a difficult topic**.

---

## 🌟 Future Improvements

* Semantic similarity using transformer embeddings
* Real-time clustering
* Teacher notifications for trending doubts
* Advanced analytics dashboard
* Deployment as a public web application

---

## 👨‍💻 Author

**Dhanunjaya Reddy**

Machine Learning & Software Development Enthusiast

---

## ⭐ If you like this project

Give the repository a **star on GitHub** to support the project.
