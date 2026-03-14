# ClassMind AI — Doubt Detector

A full-stack AI-powered classroom platform where students submit academic doubts and machine learning clusters them in real time.

## Setup & Run

```bash
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

Then open http://127.0.0.1:8000

## Usage
1. Sign up as a **Student** to submit doubts
2. Sign up as a **Teacher** to view analytics and clusters
3. Students submit doubts → ML auto-detects subject and finds similar questions
4. Teacher clicks "Re-run Clustering" to group doubts with K-Means + TF-IDF

## ML Pipeline
- **Preprocessing**: lowercase, remove stopwords, punctuation
- **Vectorization**: TF-IDF (max 500 features, n-grams 1-2)
- **Clustering**: K-Means with optimal K chosen via silhouette score
- **Similarity**: Cosine similarity for semantic search

## Stack
- Django 4.2, SQLite, scikit-learn, Bootstrap 5, Chart.js
