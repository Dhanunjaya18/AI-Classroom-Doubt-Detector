"""
ML Pipeline for Doubt Clustering using TF-IDF + K-Means
"""
import re
import json
import pickle
import os
import numpy as np
from collections import Counter

# NLP preprocessing using basic techniques (no spaCy model download needed)
STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
    'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
    'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
    'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
    'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
    'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
    'all', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
    'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
    'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll',
    'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't",
    'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't",
    'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',
    "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't",
    'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't",
    'please', 'help', 'understand', 'explain', 'know', 'want', 'need', 'get',
    'also', 'like', 'would', 'could', 'use', 'way', 'work', 'make', 'find'
}

TOPIC_KEYWORDS = {
    'Mathematics': ['math', 'calculus', 'algebra', 'geometry', 'equation', 'matrix', 'integral', 
                    'derivative', 'probability', 'statistics', 'theorem', 'proof', 'vector', 
                    'polynomial', 'trigonometry', 'differentiation', 'integration', 'function'],
    'Physics': ['physics', 'force', 'energy', 'velocity', 'acceleration', 'momentum', 'gravity',
                'quantum', 'wave', 'electromagnetic', 'thermodynamics', 'optics', 'nuclear',
                'charge', 'current', 'resistance', 'magnetic', 'electric', 'photon'],
    'Chemistry': ['chemistry', 'molecule', 'atom', 'reaction', 'element', 'compound', 'bond',
                  'acid', 'base', 'organic', 'inorganic', 'periodic', 'electron', 'proton',
                  'neutron', 'oxidation', 'reduction', 'catalyst', 'equilibrium'],
    'Computer Science': ['programming', 'algorithm', 'data structure', 'code', 'function', 
                         'recursion', 'sorting', 'graph', 'tree', 'array', 'pointer', 'memory',
                         'complexity', 'python', 'java', 'database', 'sql', 'network', 'os'],
    'Biology': ['biology', 'cell', 'dna', 'protein', 'evolution', 'genetics', 'organism',
                'photosynthesis', 'respiration', 'enzyme', 'hormone', 'neuron', 'ecosystem',
                'mitosis', 'meiosis', 'chromosome', 'bacteria', 'virus'],
    'History': ['history', 'war', 'revolution', 'empire', 'century', 'civilization', 'dynasty',
                'colonialism', 'independence', 'treaty', 'political', 'ancient', 'medieval'],
    'Literature': ['literature', 'novel', 'poem', 'author', 'character', 'theme', 'plot',
                   'metaphor', 'symbolism', 'narrative', 'genre', 'essay', 'rhetoric'],
    'Economics': ['economics', 'market', 'supply', 'demand', 'gdp', 'inflation', 'trade',
                  'monetary', 'fiscal', 'microeconomics', 'macroeconomics', 'price'],
}


def preprocess_text(text):
    """Clean and preprocess text"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    return ' '.join(tokens)


def detect_subject(text):
    """Detect subject area from doubt text"""
    text_lower = text.lower()
    scores = {}
    for subject, keywords in TOPIC_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[subject] = score
    if scores:
        return max(scores, key=scores.get)
    return 'General'


def get_top_keywords(texts, n=5):
    """Extract top keywords from a list of texts"""
    all_words = []
    for text in texts:
        processed = preprocess_text(text)
        all_words.extend(processed.split())
    counter = Counter(all_words)
    return [word for word, _ in counter.most_common(n)]


def run_clustering(doubts_queryset):
    """
    Run TF-IDF + K-Means clustering on doubts.
    Returns cluster assignments and metadata.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import normalize

    texts = [d.text for d in doubts_queryset]
    ids = [d.id for d in doubts_queryset]

    if len(texts) < 2:
        return {'status': 'insufficient_data', 'min_required': 2}

    # Preprocess
    processed = [preprocess_text(t) for t in texts]
    
    # Filter empty
    valid_indices = [i for i, p in enumerate(processed) if len(p.strip()) > 0]
    if len(valid_indices) < 2:
        return {'status': 'insufficient_data', 'min_required': 2}

    valid_texts = [processed[i] for i in valid_indices]
    valid_ids = [ids[i] for i in valid_indices]
    original_texts = [texts[i] for i in valid_indices]

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(
        max_features=500,
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True
    )
    X = vectorizer.fit_transform(valid_texts)
    X_normalized = normalize(X)

    # Determine optimal K
    n_samples = len(valid_texts)
    max_k = min(8, n_samples - 1)
    
    if max_k < 2:
        # Only 1 cluster possible
        for d in doubts_queryset:
            d.cluster_id = 0
            d.cluster_label = detect_subject(d.text)
            d.subject = detect_subject(d.text)
            d.save()
        return {'status': 'success', 'n_clusters': 1, 'silhouette': 1.0}

    # Find best K using silhouette score
    best_k = 2
    best_score = -1
    best_labels = None

    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        labels = km.fit_predict(X_normalized)
        if len(set(labels)) < 2:
            continue
        try:
            score = silhouette_score(X_normalized, labels)
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels
        except Exception:
            continue

    if best_labels is None:
        best_labels = [0] * len(valid_texts)
        best_k = 1
        best_score = 0.0

    # Assign cluster labels using subject detection
    cluster_subjects = {}
    for i, label in enumerate(best_labels):
        if label not in cluster_subjects:
            cluster_subjects[label] = []
        cluster_subjects[label].append(original_texts[i])

    cluster_names = {}
    for cluster_id, cluster_texts in cluster_subjects.items():
        subjects = [detect_subject(t) for t in cluster_texts]
        subject_counts = Counter(subjects)
        primary_subject = subject_counts.most_common(1)[0][0]
        top_kws = get_top_keywords(cluster_texts, 3)
        cluster_names[cluster_id] = f"{primary_subject}: {', '.join(top_kws[:2])}" if top_kws else primary_subject

    # Update doubts in DB
    id_to_label = {valid_ids[i]: int(best_labels[i]) for i in range(len(valid_ids))}
    
    for d in doubts_queryset:
        if d.id in id_to_label:
            cid = id_to_label[d.id]
            d.cluster_id = cid
            d.cluster_label = cluster_names.get(cid, f'Topic {cid}')
            d.subject = detect_subject(d.text)
            d.save()

    return {
        'status': 'success',
        'n_clusters': best_k,
        'silhouette_score': round(best_score, 4),
        'cluster_names': cluster_names,
    }


def find_similar_doubts(query_text, doubts_queryset, top_n=5):
    """Find similar doubts using TF-IDF cosine similarity"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    all_texts = [d.text for d in doubts_queryset]
    if not all_texts:
        return []

    all_processed = [preprocess_text(t) for t in all_texts]
    query_processed = preprocess_text(query_text)

    try:
        vectorizer = TfidfVectorizer(max_features=300, ngram_range=(1, 2), min_df=1)
        corpus = all_processed + [query_processed]
        tfidf_matrix = vectorizer.fit_transform(corpus)
        query_vec = tfidf_matrix[-1]
        doc_vecs = tfidf_matrix[:-1]
        similarities = cosine_similarity(query_vec, doc_vecs)[0]
        top_indices = similarities.argsort()[::-1][:top_n]
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.05:
                results.append({
                    'doubt': list(doubts_queryset)[idx],
                    'similarity': round(float(similarities[idx]), 3)
                })
        return results
    except Exception:
        return []
