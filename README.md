📚 AI Classroom Doubt Classifier

An intelligent NLP-based system that automatically classifies and clusters student doubts to help educators identify frequently asked questions and reduce duplicate queries.

🚀 Overview

In large classrooms or online learning platforms, students often ask similar or repetitive doubts. This project leverages Natural Language Processing (NLP) techniques to:

Detect similar questions
Cluster related doubts
Provide insights to teachers
Improve classroom efficiency
🧠 Features
🔍 Doubt Similarity Detection using TF-IDF + Cosine Similarity
📊 Clustering of Questions to group similar doubts
👨‍🎓 Student Dashboard for doubt submission & upvoting
👩‍🏫 Teacher Dashboard for analytics & insights
📈 Visualization of frequently asked topics
⚡ Real-time processing of student queries
🛠️ Tech Stack
Backend: Django
Machine Learning: Scikit-learn
NLP Techniques: TF-IDF, Cosine Similarity
Database: SQLite
Frontend: HTML, CSS, JavaScript
Visualization: Matplotlib / Seaborn
⚙️ How It Works
Student submits a doubt
System converts text → TF-IDF vector
Computes similarity with existing doubts using Cosine Similarity
If similar doubt exists → grouped together
Else → stored as a new query
Teacher dashboard shows:
Most frequent doubts
Clustered topics
Engagement metrics
📂 Project Structure
AI-Doubt-Classifier/
│── classifier/          # ML logic (TF-IDF, similarity)
│── dashboard/           # Teacher & student dashboards
│── templates/           # HTML templates
│── static/              # CSS, JS files
│── db.sqlite3           # Database
│── manage.py
│── requirements.txt
▶️ Installation & Setup
1. Clone the repository
git clone https://github.com/your-username/ai-doubt-classifier.git
cd ai-doubt-classifier
2. Create virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
3. Install dependencies
pip install -r requirements.txt
4. Run the server
python manage.py runserver
5. Open in browser
http://127.0.0.1:8000/
📊 Example Use Case
100 students ask doubts
System groups them into ~10 clusters
Teacher answers once → benefits all students

👉 Saves time and improves teaching efficiency

🔥 Future Improvements
🤖 Integrate LLMs for auto-answer suggestions
🧠 Use advanced embeddings (BERT / Sentence Transformers)
☁️ Deploy on cloud (AWS / GCP)
📱 Mobile-friendly UI
📌 Key Learnings
Practical implementation of NLP pipelines
Working with text vectorization & similarity metrics
Building full-stack ML applications
Designing real-world problem-solving systems
🤝 Contributing

Contributions are welcome!
Feel free to fork this repo and submit a pull request.

📧 Contact
Name: Dhanunjaya Reddy
Email: dhanunjaya0340@gmail.com
GitHub: https://github.com/Dhanunjaya18

I
