
# 🎬 Hybrid Movie Recommendation System (MovieLens 25M)

A modular and production-ready movie recommendation system developed during a machine learning internship. This project combines **content-based filtering** and **collaborative filtering** into a powerful **hybrid model** for personalized movie recommendations. It includes both a **Streamlit web app** and a **CLI tool**, with real-time poster integration using the **TMDB API**.

---

## 🔍 Project Highlights

- **Hybrid Recommendation Engine**
  - Merges genre similarity with predicted user ratings.
  - Designed to handle both new users and personalized recommendations.

- **Content-Based Filtering**
  - Uses TF-IDF vectorization on movie genres.
  - Computes nearest neighbors based on cosine similarity.

- **Collaborative Filtering**
  - Implements SVD (Singular Value Decomposition) from the Surprise library.
  - Trained on 25M user ratings from MovieLens.

- **Streamlit Web Interface**
  - Fully interactive movie search and recommendation page.
  - Integrates movie posters via the TMDB API.

- **Command-Line Interface (CLI)**
  - Simple CLI version for testing or quick evaluation.

- **Cached Models for Performance**
  - Includes pre-trained SVD model and vectorized genre models.

---

## 🌐 Technologies Used

| Component               | Technology                              |
|-------------------------|------------------------------------------|
| Content Filtering       | `scikit-learn`, `TF-IDF`, `NearestNeighbors` |
| Collaborative Filtering | `Surprise` SVD                          |
| UI                      | `Streamlit`                             |
| Poster Integration      | `TMDB API`                              |
| Model Caching           | `joblib`                                |
| Data Manipulation       | `pandas`                                |

---

## 📚 Dataset

- **MovieLens 25M** – contains 25 million ratings across 62,000 movies.
- Source: [GroupLens](https://grouplens.org/datasets/movielens/25m/)
- Includes: movie titles, genres, user ratings, tags, and metadata.

---

## 🧠 How It Works

1. User enters a movie they like + user ID.
2. The system finds similar movies based on genres (TF-IDF).
3. It predicts user ratings using a trained SVD model.
4. It ranks movies based on a hybrid score combining both.
5. Posters are fetched live from TMDB API.

```python
hybrid_score = 0.5 * genre_similarity + 0.5 * (predicted_rating / 5.0)
```

---

## 📁 Project Structure

```
Movie_Recommend/
├── recommend_system/
│   ├── genre_recommender.py
│   ├── rating_recommender.py
│   └── hybrid_recommender.py
│
├── models/
│   └── svd_sampled_25m.pkl
│
├── cleaned/
│   ├── tfidf_matrix.pkl
│   └── nn_model.pkl
│
├── data/ml-25m/
│   ├── movies.csv
│   ├── ratings.csv
│   └── other MovieLens CSVs...
│
├── app.py                  # Streamlit interface
├── main.py                 # CLI interface
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker build file
└── README.md               # This file
```

---

## ⚙️ Local Setup (Manual)

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Movie_Recommend.git
cd Movie_Recommend
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the App

#### Web Interface (Streamlit)

```bash
streamlit run app.py
```

#### Command-Line Interface (CLI)

```bash
python main.py
```

---

## 🐳 Run with Docker (Recommended)

### 1. Make sure Docker is installed & running

Get it from: [https://www.docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop)

### 2. Build the Docker Image

```bash
docker build -t movie-recommend-app .
```

### 3. Run the App

```bash
docker run --rm -p 8501:8501 movie-recommend-app
```

Open in browser: [http://localhost:8501](http://localhost:8501)

---

## 🔮 Future Improvements

- Dynamic weighting for hybrid model
- Personal user profiles with login system
- Use metadata like director, cast, and tags
- Deploy to cloud (Streamlit Community Cloud, Render, etc.)
- Add model evaluation metrics (MAP@k, NDCG)

---

## 👤 Author

**Omar Idriss**  
Machine Learning Intern  
GitHub: [OmarIdriss](https://github.com/OmarIdriss)

---

## 📄 Acknowledgements

- [MovieLens Dataset](https://grouplens.org/datasets/movielens/25m/)
- [TMDB API](https://www.themoviedb.org/documentation/api)
- [Surprise Library](http://surpriselib.com/)

---

*This project was developed as part of a hands-on ML internship focused on deploying recommendation systems using real-world data and modern Python tools.*
