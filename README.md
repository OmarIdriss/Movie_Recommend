# 🎮 Hybrid Movie Recommendation System (MovieLens 25M)

A modular and practical movie recommendation system built using Python, designed during a machine learning internship. It combines **content-based filtering** and **collaborative filtering** into a **hybrid model** that personalizes movie suggestions for users. The system supports both a **Streamlit web app** and a **CLI tool** and features dynamic movie poster integration via the **TMDB API**.

---

## 🔍 Project Highlights

- **Hybrid Recommendation Engine**

  - Combines genre similarity and user rating predictions
  - Balances both cold-start handling and user-specific preferences

- **Content-Based Filtering**

  - TF-IDF vectorization of genres
  - Nearest Neighbors search to find similar movies

- **Collaborative Filtering**

  - Trained using the Surprise SVD algorithm on the MovieLens 25M dataset
  - Learns latent patterns in user preferences

- **Streamlit Interface**

  - Interactive movie search and recommendation tool
  - Visual output with real-time TMDB movie poster retrieval

- **CLI Interface**

  - Fast, script-based evaluation for testing and training

- **Cached Models for Speed**

  - Preprocessed TF-IDF and Nearest Neighbors models
  - Trained collaborative filtering model stored with `joblib`

---

## 🌐 Technologies Used

| Component               | Technology                              |
| ----------------------- | --------------------------------------- |
| Content Filtering       | `sklearn`, `TF-IDF`, `NearestNeighbors` |
| Collaborative Filtering | `Surprise` SVD                          |
| UI                      | `Streamlit`                             |
| Poster Integration      | `TMDB API`                              |
| Model Caching           | `joblib`                                |
| Data Manipulation       | `pandas`                                |

---

## 📆 Dataset

- **MovieLens 25M** (25 million ratings across 62,000 movies)
- Source: [GroupLens](https://grouplens.org/datasets/movielens/25m/)
- Data includes: movie titles, genres, user ratings

---

## 📝 How It Works

1. **User Inputs** a movie they like and their user ID.
2. **Content-Based Filtering** finds similar movies using genre similarity (TF-IDF + Nearest Neighbors).
3. **Collaborative Filtering** uses the SVD model to predict how much the user would like each similar movie.
4. **Hybrid Scoring** averages genre similarity and rating prediction to rank results.
5. **Posters** are fetched using TMDB API to enhance visual presentation.

---

## 🔄 Hybrid Model Logic

```python
hybrid_score = 0.5 * genre_similarity + 0.5 * (predicted_rating / 5.0)
```

- Genre similarity: from Nearest Neighbors over TF-IDF
- Predicted rating: from SVD model trained on user ratings
- Equal weight applied for simplicity (can be tuned)

---

## 📁 Project Structure

```
Movie_Recommend/
├── cleaned/                   # Cached data: TF-IDF, NearestNeighbors
│   ├── genres_cleaned.csv
│   ├── tfidf_matrix.pkl
│   └── nn_model.pkl
│
├── models/                    # Trained SVD model
│   └── svd_sampled_25m.pkl
│
├── genre.py                   # Data preparation logic
├── genre_recommender.py       # Content-based functions
├── rating_recommender.py      # Rating-based logic
├── hybrid_recommender.py      # Hybrid logic
├── main.py                    # CLI version
├── app.py                     # Streamlit UI
└── requirements.txt           # Dependencies
```

---

## 🚀 Getting Started

### 1. Clone the Repo

```bash
git clone https://github.com/YOUR_USERNAME/Movie_Recommend.git
cd Movie_Recommend
```

### 2. Set Up Environment

```bash
python -m venv venv
source venv/bin/activate       # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 3. Run the Web App

```bash
streamlit run app.py
```

### 4. Run CLI Mode

```bash
python main.py
```

---

## 📊 Future Improvements

- Add login system for personalized user profiles
- Tune hybrid weight dynamically using regression
- Extend metadata: directors, tags, keywords, cast
- Support top-N ranking evaluation metrics (MAP\@k, NDCG\@k)
- Add visualization tools to show latent factors (SVD)

---

## 👤 Author

**Omar Idriss**\
Machine Learning Intern\
GitHub: [OmarIdriss](https://github.com/OmarIdriss)

---

## 📄 Acknowledgements

- [MovieLens 25M Dataset](https://grouplens.org/datasets/movielens/25m/)
- [TMDB API](https://www.themoviedb.org/documentation/api)
- [Surprise Library](http://surpriselib.com/)

---

*This project was completed as part of a machine learning internship focused on deploying real-world recommender systems using modern Python tools.*

