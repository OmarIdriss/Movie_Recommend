import time
start = time.time()
print("Okay Timer Started to evalute how long the process takes")

from hybrid_recommender import hybrid
# ======= genre_recommender.py =======
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from genre import prepare_genre_data

# Load and process movie data
dataM, tfidf_matrix, cosine_sim = prepare_genre_data()

def recommend_genres(title, df=dataM, cosine_sim=cosine_sim):
    idx = df[df['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    movie_indices = [i[0] for i in sim_scores]
    result = df['title'].iloc[movie_indices]
    return result
