import streamlit as st
from recommend_system.genre import prepare_genre_data
import rating_recommender as rating
import hybrid_recommender as hybrid
import joblib
import requests

api_key = '26f21c57b6a1df1afe0810a86de378b1'

# Load genre data and model
dataM, tfidf_matrix, nn_model = prepare_genre_data()
model = joblib.load('./models/svd_model.pkl')

# Streamlit UI
st.title("🎬 Hybrid Movie Recommender System")
st.markdown("Get movie recommendations based on both **genre** and **user rating history**!")

# User input
title = st.text_input("🎥 Enter a movie you like:", "Toy Story (1995)")
user_id = st.number_input("👤 Enter your user ID:", min_value=1, step=1)
top_n = st.slider("🔢 Number of recommendations:", 1, 20, 5)
def get_movie_poster(title, api_key):
    url = f"https://api.themoviedb.org/3/search/movie"
    params = {
        "api_key": api_key,
        "query": title
    }
    response = requests.get(url, params=params).json()
    if response['results']:
        poster_path = response['results'][0].get('poster_path')
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
    return None
# Button to trigger recommendations
if st.button("Get Recommendations"):
    try:
        results = hybrid.hybrid(user_id, title, dataM, model, tfidf_matrix, nn_model, rating.df_filtered, top_n=top_n)
        st.subheader("🔀 Top Recommendations:")
        for i, (movie, score) in enumerate(results, 1):
            poster_url = get_movie_poster(movie.split('(')[0].strip(), api_key)
            st.write(f"{i}. **{movie}** — Score: `{score:.2f}`")
            if poster_url:
                st.image(poster_url, width=150)
    except Exception as e:
        st.error(f"❌ Error: {e}")
