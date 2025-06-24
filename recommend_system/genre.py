import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Paths
RAW_MOVIE_PATH = './data/ml-25m/movies.csv'
CLEANED_MOVIE_PATH = './cache/genres_cleaned.csv'
TFIDF_PATH = './cache/tfidf_matrix.pkl'
NN_MODEL_PATH = './cache/nn_model.pkl'


def prepare_genre_data():
        dataM = pd.read_csv(CLEANED_MOVIE_PATH)
        tfidf_matrix = joblib.load(TFIDF_PATH)
        nn_model = joblib.load(NN_MODEL_PATH)
        return dataM, tfidf_matrix, nn_model