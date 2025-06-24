import pandas as pd
import time
import joblib
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split
from tqdm import tqdm
# ---------------- Load Ratings ---------------- #
df_data = pd.read_csv('/Users/omarmac/Downloads/Projects/Movie_recommend/data/ml-25m/ratings.csv')

df_filtered = df_data[['userId', 'movieId', 'rating']]
# ---------------- Sample the Data ---------------- #
df_sampled = df_filtered.sample(n=10_000_000, random_state=42)
# ---------------- Prepare Data ---------------- #
print("🔧 Preparing dataset...")
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df_sampled, reader)
print("DONE PREPARING")
trainset, testset = train_test_split(data, test_size=0.2)
print("Done test split")
# ---------------- Loading The Model ---------------- #
print("Model loading")
model = joblib.load('./models/svd_sampled_25m.pkl')
start_train = time.time()
print(f"✅ Model Loaded")
# ---------------- Getting Answer ---------------- #
def recommend(user_id,model,df_filtered, n=5):
 recommendations = []

 for movie_id in df_filtered['movieId'].unique():
    est_rating = model.predict(user_id, movie_id).est
    recommendations.append((movie_id, est_rating))
 recommendations.sort(key=lambda x: x[1], reverse = True)
 return recommendations[:n]