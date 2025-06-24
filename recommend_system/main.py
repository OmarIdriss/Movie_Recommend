from recommend_system.genre_recommender import start
from genre import prepare_genre_data
import rating_recommender as rating
import hybrid_recommender as hybrid
import joblib
import time
from genre_recommender import start
import genre_recommender as genre

# Load genre data from cache
dataM, tfidf_matrix, nn_model = prepare_genre_data()

# Load trained SVD model
model = joblib.load('./models/svd_sampled_25m.pkl')

# Get user input
user_id = int(input("🧑 Enter your user ID: "))
title = input("🎬 Enter a movie you like: ")
top_n = 5

# Get hybrid recommendations
results = hybrid.hybrid(user_id, title, dataM, model, tfidf_matrix, nn_model, rating.df_filtered, top_n=top_n)

# Display results
print("\n🔀 Top Recommendations:")
for i, (movie, score) in enumerate(results, 1):
    print(f"{i}. {movie} — Score: {score:.2f}")

end = time.time()
yurr = end - genre.start
print(f"The time took to finish the code is: {yurr}")