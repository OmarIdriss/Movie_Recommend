def hybrid(user_id, title, df_movies, model, tfidf_matrix, nn_model, df_filtered, top_n=5):
    matched = df_movies[df_movies['title'] == title]
    if matched.empty:
        print(f"❌ Movie '{title}' not found.")
        return []

    idx = matched.index[0]
    distances, indices = nn_model.kneighbors(tfidf_matrix[idx], n_neighbors=30)

    results = []
    for i in indices[0][1:]:  # skip self
        movie_id = df_movies.iloc[i]['movieId']
        try:
            predicted_rating = model.predict(user_id, movie_id).est
        except:
            predicted_rating = 0

        genre_score = 1 - distances[0][list(indices[0]).index(i)]  # convert distance to similarity
        hybrid_score = 0.5 * genre_score + 0.5 * (predicted_rating / 5.0)
        results.append((df_movies.iloc[i]['title'], predicted_rating))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_n]
