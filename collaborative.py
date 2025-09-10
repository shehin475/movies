# collaborative.py
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

class CollaborativeRecommender:
    def __init__(self, ratings: pd.DataFrame, movies: pd.DataFrame, n_factors=100, random_state=42):
        self.ratings = ratings
        self.movies = movies

        # Build user-item matrix
        self.user_item = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

        # Apply SVD
        svd = TruncatedSVD(n_components=n_factors, random_state=random_state)
        latent = svd.fit_transform(self.user_item)

        # Compute similarity in latent space
        self.user_factors = latent
        self.movie_factors = svd.components_.T
        self.movie_index = {m: i for i, m in enumerate(self.user_item.columns)}

    def predict(self, user_id, movie_id):
        if movie_id not in self.movie_index or user_id not in self.user_item.index:
            return 0
        u_idx = self.user_item.index.get_loc(user_id)
        m_idx = self.movie_index[movie_id]
        return np.dot(self.user_factors[u_idx], self.movie_factors[m_idx])

    def top_n_for_user(self, user_id, n=10):
        if user_id not in self.user_item.index:
            return pd.DataFrame(columns=['movieId', 'title', 'est'])

        preds = []
        for m_id in self.user_item.columns:
            if self.user_item.loc[user_id, m_id] == 0:  # not rated yet
                preds.append((m_id, self.predict(user_id, m_id)))

        preds = sorted(preds, key=lambda x: x[1], reverse=True)[:n]
        recs = pd.DataFrame(preds, columns=['movieId', 'est'])
        return recs.merge(self.movies[['movieId', 'title']], on='movieId')


