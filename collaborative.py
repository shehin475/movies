# collaborative.py
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler


class CollaborativeRecommender:
    def __init__(self, ratings: pd.DataFrame, movies: pd.DataFrame, n_factors: int = 50, random_state: int = 42):
        """
        ratings: DataFrame with columns ['userId', 'movieId', 'rating']
        movies: DataFrame with at least ['movieId', 'title', 'genres']
        """
        self.ratings = ratings
        self.movies = movies
        self.n_factors = n_factors
        self.random_state = random_state

        # Create user–item matrix
        self.user_item_matrix = self.ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

        # Fit SVD
        self.svd = TruncatedSVD(n_components=n_factors, random_state=random_state)
        self.user_factors = self.svd.fit_transform(self.user_item_matrix)
        self.item_factors = self.svd.components_.T  # shape: (n_items, n_factors)

        # Reconstruct approximated ratings
        self.pred_matrix = np.dot(self.user_factors, self.item_factors.T)
        self.pred_df = pd.DataFrame(self.pred_matrix,
                                    index=self.user_item_matrix.index,
                                    columns=self.user_item_matrix.columns)

    def top_n_for_user(self, user_id: int, top_n: int = 10) -> pd.DataFrame:
        if user_id not in self.pred_df.index:
            raise ValueError(f"User {user_id} not found in ratings data")

        # Get predicted ratings for the user
        user_preds = self.pred_df.loc[user_id]

        # Remove movies the user has already rated
        rated_movies = set(self.ratings[self.ratings['userId'] == user_id]['movieId'])
        user_preds = user_preds.drop(labels=rated_movies, errors='ignore')

        # Normalize scores for consistency
        scaler = MinMaxScaler()
        scores = scaler.fit_transform(user_preds.values.reshape(-1, 1)).flatten()

        recs = (
            pd.DataFrame({
                'movieId': user_preds.index,
                'pred_rating': scores
            })
            .merge(self.movies, on='movieId', how='left')
            .sort_values('pred_rating', ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )
        return recs

