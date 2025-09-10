# collaborative.py
from typing import List
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from utils import load_movielens



class CollaborativeRecommender:
    def __init__(self, ratings: pd.DataFrame, movies: pd.DataFrame, n_factors: int = 100, random_state: int = 42):
        self.movies = movies.copy()
        self.reader = Reader(rating_scale=(ratings['rating'].min(), ratings['rating'].max()))
        data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], self.reader)
        self.trainset = data.build_full_trainset()
        self.algo = SVD(n_factors=n_factors, random_state=random_state)

        print("Training SVD model... please wait")
        self.algo.fit(self.trainset)
        print("Training finished!")

    def top_n_for_user(self, user_id: int, top_n: int = 10, min_ratings: int = 10) -> pd.DataFrame:
        # Predict ratings for all movies not yet rated by the user
        user_inner_id = self.trainset.to_inner_uid(str(user_id)) if self.trainset.knows_user(str(user_id)) else None
        rated_movie_inner_ids = set(self.trainset.ur[user_inner_id]) if user_inner_id is not None else set()
        candidates = self.movies[~self.movies['movieId'].astype(str).isin(
            {str(self.trainset.to_raw_iid(i)) for i, _ in rated_movie_inner_ids}
        )]

        preds = []
        for mid in candidates['movieId'].tolist():
            est = self.algo.predict(str(user_id), str(mid)).est
            preds.append((int(mid), float(est)))

        preds.sort(key=lambda x: x[1], reverse=True)
        preds = preds[:top_n]

        rec_df = self.movies.set_index('movieId').loc[[m for m, _ in preds]][['title', 'genres']].copy()
        rec_df.insert(0, 'pred_rating', [round(s, 3) for _, s in preds])
        rec_df.reset_index(inplace=True)
        return rec_df


if __name__ == "__main__":
 import argparse

 parser = argparse.ArgumentParser()
 parser.add_argument('--data', type=str, default='data/ml-latest-small')
 parser.add_argument('--user', type=int, required=True)
 parser.add_argument('--topn', type=int, default=10)
 args = parser.parse_args()

 print("Training SVD model... please wait")
 print("Training finished!")


 movies, ratings = load_movielens(args.data)
 cf = CollaborativeRecommender(ratings, movies)
 print(cf.top_n_for_user(args.user, args.topn))



