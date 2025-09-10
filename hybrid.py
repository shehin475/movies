# hybrid.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from utils import load_movielens, ensure_movie_exists
from content_based import ContentRecommender
from collaborative import CollaborativeRecommender


def minmax(series: pd.Series) -> pd.Series:
    if series.nunique() <= 1:
        return pd.Series(np.zeros(len(series)), index=series.index)
    scaler = MinMaxScaler()
    vals = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
    return pd.Series(vals, index=series.index)


def hybrid_recommend(data_path: str, user_id: int, seed_title: str, top_n: int = 10, alpha: float = 0.5) -> pd.DataFrame:
    movies, ratings = load_movielens(data_path)

    # Content-based part
    cb = ContentRecommender(movies)
    seed_id = ensure_movie_exists(movies, seed_title)
    cb_df = cb.similar_by_movieId(seed_id, top_n=500)  # get a large pool
    cb_df = cb_df.set_index('movieId')

    # Collaborative part
    cf = CollaborativeRecommender(ratings, movies)
    cf_df = cf.top_n_for_user(user_id=user_id, top_n=2000).set_index('movieId')

    # Join on movieId and normalize
    joined = cb_df[['score']].join(cf_df[['pred_rating']], how='inner')
    if joined.empty:
        # fallback: just content-based
        out = cb.similar_by_movieId(seed_id, top_n=top_n)
        out.rename(columns={'score': 'hybrid_score'}, inplace=True)
        return out[['movieId', 'title', 'genres', 'hybrid_score']]

    joined['score_n'] = minmax(joined['score'])
    joined['pred_n'] = minmax(joined['pred_rating'])
    joined['hybrid'] = alpha * joined['score_n'] + (1 - alpha) * joined['pred_n']

    final = (
        joined.sort_values('hybrid', ascending=False)
        .head(top_n)
        .join(movies.set_index('movieId')[['title', 'genres']])
        .reset_index()
    )
    final.rename(columns={'hybrid': 'hybrid_score'}, inplace=True)
    return final[['movieId', 'title', 'genres', 'hybrid_score']]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/ml-latest-small')
    parser.add_argument('--user', type=int, required=True)
    parser.add_argument('--seed', type=str, required=True)
    parser.add_argument('--topn', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.5)
    args = parser.parse_args()

    recs = hybrid_recommend(args.data, args.user, args.seed, args.topn, args.alpha)
    print(recs)
