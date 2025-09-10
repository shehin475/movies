# hybrid.py
import pandas as pd
from utils import load_movielens
from content_based import ContentRecommender
from collaborative import CollaborativeRecommender

def hybrid_recommend(data_path, user_id, seed_title, topn=10, alpha=0.5):
    movies, ratings = load_movielens(data_path)

    cb = ContentRecommender(movies)
    cf = CollaborativeRecommender(ratings, movies)

    # Content similarity scores
    try:
        cb_recs = cb.similar_by_title(seed_title, topn*2)
        cb_recs['score_cb'] = 1.0 / (cb_recs.index + 1)  # simple rank-based score
    except Exception:
        cb_recs = pd.DataFrame(columns=['movieId', 'score_cb'])

    # Collaborative scores
    cf_recs = cf.top_n_for_user(user_id, topn*2)
    cf_recs.rename(columns={'est': 'score_cf'}, inplace=True)

    # Merge
    merged = pd.merge(cb_recs, cf_recs, on=['movieId', 'title'], how='outer').fillna(0)
    merged['score'] = alpha*merged['score_cb'] + (1-alpha)*merged['score_cf']

    return merged.sort_values('score', ascending=False).head(topn)[['movieId', 'title', 'score']]

