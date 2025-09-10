from typing import List
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils import load_movielens, clean_text, ensure_movie_exists


class ContentRecommender:
    def __init__(self, movies: pd.DataFrame):
        self.movies = movies.copy()
        # Build metadata soup
        self.movies['genres_tokens'] = (
            self.movies['genres']
            .str.replace('|', ' ', regex=False)
            .apply(clean_text)
        )
        self.movies['title_clean'] = self.movies['title'].apply(clean_text)
        self.movies['soup'] = self.movies['title_clean'] + ' ' + self.movies['genres_tokens']

        self.vectorizer = TfidfVectorizer(stop_words='english', min_df=2)
        self.tfidf = self.vectorizer.fit_transform(self.movies['soup'])
        self.cosine = cosine_similarity(self.tfidf)

        # Mapping movieId -> index
        self.id_to_idx = {int(mid): idx for idx, mid in enumerate(self.movies['movieId'])}

    def similar_by_movieId(self, movie_id: int, top_n: int = 10) -> pd.DataFrame:
        idx = self.id_to_idx.get(int(movie_id))
        if idx is None:
            raise ValueError(f"movie_id {movie_id} not in corpus")
        sims = self.cosine[idx]

        # Exclude itself
        indices = np.argsort(-sims)
        result = []
        for j in indices:
            if j == idx:
                continue
            result.append((int(self.movies.iloc[j]['movieId']), float(sims[j])))
            if len(result) >= top_n:
                break

        rec_df = self.movies.set_index('movieId').loc[[m for m, _ in result]][['title', 'genres']].copy()
        rec_df.insert(0, 'score', [round(s, 4) for _, s in result])
        rec_df.reset_index(inplace=True)
        return rec_df

    def similar_by_title(self, title_query: str, top_n: int = 10) -> pd.DataFrame:
        movie_id = ensure_movie_exists(self.movies, title_query)
        return self.similar_by_movieId(movie_id, top_n)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/ml-latest-small', help='Path to MovieLens folder')
    parser.add_argument('--title', type=str, required=True, help='Movie title (substring OK)')
    parser.add_argument('--topn', type=int, default=10)
    args = parser.parse_args()

    movies, _ = load_movielens(args.data)
    cb = ContentRecommender(movies)
    print(cb.similar_by_title(args.title, args.topn))
