import re
import pandas as pd


def load_movielens(path: str):
    """Load MovieLens latest-small movies & ratings."""
    movies = pd.read_csv(f"{path}/movies.csv")
    ratings = pd.read_csv(f"{path}/ratings.csv")

    # Basic cleaning
    movies['genres'] = movies['genres'].fillna('')
    movies['title'] = movies['title'].fillna('')

    # Add rating counts for popularity
    rating_counts = ratings.groupby('movieId').size().rename('rating_count')
    movies = movies.merge(rating_counts, on='movieId', how='left').fillna({'rating_count': 0})

    return movies, ratings


def clean_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def ensure_movie_exists(movies: pd.DataFrame, title: str) -> int:
    """
    Find movieId by (case-insensitive) title substring match.
    If multiple matches, return the most popular (highest rating_count).
    """
    title_low = title.lower()
    hits = movies[movies['title'].str.lower().str.contains(title_low, na=False)]
    if hits.empty:
        raise ValueError(f"No movie found matching title: {title}")

    # Pick the most popular by rating count
    best_match = hits.sort_values('rating_count', ascending=False).iloc[0]
    return int(best_match['movieId'])


