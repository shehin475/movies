# utils.py
import re
import pandas as pd




def load_movielens(path: str):
    """Load MovieLens latest-small movies & ratings."""
    movies = pd.read_csv(f"{path}/movies.csv")
    ratings = pd.read_csv(f"{path}/ratings.csv")
     # Basic cleaning
    movies['genres'] = movies['genres'].fillna('')
    movies['title'] = movies['title'].fillna('')
    return movies, ratings




def clean_text(s: str) -> str:
   s = s.lower()
   s = re.sub(r"[^a-z0-9\s]", " ", s)
   s = re.sub(r"\s+", " ", s).strip()
   return s




def ensure_movie_exists(movies: pd.DataFrame, title: str) -> int:
 """Find movieId by (case-insensitive) title substring match. Returns movieId or raises ValueError."""
 title_low = title.lower()
 hits = movies[movies['title'].str.lower().str.contains(title_low, na=False)]
 if hits.empty:
  raise ValueError(f"No movie found matching title: {title}")
# Return the best match (shortest edit distance-ish by length difference)
 return int(hits.iloc[0]['movieId'])

