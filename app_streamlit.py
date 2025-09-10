# app_streamlit.py
import streamlit as st
from utils import load_movielens
from content_based import ContentRecommender
from collaborative import CollaborativeRecommender
from hybrid import hybrid_recommend

DATA_PATH = 'data/ml-latest-small'

@st.cache_data
def load_data():
    return load_movielens(DATA_PATH)

@st.cache_resource
def get_cb(movies):
    return ContentRecommender(movies)

@st.cache_resource
def get_cf(ratings, movies):
    return CollaborativeRecommender(ratings, movies)


def main():
    st.title("🎬 Movie Recommender System")
    movies, ratings = load_data()

    st.sidebar.header("Settings")
    topn = st.sidebar.slider("Top N", 5, 30, 10)

    tabs = st.tabs(["Content-Based", "Collaborative", "Hybrid"])

    with tabs[0]:
        st.subheader("Content-Based Recommendations")
        title = st.text_input("Enter a movie title (substring works):", value="Toy Story")
        if st.button("Recommend (Content)"):
            cb = get_cb(movies)
            try:
                recs = cb.similar_by_title(title, topn)
                st.dataframe(recs)
            except Exception as e:
                st.error(str(e))

    with tabs[1]:
        st.subheader("Collaborative (SVD) Recommendations")
        user = st.number_input("User ID", min_value=1, value=1)
        if st.button("Recommend (Collaborative)"):
            cf = get_cf(ratings, movies)
            recs = cf.top_n_for_user(int(user), topn)
            st.dataframe(recs)

    with tabs[2]:
        st.subheader("Hybrid Recommendations")
        user = st.number_input("User ID for Hybrid", min_value=1, value=1, key='user_h')
        seed = st.text_input("Seed movie (content side)", value="Toy Story", key='seed_h')
        alpha = st.slider("Content weight α (0=only CF, 1=only Content)", 0.0, 1.0, 0.5, 0.05)
        if st.button("Recommend (Hybrid)"):
            try:
                recs = hybrid_recommend(DATA_PATH, int(user), seed, topn, alpha)
                st.dataframe(recs)
            except Exception as e:
                st.error(str(e))


if __name__ == "__main__":
    main()

