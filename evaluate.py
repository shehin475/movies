# evaluate.py
import math
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from utils import load_movielens


def rmse(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return math.sqrt(np.mean((y_true - y_pred) ** 2))


def build_svd_model(ratings: pd.DataFrame, n_factors: int = 100, random_state: int = 42):
    """
    Build SVD model using scikit-learn
    ratings: DataFrame with columns [userId, movieId, rating]
    """
    user_item = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

    svd = TruncatedSVD(n_components=n_factors, random_state=random_state)
    user_factors = svd.fit_transform(user_item)
    item_factors = svd.components_.T

    pred_matrix = np.dot(user_factors, item_factors.T)
    pred_df = pd.DataFrame(pred_matrix, index=user_item.index, columns=user_item.columns)

    return svd, pred_df


def evaluate_rmse(data_path: str) -> float:
    _, ratings = load_movielens(data_path)
    train, test = train_test_split(ratings, test_size=0.2, random_state=42)

    _, pred_df = build_svd_model(train)

    y_true, y_pred = [], []
    for _, row in test.iterrows():
        uid, mid, true_r = row["userId"], row["movieId"], row["rating"]
        if uid in pred_df.index and mid in pred_df.columns:
            y_true.append(true_r)
            y_pred.append(pred_df.loc[uid, mid])

    return rmse(y_true, y_pred)


def precision_recall_at_k(pred_df, ratings: pd.DataFrame, k=10, threshold=3.5):
    precisions, recalls = {}, {}

    for uid in ratings['userId'].unique():
        if uid not in pred_df.index:
            continue

        user_preds = pred_df.loc[uid].dropna()
        true_ratings = ratings[ratings['userId'] == uid].set_index("movieId")["rating"]

        # Exclude already rated movies
        user_preds = user_preds.drop(true_ratings.index, errors="ignore")

        # Scale predictions to 0–5
        scaler = MinMaxScaler(feature_range=(0, 5))
        scores = scaler.fit_transform(user_preds.values.reshape(-1, 1)).flatten()
        user_preds = pd.Series(scores, index=user_preds.index)

        # Top-k recommended
        top_k = user_preds.sort_values(ascending=False).head(k)

        # Compute metrics
        n_rel = sum(true_ratings >= threshold)
        n_rec_k = sum(top_k >= threshold)
        n_rel_and_rec_k = sum(((mid in true_ratings.index) and (true_ratings[mid] >= threshold))
                              for mid in top_k.index)

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k else 0
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel else 0

    precision = sum(precisions.values()) / len(precisions) if precisions else 0
    recall = sum(recalls.values()) / len(recalls) if recalls else 0
    return precision, recall


def evaluate_topn(data_path: str, k=10, threshold=3.5):
    _, ratings = load_movielens(data_path)
    train, test = train_test_split(ratings, test_size=0.2, random_state=42)

    _, pred_df = build_svd_model(train)
    return precision_recall_at_k(pred_df, test, k=k, threshold=threshold)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/ml-latest-small')
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--threshold', type=float, default=3.5)
    args = parser.parse_args()

    rmse_val = evaluate_rmse(args.data)
    prec, rec = evaluate_topn(args.data, k=args.k, threshold=args.threshold)

    print(f"RMSE: {rmse_val:.4f}")
    print(f"Precision@{args.k}: {prec:.4f}")
    print(f"Recall@{args.k}: {rec:.4f}")

