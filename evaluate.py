# evaluate.py
import math
import numpy as np
from scikit_surprise_py import Dataset, Reader, SVD
from scikit_surprise_py.model_selection import train_test_split
from utils import load_movielens


def rmse(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return math.sqrt(np.mean((y_true - y_pred) ** 2))


def evaluate_rmse(data_path: str) -> float:
    _, ratings = load_movielens(data_path)
    reader = Reader(rating_scale=(ratings['rating'].min(), ratings['rating'].max()))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    algo = SVD(n_factors=100, random_state=42)
    algo.fit(trainset)
    preds = algo.test(testset)
    y_true = [p.r_ui for p in preds]
    y_pred = [p.est for p in preds]
    return rmse(y_true, y_pred)


def precision_recall_at_k(predictions, k=10, threshold=3.5):
    user_est_true = {}
    for uid, _, true_r, est, _ in predictions:
        user_est_true.setdefault(uid, []).append((est, true_r))

    precisions, recalls = {}, {}
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k else 0
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel else 0

    precision = sum(precisions.values()) / len(precisions)
    recall = sum(recalls.values()) / len(recalls)
    return precision, recall


def evaluate_topn(data_path: str, k=10, threshold=3.5):
    _, ratings = load_movielens(data_path)
    reader = Reader(rating_scale=(ratings['rating'].min(), ratings['rating'].max()))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    algo = SVD(n_factors=100, random_state=42)
    algo.fit(trainset)
    predictions = algo.test(testset)
    return precision_recall_at_k(predictions, k=k, threshold=threshold)


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
