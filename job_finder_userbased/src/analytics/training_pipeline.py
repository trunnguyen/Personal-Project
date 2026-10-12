import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from imblearn.over_sampling import SMOTE

from src.utils.logger import logger
from src.analytics.job_recommender import Recommender


def train_user_model(db, user_major, user_job_title, model_path):
    all_jobs = db.get_all_jobs()

    if not all_jobs:
        logger.warning("No jobs in database, cannot train")
        return None

    y = np.array([1 if job.get('is_applied') == 1 else 0 for job in all_jobs])
    num_positives = np.sum(y == 1)
    num_negatives = np.sum(y == 0)

    if num_positives < 6 or num_negatives < 6:
        logger.warning(f"Insufficient labeled data to train (Interested: {num_positives}, Not Interested: {num_negatives})")
        return None

    recommender = Recommender()
    texts = [recommender._prepare_text(job) for job in all_jobs]

    logger.info(f"Vectorizing {len(texts)} jobs for user model")
    X = recommender.embedding_model.encode(texts, show_progress_bar=False)

    logger.info(f"Class distribution before SMOTE — Positives: {num_positives}, Negatives: {num_negatives}")

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores = []
    oof_predictions = np.zeros(len(y))

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        n_min = int(np.sum(y_train == 1))
        if n_min >= 2:
            sm = SMOTE(random_state=42, k_neighbors=min(5, n_min - 1))
            X_train, y_train = sm.fit_resample(X_train, y_train)

        fold_clf = LogisticRegression(class_weight='balanced', random_state=42)
        fold_clf.fit(X_train, y_train)
        preds = fold_clf.predict(X_test)
        oof_predictions[test_idx] = preds

        score = f1_score(y_test, preds, pos_label=1, zero_division=0)
        cv_scores.append(score)
        logger.info(f"Fold {fold} score: {score:.2f}")

    logger.info(f"Cross Validation MEAN f1: {np.mean(cv_scores):.2%}")
    report = classification_report(
        y, oof_predictions,
        target_names=["Not interested", "Interested"],
        zero_division=0
    )
    logger.info(f"Classification report:\n{report}")

    sm = SMOTE(random_state=42, k_neighbors=min(5, num_positives - 1))

    X_resampled, y_resampled = sm.fit_resample(X, y)
    logger.info(f"After SMOTE — Positives: {np.sum(y_resampled==1)}, Negatives: {np.sum(y_resampled==0)}")

    clf = LogisticRegression(class_weight='balanced')
    clf.fit(X_resampled, y_resampled)

    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)
    logger.info(f"User model saved to {model_path}")

    return {
        "mean_f1": float(np.mean(cv_scores)),
        "num_positives": int(num_positives),
        "num_negatives": int(num_negatives),
        "report": report
    }