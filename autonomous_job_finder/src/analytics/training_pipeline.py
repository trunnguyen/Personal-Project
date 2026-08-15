import os
import sys
import pickle
import numpy as np
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.metrics import f1_score, classification_report
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.utils.db_manager import JobDB
from src.utils.logger import logger
from src.analytics.job_recommender import Recommender

def run_model_training():
    base_data_path= os.path.join(PROJECT_ROOT, 'data')
    db_path= os.path.join(base_data_path, 'jobs.db')
    csv_path= os.path.join(base_data_path, 'jobs.csv')
    model_pkl_path= os.path.join(base_data_path, 'classifier.pkl')


    db= JobDB(db_path,csv_path)
    all_jobs= db.get_all_jobs()

    if not all_jobs:
        logger.warning("The database is empty")
        return

    y=np.array([1 if job.get('is_applied') == 1 else 0 for job in all_jobs])
    num_positives = np.sum(y==1)
    num_negatives = np.sum(y==0)

    if num_positives < 3 or num_negatives < 3:
        logger.warning(f"Insufficient data classification profiles to slice folds (Applied: {num_positives}, Skipped: {num_negatives})")
        logger.info("Insufficient labeled interactions ")
        return

    recommender = Recommender()
    texts = [recommender._prepare_text(job) for job in all_jobs]

    logger.info(f"Vectorizing {len(texts)} positions")
    X = recommender.embedding_model.encode(texts, show_progress_bar=False)

    #Add SMOTE distribution
    logger.info(f"Class distribution before SMOTE  - Pos: {num_positives}, Neg: {num_negatives}")
    sm= SMOTE(random_state=42, k_neighbors=min(5,num_positives - 1))

    logger.info("Evaluating the model StratifiedKFold")
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_score=[]
    oof_predictions=np.zeros(len(y))

    for fold, (train_index, test_index) in enumerate(skf.split(X, y),1):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        #Add sample to SMOTE
        X_train, y_train = sm.fit_resample(X_train, y_train)

        fold_clf= LogisticRegression(class_weight='balanced',random_state=42)
        fold_clf.fit(X_train, y_train)
        preds = fold_clf.predict(X_test)
        oof_predictions[test_index] = preds

        score = f1_score(y_test, preds, pos_label=1,zero_division=0)
        cv_score.append(score)
        logger.info(f"Fold {fold} score: {score: .2f}")

    print("\n" + "="*60)
    print(" Performance on the test set")
    print(f"Cross Validation MEAN f1: {np.mean(cv_score):.2%}")
    print(classification_report(y, oof_predictions,
                                target_names=["Not interested","Interested"],
                                zero_division=0))
    print("=" * 60)

    logger.info("Fitting model")
    #
    X_resampled, y_resampled = sm.fit_resample(X, y)
    logger.info(f"After SMOTE - Pos: {np.sum(y_resampled==1)}, Neg: {np.sum(y_resampled==0)}")
    clf = LogisticRegression(class_weight='balanced')
    clf.fit(X_resampled, y_resampled)



    with open(model_pkl_path, 'wb') as f:
        pickle.dump(clf, f)
    logger.info(f"Model saved to {model_pkl_path}")

if __name__ == "__main__":
    run_model_training()


