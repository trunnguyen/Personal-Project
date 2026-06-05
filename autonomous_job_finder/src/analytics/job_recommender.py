import os
import sys
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.utils.logger import logger



class Recommender:
    def __init__(self,model_name='all-MiniLM-L6-v2'):
        self.embedding_model = SentenceTransformer(model_name)
        self.model = None
        self.is_trained = False

        self.model_path = os.path.join(PROJECT_ROOT,"data","classifier.pkl")
        self._load_pickled_model()

    def _load_pickled_model(self):
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, "rb") as f:
                    self.model = pickle.load(f)
                self.is_trained = True
                logger.info(f"Model successfully loaded from {self.model_path}")
            except Exception as e:
                logger.error(f"Failed to parse model file: {e}.")
                self.is_trained = False
        else:
            self.is_trained = False

    def _prepare_text(self, job):
        title = job.get('title','Unknown Title')
        company = job.get('company','')
        location = job.get('location','Ho Chi Minh City')
        return f"Job Opportunity{title} at {company} located in {location}"

    def update_ai_score(self, jobs_to_score):
        if not jobs_to_score:
            return []

        texts = [self._prepare_text(job) for job in jobs_to_score]
        X_new = self.embedding_model.encode(texts,show_progress_bar=False)

        if self.is_trained and self.model is not None:
            probabilities = self.model.predict_proba(X_new)[: , 1]
            for idx, job in enumerate(jobs_to_score):
                job['ai_score'] = round(float(probabilities[idx]), 4)
        else:
            anchor_profile = "AI Engineer Intern Machine Learning Intern Data Science Intern Python PyTorch deep learning NLP LLM LangChain RAG computer vision internship entry Ho Chi Minh City Vietnam"
            anchor_vector = self.embedding_model.encode([anchor_profile])
            similarities = cosine_similarity(X_new, anchor_vector)
            for idx, job in enumerate(jobs_to_score):
                job['ai_score'] = round(float(similarities[idx][0]), 4)

        return sorted(jobs_to_score, key=lambda x: x['ai_score'], reverse=True)

    def update_scores(self, unscored_jobs):
        for job in unscored_jobs:
            text_to_score=f"{job}"