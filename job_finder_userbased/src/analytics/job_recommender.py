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
        #self._load_pickled_model()

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

    def build_anchor_from_profile(self, name, major, job_title):
        return f"{job_title} intern entry level fresh graduate {major} student Ho Chi Minh City Vietnam"

    def score_for_user(self, jobs, name, major, job_title):
        if not jobs:
            return []

        texts = [self._prepare_text(job) for job in jobs]
        job_vectors = self.embedding_model.encode(texts, show_progress_bar=False)

        if self.is_trained and self.model is not None:
            probabilities = self.model.predict_proba(job_vectors)[:, 1]
            for idx, job in enumerate(jobs):
                job['ai_score'] = round(float(probabilities[idx]), 4)
        else:
            anchor_text = self.build_anchor_from_profile(name, major, job_title)
            anchor_vector = self.embedding_model.encode([anchor_text])

            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(job_vectors, anchor_vector)

            for idx, job in enumerate(jobs):
                job['ai_score'] = round(float(similarities[idx][0]), 4)

        return sorted(jobs, key=lambda x: x['ai_score'], reverse=True)

    def load_user_model(self, model_path):
        import os
        if os.path.exists(model_path):
            try:
                with open(model_path, "rb") as f:
                    self.model = pickle.load(f)
                self.is_trained = True
                logger.info(f"User model loaded from {model_path}")
                return True
            except Exception as e:
                logger.error(f"Failed to load user model: {e}")
                self.is_trained = False
                return False
        self.model = None
        self.is_trained = False
        return False

    def update_scores(self, unscored_jobs):
        for job in unscored_jobs:
            text_to_score=f"{job}"