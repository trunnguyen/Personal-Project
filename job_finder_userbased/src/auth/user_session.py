import os
import re
from pathlib import Path
from src.auth.otp_manager import get_user_db_id

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])


def slugify_job_title(job_title: str) -> str:
    slug = job_title.strip().lower()
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    slug = re.sub(r"\s+", "-", slug)
    return slug[:50] or "search"


def get_user_data_paths(email: str, job_title: str):
    db_id = get_user_db_id(email)
    job_slug = slugify_job_title(job_title)

    user_dir = os.path.join(PROJECT_ROOT, "data", "users", db_id, job_slug)
    os.makedirs(user_dir, exist_ok=True)

    db_path = os.path.join(user_dir, "jobs.db")
    csv_path = os.path.join(user_dir, "jobs.csv")

    return db_path, csv_path