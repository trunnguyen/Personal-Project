
import os
import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from flask import Flask, request, redirect, url_for, render_template, session
from flask_session import Session
import re
import asyncio
import sqlite3
from datetime import timedelta
from dotenv import load_dotenv
import hmac

load_dotenv()

from src.auth.user_session import get_user_data_paths
from src.utils.db_manager import JobDB
from src.analytics.job_recommender import Recommender
from src.analytics.eligibility_checker import check_eligibility
from src.analytics.keyword_generator import generate_search_keywords
from src.scraper import crawler as job_crawler
from src.scraper.crawler import fetch_job_description
from src.auth.otp_manager import generate_otp, verify_otp, is_rate_limited
from src.utils.notifier import Notifier
from src.auth.admin_logger import (
    log_security_event,
    log_user_activity,
    get_recent_security_events,
    get_recent_user_activity,
)
from src.auth.otp_manager import get_user_db_id
from src.analytics.training_pipeline import train_user_model



DEV_SKIP_OTP = os.getenv("DEV_SKIP_OTP", "false").lower() == "true"

ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")
FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "")

if len(FLASK_SECRET_KEY) < 32:
    raise RuntimeError("FLASK_SECRET_KEY is missing or too short (need 32+ characters). Set it in .env.")
if not ADMIN_PASSWORD or ADMIN_PASSWORD == "changeme":
    raise RuntimeError("ADMIN_PASSWORD must be set to a real value in .env.")

if DEV_SKIP_OTP:
    print("WARNING: DEV_SKIP_OTP is ON. OTP verification is bypassed. Local testing only.")

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

def validate_inputs(name, major, job_title, email):
    field_errors = {}

    if not (1 <= len(name.strip()) <= 50) or not re.match(r"^[A-Za-zÀ-ỹ\s]+$", name):
        field_errors['name'] = "Must be 1-50 characters, letters only."

    if not (1 <= len(major.strip()) <= 50) or not re.match(r"^[A-Za-zÀ-ỹ0-9\s\-&,./]+$", major):
        field_errors['major'] = "Must be 1-50 characters."

    if not (1 <= len(job_title.strip()) <= 100) or not re.match(r"^[A-Za-zÀ-ỹ0-9\s\-&,./()]+$", job_title):
        field_errors['job_title'] = "Must be 1-100 characters."

    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        field_errors['email'] = "Please enter a valid email address."

    return field_errors

def login_required(f):
    from functools import wraps

    @wraps(f)
    def wrapper(*args, **kwargs):
        if not session.get('verified'):
            return redirect(url_for('signup'))
        return f(*args, **kwargs)
    return wrapper

app = Flask(__name__)
app.config['SECRET_KEY'] = FLASK_SECRET_KEY
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = os.path.join(PROJECT_ROOT, 'flask_app', 'flask_session')
app.config['SESSION_PERMANENT'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = os.getenv("COOKIE_SECURE", "false").lower() == "true"
Session(app)


@app.route("/", methods=["GET", "POST"])
def signup():
    form_values = {"name": "", "major": "", "job_title": "", "email": ""}
    field_errors = {}
    show_verify_modal = False

    if request.method == "POST":
        action = request.form.get("action", "signup")

        if action == "signup":
            form_values["name"] = request.form.get("name", "")
            form_values["major"] = request.form.get("major", "")
            form_values["job_title"] = request.form.get("job_title", "")
            form_values["email"] = request.form.get("email", "")

            field_errors = validate_inputs(
                form_values["name"], form_values["major"],
                form_values["job_title"], form_values["email"]
            )

            if field_errors:
                temp_hash = get_user_db_id(form_values["email"]) if form_values["email"] else "unknown"
                log_security_event(
                    "input_validation_failed",
                    temp_hash,
                    f"Fields rejected: {list(field_errors.keys())}"
                )

            if not field_errors:
                if is_rate_limited(form_values["email"]):
                    field_errors['email'] = "Too many requests, try again later."
                else:
                    notifier = Notifier()
                    code = generate_otp(form_values["email"])
                    success = notifier.send_otp_email(form_values["email"], code)

                    if success:
                        session['pending_email'] = form_values["email"]
                        session['pending_name'] = form_values["name"]
                        session['pending_major'] = form_values["major"]
                        session['pending_job_title'] = form_values["job_title"]

                        if DEV_SKIP_OTP:
                            session['verified'] = True
                            session['user_email'] = form_values["email"]
                            session['user_name'] = form_values["name"]
                            session['user_major'] = form_values["major"]
                            session['user_job_title'] = form_values["job_title"]
                            return redirect(url_for('dashboard'))

                        show_verify_modal = True
                    else:
                        field_errors['email'] = "Failed to send verification code."

        elif action == "verify":
            input_code = request.form.get("otp_code", "")
            pending_email = session.get('pending_email', '')

            if verify_otp(pending_email, input_code):
                session['verified'] = True
                session['user_email'] = pending_email
                session['user_name'] = session.get('pending_name')
                session['user_major'] = session.get('pending_major')
                session['user_job_title'] = session.get('pending_job_title')

                log_user_activity(
                    get_user_db_id(pending_email),
                    "signup_verified",
                    f"Major: {session['user_major']}, Job: {session['user_job_title']}"
                )
                return redirect(url_for('dashboard'))
            else:
                field_errors['otp'] = "Incorrect or expired code. Please try again."
                show_verify_modal = True
                form_values["name"] = session.get('pending_name', '')
                form_values["major"] = session.get('pending_major', '')
                form_values["job_title"] = session.get('pending_job_title', '')
                form_values["email"] = session.get('pending_email', '')

        elif action == "cancel_verify":
            session.pop('pending_email', None)
            session.pop('pending_name', None)
            session.pop('pending_major', None)
            session.pop('pending_job_title', None)

    return render_template(
        "signup.html",
        field_errors=field_errors,
        form_values=form_values,
        show_verify_modal=show_verify_modal,
        pending_email=session.get('pending_email', '')
    )

@app.route("/dashboard", methods=["GET", "POST"])
@login_required
def dashboard():
    db_path, csv_path = get_user_data_paths(
        session['user_email'],
        session['user_job_title']
    )
    db = JobDB(db_path, csv_path)

    status_message = None
    searched_keywords = None

    if request.method == "POST" and request.form.get("action") == "find_jobs":
        keywords = generate_search_keywords(
            session['user_major'],
            session['user_job_title']
        )
        searched_keywords = ", ".join(keywords)

        asyncio.run(job_crawler.main(
            db_path=db_path,
            csv_path=csv_path,
            search_keywords=keywords
        ))

        recommender = Recommender()
        recommender.load_user_model(db_path.replace("jobs.db", "classifier.pkl"))
        unscored = db.get_unscored_jobs()

        if unscored:
            scored = recommender.score_for_user(
                unscored,
                session['user_name'],
                session['user_major'],
                session['user_job_title']
            )
            for job in scored:
                db.update_job_score(job['id'], job['ai_score'])

        all_jobs = db.get_all_jobs()
        top_jobs = sorted(all_jobs, key=lambda x: x['ai_score'], reverse=True)[:10]

        for job in top_jobs:
            if job.get('eligibility_checked', 0) == 0:
                description = asyncio.run(fetch_job_description(job['job_url']))
                result = check_eligibility(
                    title=job['title'],
                    company=job['company'],
                    description=description,
                    user_major=session['user_major'],
                    user_job_title=session['user_job_title']
                )
                db.update_eligibility(job['id'], result)

        status_message = "Done! Jobs have been updated"
        log_user_activity(
            get_user_db_id(session['user_email']),
            "job_search_run",
            f"Keywords: {searched_keywords}"
        )

    elif request.method == "POST" and request.form.get("action") == "recheck_eligibility":
        conn = sqlite3.connect(db.db_name)
        conn.execute("UPDATE jobs SET eligibility_checked = 0")
        conn.commit()
        conn.close()
        status_message = "Eligibility flags reset. Click 'Find Jobs' to re-check with latest logic."

    elif request.method == "POST" and request.form.get("action") == "follow_search":
        followed = db.get_followed_search()

        if followed and followed['is_active'] == 1:
            db.deactivate_follow()
            status_message = "Automation turned off. You will no longer receive digest emails for this search."
        else:
            db.activate_follow()

            since_date = "2000-01-01"
            top_matches = db.get_high_matches_since(0.6, since_date)[:10]

            if top_matches:
                notifier = Notifier()
                notifier.send_job_digest(
                    session['user_email'],
                    top_matches,
                    subject_prefix=f"{session['user_job_title']} Matches"
                )
                status_message = f"Automation turned on! We sent you an email with {len(top_matches)} current top matches. You'll get updates on new matches going forward."
            else:
                status_message = "Automation turned on! No high-scoring matches yet — you'll be emailed as soon as we find some."

            db.update_last_checked()

            log_user_activity(
                get_user_db_id(session['user_email']),
                "follow_search_activated",
                f"Job: {session['user_job_title']}, Initial matches sent: {len(top_matches)}"
            )

    elif request.method == "POST" and request.form.get("action") == "toggle_interest":
        job_id = int(request.form.get("job_id"))
        new_status = int(request.form.get("new_status"))
        db.update_score_and_interest(job_id, is_applied=new_status)

    elif request.method == "POST" and request.form.get("action") == "train_model":
        model_path = db_path.replace("jobs.db", "classifier.pkl")
        training_result = train_user_model(db, session['user_major'], session['user_job_title'], model_path)

        if training_result:
            status_message = (
                f"Model trained! F1 score: {training_result['mean_f1']:.2%} "
                f"(based on {training_result['num_positives']} liked and "
                f"{training_result['num_negatives']} disliked jobs)"
            )
            recommender = Recommender()
            recommender.load_user_model(db_path.replace("jobs.db", "classifier.pkl"))
            rescored = recommender.score_for_user(
                db.get_all_jobs(),
                session['user_name'],
                session['user_major'],
                session['user_job_title']
            )
            for job in rescored:
                db.update_job_score(job['id'], job['ai_score'])
        else:
            status_message = "Not enough labeled data yet. Mark at least 6 jobs as 'Interested' and 6 as 'Not Interested' to train."

        log_user_activity(
            get_user_db_id(session['user_email']),
            "model_trained",
            status_message
        )

    jobs = db.get_all_jobs()
    scored_jobs = [j for j in jobs if j.get('ai_score', 0.0) > 0]
    scored_jobs = sorted(scored_jobs, key=lambda x: x['ai_score'], reverse=True)


    return render_template(
        "dashboard.html",
        user_name=session['user_name'],
        user_major=session['user_major'],
        user_job_title=session['user_job_title'],
        jobs=scored_jobs,
        status_message=status_message,
        searched_keywords=searched_keywords,
        followed_search=db.get_followed_search()
    )

@app.route("/dashboard/simulate-run", methods=["POST"])
@login_required
def simulate_run():
    db_path, csv_path = get_user_data_paths(
        session['user_email'],
        session['user_job_title'],
    )
    db = JobDB(db_path, csv_path)

    followed = db.get_followed_search()

    if not followed or followed['is_active'] != 1:
        return redirect(url_for('dashboard'))

    # Step 1: Generate keywords and scrape, exactly like "Find Jobs"
    keywords = generate_search_keywords(
        session['user_major'],
        session['user_job_title'],
    )

    asyncio.run(job_crawler.main(
        db_path=db_path,
        csv_path=csv_path,
        search_keywords=keywords
    ))

    # Step 2: Score any newly scraped jobs
    recommender = Recommender()
    recommender.load_user_model(db_path.replace("jobs.db", "classifier.pkl"))

    unscored = db.get_unscored_jobs()

    if unscored:
        scored = recommender.score_for_user(
            unscored,
            session['user_name'],
            session['user_major'],
            session['user_job_title']
        )
        for job in scored:
            db.update_job_score(job['id'], job['ai_score'])

    # Step 3: Run eligibility check on top jobs that haven't been checked yet
    all_jobs = db.get_all_jobs()
    top_jobs = sorted(all_jobs, key=lambda x: x['ai_score'], reverse=True)[:10]

    for job in top_jobs:
        if job.get('eligibility_checked', 0) == 0:
            description = asyncio.run(fetch_job_description(job['job_url']))
            result = check_eligibility(
                title=job['title'],
                company=job['company'],
                description=description,
                user_major=session['user_major'],
                user_job_title=session['user_job_title'],
            )
            db.update_eligibility(job['id'], result)

    # Step 4: Find and send only NEW high matches since last check
    since_date = followed['last_checked']
    new_matches = db.get_high_matches_since(0.6, since_date)[:10]

    if new_matches:
        notifier = Notifier()
        notifier.send_job_digest(
            session['user_email'],
            new_matches,
            subject_prefix=f"{session['user_job_title']} Matches (Weekly Update)"
        )

    db.update_last_checked()

    log_user_activity(
        get_user_db_id(session['user_email']),
        "simulate_scheduled_run",
        f"Keywords: {', '.join(keywords)}, Since: {since_date}, New matches sent: {len(new_matches)}"
    )

    return redirect(url_for('dashboard'))

@app.route("/admin", methods=["GET", "POST"])
def admin():
    if not session.get('admin_authenticated'):
        error = None
        if request.method == "POST":
            if is_rate_limited(f"admin:{request.remote_addr}", max_requests=5, window_seconds=900):
                error = "Too many attempts. Try again in 15 minutes."
            else:
                entered = request.form.get("password", "")
                if hmac.compare_digest(entered.encode(), ADMIN_PASSWORD.encode()):
                    session['admin_authenticated'] = True
                else:
                    error = "Incorrect password"
                    log_security_event("admin_login_failed", "admin", "Wrong password attempt")
        if not session.get('admin_authenticated'):
            return render_template("admin_login.html", error=error)

    security_events = get_recent_security_events(50)
    activity = get_recent_user_activity(50)
    return render_template("admin.html", security_events=security_events, activity=activity)

if __name__ == '__main__':

    app.run(debug = os.getenv("FLASK_DEBUG", "false") == "true", port=5000, use_reloader=False)