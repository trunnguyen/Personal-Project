import streamlit as st
import asyncio

from src.ui.signup_page import render_signup_page
from src.auth.user_session import get_user_data_paths
from src.utils.db_manager import JobDB
from src.scraper import crawler as job_crawler
from src.scraper.crawler import fetch_job_description
from src.analytics.job_recommender import Recommender
from src.analytics.eligibility_checker import check_eligibility

st.set_page_config(page_title="Job Recommender", layout="wide")

if "verified" not in st.session_state or not st.session_state.verified:
    render_signup_page()
else:
    st.title("Dashboard")
    if st.button("Refresh"):
        st.rerun()

    db_path, csv_path = get_user_data_paths(
        st.session_state.user_email,
        st.session_state.user_job_title
    )
    db = JobDB(db_path, csv_path)

    if st.button("Find Jobs For Me"):
        with st.spinner("Generating search strategy..."):
            from src.analytics.keyword_generator import generate_search_keywords

            keywords = generate_search_keywords(
                st.session_state.user_major,
                st.session_state.user_job_title
            )
            st.write(f"Searching for: {', '.join(keywords)}")

        with st.spinner("Scraping LinkedIn and scoring jobs..."):

            asyncio.run(job_crawler.main(
                db_path=db_path,
                csv_path=csv_path,
                search_keywords=keywords
            ))

            recommender = Recommender()
            unscored = db.get_unscored_jobs()

            if unscored:
                scored = recommender.score_for_user(
                    unscored,
                    st.session_state.user_name,
                    st.session_state.user_major,
                    st.session_state.user_job_title
                )
                for job in scored:
                    db.update_job_score(job['id'], job['ai_score'])

        with st.spinner("Checking eligibility on top matches..."):
            all_jobs = db.get_all_jobs()
            top_jobs = sorted(all_jobs, key=lambda x: x['ai_score'], reverse=True)[:10]

            for job in top_jobs:
                if job.get('eligibility_checked', 0) == 0:
                    description = asyncio.run(fetch_job_description(job['job_url']))
                    result = check_eligibility(
                        title=job['title'],
                        company=job['company'],
                        description=description,
                        user_major=st.session_state.user_major,
                        user_job_title=st.session_state.user_job_title
                    )
                    db.update_eligibility(job['id'], result)

        st.success("Done! Refresh below to see your matches.")

    st.write(f"Welcome, {st.session_state.user_name}")
    st.write(f"Major: {st.session_state.user_major}")
    st.write(f"Looking for: {st.session_state.user_job_title}")

    jobs = db.get_all_jobs()
    scored_jobs = [j for j in jobs if j.get('ai_score', 0.0) > 0]
    scored_jobs = sorted(scored_jobs, key=lambda x: x['ai_score'], reverse=True)

    st.write(f"You have {len(scored_jobs)} scored jobs for {st.session_state.user_job_title}")
    st.divider()

    for job in scored_jobs:
        with st.container():
            col1, col2 = st.columns([4, 1])

            with col1:
                st.markdown(f"### {job['title']}")
                st.write(f"**Company:** {job['company']} | **Location:** {job['location']}")
                st.write(f"**Match Score:** {job['ai_score']:.2%}")

                min_exp = job.get('min_experience_years', 0)
                if min_exp == 0:
                    level_label = "Entry Level / Internship"
                elif min_exp <= 2:
                    level_label = f"Junior ({min_exp}+ years)"
                else:
                    level_label = f"Senior ({min_exp}+ years)"

                if job.get('eligibility_checked', 0) == 1:
                    st.write(f"**Level:** {level_label}")
                st.markdown(f"[View Job Posting]({job['job_url']})")
                if job.get('eligibility_checked', 0) == 1:
                    if job.get('is_eligible', 1) == 1:
                        st.write(f"**Eligible** — {job.get('field_match_reason', '')}")
                    else:
                        st.write(f"**May not be eligible** — {job.get('field_match_reason', '')}")

            with col2:
                is_applied = job.get('is_applied', 0)
                if is_applied == 0:
                    if st.button("Interested", key=f"like_{job['id']}"):
                        db.update_score_and_interest(job['id'], is_applied=1)
                        st.rerun()
                else:
                    if st.button("Not Interested", key=f"dislike_{job['id']}"):
                        db.update_score_and_interest(job['id'], is_applied=0)
                        st.rerun()

            st.divider()