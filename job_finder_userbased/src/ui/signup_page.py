import streamlit as st
import re
from src.auth.otp_manager import generate_otp, verify_otp, is_rate_limited
from src.utils.notifier import Notifier


def validate_inputs(name, major, job_title, email):
    errors = []

    if not (1 <= len(name.strip()) <= 50) or not re.match(r"^[A-Za-zÀ-ỹ\s]+$", name):
        errors.append("Name must be 1-50 characters, letters only.")

    if not (1 <= len(major.strip()) <= 50) or not re.match(r"^[A-Za-zÀ-ỹ0-9\s\-&,./]+$", major):
        errors.append("Major must be 1-50 characters (letters, numbers, spaces, - & , . / only).")

    if not (1 <= len(job_title.strip()) <= 100) or not re.match(r"^[A-Za-zÀ-ỹ0-9\s\-&,./()]+$", job_title):
        errors.append("Job title must be 1-100 characters (letters, numbers, spaces, - & , . / ( ) only).")

    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        errors.append("Please enter a valid email address.")

    return errors

def render_signup_page():
    st.title("Job Recommender - Sign Up")

    if "otp_sent" not in st.session_state:
        st.session_state.otp_sent = False
    if "verified" not in st.session_state:
        st.session_state.verified = False
    if "sending" not in st.session_state:
        st.session_state.sending = False

    if not st.session_state.otp_sent:
        name = st.text_input("Name")
        major = st.text_input("Major")
        job_title = st.text_input("Desired Job")
        email = st.text_input("Email")

        if st.button("Send Verification Code", disabled=st.session_state.sending):
            st.session_state.sending = True
            errors = validate_inputs(name, major, job_title, email)
            if errors:
                for e in errors:
                    st.error(e)
                st.session_state.sending = False
            elif is_rate_limited(email):
                st.error("Too many requests, try again later.")
                st.session_state.sending = False
            else:
                notifier = Notifier()
                code = generate_otp(email)
                success = notifier.send_otp_email(email, code)

                if success:
                    st.session_state.otp_sent = True
                    st.session_state.user_email = email
                    st.session_state.user_name = name
                    st.session_state.user_major = major
                    st.session_state.user_job_title = job_title
                    st.session_state.sending = False
                    st.rerun()
                else:
                    st.error("Failed to send verification code")
                    st.session_state.sending = False


    elif not st.session_state.verified:

        st.write(f"A code was sent to {st.session_state.user_email}")

        input_code = st.text_input("Enter Verification Code")

        col1, col2 = st.columns(2)

        with col1:

            if st.button("Verify"):

                if verify_otp(st.session_state.user_email, input_code):

                    st.session_state.verified = True

                    st.rerun()

                else:

                    st.error("Incorrect or expired code. Please try again.")

        with col2:

            if st.button("Resend Code"):

                if is_rate_limited(st.session_state.user_email):

                    st.error("Too many requests, try again later.")

                else:

                    notifier = Notifier()

                    new_code = generate_otp(st.session_state.user_email)

                    success = notifier.send_otp_email(st.session_state.user_email, new_code)

                    if success:

                        st.success("A new code has been sent.")

                    else:

                        st.error("Failed to resend code.")

    else:
        st.success("Redirecting to DashBoard")