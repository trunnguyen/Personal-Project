import smtplib
import os
import sys
import dotenv
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.utils.logger import logger

dotenv.load_dotenv()

REQUIRED_ENV = ["SMTP_SERVER",
                "SMTP_PORT",
                "EMAIL_ADDRESS",
                "EMAIL_PASSWORD",
                ]


def validate_env():
    missing = [k for k in REQUIRED_ENV if not os.getenv(k, "").strip()]
    if missing:
        error_msg = f"Missing or empty environment variables: {missing}"
        logger.error(error_msg)
        raise Exception(error_msg)


class Notifier():
    def __init__(self):
        self.server = os.getenv("SMTP_SERVER", "").strip()
        self.port = os.getenv("SMTP_PORT", "").strip()
        self.sender = os.getenv("EMAIL_ADDRESS", "").strip()
        self.password = os.getenv("EMAIL_PASSWORD", "").strip()
        self.receiver = os.getenv("EMAIL_ADDRESS", "").strip()

        if not all([self.server, self.port, self.sender, self.password, self.receiver]):
            logger.error("Environment variables not set")

    def send_otp_email(self, receiver_email, code):
        msg = MIMEMultipart()
        msg['From'] = self.sender
        msg['To'] = receiver_email
        msg['Subject'] = "Your verification code"

        body = f"""
        <div style="font-family: Arial, sans-serif; padding: 20px;">
            <h2>Your verification code</h2>
            <p style="font-size: 32px; font-weight: bold; letter-spacing: 4px;">{code}</p>
            <p>This code expires in 5 minutes and can only be used once.</p>
        </div>
        """
        msg.attach(MIMEText(body, 'html'))

        try:
            server = smtplib.SMTP(self.server, int(self.port))
            server.starttls()
            server.login(self.sender, self.password)
            server.sendmail(self.sender, receiver_email, msg.as_string())
            server.quit()
            logger.info(f"OTP email sent to {receiver_email}")
            return True
        except Exception as e:
            logger.error(f"OTP email delivery failed: {e}")
            return False

    def send_job_digest(self, receiver_email, jobs, subject_prefix="New Job Matches"):
        if not jobs:
            logger.info(f"No jobs to send to {receiver_email}")
            return False

        body = f"<h3>{subject_prefix} in Ho Chi Minh City</h3><br>"

        for job in jobs:
            eligibility_note = ""
            if job.get('eligibility_checked', 0) == 1:
                if job.get('is_eligible', 1) == 1:
                    eligibility_note = f"<p style='color:#4CAF50;'>✅ {job.get('field_match_reason', '')}</p>"
                else:
                    eligibility_note = f"<p style='color:#FFA726;'>⚠️ {job.get('field_match_reason', '')}</p>"

            body += f"""
            <div style="padding: 15px; border: 1px solid #ddd; margin-bottom: 15px; border-radius: 5px; font-family: Arial, sans-serif;">
                <h3 style="margin-top: 0; color: #333;">{job['title']} at <span style="color: #0066cc;">{job['company']}</span></h3>
                <p><strong>Location:</strong> {job['location']} | <strong>Match Score:</strong> {job['ai_score']:.2%}</p>
                {eligibility_note}
                <p style="margin-bottom: 15px;">
                    <a href="{job['job_url']}" target="_blank" style="color: #0066cc; text-decoration: none; font-weight: bold;">[View Original Job Posting]</a>
                </p>
            </div>
            """

        msg = MIMEMultipart()
        msg['From'] = self.sender
        msg['To'] = receiver_email
        msg['Subject'] = f"{len(jobs)} {subject_prefix}"
        msg.attach(MIMEText(body, 'html'))

        try:
            server = smtplib.SMTP(self.server, int(self.port))
            server.starttls()
            server.login(self.sender, self.password)
            server.sendmail(self.sender, receiver_email, msg.as_string())
            server.quit()
            logger.info(f"Job digest sent to {receiver_email}")
            return True
        except Exception as e:
            logger.error(f"Job digest delivery failed: {e}")
            return False