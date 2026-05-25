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

class Notifier():
    def __init__(self):
        dotenv.load_dotenv()
        self.server = os.getenv("SMTP_SERVER")
        self.port = os.getenv("SMTP_PORT")
        self.sender = os.getenv("EMAIL_ADDRESS")
        self.password = os.getenv("EMAIL_PASSWORD")
        self.receiver = os.getenv("EMAIL_ADDRESS")

        if not all([self.server, self.port, self.sender, self.password, self.receiver]):
            logger.error("Environment variables not set")
    def send_report(self,highly_rated_jobs):
        logger.info("High-Relevance Positions Found")
        print("\n" + "=" *60)
        print("High-Relevance Roles Match Report")
        print("=" *60)
        for idx, job in enumerate(highly_rated_jobs,1):
            print(f"{idx}. {job['title']} at {job['company']}")
            print(f" Location: {job['location']}")
            print(f"Matching Score {job['ai_score'] :.2%}")
            print(f"Link: {job['job_url']} ")
            print("-" * 60)

    def send_notification(self,highly_rated_jobs):
        if not  highly_rated_jobs:
            logger.info("No High-Relevant Jobs Found")
            return
        body="<h3> New AI/ML Intern Roles Matches in Ho Chi Minh City</h3><br>"
        for job in highly_rated_jobs:
            body +=f"""
            <div style="border: 1px solid #ccc;padding: 10px;margin-bottom: 10px;">
                <b>{job['title']}</b> - <i>{job['company']}</i><br>
                Location: {job['location']}<br>
                Match Score: {job['ai_score'] :.2%}<br>
                <a href="{job['job_url']}">Apply via LinkedIn</a>
            </div>
            """
        msg = MIMEMultipart()
        msg['From'] = self.sender
        msg['To'] = self.receiver
        msg['Subject'] = f"{len(highly_rated_jobs)} New AI/ML Intern Roles Found"
        msg.attach(MIMEText(body, 'html'))

        try:
            # Commented out safety lock until you customize email keys
            server = smtplib.SMTP(self.server, int(self.port))
            server.starttls()
            server.login(self.sender, self.password)
            server.sendmail(self.sender, self.receiver, msg.as_string())
            server.quit()
            logger.info("Email notification successfully sent.")
        except Exception as e:
            logger.error(f"Email Delivery failed. {e}")

#Test
# if __name__ == "__main__":
#     notifier = Notifier()
#     test_jobs = [
#         {
#             "title": "AI Intern",
#             "company": "TechCorp",
#             "location": "Ho Chi Minh City",
#             "ai_score": 0.92,
#             "job_url": "https://linkedin.com/job/12345"
#         },
#         {
#             "title": "ML Research Assistant",
#             "company": "DataWorks",
#             "location": "Ho Chi Minh City",
#             "ai_score": 0.87,
#             "job_url": "https://linkedin.com/job/67890"
#         }
#     ]
#     notifier.send_report(test_jobs)
#     notifier.send_notification(test_jobs)