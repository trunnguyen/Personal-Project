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
        self.server = os.getenv("SMTP_SERVER", "").strip()
        self.port = os.getenv("SMTP_PORT", "").strip()
        self.sender = os.getenv("EMAIL_ADDRESS", "").strip()
        self.password = os.getenv("EMAIL_PASSWORD", "").strip()
        self.receiver = os.getenv("EMAIL_ADDRESS", "").strip()

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
    
        GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
        REPO_OWNER="trunnguyen"
        REPO_NAME="Personal-Project"

        for job in highly_rated_jobs:
                job_id=job.get('id')
                issue_url=f"https://github.com/trunnguyen/Personal-Project/issues/new?title=Applied+to+Job+{job_id}&body=mark_applied:{job_id}"
                body +=f"""
                <div style="padding: 15px; border: 1px solid #ddd; margin-bottom: 15px; border-radius: 5px; font-family: Arial, sans-serif;">
                    <h3 style="margin-top: 0; color: #333;">{job['title']} at <span style="color: #0066cc;">{job['company']}</span></h3>
                    <p><strong>Location:</strong> {job['location']} | <strong>Match Score:</strong> {job['ai_score']:.2%}</p>
                
                    <p style="margin-bottom: 15px;">
                        <a href="{job['job_url']}" target="_blank" style="color: #0066cc; text-decoration: none; font-weight: bold;">[View Original Job Posting]</a>
                    </p>
                
                    <p style="margin-top: 10px;">
                        <a href="{issue_url}" target="_blank" 
                            style="background-color: #2ea44f; color: white; padding: 8px 14px; text-decoration: none; border-radius: 4px; display: inline-block; font-size: 13px; font-weight: bold;">
                            Mark as Applied
                        </a>
                    </p>
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