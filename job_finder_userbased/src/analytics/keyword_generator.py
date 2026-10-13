import ollama
import json
from src.utils.logger import logger
import re


def build_keyword_prompt(major, job_title):
    return f"""Given a student's major and desired role, suggest realistic LinkedIn job title searches
that genuinely combine both the major and the role — not just the role alone.

Examples:
Major: Marketing, Role: HR → "Employer Branding Intern", "People Operations Intern", "Internal Communications Intern"
Major: Finance, Role: Data Analyst → "Financial Data Analyst Intern", "FP&A Intern", "Business Analyst Intern"
Major: Communication, Role: HR → "Internal Communications Intern", "Employer Branding Intern", "HR Communications Intern"

Major: {major}, Role: {job_title} → Generate 3 similar realistic search phrases that combine both fields.

Respond with ONLY JSON, no other text: {{"keywords": ["...", "...", "..."]}}
"""


def generate_search_keywords(major, job_title, model="qwen3:8b"):
    prompt = build_keyword_prompt(major, job_title)

    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={"temperature": 0},
            think=False
        )

        result = json.loads(response.message.content)
        keywords = result.get("keywords", [])

        if not keywords:
            logger.warning("Qwen returned no keywords, falling back to job title")
            return [f"{job_title} Intern"]

        pattern = re.compile(r"\b(intern|interns|internship|trainee|fresher)\b", re.I)
        cleaned = []
        for k in keywords:
            k = k.strip()
            if k and not pattern.search(k):
                k = f"{k} Intern"
            if k:
                cleaned.append(k)
        return cleaned or [f"{job_title} Intern"]

    except Exception as e:
        logger.error(f"Keyword generation failed: {e}")
        return [f"{job_title} Intern"]