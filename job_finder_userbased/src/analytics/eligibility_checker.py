import ollama
import json
from src.utils.logger import logger


def build_eligibility_prompt(title, company, description, user_major, user_job_title):
    return f"""You are analyzing a job posting to determine if it fits a student's profile.

STUDENT PROFILE:
- Major: {user_major}
- Looking for: {user_job_title}

JOB TITLE: {title}
COMPANY: {company}

JOB DESCRIPTION:
{description[:2000]}

Based ONLY on the description above, respond with a JSON object containing exactly these fields:
- "minimum_year_of_study": integer (1-5), or 0 if not mentioned
- "requires_degree": true ONLY if the posting says a completed degree is mandatory (for example "Bachelor's degree required", "must have graduated"). Answer false if it mentions "student", "undergraduate", "currently studying", "currently pursuing", "3rd year", "final year", "sinh viên", or is an internship open to students.
- "minimum_experience_years": integer, or 0 if not mentioned
- "field_match": true or false — does this role's actual responsibilities match the student's major of "{user_major}"?
- "field_match_reason": one short sentence explaining why

Respond with ONLY the JSON object, no other text."""


def detect_reasoning_boolean_conflict(field_match, reason_text):
    reason_lower = reason_text.lower()

    positive_signals = ["aligns", "align", "matches", "match", "suitable", "relevant", "fits", "fit"]
    negative_signals = ["does not align", "not align", "doesn't align", "not match", "not suited",
                         "not relevant", "not fit", "does not directly align", "not aligned"]

    has_negative_phrase = any(neg in reason_lower for neg in negative_signals)
    has_positive_word = any(pos in reason_lower for pos in positive_signals)

    if field_match == False and has_positive_word and not has_negative_phrase:
        return True

    return False

def check_eligibility(title, company, description, user_major, user_job_title, model="qwen3:8b"):
    if not description:
        return {
            "minimum_year_of_study": 0,
            "requires_degree": False,
            "minimum_experience_years": 0,
            "field_match": True,
            "field_match_reason": "No description available to verify.",
            "is_eligible": True
        }

    prompt = build_eligibility_prompt(title, company, description, user_major, user_job_title)

    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            think=False
        )

        result = json.loads(response.message.content)

        if not result.get('field_match_reason', '').strip():
            raise ValueError("Qwen returned an empty field_match_reason")

        if not result.get('field_match_reason', '').strip():
            raise ValueError("Qwen returned an empty field_match_reason")

        reason = result.get('field_match_reason', '')
        stated_match = result.get('field_match', True)

        if detect_reasoning_boolean_conflict(stated_match, reason):
            logger.warning(f"Reasoning/boolean conflict detected for '{title}': "
                          f"field_match=False but reason says '{reason}'. Correcting to True.")
            result['field_match'] = True

        result['is_eligible'] = (
            result.get('minimum_year_of_study', 0) <= 4 and
            result.get('requires_degree', False) == False and
            result.get('minimum_experience_years', 0) == 0 and
            result.get('field_match', True) == True
        )

        if not result['is_eligible']:
            disqualifiers = []
            if result.get('minimum_year_of_study', 0) > 4:
                disqualifiers.append(f"requires year {result['minimum_year_of_study']} or higher")
            if result.get('requires_degree', False):
                disqualifiers.append("requires a completed degree")
            if result.get('minimum_experience_years', 0) > 0:
                disqualifiers.append(f"requires {result['minimum_experience_years']}+ years experience")
            if not result.get('field_match', True):
                disqualifiers.append("field does not match major")

            if disqualifiers:
                result['field_match_reason'] = f"{result['field_match_reason']} (Also: {', '.join(disqualifiers)})"

        return result

    except Exception as e:
        logger.error(f"Eligibility check failed: {e}")
        return {
            "minimum_year_of_study": 0,
            "requires_degree": False,
            "minimum_experience_years": 0,
            "field_match": True,
            "field_match_reason": "Check failed, defaulting to eligible.",
            "is_eligible": True
        }