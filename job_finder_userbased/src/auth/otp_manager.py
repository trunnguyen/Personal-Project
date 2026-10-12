import random
import time
import hashlib
import secrets
import hmac

from src.auth.admin_logger import log_security_event

MAX_OTP_ATTEMPTS = 5

otp_store = {}

rate_limit_store = {}

def is_rate_limited(email: str, max_requests=5, window_seconds=3600) -> bool:
    db_id = get_user_db_id(email)
    now = time.time()

    if db_id not in rate_limit_store:
        rate_limit_store[db_id] = []

    rate_limit_store[db_id] = [t for t in rate_limit_store[db_id] if now - t < window_seconds]

    if len(rate_limit_store[db_id]) >= max_requests:
        log_security_event("rate_limit_triggered", db_id, f"{max_requests} requests in {window_seconds}s")
        return True

    rate_limit_store[db_id].append(now)
    #print(f"DEBUG: {db_id} has {len(rate_limit_store[db_id])} requests in window")
    return False

def get_user_db_id(email: str) -> str:
    return hashlib.sha256(email.strip().lower().encode()).hexdigest()

def generate_otp(email: str, expiry_seconds=300) -> str:
    code = str(secrets.randbelow(900000) + 100000)
    db_id = get_user_db_id(email)
    otp_store[db_id] = {
        "code": code,
        "expires_at": time.time() + expiry_seconds,
        "used": False,
        "attempts": 0,
    }
    return code


def verify_otp(email: str, input_code: str) -> bool:
    db_id = get_user_db_id(email)
    entry = otp_store.get(db_id)

    if not entry:
        log_security_event("otp_verify_failed", db_id, "No pending OTP found")
        return False
    if entry["used"]:
        log_security_event("otp_verify_failed", db_id, "Code already used")
        return False
    if time.time() > entry["expires_at"]:
        log_security_event("otp_verify_failed", db_id, "Code expired")
        return False
    if entry["attempts"] >= MAX_OTP_ATTEMPTS:
        log_security_event("otp_verify_failed", db_id, "Too many wrong attempts, code locked")
        return False
    if not hmac.compare_digest(entry["code"].encode(), input_code.encode()):
        entry["attempts"] += 1
        log_security_event(
            "otp_verify_failed", db_id,
            f"Incorrect code (attempt {entry['attempts']}/{MAX_OTP_ATTEMPTS})"
        )
        return False

    entry["used"] = True
    return True