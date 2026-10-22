import re
from datetime import datetime, timedelta

_UNITS = {
    "minute": timedelta(minutes=1),
    "hour": timedelta(hours=1),
    "day": timedelta(days=1),
    "week": timedelta(weeks=1),
    "month": timedelta(days=30),
    "year": timedelta(days=365),
}
_PATTERN = re.compile(r"(\d+)\s*(minute|hour|day|week|month|year)s?\s+ago", re.I)

def posted_label(time_posted, date_found, now=None):
    if not time_posted or not date_found:
        return ""
    text=str(time_posted).strip().lower()
    try:
        found = datetime.strptime(str(date_found)[:19], "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return ""

    approx = False
    if text in ("just now", "today"):
        delta = timedelta(0)
    elif text == "yesterday":
        delta = timedelta(days=1)
    else:
        m = _PATTERN.search(text)
        if not m:
            return ""
        unit = m.group(2).lower()
        delta = _UNITS[unit] * int(m.group(1))
        approx = unit in ("week", "month", "year")

    posted = found - delta
    now = now or (datetime.utcnow() + timedelta(hours=7))
    age_days = max((now.date() - posted.date()).days, 0)
    if age_days == 0:
        age = "today"
    elif age_days == 1:
        age = "1 day ago"
    else:
        age = f"{age_days} days ago"
    return f"{'about ' if approx else ''}{posted:%Y-%m-%d} ({age})"