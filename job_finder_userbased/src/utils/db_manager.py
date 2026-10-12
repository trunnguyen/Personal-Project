import sqlite3
import pandas as pd
from datetime import datetime, timedelta

def now_gmt7():
    return (datetime.utcnow() + timedelta(hours=7)).strftime('%Y-%m-%d %H:%M:%S')

class JobDB:
    def __init__(self, db_name, csv_name):
        self.db_name = db_name
        self.csv_name = csv_name
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_name) as conn:
            conn.execute(
                '''CREATE TABLE IF NOT EXISTS jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT,
                    company TEXT,
                    location TEXT,
                    time_posted TEXT,
                    job_url TEXT UNIQUE,
                    date_found TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    ai_score REAL DEFAULT 0.0,
                    is_applied BOOLEAN DEFAULT 0
                )'''
            )

            existing_cols = [row[1] for row in conn.execute("PRAGMA table_info(jobs)").fetchall()]
            new_columns = {
                "min_year_required": "INTEGER DEFAULT 0",
                "requires_degree": "BOOLEAN DEFAULT 0",
                "min_experience_years": "INTEGER DEFAULT 0",
                "field_match": "BOOLEAN DEFAULT 1",
                "field_match_reason": "TEXT DEFAULT ''",
                "is_eligible": "BOOLEAN DEFAULT 1",
                "eligibility_checked": "BOOLEAN DEFAULT 0"
            }
            for col, col_type in new_columns.items():
                if col not in existing_cols:
                    conn.execute(f"ALTER TABLE jobs ADD COLUMN {col} {col_type}")

            conn.execute('''
                CREATE TABLE IF NOT EXISTS followed_search (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    is_active BOOLEAN DEFAULT 0,
                    last_checked TIMESTAMP,
                    followed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            conn.execute(
                '''CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY, 
                    value TEXT
                )'''
            )

    def upsert_jobs(self, jobs):
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()

            new_count = 0
            for job in jobs:
                cursor.execute(
                    """
                    SELECT COUNT(*) FROM jobs
                    WHERE title = ? AND company = ?
                    AND date_found >= datetime('now', '-7 days')
                    """, (job['title'],job['company']))

                if cursor.fetchone()[0]>0:
                    continue

                cursor.execute("""
                    INSERT OR IGNORE INTO jobs (title, company, location, time_posted, job_url, date_found)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (job['title'], job['company'], job.get('location', 'N/A'), job['time'], job['link'], now_gmt7()))

                if cursor.rowcount > 0:
                    new_count += 1

            conn.commit()
            return new_count

    def export_to_csv(self, csv_name):
        with sqlite3.connect(self.db_name) as conn:
            df = pd.read_sql_query("SELECT * FROM jobs", conn)
            df.to_csv(csv_name,index=False,encoding='utf-8-sig')
            print(f"Csv exported to {csv_name}")

    def update_score_and_interest(self,job_id,is_applied):
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute(
            '''
            UPDATE jobs
            SET is_applied = ?
            WHERE id = ?
            ''',(is_applied,job_id)
            )
            conn.commit()
            rowcount = cursor.rowcount

        if rowcount > 0:
            self.export_to_csv(self.csv_name)

        return cursor.rowcount


    def get_all_jobs(self):
        with sqlite3.connect(self.db_name) as conn:
            conn.row_factory = sqlite3.Row
            cursor=conn.cursor()
            cursor.execute("SELECT * FROM jobs ORDER BY date_found DESC, is_applied DESC")
            return [dict(row) for row in cursor.fetchall()]


    def get_unscored_jobs(self):
        with sqlite3.connect(self.db_name) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                " SELECT * FROM jobs WHERE ai_score = 0.0 ORDER BY date_found DESC"
            )
            return [dict(row) for row in cursor.fetchall()]


    def update_job_score(self, job_id, ai_score):
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute(
            '''
            UPDATE jobs
            SET ai_score = ?
            WHERE id = ?
            ''',(ai_score,job_id)
            )
            conn.commit()
            rowcount = cursor.rowcount
            
        if rowcount > 0:
            self.export_to_csv(self.csv_name)

        return cursor.rowcount

    def update_eligibility(self, job_id, eligibility_result):
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute(
                '''
                UPDATE jobs
                SET min_year_required = ?,
                    requires_degree = ?,
                    min_experience_years = ?,
                    field_match = ?,
                    field_match_reason = ?,
                    is_eligible = ?,
                    eligibility_checked = 1
                WHERE id = ?
                ''',
                (
                    eligibility_result.get('minimum_year_of_study', 0),
                    eligibility_result.get('requires_degree', False),
                    eligibility_result.get('minimum_experience_years', 0),
                    eligibility_result.get('field_match', True),
                    eligibility_result.get('field_match_reason', ''),
                    eligibility_result.get('is_eligible', True),
                    job_id
                )
            )
            conn.commit()
            rowcount = cursor.rowcount

        if rowcount > 0:
            self.export_to_csv(self.csv_name)

        return rowcount

    def get_followed_search(self):
        with sqlite3.connect(self.db_name) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM followed_search ORDER BY id DESC LIMIT 1")
            row = cursor.fetchone()
            return dict(row) if row else None

    def activate_follow(self):
        with sqlite3.connect(self.db_name) as conn:
            existing = conn.execute("SELECT id FROM followed_search LIMIT 1").fetchone()
            if existing:
                conn.execute(
                    "UPDATE followed_search SET is_active = 1, last_checked = ? WHERE id = ?",
                    (now_gmt7(), existing[0])
                )
            else:
                conn.execute(
                    "INSERT INTO followed_search (is_active, last_checked, followed_at) VALUES (1, ?, ?)",
                    (now_gmt7(), now_gmt7())
                )
            conn.commit()

    def deactivate_follow(self):
        with sqlite3.connect(self.db_name) as conn:
            conn.execute("UPDATE followed_search SET is_active = 0")
            conn.commit()

    def update_last_checked(self):
        with sqlite3.connect(self.db_name) as conn:
            conn.execute("UPDATE followed_search SET last_checked = ?", (now_gmt7(),))
            conn.commit()

    def get_high_matches_since(self, threshold, since_data):
        with sqlite3.connect(self.db_name) as conn:
            conn.row_factory= sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM jobs 
                WHERE ai_score >= ?
                AND is_applied = 0
                AND date_found >= ?
                ORDER BY ai_score DESC
            """, (threshold, since_data))
            rows = [dict(row) for row in cursor.fetchall()]

        seen ={}
        for row in rows:
            key = (row['title'], row['company'])
            if key not in seen or row['date_found'] > seen[key]['date_found']:
                seen[key] = row
        return list(seen.values())

    def get_last_reported_date(self):
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT value FROM meta
                WHERE key ='last_reported_date'
            """)
            row = cursor.fetchone()
            return row[0] if row else '2000-01-01'

    def set_last_reported_date(self, date):
        with sqlite3.connect(self.db_name) as conn:
            conn.execute("""
                        CREATE TABLE IF NOT EXISTS meta 
                        (key TEXT PRIMARY KEY, value TEXT)
                         """)
            conn.execute("""
                        INSERT OR REPLACE INTO meta (key, value)
                        VALUES ('last_reported_date', ?)
                        """,(date,))
            conn.commit()