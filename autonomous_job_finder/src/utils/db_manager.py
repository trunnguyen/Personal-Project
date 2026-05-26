import sqlite3
import pandas as pd

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

    def upsert_jobs(self, jobs):
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            query='''
                INSERT OR IGNORE INTO jobs (title, company, location, time_posted, job_url)
                VALUES (?, ?, ?, ?, ?)
            '''
            data=[(j['title'],j['company'],j.get('location','N/A'),j['time'],j['link']) for j in jobs]
            cursor.executemany(query, data)
            conn.commit()

            return cursor.rowcount

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