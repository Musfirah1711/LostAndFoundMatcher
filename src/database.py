"""SQLite storage with duplicate detection and photo support."""
import json
import os
import sqlite3
import uuid
from datetime import datetime

DB_PATH = "lost_found.db"
UPLOAD_DIR = "data/uploads"

def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c

def _ensure_schema(con):
    cols = {r[1] for r in con.execute("PRAGMA table_info(reports)").fetchall()}
    if "photo_path" not in cols:
        con.execute("ALTER TABLE reports ADD COLUMN photo_path TEXT")

def init_db():
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    con = _conn()
    con.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_type TEXT NOT NULL,
            raw_text TEXT NOT NULL,
            pas_json TEXT NOT NULL,
            contact TEXT,
            photo_path TEXT,
            created_at TEXT NOT NULL
        )
    """)
    _ensure_schema(con)
    con.commit()
    con.close()

def save_upload(file_bytes, original_name):
    """Save uploaded bytes to disk, return the stored path."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    ext = original_name.rsplit(".", 1)[-1].lower() if "." in original_name else "jpg"
    if ext not in ("jpg", "jpeg", "png", "webp"):
        ext = "jpg"
    filename = f"{uuid.uuid4().hex[:12]}.{ext}"
    path = os.path.join(UPLOAD_DIR, filename)
    with open(path, "wb") as f:
        f.write(file_bytes)
    return path

def insert_report(report_type, raw_text, pas_dict, contact=None, photo_path=None):
    con = _conn()
    cur = con.cursor()
    cur.execute(
        "INSERT INTO reports (report_type, raw_text, pas_json, contact, photo_path, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (report_type, raw_text, json.dumps(pas_dict), contact, photo_path,
         datetime.utcnow().isoformat())
    )
    con.commit()
    rid = cur.lastrowid
    con.close()
    return rid

def get_reports(report_type=None):
    con = _conn()
    cur = con.cursor()
    if report_type:
        cur.execute("SELECT * FROM reports WHERE report_type=? ORDER BY id DESC",
                    (report_type,))
    else:
        cur.execute("SELECT * FROM reports ORDER BY id DESC")
    rows = cur.fetchall()
    con.close()
    return [dict(r) for r in rows]

def check_duplicate(raw_text, threshold=0.90):
    from .embeddings import embed, cosine_sim
    con = _conn()
    cur = con.cursor()
    cur.execute("SELECT id, raw_text FROM reports")
    rows = cur.fetchall()
    con.close()
    if not rows:
        return None
    new_e = embed(raw_text)
    best_id, best_score = None, 0.0
    for r in rows:
        s = cosine_sim(new_e, embed(r["raw_text"]))
        if s > best_score:
            best_score, best_id = s, r["id"]
    if best_score >= threshold:
        return {"id": best_id, "similarity": round(best_score, 3)}
    return None