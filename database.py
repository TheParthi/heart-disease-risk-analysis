import sqlite3
from datetime import datetime

def create_db():
    conn = sqlite3.connect("predictions.db")
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            age INTEGER,
            sex TEXT,
            bp INTEGER,
            chol INTEGER,
            result TEXT,
            time TEXT
        )
    """)
    conn.commit()
    conn.close()

def insert_data(age, sex, bp, chol, result):
    conn = sqlite3.connect("predictions.db")
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO predictions (age, sex, bp, chol, result, time)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (age, sex, bp, chol, result, datetime.now()))
    conn.commit()
    conn.close()

def fetch_data():
    conn = sqlite3.connect("predictions.db")
    cur = conn.cursor()
    cur.execute("SELECT * FROM predictions ORDER BY id DESC")
    rows = cur.fetchall()
    conn.close()
    return rows
