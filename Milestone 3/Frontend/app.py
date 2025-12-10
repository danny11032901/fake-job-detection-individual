from flask import Flask, render_template, request, redirect, url_for, session
import sqlite3
from datetime import datetime
import os

app = Flask(__name__)
app.secret_key = "supersecretkey123"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "analysis.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_description TEXT,
            score INTEGER,
            verdict TEXT,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    desc = request.form.get("job_description", "").strip()
    if not desc:
        return render_template("index.html", error="Job description cannot be empty")

    # Scoring logic
    score = min(len(desc) % 100 + 10, 100)  # confidence 10-100
    verdict = "Fake Job" if score > 50 else "Real Job"
    confidence = score
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Save to DB
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO analysis (job_description, score, verdict, created_at) VALUES (?, ?, ?, ?)",
        (desc, score, verdict, created_at)
    )
    conn.commit()
    conn.close()

    # Pass correct variable names to template
    return render_template(
        "result.html",
        job_description=desc,
        verdict=verdict,
        confidence=confidence,
        created_at=created_at
    )

@app.route("/admin_login", methods=["GET","POST"])
def admin_login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if username=="admin" and password=="admin123":
            session["admin"] = True
            return redirect(url_for("admin_dashboard"))
        return render_template("admin_login.html", error="Invalid credentials")
    return render_template("admin_login.html")

@app.route("/admin_logout")
def admin_logout():
    session.pop("admin", None)
    return redirect(url_for("admin_login"))

@app.route("/admin_dashboard")
def admin_dashboard():
    if "admin" not in session:
        return redirect(url_for("admin_login"))

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM analysis ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()

    results = [{"id": r[0], "job_description": r[1], "score": r[2], "verdict": r[3], "created_at": r[4]} for r in rows]
    return render_template("admin_dashboard.html", results=results)

@app.route("/history")
def history():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT job_description, verdict, score, created_at FROM analysis ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    return render_template("history.html", records=rows)

if __name__ == "__main__":
    app.run(debug=True)
