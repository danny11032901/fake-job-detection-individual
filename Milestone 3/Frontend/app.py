from flask import Flask, render_template, request, redirect, url_for, session, flash
import sqlite3
from datetime import datetime
import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)
app.secret_key = "supersecretkey123"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "analysis.db")
MODEL_FILE = os.path.join(BASE_DIR, "model.pkl")

# ---------- Initialize Database ----------
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

# ---------- Helper Functions ----------
def load_model():
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, "rb") as f:
            return pickle.load(f)
    else:
        return RandomForestClassifier()

def save_model(model):
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)

# ---------- Routes ----------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    desc = request.form.get("job_description", "").strip()
    if not desc:
        return render_template("index.html", error="Job description cannot be empty")

    # Load model
    model = load_model()
    # Use length as simple feature
    X_test = [[len(desc)]]
    score_pred = model.predict(X_test)[0] if hasattr(model, "predict") else 0
    verdict = "FAKE" if score_pred == 1 else "REAL"
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Save to DB
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO analysis (job_description, score, verdict, created_at) VALUES (?, ?, ?, ?)",
        (desc, int(score_pred*100), verdict, created_at)
    )
    conn.commit()
    conn.close()

    return render_template(
        "result.html",
        job_description=desc,
        verdict=verdict,
        confidence=int(score_pred*100),
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

    results = [{"id": r[0], "job_description": r[1], "score": r[2], "verdict": r[3], "created_at": r[4] or ""} for r in rows]
    return render_template("admin_dashboard.html", results=results)

# ---------- Retrain Model Route ----------
@app.route("/admin/retrain", methods=["POST"])
def retrain_model():
    if "admin" not in session:
        return redirect(url_for("admin_login"))

    if "dataset" not in request.files:
        flash("No dataset uploaded")
        return redirect(url_for("admin_dashboard"))

    file = request.files["dataset"]
    if file.filename == "":
        flash("No file selected")
        return redirect(url_for("admin_dashboard"))

    try:
        df = pd.read_csv(file)
        if "job_description" not in df.columns or "label" not in df.columns:
            flash("Dataset must have 'job_description' and 'label' columns")
            return redirect(url_for("admin_dashboard"))

        # Simple feature: length of job description
        df["length"] = df["job_description"].apply(len)
        X = df[["length"]]
        y = df["label"]  # 1=FAKE, 0=REAL

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        acc = round(model.score(X_test, y_test)*100,2)

        save_model(model)
        flash(f"Model retrained successfully! Accuracy: {acc}%")
        return redirect(url_for("admin_dashboard"))

    except Exception as e:
        flash(f"Error retraining model: {str(e)}")
        return redirect(url_for("admin_dashboard"))

@app.route("/history")
def history():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT job_description, verdict, score, created_at FROM analysis ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    return render_template("history.html", records=rows)

# ---------- Run ----------
if __name__ == "__main__":
    app.run(debug=True)
