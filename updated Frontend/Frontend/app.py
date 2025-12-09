# app.py
import os
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from pymongo import MongoClient
from datetime import timedelta
from bson.objectid import ObjectId

# Config
MONGO_URI = os.environ.get('MONGO_URI', 'mongodb://localhost:27017')
DB_NAME = os.environ.get('DB_NAME', 'fake_job_detector')
ADMIN_SESSION_KEY = 'fjd_admin'

app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = os.environ.get('FLASK_SECRET', 'dev-secret-change-me')
app.permanent_session_lifetime = timedelta(hours=6)

# Mongo
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
results_col = db.results
admins_col = db.admins

# Helper: create an admin (run directly or call endpoint once)
def create_admin(username: str, password: str):
    if admins_col.find_one({'username': username}):
        return False
    hash_pw = generate_password_hash(password)
    admins_col.insert_one({'username': username, 'password_hash': hash_pw})
    return True

# Set up a quick CLI helper if you run 'python app.py create_admin user pass'
import sys
if len(sys.argv) >= 2 and sys.argv[1] == 'create_admin':
    u = sys.argv[2]
    p = sys.argv[3]
    ok = create_admin(u, p)
    print("Admin created:" , ok)
    sys.exit(0)

# ROUTES

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Example server-side endpoint:
    - Accepts JSON: { job_description: "..."}
    - Returns JSON: { verdict, score, details }
    For now we run the lightweight heuristics (same as client) and return JSON.
    """
    data = request.get_json() or {}
    txt = (data.get('job_description') or '').strip()
    if not txt:
        return jsonify({'error':'no job_description provided'}), 400

    # Simple heuristics (mirrors client-side fallback but server will save)
    t = txt.lower()
    score = 10
    reasons = []
    if any(w in t for w in ['urgent','immediate','asap']): score += 15; reasons.append('Urgent language')
    if any(w in t for w in ['wire transfer','western union','upi','bank account','send money','pay via', 'pay ₹']): score += 30; reasons.append('Requests payment')
    if 'whatsapp' in t or 'contact' in t and ('@' not in t):
        score += 20; reasons.append('Direct external contact / WhatsApp')
    if any(w in t for w in ['no experience','easy money','guaranteed','100% commission']): score += 15; reasons.append('Too-good-to-be-true claims')
    if any(w in t for w in ['manager','recruiter','hr@','careers@']): score -= 10; reasons.append('Has corporate emails')
    if len(t) < 120: score += 10; reasons.append('Short posting')

    score = max(0, min(100, score))
    verdict = 'FAKE' if score >= 50 else 'LIKELY_REAL'
    details = '; '.join(reasons) if reasons else 'No obvious suspicious signals found.'

    # Save to DB (non-blocking pattern not necessary here; keep simple)
    result_doc = {
        'job_description': txt[:2000],
        'score': score,
        'verdict': verdict,
        'details': details,
        'created_at': __import__('datetime').datetime.utcnow()
    }
    results_col.insert_one(result_doc)

    return jsonify({'verdict': verdict, 'score': score, 'details': details})

# Admin login page
@app.route('/admin_login', methods=['GET','POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form.get('username','').strip()
        password = request.form.get('password','').strip()
        admin = admins_col.find_one({'username': username})
        if admin and check_password_hash(admin['password_hash'], password):
            session.permanent = True
            session[ADMIN_SESSION_KEY] = str(admin['_id'])
            flash('Signed in', 'success')
            return redirect(url_for('admin_dashboard'))
        else:
            flash('Invalid credentials', 'error')
            return redirect(url_for('admin_login'))
    return render_template('admin_login.html')

@app.route('/admin_logout')
def admin_logout():
    session.pop(ADMIN_SESSION_KEY, None)
    flash('Logged out', 'info')
    return redirect(url_for('index'))

@app.route('/admin_dashboard')
def admin_dashboard():
    if ADMIN_SESSION_KEY not in session:
        return redirect(url_for('admin_login'))
    # show latest 200 results (server-side pagination can be added)
    docs = list(results_col.find().sort('created_at', -1).limit(200))
    # convert ObjectIds and datetimes to friendly values
    for d in docs:
        d['_id'] = str(d['_id'])
        if 'created_at' in d:
            d['created_at_str'] = d['created_at'].strftime('%Y-%m-%d %H:%M:%S UTC')
    return render_template('admin_dashboard.html', results=docs)

# Simple endpoint to export one result
@app.route('/admin/result/<id>')
def admin_result(id):
    if ADMIN_SESSION_KEY not in session:
        return redirect(url_for('admin_login'))
    d = results_col.find_one({'_id': ObjectId(id)})
    if not d:
        return "Not found", 404
    d['_id'] = str(d['_id'])
    d['created_at_str'] = d['created_at'].strftime('%Y-%m-%d %H:%M:%S UTC')
    return render_template('admin_result.html', result=d)

if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get('PORT', 5000)))
