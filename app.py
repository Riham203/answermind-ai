import json
import os
import sqlite3
import time
from flask import Flask, Response, redirect, render_template, request, send_from_directory, session, stream_with_context, url_for
from werkzeug.security import check_password_hash, generate_password_hash

from logic.rag import generate_legal_answer, stream_text_by_word

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-this")
DB_PATH = "users.db"


def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL DEFAULT 'User',
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    columns = conn.execute("PRAGMA table_info(users)").fetchall()
    column_names = {row[1] for row in columns}
    if "username" not in column_names:
        conn.execute("ALTER TABLE users ADD COLUMN username TEXT NOT NULL DEFAULT 'User'")
    conn.commit()
    conn.close()


def get_user_by_email(email: str):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    user = conn.execute("SELECT * FROM users WHERE email = ?", (email.strip().lower(),)).fetchone()
    conn.close()
    return user


def create_user(username: str, email: str, password: str) -> bool:
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
            (username.strip(), email.strip().lower(), generate_password_hash(password)),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


@app.route("/")
def dashboard():
    logged_in = "user_email" in session
    return render_template(
        "dashboard.html",
        logged_in=logged_in,
        username=session.get("username", "User") if logged_in else None,
        user_email=session.get("user_email") if logged_in else None,
    )


@app.route("/chat")
def chat():
    if "user_email" not in session:
        return redirect(url_for("login"))
    return render_template("chat.html", username=session.get("username", "User"))


@app.route("/favicon.ico")
def favicon():
    return send_from_directory(app.static_folder, "favicon.svg", mimetype="image/svg+xml")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if "user_email" in session:
        return redirect(url_for("dashboard"))

    error = None
    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""
        confirm_password = request.form.get("confirm_password") or ""

        if not username or not email or not password or not confirm_password:
            error = "All fields are required."
        elif len(username) < 2:
            error = "Username must be at least 2 characters."
        elif password != confirm_password:
            error = "Passwords do not match."
        elif len(password) < 6:
            error = "Password must be at least 6 characters."
        elif not create_user(username, email, password):
            error = "An account with this email already exists."
        else:
            session["user_email"] = email
            session["username"] = username
            return redirect(url_for("dashboard"))

    return render_template("signup.html", error=error)


@app.route("/login", methods=["GET", "POST"])
def login():
    if "user_email" in session:
        return redirect(url_for("dashboard"))

    error = None
    if request.method == "POST":
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""

        user = get_user_by_email(email)
        if not user or not check_password_hash(user["password_hash"], password):
            error = "Invalid email or password."
        else:
            session["user_email"] = user["email"]
            session["username"] = user["username"] or "User"
            return redirect(url_for("dashboard"))

    return render_template("login.html", error=error)


@app.route("/logout", methods=["POST"])
def logout():
    session.pop("user_email", None)
    session.pop("username", None)
    return redirect(url_for("dashboard"))


@app.route("/stream", methods=["POST"])
def stream():
    if "user_email" not in session:
        return {"error": "Unauthorized"}, 401

    payload = request.get_json(silent=True) or {}
    query = (payload.get("message") or "").strip()

    if not query:
        return {"error": "Message is required"}, 400

    answer, citations = generate_legal_answer(query=query, top_k=3)

    def generate():
        # Initial metadata event for citations panel.
        meta = {
            "type": "meta",
            "citations": [
                {"title": c.title, "section": c.section, "excerpt": c.excerpt} for c in citations
            ],
        }
        yield f"data: {json.dumps(meta)}\n\n"

        # Stream answer token-by-token (word-by-word here).
        for token in stream_text_by_word(answer):
            chunk = {"type": "chunk", "content": token}
            yield f"data: {json.dumps(chunk)}\n\n"
            time.sleep(0.03)

        # End signal.
        done = {"type": "done"}
        yield f"data: {json.dumps(done)}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    init_db()
    app.run(debug=True, host="0.0.0.0", port=5001)
