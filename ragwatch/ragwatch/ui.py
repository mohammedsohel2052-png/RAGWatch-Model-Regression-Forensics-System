import os
import sqlite3
import json
from flask import Flask, render_template, jsonify

app = Flask(__name__)

# Try to find databases in the current working directory
EVAL_DB = "eval_results.db"
RAGWATCH_DB = "ragwatch.db"

def get_db_connection(db_path):
    if not os.path.exists(db_path):
        return None
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/runs")
def get_runs():
    conn = get_db_connection(EVAL_DB)
    if not conn:
        return jsonify([])
    try:
        runs = conn.execute("SELECT * FROM runs ORDER BY timestamp DESC LIMIT 50").fetchall()
        return jsonify([dict(r) for r in runs])
    except Exception as e:
        print(f"Error reading eval runs: {e}")
        return jsonify([])
    finally:
        conn.close()

@app.route("/api/traces")
def get_traces():
    conn = get_db_connection(RAGWATCH_DB)
    if not conn:
        return jsonify([])
    try:
        traces = conn.execute("SELECT * FROM traces ORDER BY timestamp DESC LIMIT 100").fetchall()
        return jsonify([dict(t) for t in traces])
    except Exception as e:
        print(f"Error reading traces: {e}")
        return jsonify([])
    finally:
        conn.close()

def run():
    print("[RAGWatch UI] Starting RAGWatch Local Dashboard on http://localhost:5050")
    print("Press CTRL+C to quit.")
    app.run(host="0.0.0.0", port=5050, debug=False)

if __name__ == "__main__":
    run()
