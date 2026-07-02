
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import sqlite3
from datetime import datetime

from flask import Flask, jsonify, render_template, request

from scheduler.process_generator import Process, generate_processes, clone_processes
from scheduler.runner import run_all
from scheduler.priority import detect_starvation
from ml.predict import predict_from_processes, model_is_available

app = Flask(__name__)

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "history.sqlite3")
METRICS_PATH = os.path.join(os.path.dirname(__file__), "..", "reports", "training_metrics.json")


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS run_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            num_processes INTEGER,
            avg_burst REAL,
            recommended_algorithm TEXT,
            confidence REAL,
            actual_best_algorithm TEXT,
            was_correct INTEGER
        )
    """)
    conn.commit()
    conn.close()


def log_run(num_processes, avg_burst, recommended, confidence, actual_best):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO run_history (timestamp, num_processes, avg_burst, recommended_algorithm, "
        "confidence, actual_best_algorithm, was_correct) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            datetime.utcnow().isoformat(),
            num_processes,
            avg_burst,
            recommended,
            confidence,
            actual_best,
            1 if recommended == actual_best else 0,
        ),
    )
    conn.commit()
    conn.close()


def _processes_from_payload(payload):
    """Build Process objects either from a raw process list the client
    sent (so the SAME workload can be re-simulated client-side) or from
    generation parameters."""
    if "processes" in payload:
        procs = []
        for p in payload["processes"]:
            proc = Process(
                pid=p["pid"],
                arrival_time=int(p["arrival_time"]),
                burst_time=int(p["burst_time"]),
                priority=int(p["priority"]),
                memory_required=int(p.get("memory_required", 256)),
            )
            procs.append(proc)
        return procs

    n = int(payload.get("num_processes", 10))
    max_arrival = int(payload.get("max_arrival", max(10, n * 2)))
    max_burst = int(payload.get("max_burst", 20))
    max_priority = int(payload.get("max_priority", 5))
    seed = payload.get("seed")
    seed = int(seed) if seed not in (None, "") else None

    return generate_processes(
        n=n, max_arrival=max_arrival, min_burst=1, max_burst=max_burst,
        min_priority=1, max_priority=max_priority, seed=seed,
    )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/generate", methods=["POST"])
def api_generate():
    payload = request.get_json(force=True) or {}
    n = int(payload.get("num_processes", 10))
    max_arrival = int(payload.get("max_arrival", max(10, n * 2)))
    max_burst = int(payload.get("max_burst", 20))
    max_priority = int(payload.get("max_priority", 5))
    seed = payload.get("seed")
    seed = int(seed) if seed not in (None, "") else None

    procs = generate_processes(
        n=n, max_arrival=max_arrival, min_burst=1, max_burst=max_burst,
        min_priority=1, max_priority=max_priority, seed=seed,
    )
    return jsonify({
        "processes": [
            {
                "pid": p.pid, "arrival_time": p.arrival_time, "burst_time": p.burst_time,
                "priority": p.priority, "memory_required": p.memory_required,
            }
            for p in procs
        ]
    })


@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    payload = request.get_json(force=True) or {}
    rr_quantum = int(payload.get("rr_quantum", 4))
    use_aging = bool(payload.get("use_aging", False))

    procs = _processes_from_payload(payload)
    results = run_all(procs, rr_quantum=rr_quantum, use_aging=use_aging)

    out = {name: r.to_dict() for name, r in results.items()}

    starvation = detect_starvation(results["Priority"])

    ai_prediction = None
    if model_is_available():
        ai_prediction = predict_from_processes(clone_processes(procs))


        classic = {k: v for k, v in results.items() if k in ("FCFS", "SJF", "RR", "Priority")}
        actual_best = min(classic.items(), key=lambda kv: kv[1].avg_waiting_time())[0]
        ai_prediction["actual_best_by_waiting"] = actual_best

        try:
            avg_burst = sum(p.burst_time for p in procs) / len(procs)
            log_run(len(procs), avg_burst, ai_prediction["recommended_algorithm"],
                     ai_prediction["confidence"], actual_best)
        except Exception:
            pass 

    return jsonify({
        "results": out,
        "starvation": starvation,
        "ai_prediction": ai_prediction,
        "process_count": len(procs),
    })


@app.route("/api/model-info")
def api_model_info():
    if not os.path.exists(METRICS_PATH):
        return jsonify({"available": False})
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    return jsonify({"available": True, "metrics": metrics})


@app.route("/api/history")
def api_history():
    if not os.path.exists(DB_PATH):
        return jsonify({"runs": [], "accuracy": None})
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM run_history ORDER BY id DESC LIMIT 50"
    ).fetchall()
    conn.close()

    runs = [dict(r) for r in rows]
    if runs:
        accuracy = sum(r["was_correct"] for r in runs) / len(runs)
    else:
        accuracy = None
    return jsonify({"runs": runs, "accuracy": accuracy})


if __name__ == "__main__":
    init_db()
    app.run(debug=True, host="0.0.0.0", port=5000)
