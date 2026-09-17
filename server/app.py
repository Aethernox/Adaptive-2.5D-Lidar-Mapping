"""
Flask backend for the real-time dashboard (Module 4).

Endpoints:
    GET  /                 dashboard HTML page
    GET  /api/frame        advance the pipeline by one frame, return JSON
                            (base64 PNG of the 2.5D map + live metrics)
    POST /api/reset        reset the simulation/pipeline state

Run: python -m server.app   (serves on http://0.0.0.0:5050)
"""
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, jsonify, render_template

from pipeline import Pipeline
from config import CLASSES, CLASS_COLORS

app = Flask(__name__)
_pipeline = Pipeline()
_lock = threading.Lock()


@app.route("/")
def index():
    legend = [dict(name=CLASSES[c], color="rgb(%d,%d,%d)" % CLASS_COLORS[c]) for c in CLASS_COLORS]
    return render_template("index.html", legend=legend)


@app.route("/api/frame")
def api_frame():
    with _lock:
        out = _pipeline.step()
    return jsonify(out)


@app.route("/api/reset", methods=["POST"])
def api_reset():
    with _lock:
        _pipeline.reset()
    return jsonify(dict(status="ok"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=False, threaded=True)
