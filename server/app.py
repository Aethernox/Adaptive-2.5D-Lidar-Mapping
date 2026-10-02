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

from flask import Flask, jsonify, render_template, request

from pipeline import Pipeline
from data.kitti import KittiDataError
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


@app.route("/api/seek", methods=["POST"])
def api_seek():
    """Seek the deterministic replay by rebuilding its true pipeline state.

    This is intentionally replay-based: the adaptive grid and tracker state
    are part of a frame, so changing only a displayed index would be wrong.
    """
    payload = request.get_json(silent=True) or {}
    try:
        target = int(payload.get("frame", 1))
    except (TypeError, ValueError):
        return jsonify(error="frame must be an integer"), 400
    with _lock:
        target = max(0, target)
        # Synthetic historically exposed frame 1 as its first frame. KITTI
        # exposes its native zero-padded frame indices.
        if _pipeline.dataset_type == "synthetic":
            target = max(0, min(target - 1, 499))
        out = _pipeline.seek(target)
    return jsonify(out)


@app.route("/api/reset", methods=["POST"])
def api_reset():
    with _lock:
        _pipeline.reset()
    return jsonify(dict(status="ok"))


@app.route("/api/config")
def api_config():
    with _lock:
        return jsonify(_pipeline.config_payload())


@app.route("/api/dataset/select", methods=["POST"])
@app.route("/api/kitti/select", methods=["POST"])
def api_dataset_select():
    """Select a source without changing the downstream pipeline schema."""
    global _pipeline
    payload = request.get_json(silent=True) or {}
    dataset_type = str(payload.get("dataset_type", "kitti")).lower()
    sequence = str(payload.get("sequence", "00")).zfill(2)
    dataset_root = payload.get("dataset_root")
    with _lock:
        try:
            if dataset_type == "synthetic":
                _pipeline = Pipeline(dataset_type="synthetic")
            elif dataset_type == "kitti":
                _pipeline = Pipeline(dataset_type="kitti", dataset_root=dataset_root or _pipeline.dataset_root,
                                     sequence=sequence, model_path=payload.get("model_path", _pipeline.model_path))
            else:
                return jsonify(error="dataset_type must be synthetic or kitti"), 400
            return jsonify(_pipeline.config_payload())
        except (KittiDataError, ValueError) as exc:
            return jsonify(error=str(exc)), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=False, threaded=True)
