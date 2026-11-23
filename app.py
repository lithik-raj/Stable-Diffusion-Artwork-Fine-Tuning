import os
import sys
from flask import Flask, send_from_directory, jsonify, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder=BASE_DIR)

@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")

@app.route("/<path:filename>")
def serve_static(filename):
    if os.path.exists(os.path.join(BASE_DIR, filename)):
        return send_from_directory(BASE_DIR, filename)
    return send_from_directory(BASE_DIR, "index.html")

@app.route("/health")
def health():
    return jsonify({
        "status": "online",
        "service": "Stable Diffusion LoRA Artwork Fine-Tuning Studio",
        "model": "CompVis/stable-diffusion-v1-4",
        "lora_rank": 8,
        "lora_alpha": 16,
        "target_modules": ["to_k", "to_q", "to_v", "to_out.0"]
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
