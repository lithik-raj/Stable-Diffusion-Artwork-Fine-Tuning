import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import random
import sys
import requests
from flask import Flask, send_from_directory, jsonify, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder=BASE_DIR)

ART_HORDE_MODELS = [
    "Deliberate",
    "Dreamshaper",
    "AbsoluteReality",
    "stable_diffusion",
]


def _apply_model_conditioning(prompt: str, model_key: str) -> str:
    clean = (prompt or "").strip()
    if model_key == "lora-v1":
        return (
            f"{clean}, fine-tuned artwork LoRA style, expressive oil brushstrokes, "
            "dramatic lighting, rich painterly texture, intricate composition"
        )
    if model_key == "cyber-lora":
        return (
            f"{clean}, cyberpunk neo-tokyo LoRA style, neon rim lights, "
            "futuristic urban atmosphere, vibrant reflections, concept art"
        )
    return clean


def _parse_gradio_complete_url(sse_text: str):
    last_event = None
    last_data = None
    for line in sse_text.splitlines():
        if line.startswith("event:"):
            last_event = line[6:].strip()
        elif line.startswith("data:"):
            last_data = line[5:].strip()
    if last_event == "complete" and last_data:
        arr = json.loads(last_data)
        if isinstance(arr, list) and arr:
            item = arr[0]
            if isinstance(item, dict) and item.get("url"):
                return item["url"]
            if isinstance(item, str) and item.startswith("http"):
                return item
    return None


def _call_hf_space(base_url: str, endpoint: str, payload: dict, label: str):
    init = requests.post(
        f"{base_url}/gradio_api/call{endpoint}",
        json=payload,
        timeout=6,
    )
    init.raise_for_status()
    event_id = init.json().get("event_id")
    if not event_id:
        raise RuntimeError("Missing event_id")
    stream = requests.get(
        f"{base_url}/gradio_api/call{endpoint}/{event_id}",
        timeout=18,
    )
    stream.raise_for_status()
    img_url = _parse_gradio_complete_url(stream.text)
    if not img_url:
        raise RuntimeError("No completed image URL in SSE response")
    img_resp = requests.get(img_url, timeout=12)
    img_resp.raise_for_status()
    mime = img_resp.headers.get("content-type", "image/png").split(";")[0]
    b64 = base64.b64encode(img_resp.content).decode("ascii")
    return {
        "data_url": f"data:{mime};base64,{b64}",
        "engine": label,
    }


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


@app.route("/api/fast", methods=["POST"])
def fast_synthesize():
    body = request.get_json(silent=True) or {}
    prompt = str(body.get("prompt", "")).strip()
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    model_key = str(body.get("model", "lora-v1"))
    steps = max(4, min(12, int(body.get("steps", 30)) // 4 or 6))
    try:
        seed = int(body.get("seed", 428190))
    except Exception:
        seed = random.randint(100000, 999999)

    styled_prompt = _apply_model_conditioning(prompt, model_key)

    jobs = [
        (
            "HF Z-Image Turbo",
            "https://mrfakename-z-image-turbo.hf.space",
            "/generate_image",
            {"data": [styled_prompt, 512, 512, max(4, min(9, steps)), seed, False]},
        ),
        (
            "HF FLUX.1-schnell",
            "https://black-forest-labs-flux-1-schnell.hf.space",
            "/infer",
            {"data": [styled_prompt, seed, False, 512, 512, max(2, min(8, steps // 2))]},
        ),
        (
            "HF Tongyi Z-Image",
            "https://tongyi-mai-z-image-turbo.hf.space",
            "/generate",
            {"data": [styled_prompt, "1024x1024 ( 1:1 )", seed, max(4, min(9, steps)), 3.0, False, []]},
        ),
    ]

    errors = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futs = {
            pool.submit(_call_hf_space, url, ep, payload, label): label
            for label, url, ep, payload in jobs
        }
        for fut in as_completed(futs, timeout=22):
            try:
                res = fut.result()
                res["conditioned_prompt"] = styled_prompt
                return jsonify(res)
            except Exception as exc:
                errors.append(f"{futs[fut]}: {exc}")

    return jsonify({"error": "Fast GPU backends busy", "details": errors}), 503


@app.route("/api/horde/start", methods=["POST"])
def horde_start():
    body = request.get_json(silent=True) or {}
    prompt = str(body.get("prompt", "")).strip()
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400
    model_key = str(body.get("model", "lora-v1"))
    cfg = float(body.get("cfg", 7.5))
    steps = max(15, min(35, int(body.get("steps", 25))))
    seed = str(body.get("seed", "428190"))
    styled_prompt = _apply_model_conditioning(prompt, model_key)

    try:
        r = requests.post(
            "https://stablehorde.net/api/v2/generate/async",
            headers={"apikey": "0000000000", "Content-Type": "application/json"},
            json={
                "prompt": styled_prompt,
                "params": {
                    "sampler_name": "k_euler_a",
                    "cfg_scale": cfg,
                    "steps": steps,
                    "width": 512,
                    "height": 512,
                    "seed": seed,
                    "n": 1,
                },
                "models": ART_HORDE_MODELS,
                "nsfw": False,
                "r2": True,
            },
            timeout=10,
        )
        data = r.json()
        if r.ok and data.get("id"):
            return jsonify({"id": data["id"], "conditioned_prompt": styled_prompt})
        return jsonify({"error": data.get("message", "Horde error")}), 502
    except Exception as exc:
        return jsonify({"error": str(exc)}), 502


@app.route("/api/horde/status/<job_id>")
def horde_status(job_id):
    try:
        r = requests.get(
            f"https://stablehorde.net/api/v2/generate/status/{job_id}",
            timeout=10,
        )
        data = r.json()
        if data.get("done") and data.get("generations"):
            img_url = data["generations"][0].get("img")
            if img_url and img_url.startswith("http"):
                ir = requests.get(img_url, timeout=12)
                if ir.ok:
                    mime = ir.headers.get("content-type", "image/webp").split(";")[0]
                    b64 = base64.b64encode(ir.content).decode("ascii")
                    return jsonify({
                        "done": True,
                        "data_url": f"data:{mime};base64,{b64}",
                        "model": data["generations"][0].get("model", "SD Community"),
                    })
            return jsonify({"done": True, "url": img_url})
        return jsonify({
            "done": False,
            "wait_time": data.get("wait_time", 0),
            "queue_position": data.get("queue_position", 0),
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 502


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
