"""Local backend for the dashboard. Serves the dashboard and exposes
/api/run + /api/status/<id> so a benchmark+judge pass can be triggered
from the browser instead of the terminal.

    python server.py
"""
import sys
import os
import json
import random
import threading
import uuid
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT / "src"))

from providers import PROVIDERS, guess_provider_and_model, pick_judge_provider, pick_judge_providers, RATE_LIMITER, friendly_error_response


def _friendly_setup_error(raw: str) -> str:
    """Same idea as friendly_error_response() in providers.py, but for the
    pre-flight validation messages guess_provider_and_model() raises before
    a run even starts (e.g. 'needs OPENROUTER_API_KEY', 'no free tier').
    The technical detail is printed to the server's own console; what the
    dashboard shows is just the plain-language outcome."""
    print(f"[setup] {raw}")
    return "This model is out of our free usage limit right now. Please try a different model, or try again later."
from run_benchmark import run_benchmark, load_prompts
from llm_judge import judge_records, LOCAL_JUDGE

app = Flask(__name__, static_folder=str(ROOT / "dashboard"), static_url_path="")

JOBS = {}
JOBS_LOCK = threading.Lock()


def _set(job_id, **kwargs):
    with JOBS_LOCK:
        JOBS[job_id].update(kwargs)


def _worker(job_id, params):
    try:
        _set(job_id, status="running", stage="target",
             progress={"stage": "target", "done": 0, "total": params.get("limit") or None})

        def bench_progress(i, total, rec):
            _set(job_id, progress={"stage": "target", "done": i, "total": total,
                                    "last_id": rec["id"]})

        records = run_benchmark(
            provider=params["provider"],
            model=params["model"],
            limit=params.get("limit"),
            categories=params.get("categories") or None,
            delay=params.get("target_delay", 0.0),
            stratified=params.get("stratified", True),
            # Random seed by default -- a different, shuffled subset of the
            # 123 curated prompts each time you run, instead of always the
            # same 45 in the same order. Pass an explicit "seed" in the
            # request if you ever want a specific run to be reproducible.
            seed=params.get("seed") if params.get("seed") is not None else random.randint(0, 2**31 - 1),
            progress_cb=bench_progress,
        )

        _set(job_id, stage="judging",
             progress={"stage": "judging", "done": 0, "total": len(records)})

        def judge_progress(i, total, rec, judged):
            _set(job_id, progress={"stage": "judging", "done": i, "total": total,
                                    "last_id": rec["id"], "last_label": judged.get("label")})

        judged = judge_records(
            records,
            judge_provider=params["judge_provider"],
            judge_model=params.get("judge_model"),
            delay=params.get("judge_delay", 1.2),
            retry_failed=True,
            progress_cb=judge_progress,
            fallback_providers=params.get("judge_fallback_providers"),
        )

        out_name = f"judged_{params['provider']}_{params['model']}_{job_id}.json".replace("/", "-")
        out_path = ROOT / "results" / out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(judged, f, indent=2)

        _set(job_id, status="done", result=judged, saved_to=str(out_path.relative_to(ROOT)))
    except SystemExit as e:
        _set(job_id, status="error", error=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()  # full detail stays in the server's own console
        _set(job_id, status="error",
             error="Something went wrong running this test. Please try again, or try a different model.")


@app.route("/api/providers")
def api_providers():
    out = {}
    for name, cfg in PROVIDERS.items():
        out[name] = {
            "default_model": cfg["default_model"],
            "env": cfg["env"],
            "key_set": bool(os.environ.get(cfg["env"])),
            "rate_limit": RATE_LIMITER.status(name, model=cfg["default_model"]),
        }
    prompts = load_prompts()
    categories = sorted(set(p["category"] for p in prompts))
    return jsonify({"providers": out, "categories": categories, "total_prompts": len(prompts)})


@app.route("/api/run", methods=["POST"])
def api_run():
    params = request.get_json(force=True) or {}
    raw_model = (params.get("model") or "").strip()
    if not raw_model:
        return jsonify({"error": "Type a model name, e.g. 'gemini-2.5-flash', 'gpt-4o', "
                                  "'gemini-2.5-flash-lite', 'gpt-oss-120b'."}), 400

    # Target model: honor an explicit provider override (advanced use), else auto-detect.
    override_provider = params.get("provider") or None
    note = None
    try:
        if override_provider:
            if override_provider not in PROVIDERS:
                return jsonify({"error": f"Unknown provider '{override_provider}'."}), 400
            if not os.environ.get(PROVIDERS[override_provider]["env"]):
                return jsonify({"error": "That provider isn't set up on the server yet. Pick a different one, "
                                          "or leave provider on auto-detect."}), 400
            provider, model = override_provider, raw_model
        else:
            provider, model, note = guess_provider_and_model(raw_model)
    except ValueError as e:
        return jsonify({"error": _friendly_setup_error(str(e))}), 400

    # Judge model: defaults to the offline, rule-based local judge (no API
    # key, no network, can't hit a rate limit or quota) unless the person
    # explicitly asks for an API-based judge provider or model.
    raw_judge_model = (params.get("judge_model") or "").strip()
    judge_override = params.get("judge_provider") or None
    judge_note = None
    judge_fallback_providers = []
    try:
        if judge_override == LOCAL_JUDGE:
            judge_provider, judge_model = LOCAL_JUDGE, "rule-based-v1"
        elif raw_judge_model:
            if judge_override:
                if not os.environ.get(PROVIDERS[judge_override]["env"]):
                    return jsonify({"error": "That judge provider isn't set up on the server yet."}), 400
                judge_provider, judge_model = judge_override, raw_judge_model
            else:
                judge_provider, judge_model, judge_note = guess_provider_and_model(raw_judge_model)
            judge_fallback_providers = pick_judge_providers(exclude=judge_provider)
        elif judge_override:
            if not os.environ.get(PROVIDERS[judge_override]["env"]):
                return jsonify({"error": "That judge provider isn't set up on the server yet."}), 400
            judge_provider, judge_model = judge_override, PROVIDERS[judge_override]["default_model"]
            judge_fallback_providers = pick_judge_providers(exclude=judge_provider)
        else:
            # No explicit choice at all -- use the local judge by default,
            # since it's always available and can't fail on quota/rate limits.
            judge_provider, judge_model = LOCAL_JUDGE, "rule-based-v1"
    except ValueError as e:
        return jsonify({"error": _friendly_setup_error(str(e))}), 400

    params.update({
        "provider": provider, "model": model,
        "judge_provider": judge_provider, "judge_model": judge_model,
        # Every other configured provider, in preference order, tried
        # automatically if judge_provider comes back as a provider_error --
        # this is what makes judging robust to one flaky/rate-limited
        # provider without the person having to manually pick a different
        # judge model. Not used (left empty) when judging locally.
        "judge_fallback_providers": judge_fallback_providers,
    })
    job_id = uuid.uuid4().hex[:10]
    with JOBS_LOCK:
        JOBS[job_id] = {"status": "queued", "progress": {}, "params": params}
    threading.Thread(target=_worker, args=(job_id, params), daemon=True).start()
    return jsonify({
        "job_id": job_id, "model": model, "provider": provider, "note": note,
        "judge_provider": judge_provider, "judge_model": judge_model, "judge_note": judge_note,
    })


@app.route("/api/runs")
def api_runs():
    """Lists saved run files in results/ so the dashboard can auto-load the
    most recent one and offer past runs in a dropdown, instead of requiring
    the user to manually re-upload a results file every time."""
    results_dir = ROOT / "results"
    runs = []
    if results_dir.exists():
        files = sorted(results_dir.glob("judged_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for f in files:
            provider = model = "?"
            n = 0
            try:
                with open(f) as fh:
                    data = json.load(fh)
                n = len(data)
                if data:
                    provider = data[0].get("provider", "?")
                    model = data[0].get("model", "?")
            except Exception:
                pass
            runs.append({
                "filename": f.name, "provider": provider, "model": model,
                "n": n, "modified": f.stat().st_mtime,
            })
    return jsonify({"runs": runs})


@app.route("/api/status/<job_id>")
def api_status(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return jsonify({"error": "unknown job_id"}), 404
        return jsonify(dict(job))


@app.route("/results/<path:filename>")
def results_files(filename):
    return send_from_directory(ROOT / "results", filename)


@app.route("/")
def index():
    return app.send_static_file("index.html")


if __name__ == "__main__":
    missing = [cfg["env"] for cfg in PROVIDERS.values() if not os.environ.get(cfg["env"])]
    if missing:
        print(f"NOTE: these env vars aren't set yet: {', '.join(missing)}. "
              f"Runs using those providers will fail until you export them and restart.")
    print("LLM Safety Testing & Risk Analysis System dashboard running at http://localhost:8000")
    app.run(host="127.0.0.1", port=8000, debug=False)
