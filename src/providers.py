"""
LLM Safety Testing & Risk Analysis System — Model Providers
----------------------------------
Thin wrappers around chat-completion APIs so the rest of the pipeline
(run_benchmark.py, llm_judge.py) doesn't care which provider it's talking
to. Includes guess_provider_and_model(), which lets a user type ANY model
name and be auto-routed to a way of running it for FREE -- it never
silently falls back to a paid/"pro" model.

  - Google Gemini (Google AI Studio): free tier, no card needed.
    Get a key at https://aistudio.google.com/apikey -- GEMINI_API_KEY
  - Groq: free tier, no card needed, very fast, open-weight models.
    Get a key at https://console.groq.com/keys -- GROQ_API_KEY
  - OpenRouter: also has a genuine free tier, no card needed -- models
    whose id ends in ':free' cost $0. This is what makes "type literally
    any model name" possible for most models: closed/paid names with a
    free open-weight equivalent (e.g. 'gpt-4o') get auto-substituted with
    one unless a more direct free route exists (see GitHub Models below
    for gpt- names specifically). A few closed models (Claude, Grok) have
    no free or open-weight equivalent anywhere -- those raise a clear
    error instead of guessing.
    Get a key at https://openrouter.ai/keys -- OPENROUTER_API_KEY
  - GitHub Models: free, no card, uses a GitHub Personal Access Token
    instead of a vendor API key. The most direct free route to REAL
    OpenAI models (gpt-4o, gpt-4o-mini, gpt-4.1) rather than an
    open-weight substitute -- gpt-/chatgpt- style names route here first
    if GITHUB_TOKEN is set, before falling back to OpenRouter.
    Get a token at https://github.com/settings/personal-access-tokens/new
    (fine-grained PAT, grant "Models" read-only) -- GITHUB_TOKEN
    NOTE: GitHub has announced GitHub Models is being fully retired on
    2026-07-30 (scheduled brownouts on 2026-07-16 and 2026-07-23 first).
    This code does NOT wait for it to start failing -- see
    GITHUB_MODELS_RETIRE_DATE below. From that date on, gpt-/chatgpt-/o1-
    /o3-/o4- style names skip GitHub Models automatically (even if
    GITHUB_TOKEN is still set) and go straight to the free open-weight
    gpt-oss substitute (via Groq if GROQ_API_KEY is set, else OpenRouter).
    No action needed and no payment required when the date arrives --
    this is a permanent, no-card fallback, not a stopgap.
  - DeepSeek: no native integration -- DeepSeek's own API now requires a
    paid balance (its free-token grant was discontinued), so this project
    doesn't call it directly. Any "deepseek"-ish model name routes to
    OpenRouter's free deepseek/deepseek-r1:free instead (needs
    OPENROUTER_API_KEY).
  - Anthropic / OpenAI native APIs: paid, usage-based, no free tier.
    Only used if you set ANTHROPIC_API_KEY / OPENAI_API_KEY yourself AND
    explicitly pick that provider in the dashboard's advanced options --
    never chosen automatically by the free-by-default auto-detection.

Using two DIFFERENT providers for the "model under test" and the "judge"
is methodologically preferable to using the same model for both -- it
reduces the risk that the judge shares blind spots with the model it's
grading (see docs/methodology.md, "Known limitations").
"""
import os
import re
import json
import time
import threading
import difflib
import urllib.request
import urllib.error
from datetime import date, timezone, datetime


# GitHub has announced GitHub Models retires completely on this date, with
# brownouts on 2026-07-16 and 2026-07-23 beforehand. Rather than waiting for
# calls to start failing (and, during the brownout window, failing
# *intermittently* -- the worst kind of failure to leave unhandled), routing
# checks this date directly and treats GitHub Models as unavailable for
# gpt-/chatgpt-/o1-/o3-/o4- names from this date on, even if GITHUB_TOKEN is
# still set. This is a permanent switch to the free gpt-oss open-weight
# fallback, not a temporary outage message -- no payment is ever required.
# Checked against UTC so the switchover lands consistently regardless of
# what timezone this is deployed in.
GITHUB_MODELS_RETIRE_DATE = date(2026, 7, 30)


def _github_models_retired() -> bool:
    return datetime.now(timezone.utc).date() >= GITHUB_MODELS_RETIRE_DATE


# Conservative request-per-window caps for each provider's FREE tier, kept a
# little under the documented public limits so a burst of near-simultaneous
# calls (e.g. several dashboard runs, or judge + target running close
# Requests-per-window caps (existing). (max_requests, window_seconds).
#
# GitHub Models' per-minute cap is shared across the whole account
# regardless of model, but its DAILY cap is tier-dependent -- smaller/
# cheaper "low" tier models (e.g. gpt-4o-mini, gpt-4.1-mini/nano, the
# o1-mini/o3-mini/o4-mini reasoning-lite models) get a much higher daily
# budget than "high" tier models (gpt-4o, gpt-4.1, o1, o3). A single flat
# 45/day cap for every model was both too tight for low-tier models and
# not the actual number for high-tier ones -- see GITHUB_MODEL_TIER /
# GITHUB_DAILY_CAP_BY_TIER below, which the daily window is keyed on
# instead of the bare "github" provider name.
RATE_LIMITS = {
    "gemini": [(12, 60)],                    # free tier: ~15 req/min for flash-tier models
    "groq": [(25, 60)],                      # free tier: ~30 req/min, varies by model
    # OpenRouter's free ("...":free) tier enforces BOTH a per-minute cap AND
    # a per-day cap: 20 req/min, and 50 req/day unless the account has ever
    # purchased $10+ of credits (in which case the daily cap rises to
    # 1000/day). This project can't know which bucket a given key is in, so
    # it assumes the tighter 50/day and stays a little under it.
    "openrouter": [(18, 60), (45, 86400)],
    "github": [(8, 60)],                     # per-minute cap only; daily cap is tier-aware, see below
    "anthropic": [(45, 60)],                 # paid tier defaults are much higher; still capped defensively
    # NOTE: no "openai" entry here on purpose -- OPENAI_API_KEY is never set
    # in this setup (no free tier, never auto-picked), so there's no rate
    # limit to self-impose or show on the dashboard. If it's ever added back,
    # key_set will be True and the chip will reappear automatically.
}

# GitHub Models tiers models into "low" (small/cheap) and "high"
# (large/expensive) for rate-limiting purposes. Kept as an explicit
# allowlist-with-tier rather than guessing from the name, since "mini"/
# "nano"/"-mini" suffixes are the actual signal GitHub itself uses.
GITHUB_MODEL_TIER = {
    "openai/gpt-4o": "high",
    "openai/gpt-4.1": "high",
    "openai/o1": "high",
    "openai/o3": "high",
    "openai/gpt-4o-mini": "low",
    "openai/gpt-4.1-mini": "low",
    "openai/gpt-4.1-nano": "low",
    "openai/o1-mini": "low",
    "openai/o3-mini": "low",
    "openai/o4-mini": "low",
}

# Kept a little under GitHub's own documented per-model daily limits.
GITHUB_DAILY_CAP_BY_TIER = {
    "high": 50,
    "low": 150,
}
GITHUB_DEFAULT_TIER = "high"  # unrecognized/new model -> assume the tighter cap until added above


def github_daily_cap(model: str) -> int:
    tier = GITHUB_MODEL_TIER.get(model, GITHUB_DEFAULT_TIER)
    return GITHUB_DAILY_CAP_BY_TIER[tier]

# Tokens-per-window caps. A request-count limit alone isn't enough for Groq:
# its free tier also enforces a separate, tighter tokens-per-minute (TPM)
# budget (Groq's docs put llama-3.3-70b-versatile at roughly 6,000-12,000
# TPM as of mid-2026), and the judge stage's prompts are large (rubric +
# original prompt + full model response), so a handful of judge calls can
# exhaust the TPM budget well before the 25-requests/min cap above ever
# triggers. That's what caused judge-stage 429s -> JUDGE_PARSE_ERROR even
# though the request-count limiter reported plenty of headroom. Kept
# deliberately conservative since it's an estimate, not a guarantee.
TOKEN_LIMITS = {
    "groq": [(5500, 60)],
}


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token for English) for the input
    prompt, plus a flat allowance for the ~500-token output cap every
    provider call uses -- good enough to stay a safe margin under a TPM
    cap without needing a real tokenizer (TPM budgets count input+output
    combined, so estimating input alone would undercount)."""
    return max(1, len(text) // 4) + 500


MAX_BLOCK_SECONDS = 120  # don't silently sleep longer than this -- fail with a clear message instead


class RateLimitExhausted(RuntimeError):
    """Raised instead of blocking when the wait required is too long to sit
    through silently (e.g. a daily cap is used up) -- surfaced to the person
    as a clear stop rather than the dashboard just hanging for hours."""


class RateLimiter:
    """Thread-safe sliding-window limiter, one or more windows per provider
    (e.g. GitHub Models needs both a per-minute AND a per-day cap). Every
    call_model() invocation passes through .wait(provider) first, so the
    whole pipeline -- run_benchmark.py, llm_judge.py, and any dashboard job
    thread -- automatically self-throttles instead of relying on a person
    to guess a safe --delay value. If the required wait exceeds
    MAX_BLOCK_SECONDS (i.e. a longer window like a daily cap is actually
    used up, not just a brief per-minute burst), it raises
    RateLimitExhausted rather than blocking silently for hours."""

    def __init__(self):
        self._lock = threading.Lock()
        self._timestamps = {p: [] for p in RATE_LIMITS}
        self._token_log = {p: [] for p in RATE_LIMITS}  # list of (timestamp, estimated_tokens)
        # Separate from _timestamps: tracks a daily window per (provider,
        # model) rather than per bare provider name, since GitHub Models'
        # daily cap is tier-dependent (varies per model, not per account) --
        # see GITHUB_MODEL_TIER / github_daily_cap() above. Provider-level
        # per-minute limiting above is unaffected and still shared per
        # provider regardless of model.
        self._daily_timestamps = {}  # (provider, model) -> [timestamps]

    def wait(self, provider: str, estimated_tokens: int = 0, daily_key: str = None, daily_cap: int = None):
        """daily_key/daily_cap (e.g. daily_key=model, daily_cap=github_daily_cap(model))
        adds an additional 24h window tracked per (provider, daily_key)
        instead of per bare provider -- used for GitHub Models, where the
        daily budget depends on which model tier was called."""
        req_windows = RATE_LIMITS.get(provider, [])
        tok_windows = TOKEN_LIMITS.get(provider, []) if estimated_tokens else []
        has_daily = daily_key is not None and daily_cap is not None
        if not req_windows and not tok_windows and not has_daily:
            return
        DAY_SECONDS = 86400
        while True:
            with self._lock:
                now = time.time()
                ts = self._timestamps.setdefault(provider, [])
                toks = self._token_log.setdefault(provider, [])
                keep_window = max([w for _, w in req_windows] + [w for _, w in tok_windows], default=0)
                ts[:] = [t for t in ts if now - t < keep_window]
                toks[:] = [(t, n) for t, n in toks if now - t < keep_window]

                daily_ts = None
                if has_daily:
                    dkey = (provider, daily_key)
                    daily_ts = self._daily_timestamps.setdefault(dkey, [])
                    daily_ts[:] = [t for t in daily_ts if now - t < DAY_SECONDS]

                blocking = None
                blocking_window = None
                for max_req, window in req_windows:
                    used = len([t for t in ts if now - t < window])
                    if used >= max_req:
                        oldest_in_window = min(t for t in ts if now - t < window)
                        wait_for = window - (now - oldest_in_window) + 0.05
                        if blocking is None or wait_for > blocking:
                            blocking, blocking_window = wait_for, window
                for max_tok, window in tok_windows:
                    used_tok = sum(n for t, n in toks if now - t < window)
                    if used_tok + estimated_tokens > max_tok:
                        in_window = [t for t, n in toks if now - t < window]
                        oldest_in_window = min(in_window) if in_window else now
                        wait_for = window - (now - oldest_in_window) + 0.05
                        if blocking is None or wait_for > blocking:
                            blocking, blocking_window = wait_for, window
                if has_daily and len(daily_ts) >= daily_cap:
                    oldest_in_window = min(daily_ts)
                    wait_for = DAY_SECONDS - (now - oldest_in_window) + 0.05
                    if blocking is None or wait_for > blocking:
                        blocking, blocking_window = wait_for, DAY_SECONDS
                if blocking is None:
                    ts.append(now)
                    if estimated_tokens:
                        toks.append((now, estimated_tokens))
                    if has_daily:
                        daily_ts.append(now)
                    return
                if blocking > MAX_BLOCK_SECONDS:
                    if blocking_window >= 3600 * 12:
                        label = f"{provider}/{daily_key}" if has_daily else provider
                        raise RateLimitExhausted(f"{label} API limit finished, try again tomorrow.")
                    mins = max(1, round(blocking / 60))
                    raise RateLimitExhausted(f"{provider} API limit finished, try again in about {mins} minute(s).")
                sleep_for = blocking
            time.sleep(max(sleep_for, 0.05))

    def status(self, provider: str, model: str = None):
        """Current usage for display purposes, e.g. in /api/providers.
        Returns the tightest (most-used-relative-to-cap) window. For
        provider="github", pass `model` to also account for that model's
        tier-specific daily cap; without it only the per-minute window is
        reflected."""
        windows = list(RATE_LIMITS.get(provider) or [])
        candidates = []  # (used, max, window_seconds, frac)
        if provider == "github" and model:
            daily_cap = github_daily_cap(model)
            with self._lock:
                now = time.time()
                daily_ts = self._daily_timestamps.get((provider, model), [])
                daily_used = len([t for t in daily_ts if now - t < 86400])
            candidates.append((daily_used, daily_cap, 86400, daily_used / daily_cap if daily_cap else 0))
        if not windows and not candidates:
            return None
        with self._lock:
            now = time.time()
            ts = self._timestamps.get(provider, [])
            best = None
            for max_req, window in windows:
                used = len([t for t in ts if now - t < window])
                frac = used / max_req if max_req else 0
                if best is None or frac > best[2]:
                    best = (used, max_req, frac, window)
            for used, max_req, window, frac in candidates:
                if best is None or frac > best[2]:
                    best = (used, max_req, frac, window)
        return {"used": best[0], "max": best[1], "window_seconds": best[3]}


RATE_LIMITER = RateLimiter()


def _post_json(url: str, payload: dict, headers: dict, max_retries: int = 3) -> dict:
    data = json.dumps(payload).encode()
    # Some providers (Groq in particular) sit behind Cloudflare bot-protection
    # that blocks requests with no/unusual User-Agent header (HTTP 403, error
    # code 1010). A normal browser-like User-Agent avoids that.
    full_headers = {"User-Agent": "Mozilla/5.0 (compatible; LLM Safety Testing & Risk Analysis System/1.0)"}
    full_headers.update(headers)
    req = urllib.request.Request(url, data=data, headers=full_headers, method="POST")
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="ignore")
            if e.code == 429 and attempt < max_retries - 1:
                # rate limited -- back off and retry, common on free tiers
                time.sleep(3 * (attempt + 1))
                continue
            raise RuntimeError(f"HTTP {e.code} from {url}: {body[:300]}")
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)


def call_anthropic(model: str, prompt: str, api_key: str) -> str:
    data = _post_json(
        "https://api.anthropic.com/v1/messages",
        {"model": model, "max_tokens": 500, "messages": [{"role": "user", "content": prompt}]},
        {"Content-Type": "application/json", "x-api-key": api_key, "anthropic-version": "2023-06-01"},
    )
    return "".join(b["text"] for b in data["content"] if b["type"] == "text")


def call_gemini(model: str, prompt: str, api_key: str) -> str:
    # Google AI Studio free tier. Model names like "gemini-2.5-flash".
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    data = _post_json(
        url,
        {"contents": [{"parts": [{"text": prompt}]}]},
        {"Content-Type": "application/json"},
    )
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        # e.g. blocked by Gemini's own safety filters -- still a valid,
        # informative result for a safety benchmark, so surface it clearly
        reason = data.get("candidates", [{}])[0].get("finishReason", "unknown")
        return f"[GEMINI RETURNED NO TEXT -- finishReason: {reason}. Raw: {json.dumps(data)[:200]}]"


def call_groq(model: str, prompt: str, api_key: str) -> str:
    # Groq is OpenAI-chat-completions-compatible.
    data = _post_json(
        "https://api.groq.com/openai/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 500},
        {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    return data["choices"][0]["message"]["content"]


def call_openai(model: str, prompt: str, api_key: str) -> str:
    data = _post_json(
        "https://api.openai.com/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 500},
        {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    return data["choices"][0]["message"]["content"]




def call_openrouter(model: str, prompt: str, api_key: str) -> str:
    # OpenRouter is one key that fronts hundreds of models from every major
    # vendor via a single OpenAI-compatible endpoint. This is what lets
    # LLM Safety Testing & Risk Analysis System test "any model, just type the name" without a
    # dedicated integration for every provider on earth.
    data = _post_json(
        "https://openrouter.ai/api/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 500},
        {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}",
         "HTTP-Referer": "https://llm-risk-system.local", "X-Title": "LLM Safety Testing & Risk Analysis System"},
    )
    if isinstance(data, dict) and "error" in data:
        err = data["error"]
        raise RuntimeError(err.get("message", str(err)) if isinstance(err, dict) else str(err))
    return data["choices"][0]["message"]["content"]


def call_github_models(model: str, prompt: str, api_key: str) -> str:
    # GitHub Models: free, no credit card, uses a GitHub Personal Access
    # Token (needs "Models" read-only permission) instead of a vendor API
    # key. This is the most direct free route to REAL OpenAI models
    # (gpt-4o, gpt-4o-mini, gpt-4.1) as well as Claude, Llama, DeepSeek and
    # Phi -- useful specifically when you want an actual closed model in
    # the comparison, not an open-weight stand-in. Rate limits are tight
    # and vary per model; RATE_LIMITS["github"] above throttles for this.
    # Get a token: https://github.com/settings/personal-access-tokens/new
    # Browse model IDs: https://github.com/marketplace/models
    if model not in KNOWN_GITHUB_MODELS:
        raise RuntimeError(
            f"'{model}' isn't a recognized GitHub Models id (possible typo?). "
            f"Known-good options: {', '.join(sorted(KNOWN_GITHUB_MODELS))}. "
            f"Browse the full catalog at https://github.com/marketplace/models."
        )
    data = _post_json(
        "https://models.github.ai/inference/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 500},
        {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}",
         "Accept": "application/vnd.github+json"},
    )
    if isinstance(data, dict) and "error" in data:
        err = data["error"]
        raise RuntimeError(err.get("message", str(err)) if isinstance(err, dict) else str(err))
    return data["choices"][0]["message"]["content"]


# Checked against https://github.com/marketplace/models as of mid-2026.
# This is intentionally a validation allowlist, not the full catalog --
# its job is to catch a typo (e.g. 'gpt-40' instead of 'gpt-4o') BEFORE
# spending a request, not to restrict which models can ever be used. If
# GitHub adds a model that isn't in this list yet, extend the set rather
# than bypassing the check.
KNOWN_GITHUB_MODELS = {
    "openai/gpt-4o", "openai/gpt-4o-mini", "openai/gpt-4.1", "openai/gpt-4.1-mini",
    "openai/gpt-4.1-nano", "openai/o1", "openai/o1-mini", "openai/o3", "openai/o3-mini",
    "openai/o4-mini",
}


PROVIDERS = {
    "anthropic": {"call": call_anthropic, "env": "ANTHROPIC_API_KEY", "default_model": "claude-sonnet-4-6"},
    "gemini": {"call": call_gemini, "env": "GEMINI_API_KEY", "default_model": "gemini-2.5-flash"},
    # llama-3.3-70b-versatile was flagged by Groq for deprecation (June 2026,
    # migrate to gpt-oss/qwen3.6) -- using their own recommended replacement
    # up front instead of waiting for it to start 404ing like DeepSeek did.
    "groq": {"call": call_groq, "env": "GROQ_API_KEY", "default_model": "openai/gpt-oss-120b"},
    "openai": {"call": call_openai, "env": "OPENAI_API_KEY", "default_model": "gpt-4o-mini"},
    # default_model here is a concrete free ($0) slug, not "openrouter/auto" --
    # "auto" is a router that can land on a paid model, which defeats the point.
    "openrouter": {"call": call_openrouter, "env": "OPENROUTER_API_KEY", "default_model": "meta-llama/llama-3.3-70b-instruct:free"},
    # Free, but via a GitHub PAT rather than a normal API key -- the one
    # route here to REAL gpt-4o/gpt-4o-mini rather than an open-weight
    # substitute. See call_github_models() above for setup.
    "github": {"call": call_github_models, "env": "GITHUB_TOKEN", "default_model": "openai/gpt-4o-mini"},
}

# Judge preference order when nothing is specified: free options first.
# Gemini is tried before Groq -- Groq's free tier has a tight tokens-per-
# minute cap (see TOKEN_LIMITS above) and the judge prompt (rubric + full
# prompt + full response) is large, so Groq was the most common source of
# judge-stage "provider_error" rows. Gemini's free tier tolerates these
# larger judge prompts much better. anthropic/openai only get used if the
# person has deliberately set up a paid key AND nothing free is available
# -- never chosen over a free option.
JUDGE_PREFERENCE_ORDER = ["gemini", "groq", "openrouter", "github", "anthropic", "openai"]

# Small curated safety net used only if a live OpenRouter free-model lookup
# fails (network hiccup, etc). Kept short and refreshed to genuinely free
# ($0, ":free"-suffixed) open-weight models as of mid-2026.
_FREE_FALLBACK_BY_FAMILY = {
    "llama": "meta-llama/llama-3.3-70b-instruct:free",
    "mixtral": "mistralai/mixtral-8x7b-instruct:free",
    "mistral": "mistralai/mistral-7b-instruct:free",
    "qwen": "qwen/qwen3-coder:free",
    "deepseek": "deepseek/deepseek-r1:free",
    "gemma": "google/gemma-3-27b-it:free",
    "gpt": "openai/gpt-oss-120b:free",
    "nemotron": "nvidia/nemotron-nano-9b-v2:free",
    "phi": None,
    "claude": None,   # no free/open-weight Claude exists
    "grok": None,     # no free/open-weight Grok exists
}

# name-keyword -> (native FREE provider to try first, OpenRouter vendor slug
# to search under for a free equivalent if there's no free native option).
# Only Gemini and Groq are listed as "native" here because they're the only
# genuinely free-with-no-card native integrations; gpt-/claude-/grok- style
# names always route to a free open-source stand-in via OpenRouter instead
# of their native (paid-only) APIs, unless the person overrides the
# provider explicitly in the dashboard's advanced options.
_NAME_GUESSES = [
    (("gemini",), "gemini", "google"),
    # Each Groq-native family below now ALSO carries the OpenRouter vendor
    # slug that serves the same model family for free. Previously these had
    # vendor=None, so if Groq's own id for a name went stale (exactly what
    # happened when Groq deprecated llama-3.3-70b-versatile) or the live
    # Groq catalog lookup itself failed, there was nowhere to fall back to
    # -- the request just died on Groq. Every family now gets the same
    # native-then-fallback cascade DeepSeek already had.
    (("llama",), "groq", "meta-llama"),
    (("mixtral",), "groq", "mistralai"),
    (("mistral",), "groq", "mistralai"),
    (("qwen",), "groq", "qwen"),
    (("gemma",), "groq", "google"),
    (("phi-",), "groq", "microsoft"),
    (("nemotron",), "groq", "nvidia"),
    (("gpt-oss",), "groq", "openai"),
    # Explicit request for the original R1 checkpoint specifically (not the
    # No native DeepSeek integration -- every "deepseek"-ish name (r1,
    # v4-flash, deepseek-chat, etc.) routes to OpenRouter's free
    # deepseek/deepseek-r1:free instead. Requires OPENROUTER_API_KEY.
    (("deepseek",), None, "deepseek"),
    (("gpt-", "gpt3", "gpt4", "gpt5", "chatgpt", "o1-", "o3-", "o4-", "davinci"), "github", "openai"),
    (("claude",), None, "anthropic"),
    (("grok",), None, "x-ai"),
]

_FREE_IDS_CACHE = {"ts": 0.0, "ids": None}
_FREE_IDS_TTL = 1800  # 30 min


def _openrouter_free_ids():
    """Live list of genuinely free ($0) OpenRouter model ids, cached briefly.
    Falls back to an empty set (triggering the curated table above) if the
    lookup fails -- this endpoint doesn't require an API key."""
    now = time.time()
    if _FREE_IDS_CACHE["ids"] is not None and now - _FREE_IDS_CACHE["ts"] < _FREE_IDS_TTL:
        return _FREE_IDS_CACHE["ids"]
    try:
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/models",
            headers={"User-Agent": "Mozilla/5.0 (compatible; LLM Safety Testing & Risk Analysis System/1.0)"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())

        def _is_zero(v):
            try:
                return float(v) == 0.0
            except (TypeError, ValueError):
                return False

        ids = set()
        for m in data.get("data", []):
            mid = m.get("id", "")
            pricing = m.get("pricing", {}) or {}
            if mid.endswith(":free") or (_is_zero(pricing.get("prompt")) and _is_zero(pricing.get("completion"))):
                ids.add(mid)
        _FREE_IDS_CACHE.update(ts=now, ids=ids)
        return ids
    except Exception:
        return _FREE_IDS_CACHE["ids"] or set()


def _find_free_openrouter_model(name_lower: str, vendor_hint: str = None):
    """Best-effort match of a requested model name to a genuinely free
    OpenRouter model. Tries a live lookup first, falls back to the curated
    table if that's unavailable. Returns a model id or None."""
    base = name_lower.split(":")[0]
    ids = _openrouter_free_ids()
    if ids:
        if vendor_hint:
            exact = f"{vendor_hint}/{base}:free"
            if exact in ids:
                return exact
        hits = [i for i in ids if base in i.lower()]
        if hits:
            return sorted(hits, key=len)[0]
        if vendor_hint:
            vendor_hits = [i for i in ids if i.lower().startswith(vendor_hint + "/")]
            if vendor_hits:
                return sorted(vendor_hits, key=len)[0]
    for family, slug in _FREE_FALLBACK_BY_FAMILY.items():
        if family in name_lower and slug:
            return slug
    return None


def _ensure_free_openrouter_slug(slug: str):
    """Given an explicit vendor/model slug the person typed directly, return
    a free ($0) version of it if one exists -- the slug itself if it's
    already free, its ':free' variant, or another free listing of the same
    base model. Returns None if no free version exists.

    If OpenRouter's live model-list lookup fails (network hiccup, timeout,
    OpenRouter having a bad moment) this now falls back to the same
    curated _FREE_FALLBACK_BY_FAMILY table _find_free_openrouter_model
    already uses, instead of giving up -- previously an explicit slug like
    'deepseek/deepseek-r1' had NO fallback at all if the live lookup came
    back empty, which could look exactly like "this model needs a paid
    plan" when a known-good free equivalent was sitting right there."""
    ids = _openrouter_free_ids()
    if not ids:
        if slug.endswith(":free"):
            return slug
        base = slug.split(":")[0]
        vendor = base.split("/")[0] if "/" in base else None
        name_part = base.split("/", 1)[1] if "/" in base else base
        for family, fallback_slug in _FREE_FALLBACK_BY_FAMILY.items():
            if fallback_slug and family in name_part.lower():
                return fallback_slug
        # Vendor prefix already looks free-eligible (e.g. typed the vendor
        # slug directly) but no family keyword matched -- last resort,
        # guess the ':free' variant since that's the overwhelmingly common
        # convention; call_model() will still surface a clear error if
        # this particular guess turns out to be wrong.
        return candidate if (candidate := f"{base}:free") else None
    if slug in ids:
        return slug
    candidate = slug if slug.endswith(":free") else f"{slug}:free"
    if candidate in ids:
        return candidate
    base = slug.split(":")[0]
    hits = [i for i in ids if i.split(":")[0] == base]
    return hits[0] if hits else None


_GROQ_IDS_CACHE = {"ts": 0.0, "ids": None}
_GROQ_IDS_TTL = 1800  # 30 min


def _groq_model_ids():
    """Live list of model ids Groq currently serves, cached briefly. Groq
    retires/renames models with little notice (e.g. DeepSeek was pulled
    entirely in Sept 2025; llama-3.3-70b-versatile itself is now flagged
    for deprecation) -- a static hardcoded list of "known good" Groq model
    names goes stale and silently 404s, which is exactly what happened.
    Checking the live catalog before sending a real request catches that
    up front with a clear message instead of a mysterious provider_error
    discovered only after a whole benchmark run. Returns None (skip
    validation) if there's no key yet or the lookup fails and there's no
    cached copy -- never blocks someone from testing just because this
    one lookup had a hiccup."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None
    now = time.time()
    if _GROQ_IDS_CACHE["ids"] is not None and now - _GROQ_IDS_CACHE["ts"] < _GROQ_IDS_TTL:
        return _GROQ_IDS_CACHE["ids"]
    try:
        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        ids = {m.get("id") for m in data.get("data", []) if m.get("id")}
        _GROQ_IDS_CACHE.update(ts=now, ids=ids)
        return ids
    except Exception:
        return _GROQ_IDS_CACHE["ids"]


def guess_provider_and_model(raw_name: str):
    """Given a free-typed model name like 'gpt-4o' or 'llama-3.3-70b-versatile',
    resolve it to a genuinely FREE way to run it: a native no-card-required
    provider (Gemini, Groq) when the name matches one, otherwise a free
    ($0, open-source/open-weight) model on OpenRouter. Paid/"pro" models are
    never silently used -- if a name has no free equivalent, this raises a
    clear error instead of quietly spending money. Lets the dashboard offer
    a single "type any model" box instead of a provider picker.

    Returns (provider, model, note); note explains any auto-routing/
    substitution that happened (e.g. 'gpt-4o' -> a free open-weight GPT-OSS
    model), so what actually ran is never a surprise.
    """
    name = raw_name.strip()
    if not name:
        raise ValueError("Type a model name, e.g. 'gemini-2.5-flash', 'gemini-2.5-flash-lite', 'gpt-4o', 'qwen3-coder'.")
    # Defends against stray spaces from typos/dictation/autocorrect around
    # hyphens and slashes, e.g. "deepseek- r1" or "llama- 3.3-70b" -- these
    # would otherwise silently 404 instead of matching a real model id.
    name = re.sub(r"\s*-\s*", "-", name)
    name = re.sub(r"\s*/\s*", "/", name)
    lower = name.lower()

    def has_key(p):
        return bool(os.environ.get(PROVIDERS[p]["env"]))

    # Groq's OpenAI open-weight models are namespaced "openai/gpt-oss-*",
    # which looks like an OpenRouter vendor/model slug (contains "/") but
    # is meant to run natively on Groq. Special-cased here so it isn't
    # swallowed by the generic OpenRouter-slug branch below.
    if "gpt-oss" in lower and has_key("groq"):
        canonical = name if lower.startswith("openai/") else f"openai/{name}"
        ids = _groq_model_ids()
        if ids is None or canonical in ids:
            return "groq", canonical, None
        # Groq id is stale/typo'd -- don't just die here; try the same
        # free OpenRouter fallback every other family gets before giving up.
        if has_key("openrouter"):
            free_slug = _find_free_openrouter_model(lower, "openai")
            if free_slug:
                return "openrouter", free_slug, (
                    f"'{canonical}' isn't a currently live Groq model id (typo, or Groq has since retired it). "
                    f"Fell back to a free OpenRouter equivalent instead: '{free_slug}'."
                )
        suggestion = difflib.get_close_matches(canonical, ids, n=1, cutoff=0.5) if ids else []
        hint = f" Did you mean '{suggestion[0]}'?" if suggestion else ""
        raise ValueError(
            f"'{canonical}' isn't a currently live Groq model id (typo, or Groq has since retired it).{hint} "
            f"See https://console.groq.com/docs/models for Groq's current list. No free OpenRouter "
            f"equivalent was found either."
        )

    # Already an explicit OpenRouter-style "vendor/model" slug.
    if "/" in name:
        if not has_key("openrouter"):
            raise ValueError(
                f"'{name}' looks like an OpenRouter model slug. Set OPENROUTER_API_KEY "
                f"(free, no card: https://openrouter.ai/keys) to test it."
            )
        free_slug = _ensure_free_openrouter_slug(name)
        if free_slug:
            note = None if free_slug == name else f"'{name}' isn't free; routed to its free variant '{free_slug}' instead."
            return "openrouter", free_slug, note
        raise ValueError(
            f"'{name}' doesn't have a free ($0) variant on OpenRouter. Browse free models at "
            f"https://openrouter.ai/models?max_price=0 and type one of those names directly."
        )

    for keywords, native, vendor in _NAME_GUESSES:
        if any(k in lower for k in keywords):
            # native_error stays None if the native route works OR isn't
            # configured (no key) -- it's only set when the native provider
            # IS configured but this specific model id is invalid/stale
            # there. Either way, we now fall through to the vendor's
            # OpenRouter fallback below instead of stopping here, so a
            # single provider's naming drift (e.g. Groq deprecating a
            # model id) can't permanently strand a whole model family.
            native_error = None
            github_retired_note = None
            if native == "github" and _github_models_retired():
                # GitHub Models has retired (2026-07-30) -- don't even try
                # it, regardless of GITHUB_TOKEN. Prefer Groq's own native
                # gpt-oss route (separate quota, no card, usually faster)
                # over the OpenRouter fallback below when it's available;
                # otherwise fall through to OpenRouter. Either way, no
                # error is surfaced to the person and no payment is ever
                # required -- and the note always says *why* (retirement),
                # never the misleading "GITHUB_TOKEN isn't set" (it may
                # well still be set; it just no longer works).
                github_retired_note = (
                    "GitHub Models retired on 2026-07-30, so real GPT models are no longer reachable for free "
                    "anywhere. "
                )
                if has_key("groq"):
                    return "groq", "openai/gpt-oss-120b", (
                        github_retired_note + f"Routed '{name}' to OpenAI's own open-weight gpt-oss-120b via Groq "
                        f"(free, no card, separate quota from OpenRouter) instead."
                    )
                native_error = None
            elif native and has_key(native):
                if native == "github" and "/" not in name:
                    gh_model = f"openai/{name}"
                    if gh_model in KNOWN_GITHUB_MODELS:
                        return "github", gh_model, f"Routed to real GPT via GitHub Models (free, no card): '{gh_model}'."
                    suggestion = difflib.get_close_matches(gh_model, KNOWN_GITHUB_MODELS, n=1, cutoff=0.6)
                    hint = f" Did you mean '{suggestion[0]}'?" if suggestion else ""
                    native_error = (
                        f"'{name}' isn't a recognized GitHub Models id (possible typo?).{hint} "
                        f"Known-good options: {', '.join(sorted(KNOWN_GITHUB_MODELS))}."
                    )
                elif native == "groq":
                    ids = _groq_model_ids()
                    if ids is None or name in ids:
                        return native, name, None
                    suggestion = difflib.get_close_matches(name, ids, n=1, cutoff=0.5)
                    hint = f" Did you mean '{suggestion[0]}'?" if suggestion else ""
                    native_error = (
                        f"'{name}' isn't a currently live Groq model id (typo, or Groq has since retired/renamed it).{hint}"
                    )
                else:
                    return native, name, None

            # Either native wasn't configured, or it was but this id
            # doesn't resolve there -- try the free OpenRouter equivalent
            # for this family before giving up.
            if vendor and has_key("openrouter"):
                free_slug = _find_free_openrouter_model(lower, vendor)
                if free_slug:
                    if native_error:
                        note = f"{native_error} Fell back to a free OpenRouter equivalent instead: '{free_slug}'."
                    elif github_retired_note:
                        note = github_retired_note + f"Routed to a free OpenRouter equivalent instead: '{free_slug}' (no cost)."
                    elif native:
                        note = (f"'{name}' would normally run on {native}, but {PROVIDERS[native]['env']} isn't set. "
                                 f"Routed to a free OpenRouter equivalent instead: '{free_slug}' (no cost).")
                    else:
                        note = (f"'{name}' isn't available for free here, so it was routed to a free, "
                                 f"open-source alternative instead: '{free_slug}' (no cost).")
                    return "openrouter", free_slug, note

            if native_error:
                extra = (" No free OpenRouter equivalent was found either." if has_key("openrouter") else
                         " Set OPENROUTER_API_KEY (free, no card: https://openrouter.ai/keys) to try a free "
                         "open-source alternative instead.")
                raise ValueError(native_error + extra)
            if not has_key("openrouter"):
                if github_retired_note:
                    raise ValueError(
                        github_retired_note + "Set GROQ_API_KEY or OPENROUTER_API_KEY (both free, no card) to run "
                        "a free open-weight alternative instead."
                    )
                if native:
                    raise ValueError(
                        f"'{name}' needs {PROVIDERS[native]['env']} to run there, or OPENROUTER_API_KEY "
                        f"(free, no card: https://openrouter.ai/keys) to run a free open-source alternative instead."
                    )
                raise ValueError(
                    f"'{name}' is a closed, paid-only model with no free tier of its own. Set OPENROUTER_API_KEY "
                    f"(free, no card: https://openrouter.ai/keys) to run a free open-source alternative instead."
                )
            raise ValueError(
                f"Couldn't find a free/open-source equivalent for '{name}'. Browse free models at "
                f"https://openrouter.ai/models?max_price=0 and type one of those names directly."
            )

    # Unrecognized name -- last resort: search OpenRouter's free catalog for it.
    if has_key("openrouter"):
        free_slug = _find_free_openrouter_model(lower, None)
        if free_slug:
            return "openrouter", free_slug, (
                f"Didn't recognize '{name}' by name, so it was routed to a free, open-source model instead: "
                f"'{free_slug}' (no cost)."
            )
        raise ValueError(
            f"Couldn't match '{name}' to a free model. Browse free models at "
            f"https://openrouter.ai/models?max_price=0 and type one of those names directly."
        )

    raise ValueError(
        f"Couldn't figure out how to run '{name}' for free. Set OPENROUTER_API_KEY "
        f"(free, no card: https://openrouter.ai/keys) to unlock free open-source models by name."
    )


def pick_judge_provider(exclude: str = None):
    """Auto-pick a judge provider that's configured with a key, prefers a
    free option, and is different from the target model's provider where
    possible, so the judge doesn't share blind spots with the model it's
    grading."""
    candidates = [p for p in JUDGE_PREFERENCE_ORDER if os.environ.get(PROVIDERS[p]["env"])]
    for p in candidates:
        if p != exclude:
            return p
    return candidates[0] if candidates else None


def pick_judge_providers(exclude: str = None):
    """Full ordered fallback chain of every judge-eligible provider that has
    a key configured, most-preferred first. `exclude` (the target model's
    own provider) is pushed to the back rather than dropped -- still usable
    as a last resort if nothing else has a key -- so the judge stage can
    automatically hop to the next available provider on a rate-limit/auth
    failure instead of one bad provider silently killing every judge call
    for the rest of the run."""
    candidates = [p for p in JUDGE_PREFERENCE_ORDER if os.environ.get(PROVIDERS[p]["env"])]
    if exclude in candidates:
        candidates = [p for p in candidates if p != exclude] + [exclude]
    return candidates


FRIENDLY_ERROR_MESSAGES = {
    "This model's API limit has been reached for now. Try again tomorrow, or pick a different model.",
    "This model isn't available on our free plan right now. We're working on adding paid access.",
    "This model isn't available through our free APIs right now. Try a different model name.",
    "This model took too long to respond. Please try again.",
    "This model couldn't be reached right now. Please try a different model, or try again later.",
}

# Messages that mean the SAME failure will happen on every remaining prompt
# in this run -- there's no point calling the API again and burning through
# the rest of a benchmark batch on a provider/model that's already known to
# be dead for this run (an API/daily limit reached, an account with no
# usable access, or a model name that doesn't exist). Excludes
# transient-sounding ones ("took too long", generic "couldn't be reached")
# since those genuinely can resolve case to case within the same run. Used
# by run_benchmark() to stop early instead of wasting the rest of a batch
# (and the person's time, and any remaining real quota) on guaranteed
# repeat failures.
HARD_STOP_MESSAGES = {
    "This model's API limit has been reached for now. Try again tomorrow, or pick a different model.",
    "This model isn't available on our free plan right now. We're working on adding paid access.",
    "This model isn't available through our free APIs right now. Try a different model name.",
}


def is_provider_error(text) -> bool:
    """True if `text` is one of call_model()'s friendly error messages
    (i.e. the call failed) rather than a real model response. Used
    instead of checking for a raw "[ERROR" prefix, since that technical
    prefix no longer appears in anything call_model() returns -- what it
    returns is always safe to display as-is."""
    return isinstance(text, str) and text in FRIENDLY_ERROR_MESSAGES


def is_hard_stop_error(text) -> bool:
    """True if `text` is a friendly error message that will repeat
    identically for every remaining prompt in this run (see
    HARD_STOP_MESSAGES above) -- signal to stop calling the API and skip
    the rest of the batch instead of wasting calls on guaranteed failures."""
    return isinstance(text, str) and text in HARD_STOP_MESSAGES


def friendly_error_response(raw: str) -> str:
    """Turn a raw provider error into a short, plain-language message safe
    to show directly in the dashboard (as a response or judge reasoning
    field) -- no HTTP codes, endpoint URLs, env var names, or provider
    jargon. The original technical detail is still printed to the
    server's own console (see call_model below), so nothing is lost for
    debugging -- it just doesn't surface to whoever is looking at results."""
    if not isinstance(raw, str):
        return raw
    low = raw.lower()
    if any(k in low for k in ("402", "insufficient balance", "rate limit", "429", "quota",
                               "api limit finished", "resource_exhausted")):
        return "This model's API limit has been reached for now. Try again tomorrow, or pick a different model."
    if any(k in low for k in ("401", "403", "invalid api key", "invalid_api_key", "unauthorized", "authentication", "permission")):
        return "This model isn't available on our free plan right now. We're working on adding paid access."
    if any(k in low for k in ("404", "does not exist", "model_not_found", "not found")):
        return "This model isn't available through our free APIs right now. Try a different model name."
    if any(k in low for k in ("timeout", "timed out", "connection")):
        return "This model took too long to respond. Please try again."
    if raw.startswith("[ERROR"):
        return "This model couldn't be reached right now. Please try a different model, or try again later."
    return raw


def call_model(provider: str, model: str, prompt: str) -> str:
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider '{provider}'. Choose from: {list(PROVIDERS)}")
    cfg = PROVIDERS[provider]
    api_key = os.environ.get(cfg["env"])
    if not api_key:
        raise SystemExit(f"Set {cfg['env']} environment variable before using provider '{provider}'.")
    try:
        daily_key = model if provider == "github" else None
        daily_cap = github_daily_cap(model) if provider == "github" else None
        RATE_LIMITER.wait(provider, estimated_tokens=_estimate_tokens(prompt),
                          daily_key=daily_key, daily_cap=daily_cap)  # self-throttles under request, token, AND (for github) tier-aware daily caps
    except RateLimitExhausted as e:
        # Treat exactly like any other provider failure: return an error
        # string for THIS record and let the caller keep going. Raising
        # here (as SystemExit, previously) would abort the whole
        # run_benchmark() loop and lose every already-completed record,
        # since SystemExit isn't caught by the `except Exception` below.
        print(f"[rate limit] {provider}/{model}: {e}")
        return friendly_error_response(f"[ERROR: {e}]")
    # Free-tier "out of usage" errors (429/quota/resource_exhausted) on
    # heavily-shared public endpoints (e.g. a popular free model on
    # OpenRouter) are often just the GLOBAL pool being busy for a few
    # seconds -- not a real per-account limit, since our own RATE_LIMITER
    # above already guarantees we're within OUR side of any quota. A short
    # retry-with-backoff catches these transient dips instead of recording
    # a whole case as "model unreachable" on the first blip. This does NOT
    # apply to auth (401/403) or not-found (404) errors -- those are
    # permanent for this run, so they fail immediately with no retry.
    RETRY_ATTEMPTS = 2
    RETRY_DELAYS = (3, 8)  # seconds, increasing backoff -- kept short so a batch of many cases doesn't stall
    last_raw = None
    for attempt in range(RETRY_ATTEMPTS):
        try:
            return cfg["call"](model, prompt, api_key)
        except Exception as e:
            raw = f"[ERROR calling {provider}/{model}: {e}]"
            last_raw = raw
            low = str(e).lower()
            is_transient_quota = any(
                k in low for k in ("rate limit", "429", "quota", "api limit finished", "resource_exhausted")
            )
            if not is_transient_quota or attempt == RETRY_ATTEMPTS - 1:
                print(raw)  # full technical detail stays in the server's own console
                return friendly_error_response(raw)
            wait_s = RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)]
            print(f"[retry {attempt + 1}/{RETRY_ATTEMPTS}] {provider}/{model} hit a transient quota error, "
                  f"retrying in {wait_s}s: {e}")
            time.sleep(wait_s)
    # Unreachable in practice (loop always returns or raises above), but
    # kept as a safety net so the function can't silently fall through.
    return friendly_error_response(last_raw)


if __name__ == "__main__":
    # Quick manual smoke test: python src/providers.py gemini "Say hi in 5 words"
    import sys
    provider = sys.argv[1] if len(sys.argv) > 1 else "gemini"
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Say hello in exactly 5 words."
    model = PROVIDERS[provider]["default_model"]
    print(f"Calling {provider}/{model}...")
    print(call_model(provider, model, prompt))
