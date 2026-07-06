# LLM Safety Testing & Risk Analysis System

A compact, reproducible pipeline for evaluating LLM responses across five
safety risk categories, with an explicit rubric, a rule-based prescreen,
an LLM judge, and a reliability check on the judge itself.

Built as a research work-sample project. See `docs/methodology.md` for
full positioning against prior work (AdvBench, Do-Not-Answer, HarmBench,
JailbreakBench, Project Moonshot) and stated limitations.

## Risk categories

1. **Prompt injection / jailbreak** — instruction override, persona
   hijack, hidden-prompt extraction, hypothetical framing, etc.
2. **Cyber misuse** — phishing, malware-adjacent scripting, credential
   theft framing, evasion requests (sampled from AdvBench, filtered to
   cyber-specific behaviors; explosives/weapons/bio excluded by design).
3. **Privacy leakage** — extraction/inference of personal data, session-
   memory abuse patterns.
4. **Emotional manipulation / dependency-style behavior** — isolation
   patterns, discouraging real help-seeking, manufactured intimacy,
   unconditional-agreement demands.
5. **High-stakes unsafe advice** — medical/legal/financial guidance given
   with false certainty or missing caveats.

## Project structure

```
llm-risk-system/
├── data/
│   ├── raw/                     # pulled source datasets (AdvBench, Do-Not-Answer)
│   └── curated/                 # llm-risk-system_prompts.{csv,json} — the actual eval set
├── src/
│   ├── curate_dataset.py        # builds the curated prompt set from raw sources
│   ├── run_benchmark.py         # sends prompts to a target model, saves responses
│   ├── rule_based_prescreen.py  # fast first-pass heuristic labeling
│   ├── llm_judge.py             # rubric-based LLM judge (the main evaluator)
│   └── agreement_check.py       # Cohen's kappa: judge vs. human spot-check labels
├── results/                     # output of runs (gitignored except .gitkeep)
├── dashboard/                   # standalone HTML dashboard (see below)
└── docs/
    └── methodology.md
```

## Setup

```bash
pip install pandas
```

No API costs required — this project runs entirely on free API tiers.

### Get free API keys (no credit card needed, as of mid-2026)

- **Google Gemini** (recommended as your primary "model under test" — frontier-class quality): go to https://aistudio.google.com/apikey, sign in with a Google account, click "Create API key." Free tier: ~1,500 requests/day on `gemini-2.5-flash`.
- **Groq** (recommended as your judge model, or a second model to compare — very fast, open-weight models): go to https://console.groq.com/keys, sign up, create a key. Free tier: high daily request volume on models like `gpt-oss-120b`.
- **OpenRouter** (unlocks "type literally any model name" — one key fronts 25+ free open-weight models: Llama, DeepSeek, Qwen, Gemma, GPT-OSS, and more): go to https://openrouter.ai/keys. This is what lets `guess_provider_and_model()` auto-substitute a free equivalent when you type a closed model name like `claude-sonnet-4-5`.
- **GitHub Models** (the one route to a *real* GPT model, not an open-weight stand-in): create a fine-grained personal access token at https://github.com/settings/personal-access-tokens/new with "Models" read-only permission. Typing `gpt-4o` or `chatgpt-4` routes here automatically if `GITHUB_TOKEN` is set, ahead of the OpenRouter substitute.

Set whichever you have as environment variables:
```bash
export GEMINI_API_KEY=...
export GROQ_API_KEY=...
export OPENROUTER_API_KEY=...
export GITHUB_TOKEN=...
```
(Anthropic's/OpenAI's native APIs are also supported via `ANTHROPIC_API_KEY`/`OPENAI_API_KEY` if you have credits, but they have no indefinite free tier as of mid-2026 and are only used if you explicitly pick them in the dashboard's advanced options — never chosen automatically.)

Using two *different* providers — one for the model under test, a different one as judge — is intentional, not just cost-driven: it reduces the risk that the judge shares blind spots with the model it's grading (see `docs/methodology.md`).

**Testing a product with no API at all** (Emergent, Replit Agent, or anything else you can only reach through a chat window): see `src/manual_run.py` and the "Testing an Agent With No API" panel on the dashboard — it exports a copy/paste prompt pack and imports your pasted replies back into the same record format everything else uses.

## What's original research vs. what's tooling

For anyone reviewing this as a fellowship work sample, the actual
contribution is in `data/curated/`, `src/curate_dataset.py`, and the
rubric + category-specific judging guidance in `src/llm_judge.py`:

- The 123-prompt curated set, sourced and filtered from AdvBench and
  Do-Not-Answer with source IDs preserved, plus an originally-written
  prompt-injection category built from the OWASP LLM01 taxonomy.
- The 4-label rubric (safe_refusal / borderline_partial_compliance /
  unsafe_compliance / over_refusal) and per-category judging criteria.
- The reliability check design (`agreement_check.py`, Cohen's kappa
  against a human-labeled sample) and the stated methodology/limitations
  in `docs/methodology.md`.

Everything else — `server.py`, the dashboard, `guess_provider_and_model()`'s
free-routing logic, the rate limiter, and `manual_run.py` — is tooling
built on top of that research core. It calls the same `run_benchmark()` /
`judge_records()` functions a person would otherwise run from the command
line; no prompts, rubric text, or judging logic changed to build any of it.
Worth mentioning as engineering work in a writeup, but distinct from the
scientific claims above.

## Running the full pipeline

Two ways to run it — pick whichever you want.

### Option A: dashboard (type in a model, click Run)

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=...      # free, no card: https://aistudio.google.com/apikey
export GROQ_API_KEY=...        # free, no card: https://console.groq.com/keys
export OPENROUTER_API_KEY=...  # optional: unlocks testing *any* model by name (https://openrouter.ai/keys)
python server.py
```

Open `http://localhost:8000`. In the "Run a New Test" panel there's one
field: **Model** — just type a name, e.g. `gpt-4o`, `gemini-2.5-flash`,
`gpt-oss-120b`, `deepseek-r1`, `qwen3-coder`.
It always runs for free: native no-card providers (Gemini, Groq) are used
when the name matches one; otherwise it's routed to a free ($0) open-source
model on OpenRouter. Closed/paid names that DO have a free open-weight
equivalent (e.g. `gpt-4o`) are automatically substituted with one (e.g.
OpenAI's own open-source GPT-OSS model) rather than silently running — and
billing — the real paid model. A few closed models genuinely have no free
or open-weight equivalent anywhere (Claude, Grok) — typing one of those
gives a clear "not available for free" message instead of guessing. The
dashboard tells you exactly what ran and
why in the run-status line. It also auto-picks a *different*, also-free
model as judge, so the judge doesn't share blind spots with the model
being tested. Click **Run Benchmark**; progress streams live, results
render automatically, and are saved to
`results/judged_<provider>_<model>_<job_id>.json`.

If you do have a paid ANTHROPIC_API_KEY or OPENAI_API_KEY and specifically
want to test the real paid model, set that env var and pick the provider
explicitly in "advanced options" — the free-by-default auto-detection only
applies when you leave provider on "auto-detect".

Past runs are picked up automatically too — the dashboard lists
everything already in `results/` in a "previous runs" dropdown and
opens the most recent one on load, so you never have to manually
re-upload a file you already generated.

Each run also has a **"show more"** panel that names which categories
that model scored weakest in, alongside the benchmark's own known
limitations (sample size, single-turn scope, etc.) — so a low failure
rate isn't presented as a bigger claim than the data supports.

### Option B: command line

```bash
# 1. (Already done — regenerate if you want to change sampling/seeds)
python src/curate_dataset.py

# 2. Run the target model against every curated prompt (free: Gemini)
python src/run_benchmark.py --provider gemini --model gemini-2.5-flash --out results/raw_responses.json

# 3. Judge the responses with a DIFFERENT free provider (Groq)
python src/llm_judge.py --input results/raw_responses.json --output results/judged.json --judge_provider groq

# 4. (Optional but recommended) hand-label ~15-20 cases yourself in the
#    same format as data/human_label_sample.json.example, then:
python src/agreement_check.py --judged results/judged.json --human_labels data/human_label_sample.json
```

To compare a second model (closer to the original "2-3 models tested" scope), just run step 2 again with `--provider groq --model gpt-oss-120b --out results/raw_responses_groq.json`, judge it the same way, and combine both result files in the dashboard for a head-to-head comparison.

## Testing a different model

`src/providers.py` holds five provider integrations (Anthropic, Gemini, Groq, OpenAI, OpenRouter) behind one `call_model()` function, plus `guess_provider_and_model()`, which maps any free-typed model name to a way of running it **for free** — a native no-card provider (Gemini, Groq) if the name matches one, otherwise a free ($0, open-source) model on OpenRouter. Closed/paid names with a free open-weight equivalent (e.g. `gpt-4o`) are auto-substituted with one rather than silently running (and billing) the real paid model; names with no free equivalent anywhere (e.g. Claude, Grok) raise a clear error instead of guessing. Add a new native provider by writing one function following the same pattern.

## Rate limiting

Every call to `call_model()` passes through a shared, thread-safe
sliding-window `RateLimiter` (`src/providers.py`) before it's made, keyed
per provider, with caps set a little under each provider's published
free-tier limit (Gemini ~12/min, Groq ~25/min, OpenRouter free models
~18/min, GitHub Models ~8/min *and* ~45/day — it supports more than one
window per provider specifically for GitHub's tighter daily cap). If a
run would exceed that, the limiter blocks and waits for the window to
roll rather than firing the request and hoping — so the target-model
run, the judge, and multiple dashboard jobs running close together all
self-throttle automatically, without you having to hand-tune a `--delay`
flag. `/api/providers` reports live usage (`used`/`max` per window),
which the dashboard shows as small chips under the run panel so you can
see how close to the cap a provider is in real time. A live 429 from the
provider itself still triggers a backoff-and-retry in `_post_json` as a
second line of defense. If a longer cap (like GitHub Models' daily limit)
is actually used up rather than just briefly hit, the limiter doesn't
sleep silently for hours — it stops the run right away with a clear
message (e.g. `github API limit finished, try again tomorrow.`), which
shows up directly in the dashboard's run status.

## Dashboard

`dashboard/index.html` is a self-contained, responsive dashboard (no
build step, no dependencies) that auto-loads the most recent run from
`results/` on open — no manual re-upload needed — and lets you switch
between past runs from a dropdown, or run a new test directly from the
"Run a New Test" panel. It renders: failure rate by category, a risk
heatmap (category × label), example transcripts for each failure type,
and a collapsible transparency panel that breaks down every category's
failure rate individually and lists the actual failing prompts under
each one (id, subcategory, a preview, and the label) — click one to jump
straight to that case in the log below, plus the benchmark's stated
general limitations. The heatmap cells and category bars are
clickable — click one to filter the case log to that category and/or
outcome, with a "clear filter" link to reset. Layout is fluid down to
phone widths (stacking KPI cards, a horizontally-scrollable heatmap, a
single-column run form), and panels/cases animate in on load. A
collapsible "Testing an Agent With No API" panel walks through the
`manual_run.py` export/import workflow for products like Emergent or
Replit Agent that have no callable API. Open it directly in a browser
after generating results, or serve it via `python server.py` for the
live "Run a New Test" panel to work.

## Honesty notes

- This is a ~120-prompt proof-of-concept pipeline built over about a week,
  not a validated, publishable benchmark. Sample sizes per subcategory are
  too small to support strong claims about any specific model's true
  failure rate — see `docs/methodology.md` for the full limitations list.
- Roughly 56% of prompts are drawn verbatim from published academic
  red-teaming datasets (AdvBench, Do-Not-Answer) with source IDs preserved
  for traceability; the rest are clearly labeled as originally curated for
  this project, concentrated in the emotional-dependency category where
  existing benchmarks have the least coverage.
