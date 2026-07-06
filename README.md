# LLM Safety Testing & Risk Analysis System

A benchmark and analysis pipeline for evaluating how large language models respond to risky prompts across five safety categories: **prompt injection / jailbreak, cyber misuse, privacy leakage, emotional manipulation, and high-stakes unsafe advice**.

The system tests a target model on a curated benchmark built from **AdvBench**, **Do-Not-Answer**, and **project-curated prompts** for categories with limited public coverage. Model responses are evaluated using a **rule-based prescreen** and a **rubric-based LLM judge**, then labeled as **safe refusal, borderline partial compliance, unsafe compliance, or over-refusal**. The project also includes a **judge–human agreement check using Cohen’s kappa** and a **local dashboard** for exploring category-wise failures, label distributions, and example cases.

---

## What this project does

This project is designed to test how safely an LLM handles prompts that try to trigger harmful, manipulative, privacy-violating, or high-stakes unsafe behavior.

Given a benchmark of risky prompts, the pipeline:

1. sends each prompt to a target model,
2. stores the model’s response,
3. applies a rule-based prescreen for fast first-pass screening,
4. evaluates the response with a rubric-based LLM judge,
5. assigns one of four safety labels, and
6. summarizes the results through category-level failure rates, examples, and agreement statistics.

The goal is not just to measure whether a model “refuses” or “answers,” but to analyze **how** it fails, **where** it fails, and **which safety categories are weakest**.

---

## Risk categories covered

### 1) Prompt Injection / Jailbreak

Tests whether a model can be manipulated into ignoring instructions, revealing hidden prompts, following adversarial reframing, or bypassing intended safeguards.

### 2) Cyber Misuse

Covers prompts related to phishing, credential theft framing, malware-adjacent scripting, evasion behavior, and other cyber-abuse scenarios.

### 3) Privacy Leakage

Evaluates whether the model reveals, infers, or mishandles personal, sensitive, or confidential information.

### 4) Emotional Manipulation / Dependency-Style Behavior

Tests for assistant behavior that encourages emotional dependency, discourages real-world support, reinforces manipulative dynamics, or adopts unhealthy relational framing.

### 5) High-Stakes Unsafe Advice

Covers unsafe or overconfident responses in high-impact domains such as medical, legal, and financial advice.

---

## Evaluation approach

The benchmark combines prompts adapted from existing red-teaming and refusal datasets with project-curated prompts for categories where public coverage was limited. Responses are then evaluated in two stages:

* **Rule-based prescreen**
  A lightweight first-pass layer that flags obvious risky patterns and organizes responses before detailed judging.

* **Rubric-based LLM judge**
  A category-aware evaluator that assigns one of four labels:

  * `safe_refusal`
  * `borderline_partial_compliance`
  * `unsafe_compliance`
  * `over_refusal`

To check whether the judge is behaving consistently, the project also compares judged outputs against a human-labeled sample using **Cohen’s kappa**.

---

## Project structure

```bash id="o7l0rb"
llm-risk-system/
├── data/
│   ├── raw/                     # source benchmark datasets
│   └── curated/                 # final evaluation set used by the benchmark
├── src/
│   ├── curate_dataset.py        # builds / refreshes the curated prompt set
│   ├── run_benchmark.py         # runs prompts on a target model
│   ├── rule_based_prescreen.py  # first-pass heuristic screening
│   ├── llm_judge.py             # rubric-based LLM evaluator
│   ├── agreement_check.py       # judge vs human agreement check
│   ├── manual_run.py            # workflow for testing products with no API
│   └── providers.py             # model provider integrations + routing
├── dashboard/
│   └── index.html               # local dashboard UI
├── docs/
│   └── methodology.md
├── results/                     # benchmark outputs
├── server.py                    # serves dashboard + run endpoints
├── requirements.txt
└── README.md
```

---

## Core components

### Curated benchmark dataset

The benchmark covers five LLM safety risk categories and combines:

* prompts sourced from **AdvBench** and **Do-Not-Answer**
* project-curated prompts for categories with weaker public benchmark coverage, especially around prompt injection and emotional dependency-style behavior

### Rule-based prescreen

A lightweight filter used before full judging to catch obvious patterns and make evaluation more structured.

### Rubric-based LLM judge

The main evaluator. It uses category-specific judging criteria to classify each response into one of four labels rather than using a simple safe/unsafe binary.

### Agreement check

A reliability step that compares judge labels with human labels using **Cohen’s kappa**.

### Dashboard

A local dashboard for exploring benchmark outputs, category-wise failures, label distributions, and example cases.

---

## Setup

Install dependencies:

```bash id="pk0s9p"
pip install -r requirements.txt
```

Set any API keys you want to use:

```bash id="v9e7f4"
export GEMINI_API_KEY=...
export GROQ_API_KEY=...
export OPENROUTER_API_KEY=...
export GITHUB_TOKEN=...
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
```

The project supports multiple providers, including **Gemini, Groq, OpenAI, Anthropic, OpenRouter, and GitHub Models**.

---

## Running the benchmark

You can use either the dashboard or the command line pipeline.

## Option 1: Dashboard

Start the local server:

```bash id="8qbvvc"
python server.py
```

Then open:

```bash id="8s9doj"
http://localhost:8000
```

From the dashboard you can:

* run a benchmark on a target model
* inspect failure rates by category
* view label distributions and example cases
* compare previous runs
* load benchmark results directly from the `results/` directory

---

## Option 2: Command line pipeline

### 1) Build or refresh the curated dataset

```bash id="nrx5nd"
python src/curate_dataset.py
```

### 2) Run the target model on the benchmark prompts

```bash id="vxdy7x"
python src/run_benchmark.py --provider gemini --model gemini-2.5-flash --out results/raw_responses.json
```

### 3) Judge the responses

```bash id="8v5w7g"
python src/llm_judge.py --input results/raw_responses.json --output results/judged.json --judge_provider groq
```

### 4) Run the judge–human agreement check

```bash id="n4u1ic"
python src/agreement_check.py --judged results/judged.json --human_labels data/human_label_sample.json
```

---

## Testing a product with no API

For products that can only be tested through a chat interface, use:

```bash id="bzv9wd"
python src/manual_run.py
```

This workflow exports prompts for manual testing and lets you import pasted responses back into the same evaluation format used by the main benchmark pipeline.

---

## Output

Each benchmark run produces judged outputs in `results/` and can be inspected in the dashboard.

Typical outputs include:

* overall failure rate
* category-wise failure rate
* label distribution across the benchmark
* example unsafe / borderline / over-refusal cases
* agreement statistics for judge reliability checks

---

## Dashboard features

The dashboard is designed to make benchmark results easier to inspect without manually opening JSON files.

It supports:

* **failure rate by category**
* **risk heatmaps** for category × label breakdown
* **example transcripts / failure cases**
* **previous run selection**
* **filtered inspection of unsafe, borderline, and over-refusal cases**
* **manual testing workflow support** for products without an API

---

## Supported providers

The benchmark can run models from multiple providers through a shared interface in `src/providers.py`.

Current provider support includes:

* **Gemini**
* **Groq**
* **OpenAI**
* **Anthropic**
* **OpenRouter**
* **GitHub Models**

This makes it possible to evaluate different models under the same benchmark and judging setup.

---

## Repository highlights

* Multi-category LLM safety benchmark
* Prompt set built from benchmark sources plus project curation
* Rule-based + rubric-based evaluation pipeline
* Judge reliability check using Cohen’s kappa
* Dashboard for inspecting failures and comparing runs
* Support for both API-based models and manually tested products

---

## Notes

For more detail on the benchmark design, curation approach, judging logic, and limitations, see:

* `docs/methodology.md`
