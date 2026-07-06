# LLM Safety Testing & Risk Analysis System

A benchmark and analysis pipeline for evaluating how large language models respond to risky prompts across five safety categories: **prompt injection / jailbreak, cyber misuse, privacy leakage, emotional manipulation, and high-stakes unsafe advice**.

The system runs a target model on a curated benchmark built from **AdvBench**, **Do-Not-Answer**, and **project-curated prompts** for categories with limited public coverage. Responses are evaluated using a **rule-based prescreen** and a **rubric-based LLM judge**, then labeled as **safe refusal**, **borderline partial compliance**, **unsafe compliance**, or **over-refusal**. The project also includes a **judge–human agreement check using Cohen’s kappa** and an interactive **dashboard** for exploring failure patterns, risk categories, and example cases.

---

## Risk Categories

* **Prompt Injection / Jailbreak** — instruction override, hidden prompt extraction, adversarial reframing, safeguard bypass attempts
* **Cyber Misuse** — phishing, credential theft framing, malware-adjacent scripting, evasion requests
* **Privacy Leakage** — extraction, inference, or mishandling of personal or sensitive information
* **Emotional Manipulation / Dependency** — unhealthy relational framing, discouraging real-world support, manipulative dependency patterns
* **High-Stakes Unsafe Advice** — unsafe or overconfident medical, legal, or financial guidance

---

## How It Works

The pipeline:

1. runs the benchmark prompt set on a target model
2. stores the model responses
3. applies a **rule-based prescreen**
4. evaluates responses with a **rubric-based LLM judge**
5. assigns one of four labels:

   * `safe_refusal`
   * `borderline_partial_compliance`
   * `unsafe_compliance`
   * `over_refusal`
6. optionally checks judge reliability against human labels using **Cohen’s kappa**

---

## Project Structure

```bash id="q9v1xp"
llm-risk-system/
├── data/
│   ├── raw/                     # source benchmark datasets
│   └── curated/                 # final evaluation set
├── src/
│   ├── curate_dataset.py        # builds / refreshes the curated prompt set
│   ├── run_benchmark.py         # runs prompts on a target model
│   ├── rule_based_prescreen.py  # first-pass heuristic screening
│   ├── llm_judge.py             # rubric-based LLM evaluator
│   ├── agreement_check.py       # judge vs human agreement check
│   ├── manual_run.py            # workflow for testing products with no API
│   └── providers.py             # model provider integrations and routing
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

## Setup

Install dependencies:

```bash id="cwl1x2"
pip install -r requirements.txt
```

Set the API keys for the providers you want to use in your local environment.

---

## Run Locally

### Dashboard

```bash id="7kql0s"
python server.py
```

Then open:

```bash id="f4p4x5"
http://localhost:8000
```

### Command Line

```bash id="axvukn"
python src/curate_dataset.py
python src/run_benchmark.py --provider gemini --model gemini-2.5-flash --out results/raw_responses.json
python src/llm_judge.py --input results/raw_responses.json --output results/judged.json --judge_provider groq
python src/agreement_check.py --judged results/judged.json --human_labels data/human_label_sample.json
```

---

## Dashboard Features

The dashboard is designed to make benchmark results easier to inspect without manually going through output files. It includes:

* **Run benchmarks directly from the browser**
* **View overall and category-wise failure rates**
* **Risk heatmaps and label distributions**
* **Browse example unsafe, borderline, and refusal cases**
* **Filter results by category or label**
* **Compare and reopen previous benchmark runs**
* **Automatic loading of saved benchmark results**
* **Support for manual testing of products without an API**

---

## Deployment

**Live Demo:** `YOUR_DEPLOYED_LINK_HERE`

The project includes provider integrations for **Gemini, Groq, OpenRouter, GitHub Models, OpenAI, and Anthropic**. The deployed benchmark flow is built around the default setup used in this repository, while providers such as OpenAI and Anthropic may require paid API access and are therefore not part of the default public deployment workflow.

---

## Testing a Product with No API

Use:

```bash id="nqg66m"
python src/manual_run.py
```

This exports prompts for manual testing and lets you import responses back into the same evaluation format used by the benchmark.

---

## Output

Each run produces judged results in `results/`, which can be explored in the dashboard.

Outputs include:

* overall and category-wise failure rates
* label distributions
* example unsafe / borderline / over-refusal cases
* judge agreement statistics when human labels are provided

---

## Supported Providers

The benchmark uses a modular provider layer in `src/providers.py` and supports:

* **Gemini**
* **Groq**
* **OpenRouter**
* **GitHub Models**
* **OpenAI**
* **Anthropic**

---

## Notes

More detail on benchmark design, curation, and judging logic is in `docs/methodology.md`.
