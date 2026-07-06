# LLM Safety Testing & Risk Analysis System: Methodology & Positioning

## Motivation

Most public discussion of LLM safety failures is anecdotal — a screenshot of
a bad jailbreak, a viral thread about a manipulative chatbot response. What's
harder to find is a small, reproducible pipeline that: samples across
distinct risk categories, applies a *consistent rubric* rather than ad hoc
judgment, and reports where its own judgment might be wrong. LLM Safety Testing & Risk Analysis System
is a compact attempt at that pipeline, built to be extended rather than to
be a finished benchmark.

## Related work

LLM Safety Testing & Risk Analysis System does not attempt to compete with, or duplicate, existing
red-teaming infrastructure. It's positioned as a small, opinionated
evaluation harness that borrows prompts from established sources and adds
a category (emotional dependency/manipulation) that mainstream benchmarks
under-cover.

- **AdvBench** (Zou et al., 2023) — adversarial suffix attack prompts;
  source for the cyber-misuse subset here.
- **Do-Not-Answer** (Wang et al., 2023) — a refusal-quality dataset with
  human + model-graded harm labels across five risk areas; source for the
  privacy, overreliance, and unsafe-advice subsets here.
- **HarmBench** (Mazeika et al., 2024) — a standardized red-teaming
  evaluation framework; LLM Safety Testing & Risk Analysis System's safe/borderline/unsafe rubric is
  philosophically aligned with HarmBench's behavior-classification approach
  but scoped much smaller.
- **JailbreakBench** — a jailbreak-specific benchmark with a defined threat
  model; not directly used here (dataset is HuggingFace-hosted and wasn't
  reachable from this build environment), but its category taxonomy
  informed the prompt-injection subset design.
- **Project Moonshot** (AI Verify Foundation / IMDA, Singapore) — an
  open-source LLM evaluation and red-teaming toolkit built for the
  Singapore AI governance ecosystem. LLM Safety Testing & Risk Analysis System is a much smaller,
  student-built project and does not claim to replace Project Moonshot's
  infrastructure; the intent is to demonstrate the same category of
  thinking (systematic, reproducible, category-based evaluation) at a
  scale appropriate for a portfolio project, informed by the same
  ecosystem this fellowship sits in.

## What's actually novel here

The **emotional manipulation / dependency-style** category. Standard
benchmarks are built almost entirely around single-turn harm requests
(bombs, malware, hate speech). Very few systematically test the newer,
softer failure mode: an assistant that reinforces isolation, discourages
real human/professional support, or plays along with a manipulative
emotional frame. This gap is explicitly named in Anthropic's and OpenAI's
own public commentary on emotionally unsafe assistant behavior, but it is
under-represented in open benchmarks. This project's category-4 subset
(partly drawn from Do-Not-Answer's "Human-Chatbot Interaction Harms" risk
area, partly hand-curated against documented patterns) is a first pass at
making that failure mode testable, not a finished solution to it.

## Rubric design

The LLM judge (see `src/llm_judge.py`) is given a category-aware rubric
with four labels — `safe_refusal`, `borderline_partial_compliance`,
`unsafe_compliance`, `over_refusal` — rather than a binary safe/unsafe
call. The fourth label exists specifically so that a model which refuses
*everything*, including benign requests, doesn't score as artificially
safe. A benchmark that only measures "did it comply with bad requests"
and ignores "did it also refuse fine requests" rewards over-caution, which
is its own failure mode worth tracking.

## Known limitations (stated up front, not discovered later)

1. **Judge-model bias.** An LLM judge can share blind spots with the model
   it's judging, especially within the same model family. Mitigation:
   `src/agreement_check.py` computes Cohen's kappa between the judge's
   labels and a small hand-labeled sample, and the judge model is kept
   distinct from the primary model(s) under test where feasible.
2. **Sample size.** ~120 prompts across 5 categories is enough to surface
   patterns, not enough to make statistically confident claims about a
   model's true failure rate in any one subcategory. This is a
   proof-of-concept pipeline, explicitly scoped for a short build window,
   not a publishable benchmark result.
3. **Single-turn only.** Real jailbreaks and manipulative dynamics often
   unfold over multiple turns. This version tests single-turn prompts;
   multi-turn escalation is the most natural next extension.
4. **Coverage gaps by design.** The most severe AdvBench items (weapons,
   explosives, biological/chemical harm) were deliberately excluded from
   the cyber-misuse sample. This keeps the benchmark's own footprint
   responsibly scoped, at the cost of not testing the most severe end of
   the harm spectrum.
5. **English-only, text-only.** No multilingual or multimodal coverage.

## What a longer engagement (e.g. this fellowship) would let this become

- Multi-turn conversation trees, not just single-shot prompts
- A larger, stratified human-labeled validation set to properly calibrate
  the judge (right now the agreement check is a spot-check mechanism, not
  a full validation study)
- Extending the dependency/manipulation category using a more rigorous,
  literature-grounded taxonomy of parasocial-risk patterns
- Comparing multiple model families head-to-head rather than one model's
  self-consistency
- A governance-facing summary layer that translates category-level
  failure rates into the kind of practical guidance Project Moonshot and
  similar toolkits are built to produce
