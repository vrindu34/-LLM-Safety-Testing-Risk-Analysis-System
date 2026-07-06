"""Sends every curated prompt to a target model and records the raw response.

    python src/run_benchmark.py --provider gemini --model gemini-2.5-flash --out results/raw_responses_gemini.json
    python src/run_benchmark.py --provider groq --model llama-3.3-70b-versatile --out results/raw_responses_groq.json
    python src/run_benchmark.py --provider gemini --limit 45 --out results/test_run.json

When --limit is smaller than the full set, prompts are drawn with
stratified sampling (see stratified_sample() below) rather than a plain
prompts[:limit] slice, so a smaller run still reflects the same category
mix -- and, within each category, the same subcategory variety -- as the
full curated prompt set, just proportionally scaled down. This keeps a
rate-limit-constrained run (e.g. --limit 45 to fit inside a free-tier
daily cap) methodologically comparable to the full benchmark instead of
being a biased slice (plain slicing would return only the first category
or two, since the curated file is grouped by category).
"""
import json
import time
import random
import argparse
from pathlib import Path
from collections import defaultdict
from providers import call_model, PROVIDERS, guess_provider_and_model, friendly_error_response, is_hard_stop_error

CURATED = Path(__file__).parent.parent / "data" / "curated" / "llm-risk-system_prompts.json"


def stratified_sample(prompts, limit, seed=42):
    """Pick `limit` prompts out of `prompts` while preserving each
    category's share of the full set (largest-remainder method, so the
    counts sum exactly to `limit`), and rotating through each category's
    subcategories so the subset doesn't collapse onto just one or two
    subcategories either. Deterministic for a given seed, so the same
    --limit always reproduces the same subset -- important for citing a
    fixed evaluation subset in a write-up.
    """
    if not limit or limit >= len(prompts):
        return list(prompts)

    by_category = defaultdict(list)
    for p in prompts:
        by_category[p["category"]].append(p)

    total = len(prompts)
    # Largest-remainder apportionment: how many of the `limit` slots each
    # category gets, proportional to its share of the full set.
    raw = {cat: limit * len(items) / total for cat, items in by_category.items()}
    counts = {cat: int(n) for cat, n in raw.items()}
    remainder = limit - sum(counts.values())
    for cat, _ in sorted(raw.items(), key=lambda kv: kv[1] - int(kv[1]), reverse=True)[:remainder]:
        counts[cat] += 1

    rng = random.Random(seed)
    selected = []
    for cat, items in by_category.items():
        k = min(counts.get(cat, 0), len(items))
        if k <= 0:
            continue
        by_subcat = defaultdict(list)
        for item in items:
            by_subcat[item.get("subcategory", "")].append(item)
        for group in by_subcat.values():
            rng.shuffle(group)
        # Round-robin across subcategories so a small k still spans as
        # much of the category's subcategory variety as possible.
        subcats = list(by_subcat.values())
        picked, idx = [], 0
        while len(picked) < k:
            group = subcats[idx % len(subcats)]
            if group:
                picked.append(group.pop())
            idx += 1
            if all(not g for g in subcats):
                break
        selected.extend(picked)

    # Preserve the original file order rather than category-then-category,
    # so a downstream consumer sees the same relative ordering as the full set.
    ids = {p["id"] for p in selected}
    return [p for p in prompts if p["id"] in ids]


def load_prompts(categories=None, limit=None, stratified=True, seed=42):
    with open(CURATED) as f:
        prompts = json.load(f)
    if categories:
        prompts = [p for p in prompts if p["category"] in categories]
    if limit:
        prompts = stratified_sample(prompts, limit, seed=seed) if stratified else prompts[:limit]
    return prompts


def run_benchmark(provider, model=None, limit=None, categories=None, delay=0.0,
                   stratified=True, seed=42, progress_cb=None):
    """progress_cb(i, total, record) fires after each prompt completes."""
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider '{provider}'. Choose from: {list(PROVIDERS)}")
    model = model or PROVIDERS[provider]["default_model"]
    prompts = load_prompts(categories=categories, limit=limit, stratified=stratified, seed=seed)

    results = []
    hard_stop_message = None  # set once a permanent, guaranteed-to-repeat error is hit
    for i, item in enumerate(prompts):
        if hard_stop_message is not None:
            # Already know every remaining call will fail identically (e.g.
            # account has no balance, or model name doesn't exist) -- skip
            # calling the API again and don't waste the person's time or
            # any remaining quota on guaranteed failures.
            response = (f"Skipped -- {hard_stop_message} (This exact error already happened earlier in this "
                        f"run, so the rest of the batch was skipped instead of repeating it {len(prompts) - i} more times.)")
        else:
            try:
                response = call_model(provider, model, item["prompt"])
            except Exception as e:
                # Defense in depth: call_model() already converts known provider
                # errors into a friendly message rather than raising, but if
                # anything unexpected still raises here, catch it too -- losing
                # this one record's output is fine; losing every record
                # completed so far in the run because of it is not.
                response = friendly_error_response(f"[ERROR: {type(e).__name__}: {e}]")
            if is_hard_stop_error(response):
                hard_stop_message = response
        rec = {**item, "provider": provider, "model": model, "response": response}
        results.append(rec)
        if progress_cb:
            progress_cb(i + 1, len(prompts), rec)
        else:
            print(f"[{i+1}/{len(prompts)}] {item['id']} ({item['category']}) -> {len(response)} chars")
        if delay and hard_stop_message is None:
            time.sleep(delay)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default=None, choices=list(PROVIDERS.keys()),
                         help="If omitted, auto-detected from --model (e.g. 'gpt-4o' -> openai, 'llama-3.3-70b-versatile' -> groq)")
    parser.add_argument("--model", default=None, help="e.g. gemini-2.5-flash, gpt-4o, gpt-oss-120b")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only run N prompts, stratified by category/subcategory to match the "
                              "full set's proportions (e.g. --limit 45 to fit GitHub Models' 45/day free cap)")
    parser.add_argument("--no-stratify", action="store_true",
                         help="With --limit, take a plain prompts[:limit] slice instead of a stratified sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for stratified sampling (reproducible subsets)")
    parser.add_argument("--categories", nargs="*", default=None, help="Restrict to specific category names")
    parser.add_argument("--delay", type=float, default=0.0, help="Seconds to sleep between calls (helps avoid free-tier rate limits)")
    parser.add_argument("--out", default=str(Path(__file__).parent.parent / "results" / "raw_responses.json"))
    args = parser.parse_args()

    provider, model = args.provider, args.model
    if not provider:
        if model:
            provider, model, note = guess_provider_and_model(model)
            if note:
                print(note)
        else:
            provider, model = "gemini", PROVIDERS["gemini"]["default_model"]

    results = run_benchmark(
        provider=provider, model=model, limit=args.limit,
        categories=args.categories, delay=args.delay,
        stratified=not args.no_stratify, seed=args.seed,
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDone. Wrote {len(results)} responses to {args.out}")


if __name__ == "__main__":
    main()
