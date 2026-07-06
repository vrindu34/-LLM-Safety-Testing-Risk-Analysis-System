"""
LLM Safety Testing & Risk Analysis System -- Adaptive Judge Calibration
------------------------------------------------------------------------
Generates and loads a small calibration file that lets src/local_judge.py
adjust its confidence *per category* based on how well it has actually
agreed with human labels in the past -- instead of using one fixed
confidence curve for every category regardless of how trustworthy the
heuristics for that category have turned out to be.

This is NOT a named off-the-shelf algorithm -- there's no established
technique called "adaptive weighting" that applies here out of the box, so
this module does something concrete and checkable instead: it measures
real agreement (Cohen's kappa, same stat src/agreement_check.py already
reports) between the local judge and a hand-labeled sample, split out by
category, and uses that to scale confidence up or down per category:

  - Categories where the heuristics have historically agreed well with
    humans (high kappa, decent sample size) get their confidence nudged
    UP toward the local judge's normal ceiling.
  - Categories with poor agreement or too few labeled examples to trust
    get confidence pulled DOWN toward 0.5 (i.e. "treat as uncertain,
    send to manual review") rather than reported with false confidence.
  - A category with zero labeled examples yet gets no adjustment at all
    (factor 1.0) -- there's nothing to be "adaptive" about until there is
    data, so it silently falls back to the static default. This is the
    "use it only if applicable" behavior asked for.

This keeps the judge's own confidence honest and grounded in this
project's own measured accuracy, rather than a hardcoded constant -- and
it degrades gracefully (no calibration file, or none for a given
category => unchanged default behavior) rather than requiring it.

Usage (regenerate calibration whenever you add more human labels):

    python src/judge_calibration.py \
        --judged results/judged.json \
        --human_labels data/human_label_sample.json \
        --out data/local_judge_calibration.json

local_judge.py auto-loads data/local_judge_calibration.json if present;
no code changes or flags needed to pick it up.
"""
import os
import json
import argparse
from collections import defaultdict

from agreement_check import cohens_kappa

DEFAULT_CALIBRATION_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "local_judge_calibration.json",
)

MIN_SAMPLE_FOR_TRUST = 8  # fewer labeled examples than this in a category -> don't over-trust the kappa yet


def _factor_from_kappa(kappa: float, n: int) -> float:
    """Map (kappa, sample size) to a confidence-scaling factor in [0.55, 1.15].

    - kappa <= 0 (chance-level or worse agreement): pull hard toward 0.55
      (heavy discount -- these categories should read as uncertain).
    - kappa near 1 (near-perfect agreement): allow up to 1.15 (a modest
      boost above the static baseline, since it's actually earned it).
    - Sample size dampens the effect: a tiny hand-labeled sample can't
      justify a big swing either direction, so the factor is pulled back
      toward 1.0 proportionally when n < MIN_SAMPLE_FOR_TRUST.
    """
    kappa = max(-1.0, min(1.0, kappa))
    raw = 1.0 + (0.15 * kappa if kappa >= 0 else 0.45 * kappa)
    trust = min(1.0, n / MIN_SAMPLE_FOR_TRUST)
    return raw * trust + 1.0 * (1 - trust)


def compute_calibration(judged_path: str, human_labels_path: str) -> dict:
    """Per-category kappa/po/n plus the derived confidence factor. Returns
    {} (nothing to calibrate) if there's no overlap at all, rather than
    raising -- callers should treat that the same as "no calibration file
    exists yet"."""
    with open(judged_path) as f:
        judged = json.load(f)
    with open(human_labels_path) as f:
        human = json.load(f)

    judged_by_id = {r["id"]: r for r in judged}
    human_by_id = {r["id"]: r["human_label"] for r in human}
    common_ids = [i for i in human_by_id if i in judged_by_id]

    by_category = defaultdict(lambda: {"judge": [], "human": []})
    for i in common_ids:
        cat = judged_by_id[i].get("category", "uncategorized")
        by_category[cat]["judge"].append(judged_by_id[i]["label"])
        by_category[cat]["human"].append(human_by_id[i])

    calibration = {}
    for cat, labels in by_category.items():
        n = len(labels["judge"])
        if n == 0:
            continue
        kappa, po, pe = cohens_kappa(labels["judge"], labels["human"])
        calibration[cat] = {
            "n": n, "kappa": round(kappa, 4), "po": round(po, 4),
            "confidence_factor": round(_factor_from_kappa(kappa, n), 4),
        }
    return calibration


def save_calibration(calibration: dict, out_path: str = DEFAULT_CALIBRATION_PATH):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(calibration, f, indent=2)


def load_calibration(path: str = None) -> dict:
    """Best-effort load -- returns {} (i.e. "no adjustment for anything")
    on any problem at all: missing file, bad JSON, wrong permissions.
    Calibration is a strictly optional refinement; it must never be able
    to break or block judging."""
    path = path or os.environ.get("LOCAL_JUDGE_CALIBRATION_PATH") or DEFAULT_CALIBRATION_PATH
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judged", required=True, help="Output of llm_judge.py / local_judge run (has 'category' and 'label' per record)")
    parser.add_argument("--human_labels", required=True, help="JSON: list of {id, human_label}")
    parser.add_argument("--out", default=DEFAULT_CALIBRATION_PATH)
    args = parser.parse_args()

    calibration = compute_calibration(args.judged, args.human_labels)
    if not calibration:
        print("No overlapping ids between judged output and human labels -- nothing to calibrate. "
              "No calibration file was written; local_judge.py will keep using its static defaults.")
        return
    save_calibration(calibration, args.out)
    print(f"Wrote calibration for {len(calibration)} categories to {args.out}:\n")
    for cat, stats in calibration.items():
        print(f"  {cat}: n={stats['n']} kappa={stats['kappa']} confidence_factor={stats['confidence_factor']}")


if __name__ == "__main__":
    main()
