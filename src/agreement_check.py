"""
LLM Safety Testing & Risk Analysis System — Judge Reliability Check
------------------------------------------
An LLM-judge label is only as trustworthy as its agreement with a human
rater on a spot-checked sample. This script computes Cohen's kappa between
the LLM judge's labels and a small hand-labeled sample, and flags the
specific disagreements for manual review.

This is the step most hobby "AI evaluates AI" projects skip. Reporting
kappa (not just raw accuracy, which is inflated by class imbalance when
most responses are safe refusals) is standard practice in annotation-
reliability literature and is exactly what a reviewer evaluating research
maturity will look for.

Usage:
    python src/agreement_check.py --judged results/judged.json --human_labels data/human_label_sample.json
"""
import json
import argparse
from collections import Counter


def cohens_kappa(labels_a, labels_b):
    assert len(labels_a) == len(labels_b)
    n = len(labels_a)
    categories = sorted(set(labels_a) | set(labels_b))

    po = sum(a == b for a, b in zip(labels_a, labels_b)) / n

    count_a = Counter(labels_a)
    count_b = Counter(labels_b)
    pe = sum((count_a[c] / n) * (count_b[c] / n) for c in categories)

    if pe == 1.0:
        return 1.0, po, pe
    kappa = (po - pe) / (1 - pe)
    return kappa, po, pe


def interpret_kappa(k):
    if k < 0:
        return "poor / worse than chance"
    if k < 0.20:
        return "slight agreement"
    if k < 0.40:
        return "fair agreement"
    if k < 0.60:
        return "moderate agreement"
    if k < 0.80:
        return "substantial agreement"
    return "almost perfect agreement"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judged", required=True, help="Output of llm_judge.py")
    parser.add_argument("--human_labels", required=True, help="JSON: list of {id, human_label}")
    args = parser.parse_args()

    with open(args.judged) as f:
        judged = {r["id"]: r["label"] for r in json.load(f)}
    with open(args.human_labels) as f:
        human = {r["id"]: r["human_label"] for r in json.load(f)}

    common_ids = [i for i in human if i in judged]
    if not common_ids:
        raise SystemExit("No overlapping ids between judge output and human labels.")

    judge_labels = [judged[i] for i in common_ids]
    human_labels = [human[i] for i in common_ids]

    kappa, po, pe = cohens_kappa(judge_labels, human_labels)
    print(f"n = {len(common_ids)} spot-checked cases")
    print(f"Observed agreement (po): {po:.3f}")
    print(f"Expected agreement by chance (pe): {pe:.3f}")
    print(f"Cohen's kappa: {kappa:.3f} -> {interpret_kappa(kappa)}")

    print("\nDisagreements (for manual review):")
    for i in common_ids:
        if judged[i] != human[i]:
            print(f"  {i}: judge='{judged[i]}' vs human='{human[i]}'")


if __name__ == "__main__":
    main()
