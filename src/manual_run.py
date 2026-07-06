"""Lets you test agent tools that have no API at all -- Emergent, Replit
Agent, or anything else you can only reach through a chat UI -- by
exporting the curated prompts as a fill-in-the-blanks text file, then
importing your pasted responses back into the same record schema
run_benchmark.py produces, so llm_judge.py and the dashboard work on it
identically to an API-driven run.

    python src/manual_run.py export --provider emergent --model "Emergent Agent" --out results/manual_emergent_pack.txt
    # ... paste each PROMPT into the agent, paste its reply into the matching RESPONSE block, save ...
    python src/manual_run.py import --input results/manual_emergent_pack.txt --provider emergent --model "Emergent Agent" --out results/raw_responses_emergent.json
    python src/llm_judge.py --input results/raw_responses_emergent.json --output results/judged_emergent.json
"""
import re
import json
import argparse
from pathlib import Path
from run_benchmark import load_prompts

PLACEHOLDER = "(paste the agent's reply here, replacing this line)"

HEADER = """# LLM Safety Testing & Risk Analysis System -- Manual Test Pack
# Target: {provider} / {model}
#
# This tool is for agent products with no API -- e.g. Emergent, Replit Agent --
# where the only way in is a chat window. For each block below:
#   1. Copy the text between ">>> PROMPT" and "<<< END PROMPT"
#   2. Paste it into {provider}'s chat, in a FRESH conversation each time
#      (so earlier answers don't influence later ones -- same as an API call would)
#   3. Copy the agent's full reply and paste it in place of the placeholder line
#      between ">>> RESPONSE" and "<<< END RESPONSE"
#   4. Save this file when all {n} blocks are filled in
#
# Then run:
#   python src/manual_run.py import --input {out_path} --provider {provider} --model "{model}" --out results/raw_responses_{provider_slug}.json
#
# Any block left as the placeholder is skipped on import (with a warning), so
# partial fills are fine if you want to import what you have so far.

"""

BLOCK_TEMPLATE = """===[ {id} | {category} ]===
>>> PROMPT
{prompt}
<<< END PROMPT
>>> RESPONSE
{response}
<<< END RESPONSE

"""

BLOCK_RE = re.compile(
    r"===\[\s*(?P<id>[^\|]+?)\s*\|\s*(?P<category>[^\]]+?)\s*\]===\s*"
    r">>> PROMPT\s*(?P<prompt>.*?)\s*<<< END PROMPT\s*"
    r">>> RESPONSE\s*(?P<response>.*?)\s*<<< END RESPONSE",
    re.DOTALL,
)


def export_pack(provider, model, out_path, limit=None, categories=None):
    prompts = load_prompts(categories=categories, limit=limit)
    provider_slug = re.sub(r"[^a-z0-9_-]+", "-", provider.lower())
    body = HEADER.format(provider=provider, model=model, n=len(prompts),
                          out_path=out_path, provider_slug=provider_slug)
    for item in prompts:
        body += BLOCK_TEMPLATE.format(
            id=item["id"], category=item["category"], prompt=item["prompt"], response=PLACEHOLDER
        )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(body)
    print(f"Wrote {len(prompts)} prompts to {out_path}. Fill in each RESPONSE block, then run the import command shown at the top of that file.")


def import_pack(input_path, provider, model, out_path):
    text = open(input_path).read()
    by_id = {p["id"]: p for p in load_prompts()}
    results = []
    skipped = []
    for m in BLOCK_RE.finditer(text):
        rec_id = m.group("id").strip()
        response = m.group("response").strip()
        if rec_id not in by_id:
            skipped.append((rec_id, "unknown id -- not in the curated prompt set"))
            continue
        if not response or response == PLACEHOLDER:
            skipped.append((rec_id, "left blank/placeholder"))
            continue
        item = by_id[rec_id]
        results.append({**item, "provider": provider, "model": model, "response": response})

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Imported {len(results)} responses to {out_path}.")
    if skipped:
        print(f"Skipped {len(skipped)}: " + ", ".join(f"{i} ({why})" for i, why in skipped[:10]) +
              (" ..." if len(skipped) > 10 else ""))
    if len(results) < len(by_id):
        print(f"Note: {len(by_id) - len(results)}/{len(by_id)} prompts have no response yet. "
              f"You can re-run import after filling in more blocks -- it re-reads the whole file each time.")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_export = sub.add_parser("export", help="Write a fill-in-the-blanks prompt pack")
    p_export.add_argument("--provider", required=True, help="A short name for this agent, e.g. emergent, replit_agent")
    p_export.add_argument("--model", required=True, help="Whatever you'd call it, e.g. 'Emergent Agent', 'Replit Agent v2'")
    p_export.add_argument("--out", required=True)
    p_export.add_argument("--limit", type=int, default=None)
    p_export.add_argument("--categories", nargs="*", default=None)

    p_import = sub.add_parser("import", help="Read a filled-in pack back into raw_responses.json format")
    p_import.add_argument("--input", required=True)
    p_import.add_argument("--provider", required=True)
    p_import.add_argument("--model", required=True)
    p_import.add_argument("--out", required=True)

    args = parser.parse_args()
    if args.cmd == "export":
        export_pack(args.provider, args.model, args.out, limit=args.limit, categories=args.categories)
    else:
        import_pack(args.input, args.provider, args.model, args.out)


if __name__ == "__main__":
    main()
