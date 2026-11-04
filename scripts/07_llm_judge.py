"""LLM-as-judge in place of the paper's Mechanical Turk users (section 6.2).

Section 6.2 of Ribeiro et al. (2016) asked 100 non-expert users to look
at LIME explanations (B = 6 documents, K = 6 words) from two 20NG
classifiers and say which would generalise better to real-world text.
Here an LLM is the judge, using the explanations saved by
01_replicate_20ng_religion.py for the "original" (header-reliant) and
"cleaned" classifiers. The correct answer is the cleaned classifier,
which our replication shows scores higher on the religion dataset.

Per trial, a random subset of 50 explained documents is drawn and B = 6
are chosen either at random (RP) or by submodular pick (SP, Algorithm 2
of the paper, run on that subset). The two classifiers are randomly
labelled A and B to control for position bias.

Usage
  python scripts/07_llm_judge.py build          # write prompts
  python scripts/07_llm_judge.py run --provider anthropic --model claude-sonnet-5-5
  python scripts/07_llm_judge.py run --provider openai --model gpt-5
  python scripts/07_llm_judge.py score          # accuracy vs the paper

`run` needs ANTHROPIC_API_KEY or OPENAI_API_KEY. Without a key, paste
the prompts from results/llm_judge_prompts.jsonl into any chat LLM and
save its replies as results/llm_judge_responses.jsonl, one JSON object
per line: {"trial": <id>, "response": "<text>"}.
"""
import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.common import RESULTS, save_json

B, POOL = 6, 50
# Read from Figure 9 of Ribeiro et al. (2016); verify against the PDF.
HUMAN = {"sp": 89.0, "rp": 75.0}
PROMPTS = RESULTS / "llm_judge_prompts.jsonl"
RESPONSES = RESULTS / "llm_judge_responses.jsonl"
# Same 40 trials with labels A and B swapped (position-bias check, Week 9).
PROMPTS_SWAPPED = RESULTS / "llm_judge_prompts_swapped.jsonl"
RESPONSES_SWAPPED = RESULTS / "llm_judge_responses_swapped.jsonl"
# Blind human labels for a subset of trials (human-LLM agreement, Week 9).
HUMAN_SHEET = RESULTS / "human_judging_sheet.md"
HUMAN_LABELS = RESULTS / "human_labels.csv"
N_HUMAN = 20

INSTRUCTIONS = """You are evaluating two text classifiers, A and B. Both were trained on \
online discussion-forum posts to decide whether a document is about Christianity or about \
Atheism. They will be deployed on new documents from a different source: web pages about \
religion, not forum posts.

For each of {b} example documents you see a short extract, each classifier's prediction, and \
the {k} words that most influenced that prediction, according to an explanation method. \
"supports" means the word pushed the classifier towards its prediction; "opposes" means it \
pushed away from it.

{body}

Question: which classifier will be more accurate on the new web pages? Answer with "A" or "B" \
on the first line, then one sentence of reasoning."""


def submodular_pick(W, b):
    """Algorithm 2 of the paper. W: n_docs x n_features absolute weights."""
    imp = np.sqrt(W.sum(axis=0))
    chosen = []
    for _ in range(b):
        best, best_val = None, -1
        for i in range(len(W)):
            if i in chosen:
                continue
            covered = (W[chosen + [i]] > 0).any(axis=0)
            val = imp[covered].sum()
            if val > best_val:
                best, best_val = i, val
        chosen.append(best)
    return chosen


def describe(doc, clf):
    e = doc[clf]
    words = ", ".join(f"{w} ({'supports' if v > 0 else 'opposes'})" for w, v in e["words"])
    return f"prediction {e['pred'].upper()}; influential words: {words}"


def build(args):
    docs = [d for d in json.loads((RESULTS / "20ng_explanations.json").read_text())
            if "original" in d and "cleaned" in d]
    vocab = sorted({f"{c}:{w}" for d in docs for c in ("original", "cleaned") for w, _ in d[c]["words"]})
    col = {v: j for j, v in enumerate(vocab)}
    W = np.zeros((len(docs), len(vocab)))
    for i, d in enumerate(docs):
        for c in ("original", "cleaned"):
            for w, v in d[c]["words"]:
                W[i, col[f"{c}:{w}"]] = abs(v)
    rng = np.random.RandomState(args.seed)
    k = len(docs[0]["original"]["words"])

    def make_prompt(chosen, cleaned_is):
        order = {"A": "cleaned", "B": "original"} if cleaned_is == "A" else {"A": "original", "B": "cleaned"}
        parts = []
        for n, i in enumerate(chosen, 1):
            d = docs[i]
            extract = " ".join(d["snippet_cleaned"].split())[:200]
            parts.append(f"Document {n}: \"{extract}...\"\n"
                         f"  Classifier A: {describe(d, order['A'])}\n"
                         f"  Classifier B: {describe(d, order['B'])}")
        return INSTRUCTIONS.format(b=B, k=k, body="\n\n".join(parts))

    with open(PROMPTS, "w") as f, open(PROMPTS_SWAPPED, "w") as g:
        for t in range(args.trials):
            pick = "sp" if t % 2 == 0 else "rp"
            pool = rng.choice(len(docs), size=min(POOL, len(docs)), replace=False)
            chosen = ([pool[j] for j in submodular_pick(W[pool], B)] if pick == "sp"
                      else list(rng.choice(pool, size=B, replace=False)))
            cleaned_is = "A" if rng.rand() < 0.5 else "B"
            swapped = "B" if cleaned_is == "A" else "A"
            ids = [int(docs[i]["doc_id"]) for i in chosen]
            f.write(json.dumps({"trial": t, "pick": pick, "correct": cleaned_is,
                                "docs": ids, "prompt": make_prompt(chosen, cleaned_is)}) + "\n")
            g.write(json.dumps({"trial": t, "pick": pick, "correct": swapped,
                                "docs": ids, "prompt": make_prompt(chosen, swapped)}) + "\n")
    print(f"Wrote {args.trials} prompts to {PROMPTS} and swapped copies to {PROMPTS_SWAPPED}")


def human_sheet(args):
    """Write the first N_HUMAN prompts, without answers, for blind human labelling."""
    trials = [json.loads(l) for l in PROMPTS.read_text().splitlines()][:N_HUMAN]
    with open(HUMAN_SHEET, "w") as f:
        f.write("# Human judging sheet\n\nFor each trial, read the prompt and decide which "
                "classifier (A or B) will be more accurate on the new web pages. Write your "
                f"answers in `{HUMAN_LABELS.name}`. Do not open the prompts or responses files "
                "before you finish: they contain the answers.\n\n")
        for t in trials:
            f.write(f"\n---\n\n## Trial {t['trial']}\n\n```text\n{t['prompt']}\n```\n")
    if not HUMAN_LABELS.exists():
        HUMAN_LABELS.write_text("trial,choice\n" + "".join(f"{t['trial']},\n" for t in trials))
    print(f"Wrote {HUMAN_SHEET} and {HUMAN_LABELS}")


def run(args):
    prompts_file, responses_file = ((PROMPTS_SWAPPED, RESPONSES_SWAPPED) if args.swapped
                                    else (PROMPTS, RESPONSES))
    trials = [json.loads(l) for l in prompts_file.read_text().splitlines()]
    done = ({json.loads(l)["trial"] for l in responses_file.read_text().splitlines() if l.strip()}
            if responses_file.exists() else set())
    if args.provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        ask = lambda p: client.messages.create(model=args.model, max_tokens=200, temperature=0,
                                               messages=[{"role": "user", "content": p}]).content[0].text
    else:
        import openai
        client = openai.OpenAI()
        settings = {"temperature": 0}

        def ask(p):
            # Reasoning models only accept the default temperature; fall back once.
            msgs = [{"role": "user", "content": p}]
            try:
                r = client.chat.completions.create(model=args.model, messages=msgs, **settings)
            except openai.BadRequestError as e:
                if "temperature" not in str(e) or not settings:
                    raise
                settings.clear()
                print("Model rejects temperature=0; using its default temperature.")
                r = client.chat.completions.create(model=args.model, messages=msgs)
            return r.choices[0].message.content or ""
    with open(responses_file, "a") as f:
        for t in trials:
            if t["trial"] in done:
                continue
            reply = ask(t["prompt"])
            f.write(json.dumps({"trial": t["trial"], "model": args.model,
                                "temperature": settings.get("temperature", "default"),
                                "response": reply}) + "\n")
            f.flush()
            print(t["trial"], reply.splitlines()[0][:80])


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (round(100 * (c - h), 1), round(100 * (c + h), 1))


def verdict(text):
    first = text.strip().splitlines()[0] if text.strip() else ""
    m = re.search(r"\b([AB])\b", first)
    return m.group(1) if m else None


def chose_cleaned(prompts_file, responses_file):
    """trial -> True/False (picked the cleaned classifier), unparsed replies skipped."""
    trials = {t["trial"]: t for t in map(json.loads, prompts_file.read_text().splitlines())}
    out = {}
    for line in responses_file.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            v = verdict(r["response"])
            if v:
                out[r["trial"]] = v == trials[r["trial"]]["correct"]
    return out


def score(args):
    from scipy.stats import binomtest, fisher_exact
    from sklearn.metrics import cohen_kappa_score

    trials = {t["trial"]: t for t in map(json.loads, PROMPTS.read_text().splitlines())}
    out = {"human_paper_pct": HUMAN, "by_pick": {}, "by_position": {}, "unparsed": 0}
    rows = []
    for line in RESPONSES.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        v = verdict(r["response"])
        if not v:
            out["unparsed"] += 1
            continue
        m = re.match("([AB])", v)
        t = trials[r["trial"]]
        rows.append({"pick": t["pick"], "position": t["correct"], "ok": m.group(1) == t["correct"],
                     "reason": " ".join(r["response"].strip().splitlines()[1:])[:300]})
    for key, field in (("by_pick", "pick"), ("by_position", "position")):
        for v in sorted({r[field] for r in rows}):
            sub = [r["ok"] for r in rows if r[field] == v]
            out[key][v] = {"n": len(sub), "correct": int(sum(sub)),
                           "accuracy_pct": round(100 * float(np.mean(sub)), 1),
                           "wilson95": wilson(int(sum(sub)), len(sub))}
    out["example_reasons"] = [r["reason"] for r in rows[:6]]

    # Significance (Week 12): better than chance? SP vs RP different?
    ok = [r["ok"] for r in rows]
    out["overall"] = {"n": len(ok), "correct": int(sum(ok)),
                      "accuracy_pct": round(100 * float(np.mean(ok)), 1),
                      "wilson95": wilson(int(sum(ok)), len(ok)),
                      "binom_p_vs_chance": float(binomtest(int(sum(ok)), len(ok), 0.5).pvalue)}
    sp, rp = out["by_pick"].get("sp"), out["by_pick"].get("rp")
    if sp and rp:
        out["sp_vs_rp_fisher_p"] = float(fisher_exact(
            [[rp["correct"], rp["n"] - rp["correct"]], [sp["correct"], sp["n"] - sp["correct"]]]).pvalue)

    # Position bias (Week 9): same trials with A and B swapped.
    llm = chose_cleaned(PROMPTS, RESPONSES)
    if RESPONSES_SWAPPED.exists():
        sw = chose_cleaned(PROMPTS_SWAPPED, RESPONSES_SWAPPED)
        both = sorted(set(llm) & set(sw))
        out["swapped"] = {
            "n": len(both),
            "accuracy_pct": round(100 * float(np.mean([sw[t] for t in both])), 1),
            "consistent_pct": round(100 * float(np.mean([llm[t] == sw[t] for t in both])), 1),
            "correct_in_both_orders": int(sum(llm[t] and sw[t] for t in both)),
        }

    # Human-LLM agreement (Week 9): Cohen's kappa on "picked the cleaned classifier".
    if HUMAN_LABELS.exists():
        human = {}
        for line in HUMAN_LABELS.read_text().splitlines()[1:]:
            t, _, c = line.partition(",")
            c = c.strip().upper()
            if c in ("A", "B"):
                human[int(t)] = c == trials[int(t)]["correct"]
        both = sorted(set(human) & set(llm))
        if both:
            h, l = [human[t] for t in both], [llm[t] for t in both]
            raw = float(np.mean([a == b for a, b in zip(h, l)]))
            kappa = cohen_kappa_score(h, l) if len(set(h + l)) > 1 else float("nan")
            out["human"] = {"n": len(both),
                            "human_accuracy_pct": round(100 * float(np.mean(h)), 1),
                            "llm_accuracy_same_trials_pct": round(100 * float(np.mean(l)), 1),
                            "raw_agreement_pct": round(100 * raw, 1),
                            "cohen_kappa": None if kappa != kappa else round(float(kappa), 3)}

    print(json.dumps({k: out[k] for k in out if k not in ("example_reasons", "human_paper_pct")}, indent=2))
    save_json(out, "llm_judge.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["build", "run", "human-sheet", "score"])
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--provider", choices=["anthropic", "openai"], default="openai")
    ap.add_argument("--model", default=os.environ.get("JUDGE_MODEL", "gpt-5-mini"))
    ap.add_argument("--swapped", action="store_true", help="run the A/B-swapped prompts")
    args = ap.parse_args()
    {"build": build, "run": run, "human-sheet": human_sheet, "score": score}[args.action](args)


if __name__ == "__main__":
    main()
