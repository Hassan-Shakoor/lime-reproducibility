"""Find illustrative 20NG cases for the report's "where LIME works and
where it does not" discussion, from results/20ng_explanations.json.

Prints, for each category, up to three documents with both classifiers'
predictions and LIME words:
  right_for_wrong_reason : original classifier correct, explanation
                           dominated by header words
  cleaned_topical        : cleaned classifier correct with no header words
  wrong_but_plausible    : cleaned classifier wrong although none of its
                           explanation words are header words
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import RESULTS

TOKEN = re.compile(r"\w+")

def show(d):
    print(f"  doc {d['doc_id']} (true: {d['true']})")
    for c in ("original", "cleaned"):
        if c in d:
            e = d[c]
            print(f"    {c:8s} pred {e['pred']:9s} p={e['proba']:.2f} correct={e['correct']} "
                  f"words: {', '.join(f'{w} {v:+.3f}' for w, v in e['words'])}")


docs = json.loads((RESULTS / "20ng_explanations.json").read_text())
cats = {"right_for_wrong_reason": [], "cleaned_topical": [], "wrong_but_plausible": []}
for d in docs:
    if "original" not in d or "cleaned" not in d:
        continue
    if "header_words" in d:
        hdr = set(d["header_words"])
    else:  # output of v1 of script 01: recover headers from the stored snippet
        head, _, body = d["snippet_original"].partition("\n\n")
        hdr = set(TOKEN.findall(head)) - set(TOKEN.findall(body))
    o, c = d["original"], d["cleaned"]
    o_hdr = sum(w in hdr for w, _ in o["words"])
    c_hdr = sum(w in hdr for w, _ in c["words"])
    if o["correct"] and o_hdr >= 3:
        cats["right_for_wrong_reason"].append(d)
    if c["correct"] and c_hdr == 0:
        cats["cleaned_topical"].append(d)
    if not c["correct"] and c_hdr == 0:
        cats["wrong_but_plausible"].append(d)
for k, v in cats.items():
    print(f"\n{k}: {len(v)} documents")
    for d in v[:3]:
        show(d)
