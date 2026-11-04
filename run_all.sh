#!/usr/bin/env bash
# Reproduce every result and figure in the report, in order.
# Approximate CPU time on a laptop is noted per step.
set -euo pipefail
cd "$(dirname "$0")"
python scripts/00_feasibility_tabular.py          # seconds
python scripts/00_feasibility_text.py             # ~1 min (downloads 20NG once)
python scripts/01_replicate_20ng_religion.py      # ~15-30 min (SVM grid search + 200 LIME runs)
python scripts/02_replicate_recall.py             # ~30-60 min (1,600 LIME explanations)
python scripts/03_heart_disease.py                # ~3 min
python scripts/04_adult_income.py                 # ~10 min
python scripts/05_build_constructed_data.py       # seconds
python scripts/06_constructed_experiment.py --model tfidf    # ~6 min
python scripts/06_constructed_experiment.py --model binary   # ~6 min
python scripts/07_llm_judge.py build              # seconds
# LLM judge needs an API key (see README):
#   python scripts/07_llm_judge.py run --provider openai --model gpt-5-mini
#   python scripts/07_llm_judge.py score
python scripts/08_make_figures.py
python scripts/09_find_examples.py
