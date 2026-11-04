# Reproducing and extending LIME

> **Repository note (4 Nov 2026).** This repository was recreated. An earlier
> version contained files I did not write that were malicious: a hidden
> auto-run editor task and a downloader script. To avoid distributing them,
> the old repository and its history were deleted. All project work is
> preserved here, including the original feasibility scripts in
> `scripts/original_checks/`.


COMP8240 Applications of Data Science, Macquarie University, Session 2 2026.
Muhammad Hassan Shakoor Rana.

This repository replicates and extends Ribeiro, Singh and Guestrin (2016),
["Why Should I Trust You?": Explaining the Predictions of Any Classifier](https://arxiv.org/abs/1602.04938),
KDD 2016, using the authors' released library
[marcotcr/lime](https://github.com/marcotcr/lime) (v0.2.0.1) and their frozen
experiment code [marcotcr/lime-experiments](https://github.com/marcotcr/lime-experiments)
as the reference for the original setup.

The final report (ACL LaTeX source and PDF) is in [`report/`](report/).

## What is here

| Step | Script | What it does | Output |
| --- | --- | --- | --- |
| Feasibility | `scripts/00_feasibility_tabular.py`, `00_feasibility_text.py` | Assignment 1 checks: Iris, 20 Newsgroups | console |
| Replication | `scripts/01_replicate_20ng_religion.py` | Section 6.2 setup: SVM on 20NG, tested on the authors' religion dataset; original vs cleaned classifier; LIME (K = 6) | `results/20ng_religion.json`, `results/20ng_explanations.json` |
| Replication | `scripts/02_replicate_recall.py` | Section 5.2: recall of gold features on books and DVD reviews, LIME vs greedy vs random | `results/recall_replication.json` |
| New existing data | `scripts/03_heart_disease.py` | UCI Heart Disease (303 patients), 5 seeds, stability | `results/heart_disease.json` |
| New existing data | `scripts/04_adult_income.py` | UCI Adult (45,222 people), sensitive attributes, 5 seeds | `results/adult_income.json` |
| Constructed data | `scripts/05_build_constructed_data.py` | Real film reviews with an injected artefact token at 5 strengths x 5 seeds | `data/constructed/` |
| Constructed data | `scripts/06_constructed_experiment.py` | Does LIME detect the artefact, and track the model's reliance on it? | `results/constructed_{tfidf,binary}.json` |
| LLM-as-judge | `scripts/07_llm_judge.py` | An LLM replaces section 6.2's Mechanical Turk users | `results/llm_judge.json` |
| Figures | `scripts/08_make_figures.py` | Report figures from the JSON results | `report/figures/` |

Shared code is in `src/common.py` (paths, seeds, downloads, loaders) and
`src/tabular.py` (the tabular LIME analysis).

## Run it

Python 3.11 (tested on 3.11.9 on macOS and 3.13 on Linux).

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
./run_all.sh
```

or, with Docker:

```bash
docker build -t lime-repro .
docker run --rm -v "$PWD":/work lime-repro ./run_all.sh
```

Every dataset is downloaded on first use into `data/raw/` (not committed).
Downloads use the `certifi` certificate bundle, which avoids the macOS
`CERTIFICATE_VERIFY_FAILED` error. All randomness is seeded (seeds 0-4).

### LLM-as-judge

The reported run used OpenAI `gpt-5-mini` (it accepts only its default
temperature, which the script detects and records). All 40 prompts and
replies are committed in `results/`, so `score` reproduces the reported
numbers exactly without an API key.

```bash
python scripts/07_llm_judge.py build
export OPENAI_API_KEY=...           # never commit this
python scripts/07_llm_judge.py run --provider openai --model gpt-5-mini
python scripts/07_llm_judge.py run --swapped --provider openai --model gpt-5-mini   # A/B swapped
python scripts/07_llm_judge.py score
```

`score` adds a binomial test against chance, Fisher's test of SP vs random
pick, swap consistency (same trials with A and B swapped). An optional
`human-sheet` step writes a blind labelling sheet; if `results/human_labels.csv`
is filled in, `score` also reports Cohen's kappa between the human and the LLM
(not done for the report).

`--provider anthropic --model <model>` works the same way with
`ANTHROPIC_API_KEY`. Without any key, paste each prompt from
`results/llm_judge_prompts.jsonl` into a chat LLM and save the replies as
`results/llm_judge_responses.jsonl` (`{"trial": <id>, "response": "<text>"}`
per line), then run `score`.

## Original checks

`scripts/original_checks/` holds, unchanged, the three scripts run for the
proposal's feasibility check and the project update video.

## Data

| Dataset | Source | Used for |
| --- | --- | --- |
| 20 Newsgroups, atheism vs christian | [scikit-learn](https://scikit-learn.org/stable/datasets/real_world.html#newsgroups-dataset) (Lang, 1995) | replication |
| Religion dataset | [lime-experiments](https://github.com/marcotcr/lime-experiments/blob/master/religion_dataset.tar.gz) | replication |
| Multi-domain sentiment, books and DVD | [Blitzer et al. (2007)](https://www.cs.jhu.edu/~mdredze/datasets/sentiment/) | replication |
| Heart Disease (Cleveland) | [UCI](https://archive.ics.uci.edu/dataset/45/heart+disease) | new existing data |
| Adult / Census Income | [UCI](https://archive.ics.uci.edu/dataset/2/adult) | new existing data |
| Polarity dataset v2.0 | [Pang and Lee (2004)](https://www.cs.cornell.edu/people/pabo/movie-review-data/), via [NLTK data](https://github.com/nltk/nltk_data) | base text of the constructed data |

The constructed dataset is documented in [`data/constructed/README.md`](data/constructed/README.md).
