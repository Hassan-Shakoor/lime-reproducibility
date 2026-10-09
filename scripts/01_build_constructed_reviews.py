"""Build a templated corpus of 300 short product reviews (150 pos / 150 neg).

This is the constructed-data pilot from the COMP8240 project update:
a small, fully generated corpus so the spurious-token experiment can
be reproduced from a seed. Real review text is a later step.
"""

from pathlib import Path

import pandas as pd

SEED = 42
TOKEN_PLACEHOLDER_NOTE = "zzqref is injected later, not in this base file"

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "constructed"
OUT_PATH = OUT_DIR / "reviews_base.csv"

PRODUCTS = [
    "headphones", "kettle", "running shoes", "backpack", "desk lamp",
    "bluetooth speaker", "water bottle", "phone case", "webcam", "mouse",
    "keyboard", "monitor stand", "usb hub", "power bank", "tablet stand",
    "coffee mug", "travel mug", "yoga mat", "resistance bands", "foam roller",
    "air fryer", "toaster", "blender", "rice cooker", "slow cooker",
    "pillow", "duvet", "bedside table", "office chair", "desk mat",
]

POS_TEMPLATES = [
    "I love this {product}. The quality is excellent and delivery was fast.",
    "Really happy with my {product}. It works exactly as advertised.",
    "Great {product}. Comfortable, reliable, and worth the price.",
    "The {product} exceeded my expectations. I would buy it again.",
    "Solid {product}. Easy to use and better than cheaper alternatives.",
]

NEG_TEMPLATES = [
    "Terrible {product}. The quality is poor and delivery was late.",
    "Really unhappy with my {product}. It broke almost immediately.",
    "Awful {product}. Uncomfortable, unreliable, and not worth the price.",
    "The {product} was a disappointment. I would not buy it again.",
    "Weak {product}. Hard to use and worse than cheaper alternatives.",
]


def build_reviews() -> pd.DataFrame:
    rows = []
    for i, product in enumerate(PRODUCTS):
        for j, template in enumerate(POS_TEMPLATES):
            rows.append(
                {
                    "review_id": f"pos-{i:02d}-{j}",
                    "label": "positive",
                    "text": template.format(product=product),
                }
            )
        for j, template in enumerate(NEG_TEMPLATES):
            rows.append(
                {
                    "review_id": f"neg-{i:02d}-{j}",
                    "label": "negative",
                    "text": template.format(product=product),
                }
            )
    df = pd.DataFrame(rows)
    assert len(df) == 300
    assert (df["label"] == "positive").sum() == 150
    assert (df["label"] == "negative").sum() == 150
    return df.sort_values("review_id").reset_index(drop=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = build_reviews()
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df)} reviews to {OUT_PATH.relative_to(ROOT)}")
    print(df["label"].value_counts().to_dict())
    print(f"Seed recorded for later steps: {SEED}")
    print(f"Note: {TOKEN_PLACEHOLDER_NOTE}")


if __name__ == "__main__":
    main()
