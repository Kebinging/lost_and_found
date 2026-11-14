"""Train the chatbot's intent classifier from ``intents.json``.

Two versions of the classifier can be built:

* **baseline** - the original model: whole-word TF-IDF (single words and word
  pairs) + LogisticRegression, trained only on the patterns as written.
* **improved** (the one the chatbot uses) - adds two things so typos and
  short forms do less damage:

  1. **Character n-grams.** Besides whole words, the text is also described by
     overlapping groups of 2-5 letters ("umbrela" still shares "umb", "mbr",
     "bre", ... with "umbrella"), so a misspelt word is not a complete
     unknown to the model.
  2. **Typo augmentation.** Each training pattern gets extra copies with a
     random typo (a letter swapped, dropped, doubled or replaced by a nearby
     key), so the model learns from imperfect text too.

Each pattern is preprocessed (lower-cased, punctuation removed, lemmatised
with NLTK WordNet). A classification report is printed for a held-out 20% of
the original patterns (typo copies are made from the training part only, so
the test is fair); the saved model is then refitted on *all* patterns.

Usage (from the project root)::

    python chatbot/train_intents.py

The chatbot also runs this automatically the first time it starts if
``intent_model.pkl`` does not exist yet. ``scripts/evaluate_chatbot.py``
compares the baseline and improved versions on messy messages.
"""

from __future__ import annotations

import json
import pickle
import random
import sys
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root

import config  # noqa: E402
from chatbot.nlu import preprocess  # noqa: E402

SEED = 42
TYPO_COPIES = 3   # extra typo versions of every training pattern

# Letters next to each other on a QWERTY keyboard, for realistic typos.
_KEYBOARD_ROWS = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
_NEIGHBOURS = {
    letter: "".join(row[max(i - 1, 0):i] + row[i + 1:i + 2])
    for row in _KEYBOARD_ROWS for i, letter in enumerate(row)
}


def load_intents() -> list[dict]:
    """Return the list of intents (``tag``, ``patterns``, ``responses``)."""
    with open(config.INTENTS_FILE, encoding="utf-8") as handle:
        return json.load(handle)["intents"]


def add_typo(text: str, rng: random.Random) -> str:
    """Return ``text`` with one random typo in one of its longer words."""
    words = text.split()
    candidates = [i for i, word in enumerate(words) if len(word) >= 4]
    if not candidates:
        return text
    index = rng.choice(candidates)
    word = words[index]
    position = rng.randrange(1, len(word) - 1)
    kind = rng.choice(["swap", "drop", "double", "nearby"])
    if kind == "swap":
        word = word[:position] + word[position + 1] + word[position] + word[position + 2:]
    elif kind == "drop":
        word = word[:position] + word[position + 1:]
    elif kind == "double":
        word = word[:position] + word[position] + word[position:]
    elif word[position] in _NEIGHBOURS:
        word = word[:position] + rng.choice(_NEIGHBOURS[word[position]]) + word[position + 1:]
    words[index] = word
    return " ".join(words)


def augment(texts: list[str], labels: list[str], copies: int = TYPO_COPIES) -> tuple[list, list]:
    """Return the patterns plus ``copies`` typo versions of each one."""
    rng = random.Random(SEED)
    new_texts, new_labels = list(texts), list(labels)
    for text, label in zip(texts, labels):
        for _ in range(copies):
            new_texts.append(add_typo(text, rng))
            new_labels.append(label)
    return new_texts, new_labels


def build_pipeline(improved: bool = True) -> Pipeline:
    """Return an untrained classifier (see the module docstring for the two versions)."""
    # C=10 (weaker regularisation than the default) gives sharper class
    # probabilities, which matters because predictions below
    # config.INTENT_CONFIDENCE are treated as "fallback".
    classifier = LogisticRegression(C=10, max_iter=2000, random_state=SEED)
    if not improved:
        return make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), classifier)
    features = FeatureUnion([
        ("words", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ("letters", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)),
    ])
    return make_pipeline(features, classifier)


def _examples() -> tuple[list[str], list[str]]:
    texts, labels = [], []
    for intent in load_intents():
        for pattern in intent["patterns"]:
            texts.append(pattern.lower())
            labels.append(intent["tag"])
    return texts, labels


def _fit(texts: list[str], labels: list[str], improved: bool) -> Pipeline:
    if improved:
        texts, labels = augment(texts, labels)
    return build_pipeline(improved).fit([preprocess(t) for t in texts], labels)


def train(verbose: bool = True, improved: bool = True) -> Pipeline:
    """Evaluate on a 20% split, then return a pipeline fitted on every pattern."""
    texts, labels = _examples()
    train_x, test_x, train_y, test_y = train_test_split(
        texts, labels, test_size=0.2, random_state=SEED, stratify=labels)
    if verbose:
        evaluation_model = _fit(train_x, train_y, improved)
        predictions = evaluation_model.predict([preprocess(t) for t in test_x])
        print(f"Patterns: {len(texts)} ({len(train_x)} train / {len(test_x)} test)"
              + (f", plus {TYPO_COPIES} typo copies of each training pattern" if improved else "")
              + "\n")
        print(classification_report(test_y, predictions, zero_division=0))

    return _fit(texts, labels, improved)


def train_and_save(verbose: bool = True) -> Pipeline:
    """Train the improved classifier and write it to ``config.INTENT_MODEL_FILE``."""
    model = train(verbose=verbose)
    with open(config.INTENT_MODEL_FILE, "wb") as handle:
        pickle.dump(model, handle)
    if verbose:
        print(f"Saved {config.INTENT_MODEL_FILE}")
    return model


if __name__ == "__main__":
    train_and_save()
