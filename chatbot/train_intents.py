"""Train the chatbot's intent classifier from ``intents.json``.

Each pattern is preprocessed (lower-cased, punctuation removed, lemmatised
with NLTK WordNet) and fed to ``TfidfVectorizer(ngram_range=(1, 2))`` +
``LogisticRegression``. A classification report is printed for a held-out 20%
of the patterns; the model that is saved is then refitted on *all* patterns,
because with so few examples every one counts.

Usage (from the project root)::

    python chatbot/train_intents.py

The chatbot also runs this automatically the first time it starts if
``intent_model.pkl`` does not exist yet.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root

import config  # noqa: E402
from chatbot.nlu import preprocess  # noqa: E402

SEED = 42


def load_intents() -> list[dict]:
    """Return the list of intents (``tag``, ``patterns``, ``responses``)."""
    with open(config.INTENTS_FILE, encoding="utf-8") as handle:
        return json.load(handle)["intents"]


def build_pipeline() -> Pipeline:
    # C=10 (weaker regularisation than the default) gives sharper class
    # probabilities, which matters because predictions below
    # config.INTENT_CONFIDENCE are treated as "fallback".
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2)),
        LogisticRegression(C=10, max_iter=1000, random_state=SEED),
    )


def train(verbose: bool = True) -> Pipeline:
    """Evaluate on a 20% split, then return a pipeline fitted on every pattern."""
    texts, labels = [], []
    for intent in load_intents():
        for pattern in intent["patterns"]:
            texts.append(preprocess(pattern))
            labels.append(intent["tag"])

    train_x, test_x, train_y, test_y = train_test_split(
        texts, labels, test_size=0.2, random_state=SEED, stratify=labels)
    evaluation_model = build_pipeline().fit(train_x, train_y)
    if verbose:
        print(f"Patterns: {len(texts)} ({len(train_x)} train / {len(test_x)} test)\n")
        print(classification_report(test_y, evaluation_model.predict(test_x), zero_division=0))

    return build_pipeline().fit(texts, labels)


def train_and_save(verbose: bool = True) -> Pipeline:
    """Train the classifier and write it to ``config.INTENT_MODEL_FILE``."""
    model = train(verbose=verbose)
    with open(config.INTENT_MODEL_FILE, "wb") as handle:
        pickle.dump(model, handle)
    if verbose:
        print(f"Saved {config.INTENT_MODEL_FILE}")
    return model


if __name__ == "__main__":
    train_and_save()
