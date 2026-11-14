"""Clean up a student's message before the chatbot tries to understand it.

Students type quickly: short forms ("hp", "lib", "ytd"), chat fillers ("lah",
"bro"), stretched words ("pleaseee") and typos ("umbrela", "libary"). The
rest of the chatbot only recognises proper words, so every message first
goes through :func:`normalise`, which runs these steps on each word:

1. **Short forms** - a dictionary maps abbreviations, chat words and common
   Malay words to the English word the chatbot knows ("hp" -> "phone",
   "semalam" -> "yesterday"). Fillers such as "lah" are removed.
2. **Known words are kept** - the chatbot's vocabulary (every item, colour and
   location synonym plus every word in ``intents.json``) and common English
   words are never changed.
3. **Repeated letters** - "keyy" -> "key", "pleaseee" -> "please".
4. **Spelling correction** - an unknown word is replaced by the closest
   vocabulary word if it is at most 1 edit away (words of up to 5 letters) or
   2 edits away (longer words). An edit is inserting, deleting or replacing a
   letter, or swapping two neighbouring letters ("lsot" -> "lost"); this is
   the Damerau-Levenshtein (optimal string alignment) distance. Real English
   words found in WordNet are left alone, so "bottom" never becomes "bottle".

:func:`suggest` is a looser version used for "Did you mean ...?" questions
when a word is too far from anything to be corrected automatically.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache

import config

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Dictionaries
# --------------------------------------------------------------------------- #
# Short form / chat word / Malay word -> what the chatbot understands.
# Only words that are NOT normal English words belong here ("sat" would turn
# "I sat in the library" into Saturday, so it is left out on purpose).
SHORT_FORMS = {
    # items
    "hp": "phone", "fon": "phone", "fone": "phone", "phon": "phone", "handfon": "phone",
    "telefon": "phone",
    "bttl": "bottle", "btl": "bottle", "botol": "bottle",
    "umb": "umbrella", "umbr": "umbrella", "payung": "umbrella",
    "calc": "calculator", "kalkulator": "calculator",
    "chgr": "charger", "chrgr": "charger", "pengecas": "charger",
    "pb": "power bank", "pwrbank": "power bank", "pwr": "power",
    "kunci": "key", "beg": "bag", "bp": "backpack",
    "lappy": "laptop", "lptp": "laptop", "komputer": "laptop",
    "buku": "book",
    # colours
    "blk": "black", "hitam": "black", "putih": "white", "gry": "grey", "kelabu": "grey",
    "merah": "red", "biru": "blue", "hijau": "green", "kuning": "yellow",
    "pnk": "pink", "ungu": "purple", "coklat": "brown", "perak": "silver",
    # places
    "lib": "library", "libr": "library", "perpustakaan": "library", "perpus": "library",
    "caf": "cafeteria", "kantin": "canteen", "lec": "lecture", "lect": "lecture",
    "asrama": "hostel", "makmal": "lab",
    # time
    "ytd": "yesterday", "yday": "yesterday", "yest": "yesterday", "ystd": "yesterday",
    "ystrdy": "yesterday", "semalam": "yesterday", "tdy": "today", "tday": "today",
    "mon": "monday", "tue": "tuesday", "tues": "tuesday", "weds": "wednesday",
    "thu": "thursday", "thur": "thursday", "thurs": "thursday", "fri": "friday",
    "hr": "hours", "hrs": "hours", "wk": "week",
    # actions
    "hilang": "lost", "jumpa": "found",
    # chat words
    "u": "you", "ur": "your", "r": "are", "pls": "please", "plz": "please",
    "thx": "thanks", "thnx": "thanks", "thanx": "thanks", "tks": "thanks",
    "tq": "thanks", "ty": "thanks", "tysm": "thank you so much",
    "wat": "what", "wht": "what", "whr": "where", "wer": "where", "whre": "where",
    "hw": "how", "abt": "about", "smth": "something", "sth": "something",
    "helo": "hello", "hallo": "hello", "hai": "hi",
    "hv": "have", "c": "see", "kat": "at", "dkt": "at", "sy": "i", "saya": "i", "cnt": "can't", "cant": "can't", "dont": "don't", "didnt": "didn't",
    "im": "i'm", "ive": "i've", "whats": "what is", "wheres": "where is",
}

# Chat fillers that carry no meaning for the chatbot; they are dropped.
FILLERS = {"lah", "la", "leh", "lor", "meh", "bro", "sis", "bruh", "dude", "eh", "oi", "ah"}

# Common English words that are not in WordNet (it only has nouns, verbs,
# adjectives and adverbs), so they must be protected from "correction".
COMMON_WORDS = set("""
a about above after again against all am an and any are as at be because been before being
below between both but by can can't could did didn't do does doesn't doing don't down during
each few for from further had has have having he her here hers herself him himself his how
i i'm i've if in into is isn't it it's its itself just me might more most must my myself
no nor not now of off on once only or other our ours out over own same she should so some
such than that the their theirs them themselves then there these they this those through
to too under until up upon very was we were what when where which while who whom why will
with within without would yes yet you your yours yourself anyone anything anywhere
someone something somewhere everyone everything everywhere nothing nowhere please hi
hello hey bye thanks okay ok cannot
""".split())

_WORD = re.compile(r"\b[a-z][a-z']*\b")
_REPEATS_3 = re.compile(r"(.)\1{2,}")   # 3+ of the same letter
_REPEATS_2 = re.compile(r"(.)\1+")      # 2+ of the same letter


@dataclass
class Normalised:
    """Result of :func:`normalise`: the cleaned text and what was changed."""

    text: str
    changes: list[tuple[str, str, str]] = field(default_factory=list)  # (old, new, kind)


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #
def _words(phrases) -> set[str]:
    return {word for phrase in phrases for word in _WORD.findall(phrase.lower())}


@lru_cache(maxsize=1)
def slot_vocabulary() -> frozenset[str]:
    """Words that name an item, colour or location (corrections prefer these)."""
    phrases = [w for words in config.CATEGORY_SYNONYMS.values() for w in words]
    phrases += [w for words in config.COLOUR_SYNONYMS.values() for w in words]
    phrases += [w for words in config.LOCATION_SYNONYMS.values() for w in words]
    phrases += config.CATEGORIES + list(config.COLOUR_PALETTE) + config.LOCATIONS
    return frozenset(_words(phrases))


@lru_cache(maxsize=1)
def word_counts() -> dict[str, int]:
    """How often each word appears in the ``intents.json`` patterns."""
    counts: dict[str, int] = {}
    try:
        with open(config.INTENTS_FILE, encoding="utf-8") as handle:
            for intent in json.load(handle)["intents"]:
                for pattern in intent["patterns"]:
                    for word in _WORD.findall(pattern.lower()):
                        counts[word] = counts.get(word, 0) + 1
    except (OSError, ValueError, KeyError):
        logger.warning("Could not read %s for the spelling vocabulary.", config.INTENTS_FILE)
    return counts


@lru_cache(maxsize=1)
def vocabulary() -> frozenset[str]:
    """Every word the chatbot knows: slot words, intent patterns and date words."""
    words = set(slot_vocabulary()) | set(word_counts())
    words |= _words(["monday tuesday wednesday thursday friday saturday sunday",
                     "january february march april may june july august september "
                     "october november december",
                     "yesterday today tonight morning afternoon evening night week month "
                     "days ago last lost missing found claim status reference"])
    words |= COMMON_WORDS
    return frozenset(words)


def _unzip_downloaded_wordnet() -> bool:
    """Extract a downloaded ``wordnet.zip`` into the project's NLTK folder.

    Some NLTK versions on Windows download the zip but then refuse to unzip
    it ("Path traversal blocked"), and WordNet cannot be read from the zip.
    Each archive member is checked to stay inside the target folder.
    """
    import zipfile
    from pathlib import Path

    import nltk

    target = (config.NLTK_DATA_DIR / "corpora").resolve()
    for folder in nltk.data.path:
        archive = Path(folder) / "corpora" / "wordnet.zip"
        if not archive.is_file():
            continue
        try:
            target.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive) as zipped:
                for member in zipped.namelist():
                    if not (target / member).resolve().is_relative_to(target):
                        raise ValueError(f"unsafe path in {archive}: {member}")
                zipped.extractall(target)
            return True
        except (OSError, ValueError, zipfile.BadZipFile) as error:
            logger.warning("Could not extract %s: %s", archive, error)
    return False


@lru_cache(maxsize=1)
def ensure_wordnet():
    """Return NLTK's WordNet corpus, downloading/unzipping it once; ``None`` if impossible.

    The data lives in the project's own ``nltk_data/`` folder (git-ignored),
    so it does not depend on the user's AppData folder.
    """
    try:
        import nltk
        from nltk.corpus import wordnet

        folder = str(config.NLTK_DATA_DIR)
        if folder not in nltk.data.path:
            nltk.data.path.insert(0, folder)
        for attempt in ("load", "unzip", "download", "unzip", "load"):
            try:
                wordnet.synsets("dog")
                return wordnet
            except (LookupError, PermissionError):
                if attempt == "download":
                    config.NLTK_DATA_DIR.mkdir(parents=True, exist_ok=True)
                    nltk.download("wordnet", download_dir=folder, quiet=True)
                elif attempt == "unzip":
                    _unzip_downloaded_wordnet()
        return None
    except Exception:
        logger.warning("WordNet unavailable; spelling correction may change real words.")
        return None


_wordnet = ensure_wordnet


@lru_cache(maxsize=4096)
def is_english_word(word: str) -> bool:
    """True if ``word`` is a real English word according to WordNet."""
    wordnet = _wordnet()
    return bool(wordnet and wordnet.synsets(word))


# --------------------------------------------------------------------------- #
# Edit distance
# --------------------------------------------------------------------------- #
def edit_distance(a: str, b: str) -> int:
    """Damerau-Levenshtein distance (optimal string alignment variant).

    Counts insertions, deletions, substitutions and swaps of two neighbouring
    letters: ``edit_distance("lsot", "lost") == 1``.
    """
    previous_previous: list[int] = []
    previous = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        current = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            current[j] = min(previous[j] + 1,          # deletion
                             current[j - 1] + 1,       # insertion
                             previous[j - 1] + cost)   # substitution
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                current[j] = min(current[j], previous_previous[j - 2] + 1)  # swap
        previous_previous, previous = previous, current
    return previous[-1]


def max_edits(word: str) -> int:
    """How many edits a word may be away from its correction."""
    return 1 if len(word) <= 5 else 2


@lru_cache(maxsize=4096)
def correct_word(word: str) -> str | None:
    """Return the closest vocabulary word to ``word``, or ``None`` if none is close.

    Ties are broken in favour of words used more often in the chatbot's
    training sentences ("lsot" -> "lost", not "lot"), then item / colour /
    location words, then words that start with the same letter, then
    alphabetically (so the result is always the same).
    """
    if len(word) < 4 or "'" in word:
        return None
    limit = max_edits(word)
    best: list[tuple] = []
    slot_words, counts = slot_vocabulary(), word_counts()
    for candidate in vocabulary():
        if abs(len(candidate) - len(word)) > limit or len(candidate) < 3:
            continue
        distance = edit_distance(word, candidate)
        if distance <= limit:
            best.append((distance, -counts.get(candidate, 0), candidate not in slot_words,
                         candidate[0] != word[0], candidate))
    return min(best)[-1] if best else None


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #
def _normalise_word(word: str) -> tuple[str, str | None]:
    """Return ``(replacement, kind)`` for one lower-case word (kind ``None`` = unchanged)."""
    if word in SHORT_FORMS:
        return SHORT_FORMS[word], "short form"
    if word in FILLERS:
        return "", "filler"
    known = vocabulary()
    if word in known:
        return word, None
    for squeezed in (_REPEATS_3.sub(r"\1\1", word), _REPEATS_2.sub(r"\1", word)):
        if squeezed != word and (squeezed in known or squeezed in SHORT_FORMS):
            return SHORT_FORMS.get(squeezed, squeezed), "repeated letters"
    if is_english_word(word):
        return word, None
    correction = correct_word(_REPEATS_3.sub(r"\1\1", word))
    if correction and correction != word:
        return correction, "spelling"
    return word, None


def normalise(text: str) -> Normalised:
    """Lower-case ``text`` and fix short forms, fillers, repeated letters and typos.

    Numbers and punctuation are kept as they are, so dates ("12/11", "3rd")
    and claim references ("CL-0007") still work.
    """
    changes: list[tuple[str, str, str]] = []

    def replace(match: re.Match) -> str:
        word = match.group(0)
        new, kind = _normalise_word(word)
        if kind:
            changes.append((word, new, kind))
        return new

    cleaned = _WORD.sub(replace, str(text).lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return Normalised(cleaned, changes)


# --------------------------------------------------------------------------- #
# "Did you mean ...?" suggestions
# --------------------------------------------------------------------------- #
def similarity(a: str, b: str) -> float:
    """1.0 for identical words, falling towards 0 as more edits are needed."""
    return 1 - edit_distance(a, b) / max(len(a), len(b))


def suggest(text: str, synonyms: dict[str, list[str]], min_similarity: float = 0.6) -> str | None:
    """Return the label whose synonym looks most like a word in ``text``, or ``None``.

    Looser than automatic correction: used only to ask "Did you mean ...?",
    so the student always confirms. A word also counts if it is the start of
    a synonym ("calcu" -> calculator, "blu" -> blue). Real English words are
    skipped: "tell" is not a typo of "cell", and "wallet" is not an umbrella.
    """
    best_score, best_label = 0.0, None
    for word in _WORD.findall(text.lower()):
        if len(word) < 3 or word in COMMON_WORDS or is_english_word(word):
            continue
        for label, phrases in synonyms.items():
            for phrase in phrases:
                for synonym in phrase.split():
                    if synonym.startswith(word) and len(word) >= len(synonym) / 2:
                        score = 0.9
                    else:
                        score = similarity(word, synonym)
                    if score > best_score:
                        best_score, best_label = score, label
    return best_label if best_score >= min_similarity else None
