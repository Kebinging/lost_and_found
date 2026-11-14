# Smart Lost & Found

A university Lost & Found prototype that combines **object detection** with a **chatbot**.

- **Staff** upload a photo of a found item. A YOLO11 model identifies the category, the
  system extracts the main colour and a CLIP image embedding, and the record is saved to
  an Excel file.
- **Students** describe what they lost to a chatbot (and can optionally upload a photo of
  a similar item). The chatbot extracts the category, colour, location and date, asks
  follow-up questions for anything missing, searches the found items and says whether a
  **possible match** exists. If so, the student leaves their details and receives a claim
  reference to take to the counter.

> **Privacy / anti-fraud rule:** the student side never shows a found item's photo, staff
> notes, location or item ID. It only says "possible match found" and gives a claim
> reference. Staff verify ownership in person.

## How it works

```
Staff photo ──> YOLO (custom + COCO) ──> CLIP names the item ──> crop ──> colour (KMeans + HSV) ──┐
                                                                  └────> CLIP embedding ───┼──> found_items.xlsx
                                                                    │    images/  crops/  embeddings.npy
Student chat ──> intent + slots ──> follow-up questions ──> matcher ┘
                                                              │
                                    "possible match" + claim form ──> claims.xlsx ──> staff verify
```

| Part | Technology |
|---|---|
| Object detection | Ultralytics YOLO11: custom model (Bottle, Umbrella, Calculator, Charger, Key) + COCO `yolo11n.pt` (Phone, Bag, Laptop, Book) |
| Naming items YOLO was not trained on | CLIP zero-shot classification (Earbuds, ID Card, Wallet, Spectacles, Watch, Stationery, Clothing, Toy), else "Other" |
| Colour extraction | OpenCV + scikit-learn KMeans, colour named from hue / saturation / brightness |
| Image / text similarity | `sentence-transformers` CLIP (`clip-ViT-B-32`) |
| Intent classifier | TF-IDF + LogisticRegression trained on `chatbot/intents.json` |
| Slot extraction | Keyword dictionaries + `dateparser` |
| Storage | Excel (`pandas` + `openpyxl`), `.jpg` photos, `.npy` embeddings |
| UI | Streamlit multipage app |

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
```

```bash
.venv\Scripts\activate
```

(macOS / Linux: `source .venv/bin/activate`)

```bash
pip install -r requirements.txt
```

Set the staff PIN: copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`
and choose your own `STAFF_PIN`. Staff pages stay locked until you do.

> The PIN is a **prototype-only** measure: one shared secret, no user accounts, no
> lock-out. A real deployment needs proper authentication.

## Run

```bash
streamlit run app/Home.py
```

The first run downloads the COCO weights (`models/yolo11n.pt`, ~5 MB), the CLIP model
(~600 MB) and NLTK WordNet, so it needs an internet connection and takes a few minutes.

| Page | For | What it does |
|---|---|---|
| Home | everyone | Links to both sides and shows inventory statistics |
| Staff Upload | staff (PIN) | Photo → detection → auto-filled form → save |
| Student Chatbot | students | Describe a lost item, get "match" / "no match", submit a claim |
| Inventory | staff (PIN) | Browse and filter items, change status, test the matcher |
| Claims | staff (PIN) | Review a claim next to its item, verify or reject it |

**Without `models/best.pt`** the app still runs, but only the COCO categories are
detected (phone, bag, laptop, book, bottle, umbrella). Train the custom model to add
calculator, charger and key.

## Items outside the trained classes

YOLO can only report the classes it was trained on: an item outside them is either
missed or forced into the nearest class (earbuds become a "Charger"). So every detected
object - or the whole photo, if YOLO finds nothing - is also shown to **CLIP**, which
compares the picture with written descriptions of every category
(`config.CATEGORY_PROMPTS`, e.g. "a photo of wireless earbuds") without any training
(*zero-shot classification*, `core/classifier.py`). The final category is chosen by:

1. CLIP is at least 60% sure it is a category **no YOLO model can detect** → CLIP.
2. Otherwise, if YOLO found the object → YOLO (it was trained for its classes).
3. Otherwise, if CLIP is at least 40% sure → CLIP.
4. Otherwise → **Other**; staff pick the category.

CLIP sees the YOLO box plus a 15% margin, because a tight crop loses context (a hand
holding a bottle, cropped tightly, looked like a phone to CLIP). The Staff Upload page
shows how each suggestion was made and CLIP's top three guesses, and staff can always
change the category, including to **Other**. Students can also choose "Other" in the
chatbot; their own description is then compared with the photos of items filed as Other.

All thresholds are in `config.py`. They were checked on only a few photos, so test with
real items and adjust them.

## Train on a combined dataset

To let YOLO itself detect the new categories, combine several datasets. The sources and
how each one's class names map onto our categories are listed in `dataset_sources.yaml`
(the original dataset plus earbuds, student ID, wallet, glasses and desk-object datasets
from Roboflow Universe; check each page's licence and latest version before training).
`scripts/merge_datasets.py` downloads them, renames the classes, removes duplicates,
balances the classes, adds some empty "background" images and splits 70 / 20 / 10:

```bash
python scripts/merge_datasets.py --download --pseudo-label
```

`--pseudo-label` deals with the main problem of merging: each dataset only labels its
own classes, so a phone lying next to a wallet is unlabelled and would be learned as
background. The current YOLO models label such objects (confidence ≥ 0.6) for the
categories a source does not cover.

Train on Colab with `notebooks/train_combined.ipynb` (or locally with
`python scripts/train.py --data datasets/combined/data.yaml`).

**Your own photos matter most.** The current model scores 99% mAP50 on its own test
images but worse on photos taken on campus, because they look different (background,
lighting, camera). Label 100-200 photos of real items (Roboflow or Label Studio, YOLO
format), put them in `datasets/sources/campus` and enable that source: with
`split: real_test` they become a separate real-world test set, the fairest number to
report. Once you have enough, also add a second batch to training.

## Train the custom model

Dataset: [Lost and Found (rnomo), version 2](https://universe.roboflow.com/rnomo/lost-and-found-nlb4x-e6ayk/dataset/2)
on Roboflow Universe - 4,546 images, classes `Bottle, Umbrella, Calculator, Charger, Key, None`.

**On Google Colab (recommended, free GPU):** open `notebooks/train_yolo.ipynb`, run all
cells, and save the downloaded `best.pt` as `models/best.pt`.

**Locally:**

1. Download the dataset in **YOLOv8 format** and unzip it into `datasets/raw/`, or set
   the `ROBOFLOW_API_KEY` environment variable and add `--download` below
   (`pip install roboflow` first). Never write the key into a file.
2. Clean and re-split it (drops the `None` class; 70 / 20 / 10 split, seed 42):

   ```bash
   python scripts/prepare_dataset.py
   ```

3. Train (50 epochs; very slow without a GPU) - copies the result to `models/best.pt`:

   ```bash
   python scripts/train.py
   ```

4. Evaluate on the test split (mAP50, mAP50-95, precision, recall, confusion matrix):

   ```bash
   python scripts/evaluate.py
   ```

To fill the inventory with ~15 demo items from the test images:

```bash
python scripts/seed_demo_data.py
```

## Chatbot

The intent classifier is trained automatically the first time the chatbot runs. After
editing `chatbot/intents.json`, retrain it and see the classification report with:

```bash
python chatbot/train_intents.py
```

With only ~13 example patterns per intent the held-out report is noisy (a few test
sentences per intent). The dialogue therefore also uses simple rules - e.g. a message
containing "lost", "missing" or "can't find" is always treated as a lost-item report.

### Handling typos and short forms

Students type quickly, so the chatbot has five layers that make it tolerant of messy text:

| # | Layer | Where | Example |
|---|---|---|---|
| 1 | **Short-form dictionary** - abbreviations, chat words and common Malay words; fillers like "lah" are dropped | `chatbot/normalise.py` (`SHORT_FORMS`) | "hp" → phone, "lib" → library, "ytd" / "semalam" → yesterday |
| 2 | **Spelling correction** - an unknown word becomes the closest known word within 1 edit (≤ 5 letters) or 2 edits (longer), using Damerau-Levenshtein distance; real English words (WordNet) are never changed | `chatbot/normalise.py` | "umbrela" → umbrella, "lsot" → lost, but "bottom" stays "bottom" |
| 3 | **Typo-tolerant intent model** - whole-word *and* letter (2-5 character) TF-IDF features, trained with 3 extra typo copies of every pattern | `chatbot/train_intents.py` | "how does ths work" → how it works |
| 4 | **"Did you mean ...?"** - when still unsure, the bot asks instead of giving up | `chatbot/dialogue.py` | "calcu" → "Did you mean **calculator**?" |
| 5 | **Quick-reply buttons** - item types, colours, dates, locations, yes / no | Student Chatbot page | Tap "Black" instead of typing |

Measure it on 50 messy messages (`chatbot/messy_test_set.json`) - original vs improved:

```bash
python scripts/evaluate_chatbot.py
```

| Accuracy | Before | After |
|---|---|---|
| Intent | 82% | 100% |
| Item category | 21% | 100% |
| Colour | 79% | 100% |
| Location | 67% | 100% |
| Date | 75% | 100% |

These messages were written while building the improvements, so the "after" figures are
optimistic. For a fair result, have someone who did not work on the chatbot write a fresh
set of messy messages and run the same script on them.

WordNet (used for lemmatisation and to protect real words) is downloaded into the
project's `nltk_data/` folder on first use.

The opening hours in `intents.json` are placeholders; replace them with your
university's details.

## How matching works

`core/matcher.py` only considers **Unclaimed** items, then:

1. **Hard filters** - same category; handed in between 1 day before the loss date and
   `DATE_WINDOW_DAYS` (30) after it.
2. **Score (0-1)**
   - with a photo: `0.40 image + 0.25 text + 0.25 colour + 0.10 location`
   - without: `0.45 text + 0.40 colour + 0.15 location`
   - colour: 1 same, 0.5 similar (Black/Grey/Silver, Blue/Purple, Red/Pink/Orange), 0 otherwise
3. Items scoring at least `MATCH_THRESHOLD` (0.55) are returned, at most `MAX_RESULTS` (3).

CLIP's raw text-to-image cosine similarity sits in a narrow band (about 0.15-0.35), so it
is rescaled onto 0-1 using `TEXT_SIM_RANGE` in `config.py` before weighting. All
thresholds and weights live in `config.py`.

## Tests

```bash
pytest
```

The tests use a temporary folder and fake embeddings, so they need neither the dataset
nor the YOLO / CLIP models.

## Project structure

```
lost_and_found/
├── config.py               # all paths, thresholds, category lists, colour palette
├── dataset_sources.yaml    # datasets to combine and their class mapping
├── requirements.txt
├── .streamlit/
│   └── secrets.toml.example  # template for the staff PIN
├── datasets/               # NOT committed (raw download + cleaned split)
├── notebooks/
│   ├── train_yolo.ipynb    # Colab: prepare, train, evaluate, export best.pt
│   └── train_combined.ipynb # Colab: merge several datasets, train, evaluate
├── scripts/
│   ├── prepare_dataset.py  # drop "None", remap IDs, re-split 70/20/10
│   ├── merge_datasets.py   # combine the datasets in dataset_sources.yaml
│   ├── train.py            # local training
│   ├── evaluate.py         # metrics + confusion matrix on the test split
│   ├── evaluate_chatbot.py # chatbot accuracy on messy messages, before vs after
│   └── seed_demo_data.py   # add ~15 demo found items
├── models/                 # best.pt (you train it) and yolo11n.pt (auto-downloaded)
├── data/                   # created at runtime, NOT committed
│   ├── found_items.xlsx, claims.xlsx
│   ├── images/, crops/
│   └── embeddings.npy
├── core/
│   ├── storage.py          # the only module that reads/writes data files
│   ├── detector.py         # runs custom + COCO models, merges results
│   ├── classifier.py       # CLIP zero-shot naming + YOLO/CLIP decision rule
│   ├── colour.py           # dominant colour
│   ├── embedder.py         # CLIP image + text embeddings
│   └── matcher.py          # scoring + ranking
├── chatbot/
│   ├── intents.json        # intents with patterns + responses
│   ├── train_intents.py    # trains the intent classifier (letters + typo copies)
│   ├── normalise.py        # short forms, chat words, spelling correction
│   ├── messy_test_set.json # 50 messy messages for scripts/evaluate_chatbot.py
│   ├── nlu.py              # preprocessing, intent prediction, slot extraction
│   └── dialogue.py         # conversation state machine
├── app/
│   ├── Home.py             # entry point
│   ├── common.py           # shared header, PIN gate, cached model loaders
│   └── pages/              # 1_Staff_Upload, 2_Student_Chatbot, 3_Inventory, 4_Claims
└── tests/
```

## Troubleshooting

- **"Close found_items.xlsx in Excel and try again."** - Excel locks a workbook while it
  is open. Close it and repeat the action.
- **Staff pages say no PIN is configured** - create `.streamlit/secrets.toml` (see Setup)
  and restart the app.
- **Switching to CSV** - set `STORAGE_FORMAT = "csv"` in `core/storage.py`.

## Limitations

This is a coursework prototype: a shared staff PIN instead of accounts, Excel files
instead of a database (no concurrent writers), and students' contact details stored
unencrypted in `data/claims.xlsx` (kept out of git by `.gitignore`).

## Credits

- Dataset: *Lost and Found* by rnomo, Roboflow Universe, version 2, licensed
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
  https://universe.roboflow.com/rnomo/lost-and-found-nlb4x-e6ayk/dataset/2
- [Ultralytics YOLO11](https://docs.ultralytics.com/) and
  [Sentence-Transformers CLIP](https://www.sbert.net/).
