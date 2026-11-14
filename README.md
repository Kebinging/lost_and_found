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
Staff photo ──> YOLO (custom + COCO) ──> crop ──> colour (KMeans) ──┐
                                           └────> CLIP embedding ───┼──> found_items.xlsx
                                                                    │    images/  crops/  embeddings.npy
Student chat ──> intent + slots ──> follow-up questions ──> matcher ┘
                                                              │
                                    "possible match" + claim form ──> claims.xlsx ──> staff verify
```

| Part | Technology |
|---|---|
| Object detection | Ultralytics YOLO11: custom model (Bottle, Umbrella, Calculator, Charger, Key) + COCO `yolo11n.pt` (Phone, Bag, Laptop, Book) |
| Colour extraction | OpenCV + scikit-learn KMeans, nearest palette colour in LAB space |
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

The opening hours and counter location in `intents.json` are placeholders; replace them
with your university's details.

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
├── requirements.txt
├── .streamlit/
│   └── secrets.toml.example  # template for the staff PIN
├── datasets/               # NOT committed (raw download + cleaned split)
├── notebooks/
│   └── train_yolo.ipynb    # Colab: prepare, train, evaluate, export best.pt
├── scripts/
│   ├── prepare_dataset.py  # drop "None", remap IDs, re-split 70/20/10
│   ├── train.py            # local training
│   ├── evaluate.py         # metrics + confusion matrix on the test split
│   └── seed_demo_data.py   # add ~15 demo found items
├── models/                 # best.pt (you train it) and yolo11n.pt (auto-downloaded)
├── data/                   # created at runtime, NOT committed
│   ├── found_items.xlsx, claims.xlsx
│   ├── images/, crops/
│   └── embeddings.npy
├── core/
│   ├── storage.py          # the only module that reads/writes data files
│   ├── detector.py         # runs custom + COCO models, merges results
│   ├── colour.py           # dominant colour
│   ├── embedder.py         # CLIP image + text embeddings
│   └── matcher.py          # scoring + ranking
├── chatbot/
│   ├── intents.json        # intents with patterns + responses
│   ├── train_intents.py    # trains the intent classifier
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
