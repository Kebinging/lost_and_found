"""Persistence layer: the ONLY module that reads or writes project data files.

Found items and claims are kept in spreadsheet files, photos as ``.jpg`` files
and CLIP embeddings in a single ``.npy`` file. Everything else in the project
goes through the functions below, so swapping the backend (for example to a
database) only means changing this module.

All paths are read from ``config`` at call time, which lets the tests point
storage at a temporary directory.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

import config

# Change to "csv" to store the tables as CSV files instead of Excel workbooks.
STORAGE_FORMAT = "xlsx"

ITEMS_SHEET = "items"
CLAIMS_SHEET = "claims"

ITEM_COLUMNS = [
    "item_id", "category", "colour", "description", "location_found",
    "date_found", "image_path", "crop_path", "confidence", "detector",
    "status", "created_at",
]
CLAIM_COLUMNS = [
    "claim_id", "item_id", "student_name", "student_id", "contact",
    "student_description", "match_score", "created_at", "status",
]

ITEM_ID_PREFIX = "LF"
CLAIM_ID_PREFIX = "CL"

# Columns that must always be read back as text (Excel would otherwise turn a
# numeric student ID such as "0012345" into the number 12345).
_TEXT_COLUMNS = {
    "item_id", "category", "colour", "description", "location_found",
    "image_path", "crop_path", "detector", "status",
    "claim_id", "student_name", "student_id", "contact", "student_description",
}


class StorageError(RuntimeError):
    """A data file could not be read or written; the message is user-friendly."""


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #
def _table_path(base: Path) -> Path:
    """Return the real file path for a table, honouring ``STORAGE_FORMAT``."""
    return Path(base).with_suffix(f".{STORAGE_FORMAT}")


def _items_path() -> Path:
    return _table_path(config.FOUND_ITEMS_FILE)


def _claims_path() -> Path:
    return _table_path(config.CLAIMS_FILE)


def _read_table(path: Path, sheet: str, columns: list[str]) -> pd.DataFrame:
    """Read a table, returning an empty frame with ``columns`` if it is missing."""
    if not path.exists():
        return pd.DataFrame(columns=columns)
    converters = {col: _to_text for col in columns if col in _TEXT_COLUMNS}
    try:
        if STORAGE_FORMAT == "csv":
            df = pd.read_csv(path, converters=converters)
        else:
            df = pd.read_excel(path, sheet_name=sheet, converters=converters,
                               engine="openpyxl")
    except PermissionError as exc:
        raise StorageError(f"Close {path.name} in Excel and try again.") from exc
    # Tolerate files created by an older version that lack a column.
    for col in columns:
        if col not in df.columns:
            df[col] = None
    return df[columns]


def _write_table(df: pd.DataFrame, path: Path, sheet: str) -> None:
    """Write a table, turning Excel's file lock into a friendly error."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if STORAGE_FORMAT == "csv":
            df.to_csv(path, index=False)
        else:
            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                df.to_excel(writer, sheet_name=sheet, index=False)
    except PermissionError as exc:
        # Windows locks a workbook while it is open in Excel.
        raise StorageError(f"Close {path.name} in Excel and try again.") from exc


def _to_text(value: Any) -> str:
    """Spreadsheet cell -> clean string (empty cells become '')."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def _to_date(value: Any) -> dt.date | None:
    """Coerce a cell / user value to ``datetime.date`` (``None`` if empty)."""
    if value is None or value == "" or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    parsed = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(parsed) else parsed.date()


def _to_datetime(value: Any) -> dt.datetime | None:
    """Coerce a cell value to a naive ``datetime.datetime`` (``None`` if empty)."""
    if value is None or value == "" or (isinstance(value, float) and np.isnan(value)):
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(parsed) else parsed.to_pydatetime().replace(microsecond=0)


def _normalise_items(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date_found"] = df["date_found"].map(_to_date).astype(object)
    df["created_at"] = df["created_at"].map(_to_datetime).astype(object)
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0)
    return df


def _normalise_claims(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["created_at"] = df["created_at"].map(_to_datetime).astype(object)
    df["match_score"] = pd.to_numeric(df["match_score"], errors="coerce").fillna(0.0)
    return df


def _append_row(df: pd.DataFrame, record: dict, columns: list[str]) -> pd.DataFrame:
    rows = df.to_dict("records")
    rows.append({col: record.get(col) for col in columns})
    return pd.DataFrame(rows, columns=columns)


def _next_id(existing: pd.Series, prefix: str) -> str:
    """Next zero-padded ID after the highest one in ``existing``."""
    numbers = (
        existing.astype(str).str.extract(rf"^{prefix}-(\d+)$", expand=False)
        .dropna().astype(int)
    )
    highest = int(numbers.max()) if len(numbers) else 0
    return f"{prefix}-{highest + 1:04d}"


def _stored_path(path: Path) -> str:
    """Path as written to the spreadsheet: relative to the project root if possible."""
    try:
        return path.resolve().relative_to(config.PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def _save_jpeg(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path, format="JPEG", quality=90)


def _now() -> dt.datetime:
    return dt.datetime.now().replace(microsecond=0)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def resolve_path(stored: str) -> Path:
    """Turn an ``image_path`` / ``crop_path`` cell back into an absolute path."""
    path = Path(stored)
    return path if path.is_absolute() else config.PROJECT_ROOT / path


def init_storage() -> None:
    """Create the data folders and empty tables (with headers) if they are missing."""
    for folder in (config.DATA_DIR, config.IMAGES_DIR, config.CROPS_DIR):
        Path(folder).mkdir(parents=True, exist_ok=True)
    if not _items_path().exists():
        _write_table(pd.DataFrame(columns=ITEM_COLUMNS), _items_path(), ITEMS_SHEET)
    if not _claims_path().exists():
        _write_table(pd.DataFrame(columns=CLAIM_COLUMNS), _claims_path(), CLAIMS_SHEET)


def next_item_id() -> str:
    """Return the next free item ID, e.g. ``LF-0001``."""
    return _next_id(_read_table(_items_path(), ITEMS_SHEET, ITEM_COLUMNS)["item_id"],
                    ITEM_ID_PREFIX)


def next_claim_id() -> str:
    """Return the next free claim ID, e.g. ``CL-0001``."""
    return _next_id(_read_table(_claims_path(), CLAIMS_SHEET, CLAIM_COLUMNS)["claim_id"],
                    CLAIM_ID_PREFIX)


def add_found_item(record: dict, image: Image.Image, crop: Image.Image) -> str:
    """Save a found item and its photos; return the new item ID.

    ``record`` supplies ``category``, ``colour``, ``location_found`` and
    optionally ``description``, ``date_found``, ``confidence``, ``detector``.
    The ID, file paths, status and timestamp are filled in here. If the
    spreadsheet cannot be written the saved photos are removed again, so a
    failed save leaves nothing behind.
    """
    init_storage()
    df = _normalise_items(_read_table(_items_path(), ITEMS_SHEET, ITEM_COLUMNS))
    item_id = _next_id(df["item_id"], ITEM_ID_PREFIX)

    image_path = Path(config.IMAGES_DIR) / f"{item_id}.jpg"
    crop_path = Path(config.CROPS_DIR) / f"{item_id}.jpg"
    _save_jpeg(image, image_path)
    _save_jpeg(crop, crop_path)

    row = {
        "item_id": item_id,
        "category": _to_text(record.get("category")),
        "colour": _to_text(record.get("colour")),
        "description": _to_text(record.get("description")),
        "location_found": _to_text(record.get("location_found")),
        "date_found": _to_date(record.get("date_found")) or dt.date.today(),
        "image_path": _stored_path(image_path),
        "crop_path": _stored_path(crop_path),
        "confidence": round(float(record.get("confidence") or 0.0), 4),
        "detector": _to_text(record.get("detector")) or "manual",
        "status": "Unclaimed",
        "created_at": _now(),
    }
    try:
        _write_table(_append_row(df, row, ITEM_COLUMNS), _items_path(), ITEMS_SHEET)
    except StorageError:
        image_path.unlink(missing_ok=True)
        crop_path.unlink(missing_ok=True)
        raise
    return item_id


def load_found_items(status: str | None = None) -> pd.DataFrame:
    """Return all found items, optionally only those with the given status."""
    df = _normalise_items(_read_table(_items_path(), ITEMS_SHEET, ITEM_COLUMNS))
    if status is not None:
        df = df[df["status"] == status]
    return df.reset_index(drop=True)


def update_item(item_id: str, **fields: Any) -> None:
    """Update columns of one found item, e.g. ``update_item("LF-0001", status="Claimed")``."""
    unknown = set(fields) - set(ITEM_COLUMNS)
    if unknown or "item_id" in fields:
        raise ValueError(f"Cannot update item field(s): {sorted(unknown or {'item_id'})}")
    if "status" in fields and fields["status"] not in config.ITEM_STATUSES:
        raise ValueError(f"Invalid item status: {fields['status']!r}")

    df = _normalise_items(_read_table(_items_path(), ITEMS_SHEET, ITEM_COLUMNS))
    mask = df["item_id"] == item_id
    if not mask.any():
        raise KeyError(f"No found item with ID {item_id!r}")
    for column, value in fields.items():
        if column == "date_found":
            value = _to_date(value)
        df.loc[mask, column] = value
    _write_table(df, _items_path(), ITEMS_SHEET)


def add_claim(record: dict) -> str:
    """Save a student's claim and mark the item "Pending Claim"; return the claim ID.

    ``record`` supplies ``item_id``, ``student_name``, ``student_id``,
    ``contact``, ``student_description`` and ``match_score``.
    """
    init_storage()
    item_id = _to_text(record.get("item_id"))
    items = load_found_items()
    if item_id not in set(items["item_id"]):
        raise KeyError(f"No found item with ID {item_id!r}")

    df = _normalise_claims(_read_table(_claims_path(), CLAIMS_SHEET, CLAIM_COLUMNS))
    claim_id = _next_id(df["claim_id"], CLAIM_ID_PREFIX)
    row = {
        "claim_id": claim_id,
        "item_id": item_id,
        "student_name": _to_text(record.get("student_name")),
        "student_id": _to_text(record.get("student_id")),
        "contact": _to_text(record.get("contact")),
        "student_description": _to_text(record.get("student_description")),
        "match_score": round(float(record.get("match_score") or 0.0), 4),
        "created_at": _now(),
        "status": "Pending",
    }
    _write_table(_append_row(df, row, CLAIM_COLUMNS), _claims_path(), CLAIMS_SHEET)
    update_item(item_id, status="Pending Claim")
    return claim_id


def load_claims() -> pd.DataFrame:
    """Return all claims."""
    df = _normalise_claims(_read_table(_claims_path(), CLAIMS_SHEET, CLAIM_COLUMNS))
    return df.reset_index(drop=True)


def update_claim(claim_id: str, **fields: Any) -> None:
    """Update columns of one claim, e.g. ``update_claim("CL-0001", status="Verified")``."""
    unknown = set(fields) - set(CLAIM_COLUMNS)
    if unknown or "claim_id" in fields:
        raise ValueError(f"Cannot update claim field(s): {sorted(unknown or {'claim_id'})}")
    if "status" in fields and fields["status"] not in config.CLAIM_STATUSES:
        raise ValueError(f"Invalid claim status: {fields['status']!r}")

    df = _normalise_claims(_read_table(_claims_path(), CLAIMS_SHEET, CLAIM_COLUMNS))
    mask = df["claim_id"] == claim_id
    if not mask.any():
        raise KeyError(f"No claim with ID {claim_id!r}")
    for column, value in fields.items():
        df.loc[mask, column] = value
    _write_table(df, _claims_path(), CLAIMS_SHEET)


def load_embeddings() -> dict[str, np.ndarray]:
    """Return ``{item_id: CLIP vector}`` (empty if nothing has been saved yet)."""
    path = Path(config.EMBEDDINGS_FILE)
    if not path.exists():
        return {}
    # allow_pickle is needed because the file holds a dict. It is only ever
    # written by save_embedding() below; never load an .npy from elsewhere.
    return dict(np.load(path, allow_pickle=True).item())


def save_embedding(item_id: str, vector: np.ndarray) -> None:
    """Store (or replace) the embedding of one item."""
    embeddings = load_embeddings()
    embeddings[item_id] = np.asarray(vector, dtype=np.float32)
    path = Path(config.EMBEDDINGS_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        np.save(path, embeddings, allow_pickle=True)
    except PermissionError as exc:
        raise StorageError(f"Could not write {path.name}. Close any program "
                           "using it and try again.") from exc
