
import base64
import io
import json
import os
import re
import sqlite3
import threading
import time
import uuid
from contextlib import closing

import gradio as gr
import numpy as np
import pandas as pd
import requests
from PIL import Image

# ---------- config (override with environment variables) ----------
DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
# Inside Docker, "localhost" is the container itself. host.docker.internal reaches your Mac.
EMBED_URL = os.environ.get("EMBED_URL", "http://host.docker.internal:9931").rstrip("/")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "ggml-org/embeddinggemma-2-GGUF:Q8_0")
EMBED_API_KEY = os.environ.get("EMBED_API_KEY", "no-key")
EMBED_DIM = int(os.environ.get("EMBED_DIM", "768"))  # 768 native; 512/256/128 use Matryoshka truncation

COVER_DIR = os.path.join(DATA_DIR, "covers")
EXPORT_DIR = os.path.join(DATA_DIR, "exports")
SHELF_PHOTO_DIR = os.path.join(DATA_DIR, "shelf_photos")
DB_PATH = os.path.join(DATA_DIR, "books.db")
os.makedirs(COVER_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)
os.makedirs(SHELF_PHOTO_DIR, exist_ok=True)

COLUMNS = ["image_name", "title", "author", "publisher", "isbn", "notes", "group_name", "cover_path"]
FIELD_WEIGHTS = {"title": 3.0, "isbn": 3.0, "author": 2.0, "publisher": 1.0, "notes": 1.0, "group_name": 2.5}
MAXW = max(FIELD_WEIGHTS.values())
STOP = {"a", "an", "the", "of", "and", "by", "in", "on", "to", "for", "with"}
lock = threading.Lock()

PLACEHOLDER = os.path.join(COVER_DIR, "_placeholder.png")
if not os.path.exists(PLACEHOLDER):
    Image.new("RGB", (300, 450), (200, 200, 200)).save(PLACEHOLDER)


# ---------- embedding server client ----------
class EmbedError(Exception):
    pass


def embed_server(texts):
    """Embed text or multimodal items through the local server. Tries OpenAI-style /v1/embeddings,
    then Ollama's /api/embed. If an image-bearing payload is rejected, it falls back to text-only."""
    headers = {"Content-Type": "application/json"}
    if EMBED_API_KEY:
        headers["Authorization"] = f"Bearer {EMBED_API_KEY}"

    def normalize_multimodal(payload):
        if isinstance(payload, str):
            return payload
        if not isinstance(payload, list):
            return payload
        out = []
        for item in payload:
            if isinstance(item, dict) and "text" in item and "image" in item:
                out.append({
                    "content": [
                        {"type": "image_url", "image_url": {"url": item["image"]}},
                        {"type": "text", "text": item["text"]},
                    ]
                })
            elif isinstance(item, dict) and "content" in item:
                out.append(item)
            elif isinstance(item, str):
                out.append(item)
            else:
                out.append({"content": [{"type": "text", "text": str(item)}]})
        return out

    def has_multimodal_content(payload):
        if not isinstance(payload, list):
            return False
        for item in payload:
            if not isinstance(item, dict):
                continue
            if "image" in item:
                return True
            content = item.get("content")
            if isinstance(content, list) and any(
                isinstance(block, dict) and block.get("type") == "image_url" and isinstance(block.get("image_url"), dict)
                for block in content
            ):
                return True
        return False

    def send(payload):
        try:
            final_payload = payload
            if has_multimodal_content(final_payload):
                final_payload = normalize_multimodal(final_payload)
            r = requests.post(f"{EMBED_URL}/v1/embeddings", headers=headers, timeout=300,
                              json={"model": EMBED_MODEL, "input": final_payload})
            if r.status_code == 404:
                r = requests.post(f"{EMBED_URL}/api/embed", headers=headers, timeout=300,
                                  json={"model": EMBED_MODEL, "input": payload})
            r.raise_for_status()
            body = r.json()
            if "data" in body:
                vecs = [d["embedding"] for d in sorted(body["data"], key=lambda d: d.get("index", 0))]
            else:
                vecs = body["embeddings"]
            return np.asarray(vecs, dtype=np.float32)
        except requests.ConnectionError:
            raise EmbedError(f"Could not reach the embedding server at {EMBED_URL}. "
                             "From inside Docker use http://host.docker.internal:9931, not localhost.")
        except requests.HTTPError as e:
            raise EmbedError(f"Embedding server returned an error: {e.response.status_code} {e.response.text[:200]}")
        except (requests.RequestException, KeyError, ValueError) as e:
            raise EmbedError(f"Unexpected response from the embedding server: {e}")

    payload = texts
    try:
        arr = send(payload)
    except EmbedError as exc:
        if has_multimodal_content(payload):
            fallback = [{"content": [{"type": "text", "text": item["text"]}]} if isinstance(item, dict) and "text" in item else item for item in payload]
            arr = send(fallback)
        else:
            raise exc

    if EMBED_DIM and arr.shape[1] > EMBED_DIM:
        arr = arr[:, :EMBED_DIM]
    arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return arr


def embed_batches(texts, progress=None, desc="Embedding"):
    out = []
    total = max(len(texts), 1)
    for s in range(0, len(texts), 16):
        if progress:
            pct = min(1.0, (s + 16) / total)
            progress(pct, desc=f"{desc} ({min(s + 16, len(texts))}/{len(texts)})")
        out.append(embed_server(texts[s:s + 16]))
    if progress and total:
        progress(1.0, desc=f"{desc} complete")
    return np.vstack(out) if out else np.zeros((0, EMBED_DIM), np.float32)


def file_or_placeholder(path, fallback=PLACEHOLDER):
    candidate = clean(path)
    if not candidate:
        return fallback
    if os.path.isdir(candidate):
        return fallback
    if os.path.isfile(candidate):
        return candidate
    if os.path.basename(candidate):
        alt = os.path.join(COVER_DIR, os.path.basename(candidate))
        if os.path.isfile(alt):
            return alt
    return fallback


def cover_image_data_url(path):
    real_path = file_or_placeholder(path, fallback="")
    if not real_path:
        return ""
    try:
        with open(real_path, "rb") as fh:
            data = fh.read()
        ext = os.path.splitext(real_path)[1].lower()
        mime = "image/jpeg" if ext in {".jpg", ".jpeg"} else "image/png" if ext == ".png" else "image/webp" if ext == ".webp" else "application/octet-stream"
        return f"data:{mime};base64,{base64.b64encode(data).decode()}"
    except Exception:
        return ""


# EmbeddingGemma 2 task prefixes (the server does not add them)
def doc_text(r):
    title = clean(r.get("title")) or "none"
    parts = []
    if clean(r.get("author")):
        parts.append("by " + clean(r.get("author")))
    if clean(r.get("publisher")):
        parts.append("published by " + clean(r.get("publisher")))
    if clean(r.get("group_name")):
        parts.append("shelf " + clean(r.get("group_name")))
    if clean(r.get("notes")):
        parts.append(clean(r.get("notes"))[:1500])
    content = ". ".join(parts) if parts else title
    return f"title: {title} | text: {content}"


def doc_embedding_input(r):
    text = doc_text(r)
    image = cover_image_data_url(clean(r.get("cover_path") or r.get("image_path")))
    if image:
        return {
            "content": [
                {"type": "image_url", "image_url": {"url": image}},
                {"type": "text", "text": text},
            ]
        }
    return text


def query_text(q):
    return f"task: search result | query: {q}"


# ---------- helpers ----------
def clean(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return str(v).strip()


def norm_isbn(s):
    return re.sub(r"[^0-9Xx]", "", clean(s)).upper()


def row_key(r):
    isbn = norm_isbn(r.get("isbn"))
    if len(isbn) in (10, 13):
        return "isbn:" + isbn
    return "ta:" + clean(r.get("title")).lower() + "|" + clean(r.get("author")).lower()


def save_cover(b64):
    b64 = clean(b64)
    if not b64:
        return ""
    try:
        img = Image.open(io.BytesIO(base64.b64decode(b64.split(",")[-1]))).convert("RGB")
        path = os.path.join(COVER_DIR, f"{uuid.uuid4().hex}.jpg")
        img.save(path)
        return path
    except Exception:
        return ""


def cover_b64(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return ""


def save_group_photo(image):
    if image in (None, "", [], {}):
        return ""
    if isinstance(image, (list, tuple)):
        image = image[0] if image else ""
    src = image if isinstance(image, str) else getattr(image, "name", "")
    if not src or os.path.isdir(src) or not os.path.isfile(src):
        return ""
    try:
        with Image.open(src) as img:
            target = os.path.join(SHELF_PHOTO_DIR, f"{uuid.uuid4().hex}.jpg")
            img.convert("RGB").save(target, quality=90)
            return target
    except Exception:
        return ""


# ---------- database ----------
def db():
    return sqlite3.connect(DB_PATH, timeout=30)


def init_db():
    with closing(db()) as con, con:
        con.execute("""CREATE TABLE IF NOT EXISTS books(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT UNIQUE,
            image_name TEXT, title TEXT NOT NULL, author TEXT, publisher TEXT,
            isbn TEXT, notes TEXT, group_name TEXT, cover_path TEXT, embedding BLOB)""")
        book_cols = {row[1] for row in con.execute("PRAGMA table_info(books)").fetchall()}
        if "group_name" not in book_cols:
            con.execute("ALTER TABLE books ADD COLUMN group_name TEXT")

        con.execute("CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT)")
        con.execute("""CREATE TABLE IF NOT EXISTS import_history(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT NOT NULL,
            imported_at TEXT NOT NULL,
            rows_added INTEGER NOT NULL,
            status TEXT NOT NULL)""")
        import_cols = {row[1] for row in con.execute("PRAGMA table_info(import_history)").fetchall()}
        for col_name, col_type in [
            ("rows_read", "INTEGER"),
            ("no_title", "INTEGER"),
            ("duplicates", "INTEGER"),
            ("embeddings_reused", "INTEGER"),
            ("newly_embedded", "INTEGER"),
            ("covers_failed", "INTEGER"),
        ]:
            if col_name not in import_cols:
                con.execute(f"ALTER TABLE import_history ADD COLUMN {col_name} {col_type}")
        con.execute("""CREATE TABLE IF NOT EXISTS app_state(
            k TEXT PRIMARY KEY,
            v TEXT NOT NULL)""")
        con.execute("""CREATE TABLE IF NOT EXISTS groups(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            image_path TEXT)""")
        group_cols = {row[1] for row in con.execute("PRAGMA table_info(groups)").fetchall()}
        if "image_path" not in group_cols:
            con.execute("ALTER TABLE groups ADD COLUMN image_path TEXT")
        con.execute("""CREATE TABLE IF NOT EXISTS book_corrections(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            book_id INTEGER NOT NULL,
            old_title TEXT,
            old_author TEXT,
            old_publisher TEXT,
            old_isbn TEXT,
            old_notes TEXT,
            old_group_name TEXT,
            new_title TEXT,
            new_author TEXT,
            new_publisher TEXT,
            new_isbn TEXT,
            new_notes TEXT,
            new_group_name TEXT,
            corrected_at TEXT NOT NULL)""")
        con.execute("""CREATE TABLE IF NOT EXISTS book_review_queue(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_file TEXT NOT NULL,
            image_name TEXT,
            title TEXT,
            author TEXT,
            publisher TEXT,
            isbn TEXT,
            notes TEXT,
            group_name TEXT,
            cover_path TEXT,
            image_base64 TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TEXT NOT NULL,
            reviewed_at TEXT)""")

def set_meta(con, model, dim):
    con.execute("INSERT OR REPLACE INTO meta(k,v) VALUES('model',?)", (model,))
    con.execute("INSERT OR REPLACE INTO meta(k,v) VALUES('dim',?)", (str(dim),))


df = pd.DataFrame(columns=["id", "key"] + COLUMNS)
emb = np.zeros((0, EMBED_DIM), np.float32)
STALE = False


def reload():
    """Load the library from SQLite into memory for fast searching."""
    global df, emb, STALE
    with closing(db()) as con:
        rows = con.execute("SELECT id,key,image_name,title,author,publisher,isbn,notes,group_name,cover_path,embedding "
                           "FROM books ORDER BY id").fetchall()
        meta = dict(con.execute("SELECT k,v FROM meta").fetchall())
    df = pd.DataFrame([[x if x is not None else "" for x in r[:-1]] for r in rows],
                      columns=["id", "key"] + COLUMNS)
    vecs = [np.frombuffer(r[-1], dtype=np.float32) if r[-1] else None for r in rows]
    dim = next((len(v) for v in vecs if v is not None), EMBED_DIM)
    emb = (np.vstack([v if v is not None else np.zeros(dim, np.float32) for v in vecs])
           if rows else np.zeros((0, EMBED_DIM), np.float32))
    STALE = bool(rows) and (meta.get("model") != EMBED_MODEL or dim != EMBED_DIM)


def get_group_names():
    with closing(db()) as con:
        rows = con.execute("SELECT name FROM groups ORDER BY name").fetchall()
    return [r[0] for r in rows]


def group_summary_table():
    with closing(db()) as con:
        rows = con.execute(
            "SELECT g.name, COUNT(b.id) FROM groups g LEFT JOIN books b ON b.group_name = g.name GROUP BY g.name ORDER BY g.name"
        ).fetchall()
    if not rows:
        return pd.DataFrame(columns=["Group", "Books"])
    return pd.DataFrame(rows, columns=["Group", "Books"])


def group_photo_gallery():
    with closing(db()) as con:
        rows = con.execute("SELECT name, image_path FROM groups ORDER BY name").fetchall()
    gallery = []
    for name, image_path in rows:
        pic = file_or_placeholder(image_path)
        gallery.append((pic, name))
    return gallery


def add_group(group_name, group_image=None):
    name = clean(group_name)
    image_path = save_group_photo(group_image)
    if not name:
        return ("Please enter a shelf/group name.", group_summary_table(), group_assignments_table(),
                gr.update(choices=get_group_names()), gr.update(value=""), gr.update(choices=get_group_names(), value=None),
                group_photo_gallery())
    with closing(db()) as con, con:
        con.execute(
            "INSERT INTO groups(name, image_path) VALUES(?, ?) ON CONFLICT(name) DO UPDATE SET image_path=excluded.image_path",
            (name, image_path),
        )
    choices = get_group_names()
    return (f"✅ Added shelf/group: {name}", group_summary_table(), group_assignments_table(),
            gr.update(choices=choices, value=name), gr.update(value=""), gr.update(choices=choices, value=None),
            group_photo_gallery())


def delete_empty_group(group_name):
    name = clean(group_name)
    if not name:
        return ("Select a shelf/group to delete.", group_summary_table(), group_assignments_table(),
                gr.update(choices=get_group_names(), value=None), gr.update(choices=get_group_names(), value=None),
                group_photo_gallery())

    with closing(db()) as con:
        book_count = con.execute("SELECT COUNT(*) FROM books WHERE group_name = ?", (name,)).fetchone()[0]
        if book_count > 0:
            return (f"Cannot delete shelf/group '{name}' because {book_count} book(s) are still assigned to it.",
                    group_summary_table(), group_assignments_table(),
                    gr.update(choices=get_group_names(), value=name),
                    gr.update(choices=get_group_names(), value=None),
                    group_photo_gallery())

    with closing(db()) as con, con:
        con.execute("DELETE FROM groups WHERE name = ?", (name,))
    reload()
    choices = get_group_names()
    return (f"✅ Deleted empty shelf/group: {name}", group_summary_table(), group_assignments_table(),
            gr.update(choices=choices, value=None), gr.update(choices=choices, value=None), group_photo_gallery())


def group_assignments_table():
    if len(df) == 0:
        return pd.DataFrame(columns=["ID", "Title", "Author", "Group"])
    out = df[["id", "title", "author", "group_name"]].copy()
    out.columns = ["ID", "Title", "Author", "Group"]
    return out


def save_group_assignments(rows):
    if rows is None:
        return ("No book assignments to save.", group_summary_table(), group_assignments_table(),
                gr.update(choices=get_group_names()), gr.update(choices=get_group_names(), value=None), group_photo_gallery())
    try:
        updates = []
        frame = pd.DataFrame(rows)
        for _, r in frame.iterrows():
            book_id = r.get("ID") or r.get("id")
            group_name = clean(r.get("Group") or r.get("group_name"))
            if book_id is None or book_id == "":
                continue
            updates.append((group_name, int(book_id)))
        with closing(db()) as con, con:
            for group_name, book_id in updates:
                con.execute("UPDATE books SET group_name=? WHERE id=?", (group_name, book_id))
        reload()
        choices = get_group_names()
        return ("✅ Updated group assignments.", group_summary_table(), group_assignments_table(),
                gr.update(choices=choices), gr.update(choices=choices, value=None), group_photo_gallery())
    except Exception as e:
        choices = get_group_names()
        return (f"❌ Could not save groups: {e}", group_summary_table(), group_assignments_table(),
                gr.update(choices=choices), gr.update(choices=choices, value=None), group_photo_gallery())


def assign_selected_books_to_group(selected_ids, group_name):
    ids = []
    for value in selected_ids or []:
        if value is None:
            continue
        if isinstance(value, int):
            ids.append(value)
            continue
        text = clean(value)
        if not text:
            continue
        cell = re.match(r"(\d+)", text)
        if cell:
            ids.append(int(cell.group(1)))
        elif text.isdigit():
            ids.append(int(text))

    ids = sorted(set(int(v) for v in ids))
    name = clean(group_name)
    if not ids:
        return ("Select at least one book from the search results before assigning it to a shelf.",
                group_summary_table(), group_assignments_table(), gr.update(choices=get_group_names()),
                gr.update(value=[]), gr.update(choices=get_group_names(), value=None), group_photo_gallery())
    if not name:
        return ("Choose a shelf/group first, then assign the selected books.",
                group_summary_table(), group_assignments_table(), gr.update(choices=get_group_names()),
                gr.update(value=[]), gr.update(choices=get_group_names(), value=None), group_photo_gallery())

    with closing(db()) as con, con:
        con.execute("INSERT OR IGNORE INTO groups(name) VALUES(?)", (name,))
        for book_id in ids:
            con.execute("UPDATE books SET group_name=? WHERE id=?", (name, int(book_id)))
    reload()
    choices = get_group_names()
    return (f"✅ Assigned {len(ids)} book(s) to shelf/group: {name}",
            group_summary_table(), group_assignments_table(),
            gr.update(choices=choices, value=name), gr.update(value=[]), gr.update(choices=choices, value=None),
            group_photo_gallery())


def import_history_table():
    with closing(db()) as con:
        cols = [row[1] for row in con.execute("PRAGMA table_info(import_history)").fetchall()]
        if all(col not in cols for col in ["rows_read", "no_title", "duplicates", "embeddings_reused", "newly_embedded", "covers_failed"]):
            rows = con.execute(
                "SELECT file_name, imported_at, rows_added, status FROM import_history ORDER BY imported_at DESC LIMIT 50"
            ).fetchall()
            if not rows:
                return pd.DataFrame(columns=["File", "Imported at", "Rows added", "Status"])
            return pd.DataFrame(rows, columns=["File", "Imported at", "Rows added", "Status"])

        rows = con.execute(
            "SELECT file_name, imported_at, rows_added, status, COALESCE(rows_read,0), COALESCE(no_title,0), "
            "COALESCE(duplicates,0), COALESCE(embeddings_reused,0), COALESCE(newly_embedded,0), COALESCE(covers_failed,0) "
            "FROM import_history ORDER BY imported_at DESC LIMIT 50"
        ).fetchall()
    if not rows:
        return pd.DataFrame(columns=["File", "Imported at", "Rows added", "Status", "Rows read", "No title", "Duplicates",
                                     "Embeddings reused", "Newly embedded", "Covers failed"])
    return pd.DataFrame(rows, columns=["File", "Imported at", "Rows added", "Status", "Rows read", "No title", "Duplicates",
                                       "Embeddings reused", "Newly embedded", "Covers failed"])


def human_review_queue_table():
    with closing(db()) as con:
        rows = con.execute(
            "SELECT id, image_name, title, author, publisher, isbn, notes, group_name, source_file, created_at "
            "FROM book_review_queue WHERE status='pending' ORDER BY created_at DESC"
        ).fetchall()
    if not rows:
        return pd.DataFrame(columns=["ID", "Image", "Title", "Author", "Publisher", "ISBN", "Notes", "Shelf", "Source", "Created"])
    return pd.DataFrame(rows, columns=["ID", "Image", "Title", "Author", "Publisher", "ISBN", "Notes", "Shelf", "Source", "Created"])


def human_review_choices():
    with closing(db()) as con:
        rows = con.execute("SELECT id FROM book_review_queue WHERE status='pending' ORDER BY created_at DESC").fetchall()
    return [str(row[0]) for row in rows]


def human_review_item(review_id):
    if review_id in (None, ""):
        return None, "", "", "", "", "", "", ""
    try:
        review_id = int(review_id)
    except (TypeError, ValueError):
        return None, "", "", "", "", "", "", ""
    with closing(db()) as con:
        row = con.execute(
            "SELECT id, image_name, title, author, publisher, isbn, notes, group_name, cover_path, source_file FROM book_review_queue WHERE id=? AND status='pending'",
            (review_id,),
        ).fetchone()
    if not row:
        return None, "", "", "", "", "", "", ""
    image_value = file_or_placeholder(row[8] or row[1], fallback=None)
    return (image_value, row[2] or "", row[3] or "", row[4] or "", row[5] or "", row[6] or "", row[7] or "", row[9] or "")


def save_human_review(review_id, title, author, publisher, isbn, notes, group_name):
    review_id = clean(review_id)
    title = clean(title)
    if not review_id:
        return ("Select a pending row to review first.", human_review_queue_table(), gr.update(choices=human_review_choices(), value=None))
    if not title:
        return ("A title is required before the item can be added to the library.", human_review_queue_table(), gr.update(choices=human_review_choices(), value=None))
    try:
        review_id_int = int(review_id)
    except ValueError:
        return ("The review item is invalid.", human_review_queue_table(), gr.update(choices=human_review_choices(), value=None))

    with closing(db()) as con, con:
        item = con.execute(
            "SELECT image_name, title, author, publisher, isbn, notes, group_name, cover_path, source_file, image_base64 FROM book_review_queue WHERE id=? AND status='pending'",
            (review_id_int,),
        ).fetchone()
        if item is None:
            return ("That review item is no longer pending.", human_review_queue_table(), gr.update(choices=human_review_choices(), value=None))
        image_name, _, author_old, publisher_old, isbn_old, notes_old, group_name_old, cover_path, source_file, image_base64 = item
        row = {
            "image_name": clean(image_name),
            "title": title,
            "author": clean(author),
            "publisher": clean(publisher),
            "isbn": clean(isbn),
            "notes": clean(notes),
            "group_name": clean(group_name),
            "cover_path": clean(cover_path),
        }
        if row["cover_path"]:
            row["cover_path"] = row["cover_path"]
        elif clean(image_base64):
            row["cover_path"] = save_cover(image_base64)
        key = row_key(row)
        con.execute(
            "INSERT OR IGNORE INTO books(key,image_name,title,author,publisher,isbn,notes,group_name,cover_path,embedding) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (key, row["image_name"], row["title"], row["author"], row["publisher"], row["isbn"], row["notes"], row["group_name"], row["cover_path"], b""),
        )
        con.execute("UPDATE book_review_queue SET status='done', reviewed_at=? WHERE id=?", (time.strftime("%Y-%m-%d %H:%M:%S"), review_id_int))
    reload()
    return (
        f"✅ Saved reviewed item #{review_id_int} to the library from {source_file}.",
        human_review_queue_table(),
        gr.update(choices=human_review_choices(), value=None),
    )


def get_app_state(key, default=""):
    with closing(db()) as con:
        value = con.execute("SELECT v FROM app_state WHERE k=?", (key,)).fetchone()
    return value[0] if value else default


def set_app_state(key, value):
    with closing(db()) as con, con:
        con.execute("INSERT INTO app_state(k,v) VALUES(?, ?) ON CONFLICT(k) DO UPDATE SET v=excluded.v",
                    (key, value))


def import_help_text():
    default = (
        "Columns: `image_name, title, author, publisher, isbn, notes, image_base64`. "
        "Files exported from this app also carry `embedding` and `embedding_model`; those books are "
        "restored without calling the embedding server. Rows without a title are skipped and duplicates "
        "(same ISBN, or title+author) are ignored."
    )
    with closing(db()) as con, con:
        con.execute(
            "INSERT OR IGNORE INTO app_state(k, v) VALUES(?, ?)",
            ("import_help_text", default),
        )
    return get_app_state("import_help_text", default)


init_db()
reload()


# ---------- hybrid search ----------
def parse_query(q):
    phrases = [p.strip().lower() for p in re.findall(r'"([^"]+)"', q) if p.strip()]
    rest = re.sub(r'"[^"]*"', " ", q).lower()
    words = [w for w in re.findall(r"[\w'-]+", rest) if w not in STOP]
    return phrases + words


def keyword_scores(query):
    """Whole-word, case-insensitive matching. Returns (score 0-1, row contains every term)."""
    n = len(df)
    terms = parse_query(query)
    if not terms or n == 0:
        return np.zeros(n), np.zeros(n, dtype=bool)

    low = {f: df[f].str.lower().tolist() for f in FIELD_WEIGHTS}
    kw = np.zeros(n)
    hit_all = np.ones(n, dtype=bool)
    for t in terms:
        rx = re.compile(r"(?<!\w)" + re.escape(t) + r"(?!\w)")
        for i in range(n):
            best = 0.0
            for f, w in FIELD_WEIGHTS.items():
                if w > best and rx.search(low[f][i]):
                    best = w
            kw[i] += best / MAXW
            if best == 0:
                hit_all[i] = False
    kw /= len(terms)

    q_isbn = norm_isbn(query) if re.fullmatch(r"[0-9Xx\-\s]+", query.strip()) else ""
    if len(q_isbn) in (10, 13):
        for i, v in enumerate(df["isbn"].map(norm_isbn)):
            if v == q_isbn:
                kw[i], hit_all[i] = 1.0, True
    return kw, hit_all


TABLE_HEADERS = ["Title", "Author", "Publisher", "ISBN", "Semantic", "Keyword"]


def search(query, mode, sem_weight, k):
    if len(df) == 0:
        return [], [], "Library is empty. Import some CSVs first.", gr.update(choices=[], value=[]), gr.update(choices=get_group_names())
    if not query or not query.strip():
        return [], [], "", gr.update(choices=[], value=[]), gr.update(choices=get_group_names())
    k = int(k)
    kw, hit_all = keyword_scores(query)

    note, sem = "", None
    if mode != "Exact words only":
        if STALE:
            note = ("⚠️ The stored embeddings were made with a different model or size. "
                    "Click “Rebuild all embeddings” on the Library tab. Showing keyword matches only.")
        else:
            try:
                sem = emb @ embed_server([query_text(query)])[0]
            except EmbedError as e:
                note = f"⚠️ {e} Showing keyword matches only."

    if sem is not None:
        sem_n = (sem - sem.min()) / (sem.max() - sem.min() + 1e-9)
    else:
        sem, sem_n = np.zeros(len(df)), np.zeros(len(df))

    if mode == "Exact words only":
        score = np.where(hit_all, kw + 0.01 * sem_n, -np.inf)
    elif note:  # semantic part unavailable
        score = np.where(kw > 0, kw, -np.inf)
    elif mode == "Semantic only":
        score = sem_n
    else:
        score = sem_weight * sem_n + (1 - sem_weight) * kw

    order = [int(i) for i in np.argsort(-score) if np.isfinite(score[i])][:k]
    if not order:
        return [], [], note or "No book contains all of those words. Try Hybrid mode.", gr.update(choices=[], value=[]), gr.update(choices=get_group_names())

    gallery, table, choices, selected_ids = [], [], [], []
    for i in order:
        r = df.iloc[i]
        path = file_or_placeholder(r["cover_path"])
        title = clean(r["title"]) or "Untitled"
        author = clean(r["author"])
        shelf = clean(r["group_name"])
        label = f"{int(r['id'])}: {title}"
        if author:
            label += f" — {author}"
        if shelf:
            label += f" | Shelf: {shelf}"
        gallery.append((path, label))
        table.append([title, author, r["publisher"], r["isbn"],
                      round(float(sem[i]), 3), round(float(kw[i]), 2)])
        selected_ids.append(int(r["id"]))
        choices.append(label)
    return gallery, table, note or f"{len(order)} results", gr.update(choices=choices, value=[]), gr.update(choices=get_group_names(), value=None)


# ---------- import ----------
def import_csvs(files, progress=gr.Progress()):
    if not files:
        return "Choose one or more CSV files first.", import_history_table()
    if STALE:
        return ("⚠️ The library's embeddings were made with a different model or size. "
                "Click “Rebuild all embeddings” on the Library tab first.", import_history_table())

    with lock:
        seen = set(df["key"])
        new_rows, log = [], []
        stats = {"rows": 0, "no_title": 0, "dupes": 0, "reused": 0}
        import_records = []

        for f in files:
            path = f if isinstance(f, str) else f.name
            name = os.path.basename(path)
            file_stats = {"rows": 0, "no_title": 0, "dupes": 0, "reused": 0, "newly_embedded": 0, "covers_failed": 0}
            try:
                raw = pd.read_csv(path, dtype=str, keep_default_na=False, encoding_errors="replace")
            except Exception as e:
                log.append(f"❌ {name}: could not read ({e})")
                import_records.append((name, time.strftime("%Y-%m-%d %H:%M:%S"), 0, "failed", 0, 0, 0, 0, 0, 0))
                continue
            raw.columns = [c.strip().lower() for c in raw.columns]
            added = 0
            for rec in raw.to_dict("records"):
                file_stats["rows"] += 1
                stats["rows"] += 1
                row = {c: clean(rec.get(c)) for c in ["image_name", "title", "author", "publisher", "isbn", "notes", "group_name"]}
                b64 = clean(rec.get("image_base64"))
                if not row.get("title"):
                    file_stats["no_title"] += 1
                    stats["no_title"] += 1
                    cover_path = save_cover(b64)
                    with closing(db()) as con, con:
                        con.execute(
                            "INSERT INTO book_review_queue(source_file, image_name, title, author, publisher, isbn, notes, group_name, cover_path, image_base64, status, created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                            (name, row.get("image_name"), row.get("title"), row.get("author"), row.get("publisher"), row.get("isbn"), row.get("notes"), row.get("group_name"), cover_path, b64, "pending", time.strftime("%Y-%m-%d %H:%M:%S")),
                        )
                    continue
                row["key"] = row_key(row)
                if row["key"] in seen:
                    file_stats["dupes"] += 1
                    stats["dupes"] += 1
                    continue
                seen.add(row["key"])
                row["_b64"] = b64

                # reuse a stored embedding if it came from the same model and size
                row["_vec"] = None
                if clean(rec.get("embedding")) and clean(rec.get("embedding_model")) == EMBED_MODEL:
                    try:
                        v = np.asarray(json.loads(rec["embedding"]), dtype=np.float32)
                        if v.shape == (EMBED_DIM,):
                            row["_vec"] = v / (np.linalg.norm(v) + 1e-12)
                            file_stats["reused"] += 1
                            stats["reused"] += 1
                    except Exception:
                        pass
                new_rows.append(row)
                added += 1
            log.append(f"✅ {name}: {added} new books")
            file_stats["newly_embedded"] = added
            import_records.append((name, time.strftime("%Y-%m-%d %H:%M:%S"), added, "ok",
                                  file_stats["rows"], file_stats["no_title"], file_stats["dupes"],
                                  file_stats["reused"], file_stats["newly_embedded"], file_stats["covers_failed"]))

        if not new_rows:
            with closing(db()) as con, con:
                con.executemany(
                    "INSERT INTO import_history(file_name, imported_at, rows_added, status, rows_read, no_title, duplicates, embeddings_reused, newly_embedded, covers_failed) VALUES(?,?,?,?,?,?,?,?,?,?)",
                    import_records,
                )
            return "\n\n".join(log) + "\n\nNothing new to add.", import_history_table()

        bad_cover = 0
        for r in new_rows:
            r["cover_path"] = save_cover(r["_b64"])
            if r["_b64"] and not r["cover_path"]:
                bad_cover += 1

        todo = [r for r in new_rows if r["_vec"] is None]
        try:
            vecs = embed_batches([doc_embedding_input(r) for r in todo], progress, "Embedding books")
        except EmbedError as e:
            with closing(db()) as con, con:
                con.executemany(
                    "INSERT INTO import_history(file_name, imported_at, rows_added, status, rows_read, no_title, duplicates, embeddings_reused, newly_embedded, covers_failed) VALUES(?,?,?,?,?,?,?,?,?,?)",
                    import_records,
                )
            return f"❌ {e}\n\nNothing was imported.", import_history_table()
        for r, v in zip(todo, vecs):
            r["_vec"] = v

        with closing(db()) as con, con:
            for r in new_rows:
                con.execute(
                    "INSERT OR IGNORE INTO books(key,image_name,title,author,publisher,isbn,notes,group_name,cover_path,embedding)"
                    " VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (r["key"], r["image_name"], r["title"], r["author"], r["publisher"], r["isbn"],
                     r["notes"], r.get("group_name", ""), r["cover_path"], r["_vec"].astype(np.float32).tobytes()))
            con.executemany(
                "INSERT INTO import_history(file_name, imported_at, rows_added, status, rows_read, no_title, duplicates, embeddings_reused, newly_embedded, covers_failed) VALUES(?,?,?,?,?,?,?,?,?,?)",
                import_records,
            )
            set_meta(con, EMBED_MODEL, EMBED_DIM)
        reload()

    return (f"**Imported {len(new_rows)} books. Library now has {len(df)}.**\n\n"
            f"Rows read: {stats['rows']} · no title: {stats['no_title']} · duplicates: {stats['dupes']} · "
            f"embeddings reused from CSV: {stats['reused']} · newly embedded: {len(todo)} · "
            f"covers that failed to decode: {bad_cover}\n\n" + "\n\n".join(log), import_history_table())


# ---------- library tools ----------
def test_server():
    try:
        v = embed_server([query_text("test")])
        return f"✅ Connected to `{EMBED_URL}` · model `{EMBED_MODEL}` · vector size {v.shape[1]}"
    except EmbedError as e:
        return f"❌ {e}"


def rebuild_embeddings(progress=gr.Progress()):
    with lock:
        if len(df) == 0:
            return "Library is empty."
        try:
            vecs = embed_batches([doc_embedding_input(r) for r in df.to_dict("records")], progress, "Re-embedding")
        except EmbedError as e:
            return f"❌ {e}"
        with closing(db()) as con, con:
            con.executemany("UPDATE books SET embedding=? WHERE id=?",
                            [(v.astype(np.float32).tobytes(), int(i)) for v, i in zip(vecs, df["id"])])
            set_meta(con, EMBED_MODEL, vecs.shape[1])
        reload()
    return f"✅ Re-embedded {len(df)} books with `{EMBED_MODEL}`."


def export_csv(include_covers):
    if len(df) == 0:
        return None, "Library is empty."
    out = pd.DataFrame({
        "image_name": df["image_name"], "title": df["title"], "author": df["author"],
        "publisher": df["publisher"], "isbn": df["isbn"], "notes": df["notes"], "group_name": df["group_name"],
    })
    if include_covers:
        out["image_base64"] = [cover_b64(p) if p else "" for p in df["cover_path"]]
    out["embedding_model"] = EMBED_MODEL
    out["embedding"] = [json.dumps(np.round(v, 5).tolist()) for v in emb]
    path = os.path.join(EXPORT_DIR, f"books_export_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    out.to_csv(path, index=False)
    return path, f"✅ Exported {len(out)} books. Import this file on the Import tab to restore them without re-embedding."


def library_view():
    status = (f"**{len(df)} books** · server `{EMBED_URL}` · model `{EMBED_MODEL}` · size {EMBED_DIM}"
              + (" · ⚠️ stored embeddings are out of date, rebuild them" if STALE else ""))
    table = df[["title", "author", "publisher", "isbn", "group_name", "notes"]].copy()
    table.columns = ["Title", "Author", "Publisher", "ISBN", "Group", "Notes"]
    return status, table


def library_cards():
    if len(df) == 0:
        return []
    cards = []
    for _, r in df.iterrows():
        path = file_or_placeholder(r["cover_path"])
        title = clean(r["title"]) or "Untitled"
        author = clean(r["author"]) or "Unknown author"
        publisher = clean(r["publisher"]) or "Unknown publisher"
        isbn = clean(r["isbn"]) or "No ISBN"
        shelf = clean(r["group_name"])
        notes = clean(r["notes"])[:180]
        caption = f"{title}\n{author}\n{publisher}\nISBN: {isbn}"
        if shelf:
            caption += f"\nShelf: {shelf}"
        if notes:
            caption += f"\nNotes: {notes}"
        cards.append((path, caption))
    return cards


def library_details_view(cols=4):
    cols = max(1, int(cols))
    if len(df) == 0:
        return "**0 books**", gr.update(value=[], columns=cols), []

    gallery = []
    for _, r in df.iterrows():
        path = file_or_placeholder(r["cover_path"])
        title = clean(r["title"]) or "Untitled"
        author = clean(r["author"])
        shelf = clean(r["group_name"])
        label = title + (f" — {author}" if author else "")
        if shelf:
            label += f" | Shelf: {shelf}"
        gallery.append((path, label))

    details = df[["title", "author", "publisher", "isbn", "group_name", "notes"]].copy()
    details.columns = ["Title", "Author", "Publisher", "ISBN", "Group", "Notes"]
    status = (f"**{len(df)} books** · server `{EMBED_URL}` · model `{EMBED_MODEL}` · size {EMBED_DIM}"
              + (" · ⚠️ stored embeddings are out of date, rebuild them" if STALE else ""))
    return status, gr.update(value=gallery, columns=cols), details


def book_edit_choices():
    with closing(db()) as con:
        rows = con.execute("SELECT id, title FROM books ORDER BY title, id").fetchall()
    return [str(row[0]) for row in rows]


def book_for_edit(book_id):
    if book_id in (None, ""):
        return "", "", "", "", "", ""
    try:
        book_id = int(book_id)
    except (TypeError, ValueError):
        return "", "", "", "", "", ""
    with closing(db()) as con:
        row = con.execute(
            "SELECT title, author, publisher, isbn, notes, group_name FROM books WHERE id=?",
            (book_id,),
        ).fetchone()
    if not row:
        return "", "", "", "", "", ""
    return row[0], row[1], row[2], row[3], row[4], row[5]


def update_book_details(book_id, title, author, publisher, isbn, notes, group_name):
    book_id = clean(book_id)
    title = clean(title)
    author = clean(author)
    publisher = clean(publisher)
    isbn = clean(isbn)
    notes = clean(notes)
    group_name = clean(group_name)

    if not book_id:
        return ("Select a book first.", gr.update(choices=book_edit_choices()), gr.update(choices=get_group_names(), value=None))

    try:
        book_id_int = int(book_id)
    except ValueError:
        return ("Book ID is invalid.", gr.update(choices=book_edit_choices()), gr.update(choices=get_group_names(), value=None))

    if not title:
        return ("A title is required for the corrected record.", gr.update(choices=book_edit_choices()), gr.update(choices=get_group_names(), value=None))

    with closing(db()) as con, con:
        current = con.execute(
            "SELECT title, author, publisher, isbn, notes, group_name FROM books WHERE id=?",
            (book_id_int,),
        ).fetchone()
        if current is None:
            return ("Book not found.", gr.update(choices=book_edit_choices()), gr.update(choices=get_group_names(), value=None))

        old_title, old_author, old_publisher, old_isbn, old_notes, old_group_name = current
        if group_name:
            con.execute("INSERT OR IGNORE INTO groups(name) VALUES(?)", (group_name,))
        con.execute(
            "UPDATE books SET title=?, author=?, publisher=?, isbn=?, notes=?, group_name=? WHERE id=?",
            (title, author, publisher, isbn, notes, group_name, book_id_int),
        )
        con.execute(
            "INSERT INTO book_corrections(book_id, old_title, old_author, old_publisher, old_isbn, old_notes, old_group_name, "
            "new_title, new_author, new_publisher, new_isbn, new_notes, new_group_name, corrected_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                book_id_int,
                old_title,
                old_author,
                old_publisher,
                old_isbn,
                old_notes,
                old_group_name,
                title,
                author,
                publisher,
                isbn,
                notes,
                group_name,
                time.strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
    reload()
    return (
        f"✅ Updated book #{book_id_int} and saved the correction record separately.",
        gr.update(choices=book_edit_choices(), value=str(book_id_int)),
        gr.update(choices=get_group_names(), value=group_name or None),
    )


def library_refresh():
    status, table = library_view()
    detail_status, gallery, details = library_details_view()
    return status, table, detail_status, gallery, details


# ---------- UI ----------
with gr.Blocks(title="Book Search") as demo:
    gr.Markdown("# Book Search")

    gr.Markdown(
        """
        ## Search
        Try natural-language searches like:
        - books with dino
        - books about body
        - board books
        - math books
        - level 1 i can read
        - puppet book

        Put words in quotes like `\"pete the cat\"` to require an exact phrase.
        """
    )
    with gr.Row():
        query = gr.Textbox(label="Search", placeholder="title, author, publisher, ISBN or a description", scale=4)
        btn = gr.Button("Search", variant="primary", scale=1)
    with gr.Row():
        mode = gr.Radio(["Hybrid", "Exact words only", "Semantic only"], value="Hybrid", label="Mode")
        sem_weight = gr.Slider(0, 1, value=0.5, step=0.05, label="Hybrid: semantic weight (0 = keywords only)")
        k = gr.Slider(1, 100, value=20, step=1, label="Results")
    msg = gr.Markdown()
    gallery = gr.Gallery(columns=3, height="auto", object_fit="contain", label="Results")
    table = gr.Dataframe(headers=TABLE_HEADERS, interactive=False, label="Details")

    with gr.Tab("Groups / Shelves"):
        group_message = gr.Markdown()
        groups_list = gr.Dataframe(interactive=False, label="Existing groups")
        group_gallery = gr.Gallery(label="Shelf photos", columns=3, height=220, object_fit="cover")
        assignment_table = gr.Dataframe(headers=["ID", "Title", "Author", "Group"], interactive=True,
                                       label="Assign books to shelves")
        search_selection = gr.CheckboxGroup(label="Mark books to add to a shelf", choices=[], value=[], info="Search first, then tick the books to assign.")
        search_shelf = gr.Dropdown(choices=get_group_names(), label="Shelf/group", value=None)
        search_assign_msg = gr.Markdown()
        assign_selected_btn = gr.Button("Add selected books to shelf", variant="primary")

        with gr.Row():
            new_group_name = gr.Textbox(label="Create shelf / group", placeholder="Enter a shelf name", scale=3)
            group_image = gr.Image(label="Shelf photo (optional)", type="filepath", scale=2)
            add_group_btn = gr.Button("Create shelf", variant="primary", scale=1)

        with gr.Row():
            delete_group_name = gr.Dropdown(choices=get_group_names(), label="Delete empty shelf / group", value=None, scale=3)
            delete_group_btn = gr.Button("Delete empty shelf", variant="secondary", scale=1)

        save_groups_btn = gr.Button("Save group assignments", variant="primary")

        add_group_btn.click(add_group, [new_group_name, group_image],
                            [group_message, groups_list, assignment_table, search_shelf, new_group_name, delete_group_name,
                             group_gallery])
        delete_group_btn.click(delete_empty_group, delete_group_name,
                              [group_message, groups_list, assignment_table, search_shelf, delete_group_name, group_gallery])
        save_groups_btn.click(save_group_assignments, assignment_table,
                             [group_message, groups_list, assignment_table, search_shelf, delete_group_name, group_gallery])
        assign_selected_btn.click(assign_selected_books_to_group, [search_selection, search_shelf],
                                  [search_assign_msg, groups_list, assignment_table, search_shelf, search_selection, group_gallery])
        demo.load(lambda: (group_summary_table(), group_assignments_table(), group_photo_gallery()), None,
                  [groups_list, assignment_table, group_gallery])
        demo.load(lambda: "", None, new_group_name)

    for trigger in (btn.click, query.submit):
        trigger(search, [query, mode, sem_weight, k], [gallery, table, msg, search_selection, search_shelf])

    with gr.Tab("Import CSVs"):
        import_help = gr.Markdown(value=import_help_text())
        files = gr.File(label="CSV files", file_count="multiple", file_types=[".csv"])
        import_btn = gr.Button("Import", variant="primary")
        import_out = gr.Markdown()
        import_history = gr.Dataframe(interactive=False, label="Previously imported CSV files")
        import_btn.click(import_csvs, files, [import_out, import_history], show_progress="full")
        demo.load(import_history_table, None, import_history)
        demo.load(lambda: import_help_text(), None, import_help)
        import_btn.click(lambda: import_help_text(), None, import_help)

    with gr.Tab("Human review"):
        review_queue = gr.Dataframe(interactive=False, label="Items needing a title and review")
        review_select = gr.Dropdown(choices=human_review_choices(), label="Select item to review", value=None)
        review_image = gr.Image(label="Cover image", type="filepath")
        review_title = gr.Textbox(label="Title")
        review_author = gr.Textbox(label="Author")
        review_publisher = gr.Textbox(label="Publisher")
        review_isbn = gr.Textbox(label="ISBN")
        review_notes = gr.Textbox(label="Notes", lines=3)
        review_group = gr.Dropdown(choices=get_group_names(), label="Shelf / group", value=None)
        review_msg = gr.Markdown("No pending items yet. Import a CSV with rows missing a title to populate this queue.")
        review_load = gr.Button("Load selected review item")
        review_save = gr.Button("Save reviewed book", variant="primary")

        demo.load(human_review_queue_table, None, review_queue)
        demo.load(lambda: gr.update(choices=human_review_choices(), value=None), None, review_select)
        review_load.click(human_review_item, review_select,
                          [review_image, review_title, review_author, review_publisher, review_isbn, review_notes, review_group, review_select])
        review_select.change(human_review_item, review_select,
                             [review_image, review_title, review_author, review_publisher, review_isbn, review_notes, review_group, review_select])
        review_save.click(save_human_review,
                          [review_select, review_title, review_author, review_publisher, review_isbn, review_notes, review_group],
                          [review_msg, review_queue, review_select])
        review_save.click(lambda: human_review_queue_table(), None, review_queue)

    with gr.Tab("Library"):
        lib_status = gr.Markdown()
        with gr.Row():
            test_btn = gr.Button("Test embedding server")
            refresh = gr.Button("Refresh")
            rebuild = gr.Button("Rebuild all embeddings")
        tool_out = gr.Markdown()
        lib_table = gr.Dataframe(interactive=False)
        gr.Markdown("### Export to CSV")
        with gr.Row():
            with_covers = gr.Checkbox(value=True, label="Include cover images (larger file)")
            export_btn = gr.Button("Export CSV with embeddings", variant="primary")
        export_out = gr.Markdown()
        export_file = gr.File(label="Download")

        test_btn.click(test_server, None, tool_out)
        refresh.click(library_view, None, [lib_status, lib_table])
        rebuild.click(rebuild_embeddings, None, tool_out, show_progress="full").then(library_view, None, [lib_status, lib_table])
        export_btn.click(export_csv, with_covers, [export_file, export_out])

    with gr.Tab("Library details"):
        details_status = gr.Markdown()
        detail_cols = gr.Slider(1, 8, value=4, step=1, label="Columns")
        details_gallery = gr.Gallery(columns=4, height="auto", object_fit="contain",
                                    label="Books with cover and metadata")
        details_table = gr.Dataframe(interactive=False, label="Book details")
        refresh_details = gr.Button("Refresh library details")
        refresh_details.click(library_details_view, detail_cols, [details_status, details_gallery, details_table])
        detail_cols.change(library_details_view, detail_cols, [details_status, details_gallery, details_table])

        edit_book_id = gr.Dropdown(choices=book_edit_choices(), label="Select book to correct", value=None)
        edit_title = gr.Textbox(label="Correct title")
        edit_author = gr.Textbox(label="Correct author")
        edit_publisher = gr.Textbox(label="Correct publisher")
        edit_isbn = gr.Textbox(label="Correct ISBN")
        edit_notes = gr.Textbox(label="Correct notes", lines=3)
        edit_group = gr.Dropdown(choices=get_group_names(), label="Correct shelf/group", value=None)
        save_book_edit_btn = gr.Button("Save corrected details", variant="primary")
        book_edit_msg = gr.Markdown()

        edit_book_id.change(book_for_edit, edit_book_id,
                            [edit_title, edit_author, edit_publisher, edit_isbn, edit_notes, edit_group])
        save_book_edit_btn.click(update_book_details,
                                [edit_book_id, edit_title, edit_author, edit_publisher, edit_isbn, edit_notes, edit_group],
                                [book_edit_msg, edit_book_id, edit_group])
        save_book_edit_btn.click(library_details_view, detail_cols, [details_status, details_gallery, details_table])

    demo.load(library_view, None, [lib_status, lib_table])
    demo.load(library_details_view, detail_cols, [details_status, details_gallery, details_table])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0",
                server_port=int(os.environ.get("BOOKS_PORT", "7861")),
                allowed_paths=[DATA_DIR])