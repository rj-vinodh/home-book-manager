import os
import sqlite3

import numpy as np
import pandas as pd
from PIL import Image

from app import toggle_selection
import books_app


class DummyEvent:
    def __init__(self, index):
        self.index = index


def test_toggle_selection_without_pagination():
    items = [{"name": "frame_1.jpg", "thumb": "thumb_1.jpg", "thumb_sel": "thumb_1_sel.jpg"}]
    evt = DummyEvent(0)

    result = toggle_selection(items, [], 3, evt)

    assert result[0] == [0]


def test_assign_selected_books_to_group_updates_db(tmp_path):
    db_path = tmp_path / "books.db"
    books_app.DB_PATH = str(db_path)

    with sqlite3.connect(db_path) as con:
        con.execute(
            "CREATE TABLE books(id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT UNIQUE, image_name TEXT, title TEXT NOT NULL, author TEXT, publisher TEXT, isbn TEXT, notes TEXT, group_name TEXT, cover_path TEXT, embedding BLOB)"
        )
        con.execute(
            "INSERT INTO books(key, image_name, title, author, publisher, isbn, notes, group_name, cover_path, embedding) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("k1", "img1", "Book One", "Author A", "Pub A", "123", "note", "", "", b""),
        )
        con.execute(
            "INSERT INTO books(key, image_name, title, author, publisher, isbn, notes, group_name, cover_path, embedding) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("k2", "img2", "Book Two", "Author B", "Pub B", "456", "note", "", "", b""),
        )
        con.execute("CREATE TABLE IF NOT EXISTS groups(id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL UNIQUE)")
        con.execute("INSERT OR IGNORE INTO groups(name) VALUES(?)", ("Shelf A",))

    books_app.init_db()
    books_app.reload()

    result = books_app.assign_selected_books_to_group([1, 2], "Shelf A")

    assert "✅" in result[0]
    with sqlite3.connect(db_path) as con:
        rows = con.execute("SELECT group_name FROM books ORDER BY id").fetchall()
    assert [row[0] for row in rows] == ["Shelf A", "Shelf A"]


def test_add_group_with_photo_persists_image(tmp_path):
    db_path = tmp_path / "books.db"
    photo_dir = tmp_path / "shelf_photos"
    photo_dir.mkdir()
    books_app.DB_PATH = str(db_path)
    books_app.DATA_DIR = str(tmp_path)
    books_app.COVER_DIR = str(photo_dir)

    image_path = tmp_path / "shelf.png"
    Image.new("RGB", (20, 20), color="blue").save(image_path)

    with sqlite3.connect(db_path) as con:
        con.execute("CREATE TABLE IF NOT EXISTS books(id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT UNIQUE, image_name TEXT, title TEXT NOT NULL, author TEXT, publisher TEXT, isbn TEXT, notes TEXT, group_name TEXT, cover_path TEXT, embedding BLOB)")
        con.execute("CREATE TABLE IF NOT EXISTS groups(id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL UNIQUE, image_path TEXT)")

    books_app.init_db()
    result = books_app.add_group("Shelf B", str(image_path))

    assert "✅" in result[0]
    with sqlite3.connect(db_path) as con:
        row = con.execute("SELECT name, image_path FROM groups WHERE name=?", ("Shelf B",)).fetchone()
    assert row[0] == "Shelf B"
    assert row[1] and os.path.exists(row[1])


def test_keyword_scores_include_group_name():
    books_app.df = pd.DataFrame([
        {
            "id": 1,
            "key": "k1",
            "image_name": "img1",
            "title": "Book One",
            "author": "Author A",
            "publisher": "Pub A",
            "isbn": "123",
            "notes": "note",
            "group_name": "Shelf A",
            "cover_path": "",
        }
    ])

    kw, hit_all = books_app.keyword_scores("Shelf A")

    assert kw[0] > 0
    assert bool(hit_all[0]) is True


def test_embed_server_includes_image_payload(monkeypatch):
    captured = {}

    def fake_post(url, headers=None, timeout=None, json=None):
        captured["json"] = json
        class Resp:
            status_code = 200
            def raise_for_status(self):
                pass
            def json(self):
                return {"data": [{"embedding": [0.1, 0.2], "index": 0}]}
        return Resp()

    monkeypatch.setattr(books_app.requests, "post", fake_post)

    result = books_app.embed_server([{"text": "Book One", "image": "data:image/jpeg;base64,abcd"}])

    assert result.shape == (1, 2)
    assert captured["json"]["input"][0]["content"][0]["type"] == "image_url"
    assert captured["json"]["input"][0]["content"][0]["image_url"]["url"] == "data:image/jpeg;base64,abcd"
    assert captured["json"]["input"][0]["content"][1]["text"] == "Book One"


def test_doc_embedding_input_includes_cover_image(tmp_path):
    image_path = tmp_path / "cover.jpg"
    image_path.write_bytes(b"not-a-real-image")

    item = books_app.doc_embedding_input({"title": "Book One", "cover_path": str(image_path), "author": "Author A"})

    assert isinstance(item, dict)
    assert item["content"][0]["type"] == "image_url"
    assert item["content"][1]["type"] == "text"
    assert item["content"][1]["text"].startswith("title: Book One")
    assert item["content"][0]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_embed_server_falls_back_to_text_only_on_multimodal_error(monkeypatch):
    calls = []

    def fake_post(url, headers=None, timeout=None, json=None):
        calls.append(json["input"])

        class ErrResp:
            status_code = 400
            text = '{"error":{"message":"prompt elements must be a string"}}'
            def raise_for_status(self):
                raise books_app.requests.HTTPError(response=self)

        class OkResp:
            status_code = 200
            def raise_for_status(self):
                pass
            def json(self):
                return {"data": [{"embedding": [0.1, 0.2], "index": 0}]}

        if isinstance(json["input"], list):
            first = json["input"][0]
            if isinstance(first, dict) and "content" in first and any(
                block.get("type") == "image_url" for block in first.get("content", [])
            ):
                return ErrResp()
        return OkResp()

    monkeypatch.setattr(books_app.requests, "post", fake_post)

    result = books_app.embed_server([{"text": "Book One", "image": "data:image/jpeg;base64,abcd"}])

    assert result.shape == (1, 2)
    assert calls[0][0]["content"][0]["image_url"]["url"] == "data:image/jpeg;base64,abcd"
    assert calls[1] == [{"content": [{"type": "text", "text": "Book One"}]}]


def test_delete_empty_group_only_when_zero_books(tmp_path):
    db_path = tmp_path / "books.db"
    books_app.DB_PATH = str(db_path)

    with sqlite3.connect(db_path) as con:
        con.execute(
            "CREATE TABLE books(id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT UNIQUE, image_name TEXT, title TEXT NOT NULL, author TEXT, publisher TEXT, isbn TEXT, notes TEXT, group_name TEXT, cover_path TEXT, embedding BLOB)"
        )
        con.execute(
            "INSERT INTO books(key, image_name, title, author, publisher, isbn, notes, group_name, cover_path, embedding) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("k1", "img1", "Book One", "Author A", "Pub A", "123", "note", "Shelf A", "", b""),
        )
        con.execute("CREATE TABLE IF NOT EXISTS groups(id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL UNIQUE, image_path TEXT)")
        con.execute("INSERT INTO groups(name, image_path) VALUES(?, ?)", ("Shelf A", ""))
        con.execute("INSERT INTO groups(name, image_path) VALUES(?, ?)", ("Empty Shelf", ""))

    books_app.init_db()

    blocked = books_app.delete_empty_group("Shelf A")
    assert "Cannot delete" in blocked[0]

    ok = books_app.delete_empty_group("Empty Shelf")
    assert "✅ Deleted" in ok[0]

    with sqlite3.connect(db_path) as con:
        assert con.execute("SELECT COUNT(*) FROM groups WHERE name='Empty Shelf'").fetchone()[0] == 0
        assert con.execute("SELECT COUNT(*) FROM groups WHERE name='Shelf A'").fetchone()[0] == 1


def test_update_book_details_persists_corrected_fields(tmp_path):
    db_path = tmp_path / "books.db"
    books_app.DB_PATH = str(db_path)

    with sqlite3.connect(db_path) as con:
        con.execute(
            "CREATE TABLE books(id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT UNIQUE, image_name TEXT, title TEXT NOT NULL, author TEXT, publisher TEXT, isbn TEXT, notes TEXT, group_name TEXT, cover_path TEXT, embedding BLOB)"
        )
        con.execute(
            "INSERT INTO books(key, image_name, title, author, publisher, isbn, notes, group_name, cover_path, embedding) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("k1", "img1", "Wrong Title", "Wrong Author", "Wrong Pub", "111", "old note", "Shelf A", "", b""),
        )
        con.execute("CREATE TABLE IF NOT EXISTS groups(id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL UNIQUE, image_path TEXT)")
        con.execute("INSERT INTO groups(name, image_path) VALUES(?, ?)", ("Shelf A", ""))

    books_app.init_db()
    books_app.reload()

    result = books_app.update_book_details(1, "Correct Title", "Correct Author", "Correct Pub", "222", "fixed note", "Shelf A")

    assert "✅ Updated" in result[0]
    with sqlite3.connect(db_path) as con:
        row = con.execute("SELECT title, author, publisher, isbn, notes, group_name FROM books WHERE id=1").fetchone()
        correction = con.execute(
            "SELECT old_title, new_title, old_author, new_author, old_publisher, new_publisher, old_isbn, new_isbn, old_notes, new_notes, old_group_name, new_group_name FROM book_corrections WHERE book_id=1"
        ).fetchone()
    assert row == ("Correct Title", "Correct Author", "Correct Pub", "222", "fixed note", "Shelf A")
    assert correction == (
        "Wrong Title", "Correct Title",
        "Wrong Author", "Correct Author",
        "Wrong Pub", "Correct Pub",
        "111", "222",
        "old note", "fixed note",
        "Shelf A", "Shelf A",
    )


def test_import_history_table_shows_full_summary(tmp_path):
    db_path = tmp_path / "books.db"
    books_app.DB_PATH = str(db_path)

    with sqlite3.connect(db_path) as con:
        con.execute(
            "CREATE TABLE import_history(file_name TEXT, imported_at TEXT, rows_added INTEGER, status TEXT)"
        )
        con.execute(
            "INSERT INTO import_history(file_name, imported_at, rows_added, status) VALUES(?,?,?,?)",
            ("demo.csv", "2024-01-01 00:00:00", 12, "ok"),
        )

    books_app.init_db()
    with sqlite3.connect(db_path) as con:
        con.execute("UPDATE import_history SET rows_read=12, no_title=0, duplicates=5, embeddings_reused=0, newly_embedded=7, covers_failed=0 WHERE file_name='demo.csv'")
    table = books_app.import_history_table()

    assert "Rows read" in table.columns
    assert "No title" in table.columns
    assert "Duplicates" in table.columns
    assert "Newly embedded" in table.columns
    assert table.iloc[0]["File"] == "demo.csv"


def test_file_or_placeholder_rejects_directories():
    assert books_app.file_or_placeholder("/app") == books_app.PLACEHOLDER
    assert books_app.cover_image_data_url("/app") == ""


def test_import_csvs_stores_missing_title_rows_in_human_review_queue(tmp_path, monkeypatch):
    db_path = tmp_path / "books.db"
    books_app.DB_PATH = str(db_path)
    books_app.DATA_DIR = str(tmp_path)
    books_app.COVER_DIR = str(tmp_path / "covers")
    os.makedirs(books_app.COVER_DIR, exist_ok=True)

    csv_path = tmp_path / "needs_review.csv"
    csv_path.write_text(
        "image_name,title,author,publisher,isbn,notes,image_base64\n"
        "c1.jpg,,Author One,Pub One,111,Note one,\n"
        "c2.jpg,Book Two,Author Two,Pub Two,222,Note two,\n",
        encoding="utf-8",
    )

    def fake_embed_batches(items, progress=None, desc="Embedding"):
        return np.zeros((len(items), books_app.EMBED_DIM), dtype=np.float32)

    monkeypatch.setattr(books_app, "embed_batches", fake_embed_batches)

    books_app.init_db()
    books_app.reload()

    result = books_app.import_csvs([str(csv_path)])

    assert "Imported 1 books" in result[0]
    with sqlite3.connect(db_path) as con:
        pending = con.execute("SELECT COUNT(*) FROM book_review_queue WHERE status='pending'").fetchone()[0]
        queued = con.execute("SELECT title, author, source_file FROM book_review_queue WHERE status='pending'").fetchone()
    assert pending == 1
    assert queued[0] == ""
    assert queued[1] == "Author One"
    assert queued[2] == "needs_review.csv"
