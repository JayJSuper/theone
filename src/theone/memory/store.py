"""Minimal persistent memory store (SQLite). CC-10.
Real persistence, real deletion (sovereignty: delete means gone), full provenance
(source + timestamp mandatory), export = take-your-data-with-you right."""
from __future__ import annotations
import json
import sqlite3
import time
from pathlib import Path


class MemoryStore:
    def __init__(self, path: str) -> None:
        self.path = str(path)
        self._con = sqlite3.connect(self.path)
        self._con.execute(
            "CREATE TABLE IF NOT EXISTS memory ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT NOT NULL, "
            "value TEXT NOT NULL, source TEXT NOT NULL, ts REAL NOT NULL)")
        self._con.commit()

    def put(self, key: str, value, source: str, ts: float | None = None) -> int:
        if not source:
            raise ValueError("source is mandatory (provenance rule)")
        cur = self._con.execute(
            "INSERT INTO memory (key, value, source, ts) VALUES (?,?,?,?)",
            (key, json.dumps(value, ensure_ascii=False), source,
             time.time() if ts is None else float(ts)))
        self._con.commit()
        return int(cur.lastrowid)

    def get(self, mem_id: int):
        row = self._con.execute(
            "SELECT id,key,value,source,ts FROM memory WHERE id=?", (mem_id,)).fetchone()
        if row is None:
            return None
        return {"id": row[0], "key": row[1], "value": json.loads(row[2]),
                "source": row[3], "ts": row[4]}

    def search(self, key_prefix: str) -> list:
        rows = self._con.execute(
            "SELECT id,key,value,source,ts FROM memory WHERE key LIKE ? ORDER BY id",
            (key_prefix + "%",)).fetchall()
        return [{"id": r[0], "key": r[1], "value": json.loads(r[2]),
                 "source": r[3], "ts": r[4]} for r in rows]

    def delete(self, mem_id: int) -> bool:
        cur = self._con.execute("DELETE FROM memory WHERE id=?", (mem_id,))
        self._con.commit()
        return cur.rowcount > 0

    def export(self) -> str:
        """JSONL export - the take-your-data right."""
        rows = self.search("")
        return "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)

    def close(self) -> None:
        self._con.close()
