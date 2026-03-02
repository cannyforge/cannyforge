#!/usr/bin/env python3
"""
CannyForge Storage Backends

Abstraction for persistent storage of errors, successes, step errors, and rules.
Implementations: JSONFileBackend (legacy), SQLiteBackend (default for production).
"""

import json
import logging
import sqlite3
import threading
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger("Storage")


class StorageBackend(ABC):
    """Abstract storage backend for CannyForge data."""

    # -- Error records --
    @abstractmethod
    def store_error(self, data: Dict[str, Any]) -> None: ...

    @abstractmethod
    def get_errors(self, skill_name: Optional[str] = None,
                   error_type: Optional[str] = None,
                   limit: Optional[int] = None) -> List[Dict[str, Any]]: ...

    @abstractmethod
    def clear_errors(self) -> None: ...

    # -- Success records --
    @abstractmethod
    def store_success(self, data: Dict[str, Any]) -> None: ...

    @abstractmethod
    def get_successes(self, skill_name: Optional[str] = None,
                      limit: Optional[int] = None) -> List[Dict[str, Any]]: ...

    @abstractmethod
    def clear_successes(self) -> None: ...

    # -- Step error records --
    @abstractmethod
    def store_step_error(self, data: Dict[str, Any]) -> None: ...

    @abstractmethod
    def get_step_errors(self, skill_name: Optional[str] = None,
                        error_type: Optional[str] = None) -> List[Dict[str, Any]]: ...

    @abstractmethod
    def clear_step_errors(self) -> None: ...

    # -- Rules --
    @abstractmethod
    def store_rules(self, rules_by_skill: Dict[str, List[Dict[str, Any]]]) -> None: ...

    @abstractmethod
    def load_rules(self) -> Dict[str, List[Dict[str, Any]]]: ...


class JSONFileBackend(StorageBackend):
    """Legacy JSON/JSONL file-based storage (current behavior)."""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _append_jsonl(self, filename: str, data: Dict[str, Any]):
        filepath = self.data_dir / filename
        with open(filepath, 'a') as f:
            f.write(json.dumps(data) + '\n')

    def _read_jsonl(self, filename: str) -> List[Dict[str, Any]]:
        filepath = self.data_dir / filename
        if not filepath.exists():
            return []
        records = []
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return records

    def _clear_file(self, filename: str):
        filepath = self.data_dir / filename
        if filepath.exists():
            filepath.unlink()

    # Errors
    def store_error(self, data: Dict[str, Any]) -> None:
        self._append_jsonl("errors.jsonl", data)

    def get_errors(self, skill_name=None, error_type=None, limit=None):
        records = self._read_jsonl("errors.jsonl")
        if skill_name:
            records = [r for r in records if r.get('skill') == skill_name]
        if error_type:
            records = [r for r in records if r.get('error_type') == error_type]
        if limit:
            records = records[-limit:]
        return records

    def clear_errors(self):
        self._clear_file("errors.jsonl")

    # Successes
    def store_success(self, data: Dict[str, Any]) -> None:
        self._append_jsonl("successes.jsonl", data)

    def get_successes(self, skill_name=None, limit=None):
        records = self._read_jsonl("successes.jsonl")
        if skill_name:
            records = [r for r in records if r.get('skill') == skill_name]
        if limit:
            records = records[-limit:]
        return records

    def clear_successes(self):
        self._clear_file("successes.jsonl")

    # Step errors
    def store_step_error(self, data: Dict[str, Any]) -> None:
        self._append_jsonl("step_errors.jsonl", data)

    def get_step_errors(self, skill_name=None, error_type=None):
        records = self._read_jsonl("step_errors.jsonl")
        if skill_name:
            records = [r for r in records if r.get('skill') == skill_name]
        if error_type:
            records = [r for r in records if r.get('error_type') == error_type]
        return records

    def clear_step_errors(self):
        self._clear_file("step_errors.jsonl")

    # Rules
    def store_rules(self, rules_by_skill: Dict[str, List[Dict[str, Any]]]) -> None:
        filepath = self.data_dir / "rules.json"
        filepath.write_text(json.dumps(rules_by_skill, indent=2))

    def load_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        filepath = self.data_dir / "rules.json"
        if not filepath.exists():
            return {}
        return json.loads(filepath.read_text())


class SQLiteBackend(StorageBackend):
    """
    SQLite-based storage backend for production use.

    Provides atomic writes, bounded memory (query by time range instead of
    loading everything), and concurrent access safety via threading locks.
    """

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "cannyforge.db"
        self._lock = threading.Lock()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self):
        with self._lock:
            conn = self._get_conn()
            try:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS errors (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        skill TEXT NOT NULL,
                        task TEXT NOT NULL,
                        error_type TEXT NOT NULL,
                        error_message TEXT NOT NULL,
                        context TEXT DEFAULT '{}',
                        rules_applied TEXT DEFAULT '[]',
                        created_at TEXT DEFAULT (datetime('now'))
                    );
                    CREATE INDEX IF NOT EXISTS idx_errors_skill ON errors(skill);
                    CREATE INDEX IF NOT EXISTS idx_errors_type ON errors(error_type);

                    CREATE TABLE IF NOT EXISTS successes (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        skill TEXT NOT NULL,
                        task TEXT NOT NULL,
                        context TEXT DEFAULT '{}',
                        rules_applied TEXT DEFAULT '[]',
                        execution_time_ms REAL DEFAULT 0,
                        created_at TEXT DEFAULT (datetime('now'))
                    );
                    CREATE INDEX IF NOT EXISTS idx_successes_skill ON successes(skill);

                    CREATE TABLE IF NOT EXISTS step_errors (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        skill TEXT NOT NULL,
                        task TEXT NOT NULL,
                        step INTEGER NOT NULL,
                        tool TEXT NOT NULL,
                        error_type TEXT NOT NULL,
                        error_message TEXT NOT NULL,
                        recovery_applied TEXT DEFAULT '[]',
                        recovery_succeeded INTEGER DEFAULT 0,
                        context TEXT DEFAULT '{}',
                        created_at TEXT DEFAULT (datetime('now'))
                    );
                    CREATE INDEX IF NOT EXISTS idx_step_errors_skill ON step_errors(skill);

                    CREATE TABLE IF NOT EXISTS rules (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        skill TEXT NOT NULL,
                        rule_data TEXT NOT NULL,
                        updated_at TEXT DEFAULT (datetime('now'))
                    );
                    CREATE INDEX IF NOT EXISTS idx_rules_skill ON rules(skill);
                """)
                conn.commit()
            finally:
                conn.close()

    # -- Errors --

    def store_error(self, data: Dict[str, Any]) -> None:
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    "INSERT INTO errors (timestamp, skill, task, error_type, error_message, context, rules_applied) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (data['timestamp'], data['skill'], data['task'],
                     data['error_type'], data['error_message'],
                     json.dumps(data.get('context', {})),
                     json.dumps(data.get('rules_applied', [])))
                )
                conn.commit()
            finally:
                conn.close()

    def get_errors(self, skill_name=None, error_type=None, limit=None):
        with self._lock:
            conn = self._get_conn()
            try:
                query = "SELECT * FROM errors WHERE 1=1"
                params = []
                if skill_name:
                    query += " AND skill = ?"
                    params.append(skill_name)
                if error_type:
                    query += " AND error_type = ?"
                    params.append(error_type)
                query += " ORDER BY id"
                if limit:
                    query += " LIMIT ?"
                    params.append(limit)
                rows = conn.execute(query, params).fetchall()
                return [self._row_to_error_dict(r) for r in rows]
            finally:
                conn.close()

    def _row_to_error_dict(self, row) -> Dict[str, Any]:
        return {
            'timestamp': row['timestamp'],
            'skill': row['skill'],
            'task': row['task'],
            'error_type': row['error_type'],
            'error_message': row['error_message'],
            'context': json.loads(row['context']),
            'rules_applied': json.loads(row['rules_applied']),
        }

    def clear_errors(self):
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute("DELETE FROM errors")
                conn.commit()
            finally:
                conn.close()

    # -- Successes --

    def store_success(self, data: Dict[str, Any]) -> None:
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    "INSERT INTO successes (timestamp, skill, task, context, rules_applied, execution_time_ms) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (data['timestamp'], data['skill'], data['task'],
                     json.dumps(data.get('context', {})),
                     json.dumps(data.get('rules_applied', [])),
                     data.get('execution_time_ms', 0))
                )
                conn.commit()
            finally:
                conn.close()

    def get_successes(self, skill_name=None, limit=None):
        with self._lock:
            conn = self._get_conn()
            try:
                query = "SELECT * FROM successes WHERE 1=1"
                params = []
                if skill_name:
                    query += " AND skill = ?"
                    params.append(skill_name)
                query += " ORDER BY id"
                if limit:
                    query += " LIMIT ?"
                    params.append(limit)
                rows = conn.execute(query, params).fetchall()
                return [self._row_to_success_dict(r) for r in rows]
            finally:
                conn.close()

    def _row_to_success_dict(self, row) -> Dict[str, Any]:
        return {
            'timestamp': row['timestamp'],
            'skill': row['skill'],
            'task': row['task'],
            'context': json.loads(row['context']),
            'rules_applied': json.loads(row['rules_applied']),
            'execution_time_ms': row['execution_time_ms'],
        }

    def clear_successes(self):
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute("DELETE FROM successes")
                conn.commit()
            finally:
                conn.close()

    # -- Step errors --

    def store_step_error(self, data: Dict[str, Any]) -> None:
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    "INSERT INTO step_errors (timestamp, skill, task, step, tool, error_type, "
                    "error_message, recovery_applied, recovery_succeeded, context) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (data['timestamp'], data['skill'], data['task'],
                     data['step'], data['tool'],
                     data['error_type'], data['error_message'],
                     json.dumps(data.get('recovery_applied', [])),
                     int(data.get('recovery_succeeded', False)),
                     json.dumps(data.get('context', {})))
                )
                conn.commit()
            finally:
                conn.close()

    def get_step_errors(self, skill_name=None, error_type=None):
        with self._lock:
            conn = self._get_conn()
            try:
                query = "SELECT * FROM step_errors WHERE 1=1"
                params = []
                if skill_name:
                    query += " AND skill = ?"
                    params.append(skill_name)
                if error_type:
                    query += " AND error_type = ?"
                    params.append(error_type)
                query += " ORDER BY id"
                rows = conn.execute(query, params).fetchall()
                return [self._row_to_step_error_dict(r) for r in rows]
            finally:
                conn.close()

    def _row_to_step_error_dict(self, row) -> Dict[str, Any]:
        return {
            'timestamp': row['timestamp'],
            'skill': row['skill'],
            'task': row['task'],
            'step': row['step'],
            'tool': row['tool'],
            'error_type': row['error_type'],
            'error_message': row['error_message'],
            'recovery_applied': json.loads(row['recovery_applied']),
            'recovery_succeeded': bool(row['recovery_succeeded']),
            'context': json.loads(row['context']),
        }

    def clear_step_errors(self):
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute("DELETE FROM step_errors")
                conn.commit()
            finally:
                conn.close()

    # -- Rules --

    def store_rules(self, rules_by_skill: Dict[str, List[Dict[str, Any]]]) -> None:
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute("DELETE FROM rules")
                for skill, rules in rules_by_skill.items():
                    conn.execute(
                        "INSERT INTO rules (skill, rule_data) VALUES (?, ?)",
                        (skill, json.dumps(rules))
                    )
                conn.commit()
            finally:
                conn.close()

    def load_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        with self._lock:
            conn = self._get_conn()
            try:
                rows = conn.execute("SELECT skill, rule_data FROM rules").fetchall()
                result = {}
                for row in rows:
                    result[row['skill']] = json.loads(row['rule_data'])
                return result
            finally:
                conn.close()
