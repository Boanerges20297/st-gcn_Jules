import argparse
import getpass
import hashlib
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from secrets import token_bytes


PROJECT_ROOT = Path(__file__).resolve().parent
USERS_DIR = PROJECT_ROOT / "data" / "users"
DB_PATH = USERS_DIR / "telegram_auth.sqlite3"


def normalize_username(username: str) -> str:
    return username.strip()


def ensure_db() -> None:
    USERS_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_salt TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS auth_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                chat_id TEXT NOT NULL,
                telegram_user_id TEXT,
                username TEXT,
                details_json TEXT,
                created_at INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS auth_controls (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def hash_password(password: str, salt_hex: str) -> str:
    derived = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt_hex),
        100000,
    )
    return derived.hex()


def prompt_password(confirm: bool) -> str:
    password = getpass.getpass("Senha: ")
    if not password:
        raise ValueError("Senha vazia nao e permitida.")
    if confirm:
        confirmation = getpass.getpass("Confirmar senha: ")
        if password != confirmation:
            raise ValueError("As senhas nao conferem.")
    return password


def add_user(username: str, password: str | None) -> None:
    normalized = normalize_username(username)
    if not normalized:
        raise ValueError("Usuario invalido.")
    password_value = password or prompt_password(confirm=True)
    salt_hex = token_bytes(16).hex()
    password_hash = hash_password(password_value, salt_hex)
    now = datetime.now().isoformat(timespec="seconds")

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO users (username, password_salt, password_hash, is_active, created_at, updated_at)
            VALUES (?, ?, ?, 1, ?, ?)
            """,
            (normalized, salt_hex, password_hash, now, now),
        )
        conn.commit()

    print(f"Usuario '{normalized}' cadastrado em {DB_PATH}")


def set_password(username: str, password: str | None) -> None:
    normalized = normalize_username(username)
    if not normalized:
        raise ValueError("Usuario invalido.")
    password_value = password or prompt_password(confirm=True)
    salt_hex = token_bytes(16).hex()
    password_hash = hash_password(password_value, salt_hex)
    now = datetime.now().isoformat(timespec="seconds")

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            """
            UPDATE users
            SET password_salt = ?, password_hash = ?, updated_at = ?
            WHERE lower(username) = lower(?)
            """,
            (salt_hex, password_hash, now, normalized),
        )
        conn.commit()

    if cursor.rowcount == 0:
        raise ValueError(f"Usuario '{normalized}' nao encontrado.")
    print(f"Senha atualizada para '{normalized}'")


def set_active(username: str, active: bool) -> None:
    normalized = normalize_username(username)
    if not normalized:
        raise ValueError("Usuario invalido.")
    now = datetime.now().isoformat(timespec="seconds")

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            """
            UPDATE users
            SET is_active = ?, updated_at = ?
            WHERE lower(username) = lower(?)
            """,
            (1 if active else 0, now, normalized),
        )
        conn.commit()

    if cursor.rowcount == 0:
        raise ValueError(f"Usuario '{normalized}' nao encontrado.")
    state = "ativado" if active else "desativado"
    print(f"Usuario '{normalized}' {state}")


def list_users(show_inactive: bool) -> None:
    query = "SELECT username, is_active, created_at, updated_at FROM users"
    params: tuple = ()
    if not show_inactive:
        query += " WHERE is_active = 1"
    query += " ORDER BY lower(username)"

    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(query, params).fetchall()

    if not rows:
        print("Nenhum usuario cadastrado.")
        return

    for username, is_active, created_at, updated_at in rows:
        status = "ativo" if int(is_active) == 1 else "inativo"
        print(f"{username}\t{status}\tcriado={created_at}\tatualizado={updated_at}")


def user_exists(username: str) -> bool:
    normalized = normalize_username(username)
    if not normalized:
        return False

    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT 1 FROM users WHERE lower(username) = lower(?) LIMIT 1",
            (normalized,),
        ).fetchone()
    return row is not None


def upsert_user(username: str, password: str) -> str:
    if user_exists(username):
        set_password(username, password)
        return "updated"
    add_user(username, password)
    return "created"


def set_global_lock(active: bool, reason: str | None = None) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    payload = {
        "active": bool(active),
        "reason": (reason or "").strip(),
    }

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO auth_controls (key, value, updated_at)
            VALUES ('global_lock', ?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
            """,
            (json.dumps(payload, ensure_ascii=False), now),
        )
        conn.commit()

    state = "ativado" if active else "desativado"
    print(f"Bloqueio global {state}.")
    if payload["reason"]:
        print(f"Motivo: {payload['reason']}")


def get_global_lock() -> dict:
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT value, updated_at FROM auth_controls WHERE key = 'global_lock' LIMIT 1"
        ).fetchone()

    if not row:
        return {"active": False, "reason": "", "updated_at": None}

    try:
        payload = json.loads(row[0])
    except json.JSONDecodeError:
        payload = {"active": False, "reason": ""}

    return {
        "active": bool(payload.get("active")),
        "reason": str(payload.get("reason") or ""),
        "updated_at": row[1],
    }


def fetch_auth_audit(limit: int = 50, username: str | None = None, event_type: str | None = None) -> list[dict]:
    limit = max(1, min(int(limit), 500))
    clauses: list[str] = []
    params: list[object] = []

    if username:
        clauses.append("lower(username) = lower(?)")
        params.append(normalize_username(username))

    if event_type:
        clauses.append("event_type = ?")
        params.append(event_type.strip())

    query = "SELECT id, event_type, chat_id, telegram_user_id, username, details_json, created_at FROM auth_audit"
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(query, params).fetchall()

    records = []
    for row in rows:
        try:
            details = json.loads(row[5]) if row[5] else {}
        except json.JSONDecodeError:
            details = {"raw": row[5]}
        records.append(
            {
                "id": row[0],
                "event_type": row[1],
                "chat_id": row[2],
                "telegram_user_id": row[3],
                "username": row[4],
                "details": details,
                "created_at": row[6],
            }
        )
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gerencia usuarios do bot Telegram no SQLite local")
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_parser = subparsers.add_parser("add", help="Cadastra um novo usuario")
    add_parser.add_argument("username")
    add_parser.add_argument("--password")

    password_parser = subparsers.add_parser("set-password", help="Atualiza a senha de um usuario")
    password_parser.add_argument("username")
    password_parser.add_argument("--password")

    activate_parser = subparsers.add_parser("activate", help="Ativa um usuario")
    activate_parser.add_argument("username")

    deactivate_parser = subparsers.add_parser("deactivate", help="Desativa um usuario")
    deactivate_parser.add_argument("username")

    list_parser = subparsers.add_parser("list", help="Lista usuarios cadastrados")
    list_parser.add_argument("--all", action="store_true", help="Inclui usuarios inativos")

    return parser


def main() -> int:
    ensure_db()
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "add":
            add_user(args.username, args.password)
        elif args.command == "set-password":
            set_password(args.username, args.password)
        elif args.command == "activate":
            set_active(args.username, True)
        elif args.command == "deactivate":
            set_active(args.username, False)
        elif args.command == "list":
            list_users(args.all)
        else:
            parser.error("Comando invalido")
    except sqlite3.IntegrityError:
        print(f"Usuario '{args.username}' ja existe.", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
