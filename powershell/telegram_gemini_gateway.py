import argparse
import hashlib
import hmac
import json
import logging
import os
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


DEFAULT_SESSION_TTL_SECONDS = 8 * 60 * 60
DEFAULT_MAX_FAILED_ATTEMPTS = 5
DEFAULT_LOCKOUT_SECONDS = 15 * 60


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def load_dotenv(path: Path | None) -> dict[str, str]:
    data: dict[str, str] = {}
    if path is None or not path.exists():
        return data
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip().strip('"').strip("'")
    return data


def parse_int_env(env_data: dict[str, str], key: str, default: int) -> int:
    raw_value = (env_data.get(key) or "").strip()
    if not raw_value:
        return default
    try:
        parsed = int(raw_value)
        return parsed if parsed > 0 else default
    except ValueError:
        return default


def infer_scope(message: str) -> str:
    lowered = message.lower()
    if "fortaleza" in lowered or "bairro" in lowered or "bairros" in lowered:
        return "fortaleza"
    if "rmf" in lowered or "região metropolitana" in lowered or "regiao metropolitana" in lowered:
        return "rmf"
    if "interior" in lowered:
        return "interior"
    return "geral"


class TelegramGeminiGateway:
    def __init__(
        self,
        project_root: Path,
        hermes_workspace: Path | None,
        gemini_model: str,
        wrapper_path: Path | None = None,
        chat_runtime_dir: Path | None = None,
    ) -> None:
        self.project_root = project_root
        self.hermes_workspace = hermes_workspace
        self.gemini_model = gemini_model
        self.wrapper_path = wrapper_path or self._default_wrapper_path()
        self.chat_dir = chat_runtime_dir or self._default_chat_dir()
        self.chat_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.chat_dir / "telegram_gateway_state.json"
        self.log_path = self.chat_dir / "telegram_gemini_gateway.log"
        self.users_dir = self.project_root / "data" / "users"
        self.auth_db_path = self.users_dir / "telegram_auth.sqlite3"
        self.project_dotenv_path = self.project_root / ".env"
        self.context_dotenv_path = self._detect_context_dotenv()
        self.users_dir.mkdir(parents=True, exist_ok=True)
        self.project_env_data = load_dotenv(self.project_dotenv_path)
        self.env_data = {**self.project_env_data, **load_dotenv(self.context_dotenv_path)}
        self.session_ttl_seconds = parse_int_env(self.project_env_data, "TELEGRAM_AUTH_SESSION_TTL_SECONDS", DEFAULT_SESSION_TTL_SECONDS)
        self.max_failed_attempts = parse_int_env(self.project_env_data, "TELEGRAM_AUTH_MAX_FAILED_ATTEMPTS", DEFAULT_MAX_FAILED_ATTEMPTS)
        self.lockout_seconds = parse_int_env(self.project_env_data, "TELEGRAM_AUTH_LOCKOUT_SECONDS", DEFAULT_LOCKOUT_SECONDS)
        self._ensure_auth_db()
        self.token = self.env_data.get("TELEGRAM_BOT_TOKEN", "")
        if not self.token:
            searched = [str(self.project_dotenv_path)]
            if self.context_dotenv_path:
                searched.append(str(self.context_dotenv_path))
            raise RuntimeError(f"TELEGRAM_BOT_TOKEN ausente nos envs pesquisados: {', '.join(searched)}")
        self.state = self._load_state()
        self.offset = int(self.state.get("offset", 0))
        self.chat_sessions = self.state.get("chat_sessions", {})

        logging.basicConfig(
            filename=self.log_path,
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            encoding="utf-8",
        )
        logging.info(
            "Gateway iniciado com auth_db=%s, wrapper=%s, chat_dir=%s, session_ttl=%s, max_failed_attempts=%s, lockout_seconds=%s",
            self.auth_db_path,
            self.wrapper_path,
            self.chat_dir,
            self.session_ttl_seconds,
            self.max_failed_attempts,
            self.lockout_seconds,
        )

    def _default_wrapper_path(self) -> Path:
        if os.name == "nt":
            return self.project_root / "powershell" / "ask_gemini_with_hermes_memory.ps1"
        return self.project_root / "scripts" / "linux" / "ask_gemini_with_mempalace.py"

    def _default_chat_dir(self) -> Path:
        if os.name == "nt":
            return self.project_root / "outputs" / "hermes" / "chat"
        return self.project_root / "outputs" / "mempalace" / "chat"

    def _detect_context_dotenv(self) -> Path | None:
        candidates: list[Path] = []
        if self.hermes_workspace:
            candidates.extend(
                [
                    self.hermes_workspace / ".mempalace" / ".env",
                    self.hermes_workspace / ".hermes" / ".env",
                ]
            )
        candidates.extend(
            [
                self.project_root / ".mempalace" / ".env",
                self.project_root / ".hermes" / ".env",
            ]
        )
        for path in candidates:
            if path.exists():
                return path
        return None

    def _build_wrapper_command(self, query: str, scope: str) -> list[str]:
        wrapper_suffix = self.wrapper_path.suffix.lower()
        common_args = [
            "--scope",
            scope,
            "--query",
            query,
            "--gemini-model",
            self.gemini_model,
            "--project-root",
            str(self.project_root),
            "--chat-dir",
            str(self.chat_dir),
        ]

        if self.hermes_workspace:
            common_args.extend(["--context-root", str(self.hermes_workspace)])

        if wrapper_suffix == ".ps1":
            command = [
                "powershell",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(self.wrapper_path),
                "-Scope",
                scope,
                "-Query",
                query,
                "-GeminiModel",
                self.gemini_model,
                "-ProjectRoot",
                str(self.project_root),
            ]
            if self.hermes_workspace:
                command.extend(["-HermesWorkspace", str(self.hermes_workspace)])
            return command

        if wrapper_suffix == ".py":
            return [sys.executable, str(self.wrapper_path), *common_args]

        if wrapper_suffix == ".sh":
            return ["bash", str(self.wrapper_path), *common_args]

        raise RuntimeError(f"Wrapper nao suportado: {self.wrapper_path}")

    def _ensure_auth_db(self) -> None:
        with sqlite3.connect(self.auth_db_path) as conn:
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
            conn.commit()

    def _audit_auth_event(
        self,
        event_type: str,
        chat_id: int,
        user_id: int | None = None,
        username: str | None = None,
        details: dict | None = None,
    ) -> None:
        payload = json.dumps(details or {}, ensure_ascii=False)
        with sqlite3.connect(self.auth_db_path) as conn:
            conn.execute(
                """
                INSERT INTO auth_audit (event_type, chat_id, telegram_user_id, username, details_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    event_type,
                    str(chat_id),
                    str(user_id) if user_id is not None else None,
                    username,
                    payload,
                    self._now(),
                ),
            )
            conn.commit()

    def _get_global_lock(self) -> dict:
        with sqlite3.connect(self.auth_db_path) as conn:
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

    def _load_state(self) -> dict:
        if not self.state_path.exists():
            return {"offset": 0, "chat_sessions": {}}
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return {"offset": 0, "chat_sessions": {}}
            data.setdefault("offset", 0)
            data.setdefault("chat_sessions", {})
            return data
        except Exception:
            return {"offset": 0, "chat_sessions": {}}

    def _save_state(self) -> None:
        payload = {
            "offset": self.offset,
            "chat_sessions": self.chat_sessions,
            "updated_at": int(time.time()),
        }
        self.state_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    def _get_session(self, chat_id: int) -> dict:
        session = self.chat_sessions.get(str(chat_id))
        if isinstance(session, dict):
            return session
        return {}

    def _now(self) -> int:
        return int(time.time())

    def _set_session(self, chat_id: int, session: dict) -> None:
        self.chat_sessions[str(chat_id)] = session
        self._save_state()

    def _clear_session(self, chat_id: int) -> None:
        self.chat_sessions.pop(str(chat_id), None)
        self._save_state()

    def _hash_password(self, password: str, salt_hex: str) -> str:
        derived = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            bytes.fromhex(salt_hex),
            100000,
        )
        return derived.hex()

    def _verify_credentials(self, username: str, password: str) -> str | None:
        normalized = username.strip().lower()
        if not normalized or not password:
            return None

        with sqlite3.connect(self.auth_db_path) as conn:
            row = conn.execute(
                """
                SELECT username, password_salt, password_hash
                FROM users
                WHERE lower(username) = ? AND is_active = 1
                LIMIT 1
                """,
                (normalized,),
            ).fetchone()

        if not row:
            return None

        expected_hash = self._hash_password(password, row[1])
        if hmac.compare_digest(expected_hash, row[2]):
            return str(row[0])
        return None

    def _get_lockout_remaining(self, session: dict) -> int:
        lock_until = int(session.get("lock_until", 0) or 0)
        return max(0, lock_until - self._now())

    def _is_locked(self, session: dict) -> bool:
        return self._get_lockout_remaining(session) > 0

    def _prune_expired_session(self, chat_id: int) -> bool:
        session = self._get_session(chat_id)
        authenticated_at = int(session.get("authenticated_at", 0) or 0)
        if not session.get("authenticated") or authenticated_at <= 0:
            return False
        if self._now() - authenticated_at <= self.session_ttl_seconds:
            return False

        self._set_session(
            chat_id,
            {
                "authenticated": False,
                "awaiting": "username",
                "session_expired": True,
                "failed_attempts": 0,
                "lock_until": 0,
            },
        )
        return True

    def _is_authenticated(self, chat_id: int) -> bool:
        if self._prune_expired_session(chat_id):
            return False
        session = self._get_session(chat_id)
        return bool(session.get("authenticated"))

    def _prompt_for_username(self, chat_id: int, session: dict | None = None, message: str | None = None) -> None:
        current = session or self._get_session(chat_id)
        next_session = {
            "authenticated": False,
            "awaiting": "username",
            "failed_attempts": int(current.get("failed_attempts", 0) or 0),
            "lock_until": int(current.get("lock_until", 0) or 0),
        }
        self._set_session(chat_id, next_session)
        self._send_message(chat_id, message or "Informe seu usuario para liberar o acesso.")

    def _set_authenticated_session(self, chat_id: int, username: str) -> None:
        self._set_session(
            chat_id,
            {
                "authenticated": True,
                "username": username,
                "authenticated_at": self._now(),
                "failed_attempts": 0,
                "lock_until": 0,
            },
        )

    def _touch_authenticated_session(self, chat_id: int) -> None:
        session = self._get_session(chat_id)
        if not session.get("authenticated"):
            return
        session["authenticated_at"] = self._now()
        self._set_session(chat_id, session)

    def _handle_auth_message(self, chat_id: int, user_id: int, text: str) -> bool:
        session = self._get_session(chat_id)
        if session.get("session_expired"):
            session.pop("session_expired", None)
            self._set_session(chat_id, session)
            self._send_message(chat_id, "Sua sessao expirou. Informe usuario e senha novamente.")
            session = self._get_session(chat_id)

        if self._is_locked(session):
            remaining = self._get_lockout_remaining(session)
            minutes = max(1, int((remaining + 59) / 60))
            self._audit_auth_event(
                "lockout_active",
                chat_id,
                user_id=user_id,
                username=session.get("pending_username") or session.get("username"),
                details={"remaining_seconds": remaining},
            )
            self._send_message(chat_id, f"Acesso temporariamente bloqueado por tentativas invalidas. Tente novamente em cerca de {minutes} minuto(s).")
            return True

        awaiting = session.get("awaiting")

        if text.startswith("/logout"):
            self._audit_auth_event("logout", chat_id, user_id=user_id, username=session.get("username"))
            self._prompt_for_username(chat_id, message="Sessao encerrada. Informe seu usuario.")
            return True

        if awaiting == "password":
            username = session.get("pending_username", "")
            verified_username = self._verify_credentials(username, text)
            if verified_username:
                self._set_authenticated_session(chat_id, verified_username)
                self._audit_auth_event("login_success", chat_id, user_id=user_id, username=verified_username)
                self._send_message(chat_id, f"Acesso liberado para {verified_username}. Envie sua pergunta.")
            else:
                failed_attempts = int(session.get("failed_attempts", 0) or 0) + 1
                self._audit_auth_event(
                    "login_failure",
                    chat_id,
                    user_id=user_id,
                    username=username,
                    details={"failed_attempts": failed_attempts},
                )
                if failed_attempts >= self.max_failed_attempts:
                    locked_session = {
                        "authenticated": False,
                        "awaiting": "username",
                        "failed_attempts": 0,
                        "lock_until": self._now() + self.lockout_seconds,
                    }
                    self._set_session(chat_id, locked_session)
                    self._audit_auth_event(
                        "login_lockout",
                        chat_id,
                        user_id=user_id,
                        username=username,
                        details={"lockout_seconds": self.lockout_seconds},
                    )
                    lockout_minutes = max(1, int((self.lockout_seconds + 59) / 60))
                    self._send_message(chat_id, f"Muitas tentativas invalidas. Acesso bloqueado temporariamente por {lockout_minutes} minuto(s).")
                else:
                    retry_session = {
                        "authenticated": False,
                        "awaiting": "username",
                        "failed_attempts": failed_attempts,
                        "lock_until": 0,
                    }
                    self._set_session(chat_id, retry_session)
                    remaining_attempts = self.max_failed_attempts - failed_attempts
                    self._send_message(chat_id, f"Usuario ou senha invalidos. Informe o usuario novamente. Restam {remaining_attempts} tentativa(s) antes do bloqueio temporario.")
            return True

        username = text.strip()
        if not username or username.startswith("/"):
            self._prompt_for_username(chat_id, session=session)
            return True

        self._set_session(
            chat_id,
            {
                "authenticated": False,
                "awaiting": "password",
                "pending_username": username,
                "failed_attempts": int(session.get("failed_attempts", 0) or 0),
                "lock_until": int(session.get("lock_until", 0) or 0),
            },
        )
        self._send_message(chat_id, "Usuario recebido. Agora informe a senha.")
        return True

    def _run_with_feedback(self, chat_id: int, query: str, scope: str) -> str:
        self._send_message(chat_id, "Analisando sua solicitacao. Isso pode levar alguns instantes...")
        stop_event = threading.Event()

        def keep_typing() -> None:
            while not stop_event.is_set():
                self._send_typing(chat_id)
                stop_event.wait(4)

        worker = threading.Thread(target=keep_typing, daemon=True)
        worker.start()
        try:
            return self._run_query(query, scope)
        finally:
            stop_event.set()
            worker.join(timeout=1)

    def _api(self, method: str, payload: dict | None = None) -> dict:
        payload = payload or {}
        url = f"https://api.telegram.org/bot{self.token}/{method}"
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=90) as response:
            return json.loads(response.read().decode("utf-8"))

    def _send_message(self, chat_id: int, text: str) -> None:
        chunks = []
        text = text.strip()
        while text:
            chunks.append(text[:3800])
            text = text[3800:]
        for chunk in chunks or ["Resposta vazia."]:
            self._api("sendMessage", {"chat_id": chat_id, "text": chunk})

    def _send_typing(self, chat_id: int) -> None:
        try:
            self._api("sendChatAction", {"chat_id": chat_id, "action": "typing"})
        except Exception:
            logging.exception("Falha ao enviar typing para chat %s", chat_id)

    def _extract_answer_body(self, scope: str) -> str:
        latest_path = self.chat_dir / f"gemini_chat_{scope}_latest.md"
        text = read_text(latest_path)
        if not text:
            return "Nao foi possivel localizar a resposta gerada."
        parts = text.split("\n\n", 2)
        if len(parts) == 3:
            return parts[2].strip()
        return text.strip()

    def _run_query(self, query: str, scope: str) -> str:
        command = self._build_wrapper_command(query, scope)
        completed = subprocess.run(
            command,
            cwd=self.project_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=180,
        )
        if completed.returncode != 0:
            logging.error("Falha ao processar pergunta. stdout=%s stderr=%s", completed.stdout, completed.stderr)
            return "Falha ao gerar a resposta analitica no momento. Verifique o gateway Gemini local."
        logging.info("Pergunta processada com scope=%s", scope)
        return self._extract_answer_body(scope)

    def handle_update(self, update: dict) -> None:
        message = update.get("message") or update.get("edited_message")
        if not message:
            return
        text = (message.get("text") or "").strip()
        if not text:
            return
        chat_id = int(message["chat"]["id"])
        user_id = int(message.get("from", {}).get("id", chat_id))
        logging.info("Mensagem recebida chat=%s user=%s texto=%s", chat_id, user_id, text)

        global_lock = self._get_global_lock()
        if global_lock.get("active"):
            reason = global_lock.get("reason") or "Acesso temporariamente bloqueado pela administracao."
            self._audit_auth_event(
                "global_lock_block",
                chat_id,
                user_id=user_id,
                username=self._get_session(chat_id).get("username"),
                details={"reason": reason, "updated_at": global_lock.get("updated_at")},
            )
            self._send_message(chat_id, reason)
            return

        if text.startswith("/start"):
            if self._is_authenticated(chat_id):
                username = self._get_session(chat_id).get("username", "usuario")
                self._send_message(chat_id, f"Bot ativo. Sessao autenticada como {username}. Envie sua pergunta.")
            else:
                self._prompt_for_username(chat_id)
            return

        if text.startswith("/status"):
            if self._is_authenticated(chat_id):
                session = self._get_session(chat_id)
                expiry_seconds = max(0, self.session_ttl_seconds - (self._now() - int(session.get('authenticated_at', 0) or 0)))
                expiry_minutes = max(1, int((expiry_seconds + 59) / 60))
                self._send_message(chat_id, f"Gateway Gemini ativo. Autenticado como {session.get('username', 'usuario')}. Sessao expira em cerca de {expiry_minutes} minuto(s).")
            else:
                self._prompt_for_username(chat_id, message="Gateway Gemini ativo, mas este chat ainda nao foi autenticado. Informe seu usuario.")
            return

        if text.startswith("/logout"):
            current_session = self._get_session(chat_id)
            self._audit_auth_event("logout", chat_id, user_id=user_id, username=current_session.get("username"))
            self._prompt_for_username(chat_id, message="Sessao encerrada. Informe seu usuario.")
            return

        if not self._is_authenticated(chat_id):
            self._handle_auth_message(chat_id, user_id, text)
            return

        self._touch_authenticated_session(chat_id)
        scope = infer_scope(text)
        answer = self._run_with_feedback(chat_id, text, scope)
        self._send_message(chat_id, answer)

    def run(self) -> None:
        while True:
            try:
                payload = {"timeout": 30}
                if self.offset:
                    payload["offset"] = self.offset
                result = self._api("getUpdates", payload)
                for update in result.get("result", []):
                    self.offset = int(update["update_id"]) + 1
                    self._save_state()
                    self.handle_update(update)
            except urllib.error.URLError:
                logging.exception("Falha de rede no polling do Telegram")
                time.sleep(5)
            except subprocess.TimeoutExpired:
                logging.exception("Timeout ao processar consulta")
                time.sleep(2)
            except Exception:
                logging.exception("Erro inesperado no gateway")
                time.sleep(5)


def main() -> int:
    parser = argparse.ArgumentParser(description="Gateway Telegram -> Gemini com memoria operacional do projeto")
    parser.add_argument("--project-root", default=r"C:\Users\Boanerges\Desktop\Projetos\Report Preview" if os.name == "nt" else str(Path.cwd()))
    parser.add_argument("--hermes-workspace", default=r"E:\Hermes_Workspace" if os.name == "nt" else "")
    parser.add_argument("--gemini-model", default="gemini-2.5-flash")
    parser.add_argument("--wrapper-path", default="")
    parser.add_argument("--chat-runtime-dir", default="")
    args = parser.parse_args()

    gateway = TelegramGeminiGateway(
        project_root=Path(args.project_root),
        hermes_workspace=Path(args.hermes_workspace) if args.hermes_workspace else None,
        gemini_model=args.gemini_model,
        wrapper_path=Path(args.wrapper_path) if args.wrapper_path else None,
        chat_runtime_dir=Path(args.chat_runtime_dir) if args.chat_runtime_dir else None,
    )
    gateway.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())