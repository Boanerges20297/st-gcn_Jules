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


DEFAULT_SESSION_TTL_SECONDS = 15 * 60
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
        k = key.strip()
        v = value.strip().strip('"').strip("'")
        data[k] = v
        os.environ[k] = v
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


_MUNICIPIOS_RMF = {
    "caucaia", "maracanau", "maranguape", "aquiraz", "eusebio", "horizonte",
    "pacajus", "itaitinga", "chorozinho", "pindoretama", "guaiuba", "pacatuba",
    "cascavel", "sao goncalo do amarante",
}

_TERMOS_TATICOS = {
    "faccao", "massa", "cvli", "cvp", "gangue", "dominio", "controle",
    "territorio", "territórios", "milicia", "trafico", "facção",
}


def infer_scope(message: str) -> str:
    lowered = message.lower()
    # Termos táticos sempre usam escopo geral (CSV consolidado, não recorte)
    if any(t in lowered for t in _TERMOS_TATICOS):
        return "geral"
    # Municípios da RMF não aparecem no CSV de Fortaleza
    if any(m in lowered for m in _MUNICIPIOS_RMF):
        return "rmf"
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
        self.last_request_time = 0.0

        # Configure logging to both console (stdout) and file in a robust way
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        
        # File handler
        file_handler = logging.FileHandler(self.log_path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(threadName)s] %(message)s"))
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(threadName)s] %(message)s"))
        logger.addHandler(console_handler)

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
        if self.hermes_workspace and self.hermes_workspace.exists():
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

    def _build_wrapper_command(self, query: str, scope: str, chat_id: int | None = None) -> list[str]:
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
            if chat_id is not None:
                command.extend(["-ChatId", str(chat_id)])
            return command

        if wrapper_suffix == ".py":
            args = [sys.executable, str(self.wrapper_path), *common_args]
            if chat_id is not None:
                args.extend(["--chat-id", str(chat_id)])
            return args

        if wrapper_suffix == ".sh":
            args = ["bash", str(self.wrapper_path), *common_args]
            if chat_id is not None:
                args.extend(["--chat-id", str(chat_id)])
            return args

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
                    acesso TEXT NOT NULL DEFAULT 'user',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            # Migração: adicionar a coluna acesso se ela não existir
            cursor = conn.execute("PRAGMA table_info(users)")
            columns = [row[1] for row in cursor.fetchall()]
            if "acesso" not in columns:
                conn.execute("ALTER TABLE users ADD COLUMN acesso TEXT NOT NULL DEFAULT 'user'")
                conn.commit()

            # Forçar boanerges como admin e todos os outros como user
            conn.execute("UPDATE users SET acesso = 'admin' WHERE lower(username) = 'boanerges'")
            conn.execute("UPDATE users SET acesso = 'user' WHERE lower(username) != 'boanerges'")
            
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
        old_session = self.chat_sessions.get(str(chat_id))
        if isinstance(old_session, dict) and "message_ids" in old_session:
            if "message_ids" not in session:
                session["message_ids"] = old_session["message_ids"]
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

        # 1. Armazenar o evento de timeout no log primeiro
        self._log_system_event(chat_id, "Sessão expirada por inatividade (Timeout)")
        # 2. Deletar ativamente as mensagens do cliente
        self._clear_chat_history(chat_id)

        self._set_session(
            chat_id,
            {
                "authenticated": False,
                "awaiting": None,
                "session_expired": False,  # Reset immediate as we notify proactively
                "failed_attempts": 0,
                "lock_until": 0,
            },
        )
        self._send_message(chat_id, "Sua sessão expirou por inatividade. Digite /start para iniciar")
        return True

    def _logout_session(self, chat_id: int, user_id: int, trigger_source: str, trigger_msg_id: int | None = None) -> None:
        session = self._get_session(chat_id)
        username = session.get("username")
        
        # 1. Audit event
        self._audit_auth_event("logout", chat_id, user_id=user_id, username=username)
        
        # 2. Track the message ID of the logout trigger command/callback if present
        if trigger_msg_id:
            msg_ids = session.setdefault("message_ids", [])
            if trigger_msg_id not in msg_ids:
                msg_ids.append(trigger_msg_id)
                self._set_session(chat_id, session)

        # 3. Log the system event to conversation log first
        self._log_system_event(chat_id, f"Sessão encerrada pelo usuário (Origem: {trigger_source})")
        
        # 4. Clear chat history client-side
        self._clear_chat_history(chat_id)
        
        # 5. Reset session and prompt to type /start
        self._set_session(
            chat_id,
            {
                "authenticated": False,
                "awaiting": None,
                "failed_attempts": 0,
                "lock_until": 0,
            },
        )
        self._send_message(chat_id, "Sessão encerrada com sucesso. Digite /start para iniciar")

    def _prune_all_expired_sessions(self) -> None:
        chat_ids = list(self.chat_sessions.keys())
        for str_chat_id in chat_ids:
            try:
                chat_id = int(str_chat_id)
                self._prune_expired_session(chat_id)
            except Exception as e:
                logging.error("Erro ao podar sessao expirada para %s: %s", str_chat_id, e)

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
        acesso = "user"
        try:
            with sqlite3.connect(self.auth_db_path) as conn:
                row = conn.execute(
                    "SELECT acesso FROM users WHERE lower(username) = ? LIMIT 1",
                    (username.strip().lower(),)
                ).fetchone()
                if row:
                    acesso = str(row[0])
        except Exception as e:
            logging.warning("Erro ao obter acesso do usuario %s: %s", username, e)

        self._set_session(
            chat_id,
            {
                "authenticated": True,
                "username": username,
                "acesso": acesso,
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

    def _delete_message(self, chat_id: int, message_id: int) -> None:
        if not message_id:
            return
        try:
            self._api("deleteMessage", {"chat_id": chat_id, "message_id": message_id})
            logging.info("Mensagem de credencial apagada do chat=%s message_id=%s", chat_id, message_id)
        except Exception as e:
            logging.warning("Nao foi possivel apagar a mensagem chat=%s message_id=%s: %s", chat_id, message_id, e)

    def _handle_auth_message(self, chat_id: int, user_id: int, text: str, message_id: int | None = None) -> bool:
        session = self._get_session(chat_id)
        if session.get("session_expired"):
            # If session is expired, reset to initial state and prompt
            self._set_session(
                chat_id,
                {
                    "authenticated": False,
                    "awaiting": None,
                    "failed_attempts": int(session.get("failed_attempts", 0) or 0),
                    "lock_until": int(session.get("lock_until", 0) or 0),
                },
            )
            self._send_message(chat_id, "Sua sessão expirou. Digite /start para iniciar")
            return True

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

        if text.startswith("/logout") or text.startswith("/exit") or text.lower() == "exit":
            self._logout_session(chat_id, user_id, trigger_source=text, trigger_msg_id=message_id)
            return True

        if awaiting == "password":
            username = session.get("pending_username", "")
            verified_username = self._verify_credentials(username, text)
            
            # Deletar mensagens de credenciais do chat para segurança
            username_msg_id = session.get("username_message_id")
            if message_id:
                self._delete_message(chat_id, message_id)
            if username_msg_id:
                self._delete_message(chat_id, int(username_msg_id))

            if verified_username:
                self._clear_chat_history(chat_id)
                self._set_authenticated_session(chat_id, verified_username)
                self._audit_auth_event("login_success", chat_id, user_id=user_id, username=verified_username)
                self._send_main_menu(chat_id, message=f"🔓 Acesso liberado para *{verified_username}*.\n\nSelecione um módulo tático abaixo para iniciar:")
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

        if awaiting == "username":
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
                    "username_message_id": message_id,
                },
            )
            self._send_message(chat_id, "Usuario recebido. Agora informe a senha.")
            return True

        # Se não estiver no meio do fluxo de login (awaiting não é username nem password), exigir /start
        self._set_session(
            chat_id,
            {
                "authenticated": False,
                "awaiting": None,
                "failed_attempts": int(session.get("failed_attempts", 0) or 0),
                "lock_until": int(session.get("lock_until", 0) or 0),
            },
        )
        self._send_message(chat_id, "Digite /start para iniciar")
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
            return self._run_query(query, scope, chat_id)
        finally:
            stop_event.set()
            worker.join(timeout=1)

    def _api(self, method: str, payload: dict | None = None) -> dict:
        payload = payload or {}
        url = f"https://api.telegram.org/bot{self.token}/{method}"
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=90) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            
            # Auto-track message_id for session deletion
            if method in ("sendMessage", "sendPhoto", "sendDocument", "sendVideo", "editMessageText") and res_data.get("ok"):
                result = res_data.get("result")
                if isinstance(result, dict):
                    msg_id = result.get("message_id")
                    chat_id = payload.get("chat_id") or result.get("chat", {}).get("id")
                    if chat_id and msg_id:
                        try:
                            session = self._get_session(int(chat_id))
                            msg_ids = session.setdefault("message_ids", [])
                            if msg_id not in msg_ids:
                                msg_ids.append(msg_id)
                                self._set_session(int(chat_id), session)
                        except Exception as e:
                            logging.warning("Erro ao rastrear message_id na sessao: %s", e)
            return res_data

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

    def _extract_answer_body(self, scope: str, chat_id: int | None = None) -> str:
        # Prefer per-chat latest file if available, fall back to scope-global latest
        latest_path = None
        if chat_id is not None:
            candidate = self.chat_dir / f"gemini_chat_{scope}_{chat_id}_latest.md"
            if candidate.exists():
                latest_path = candidate
        if latest_path is None:
            latest_path = self.chat_dir / f"gemini_chat_{scope}_latest.md"
        text = read_text(latest_path)
        if not text:
            return "Nao foi possivel localizar a resposta gerada."
        parts = text.split("\n\n", 2)
        if len(parts) == 3:
            return parts[2].strip()
        return text.strip()

    def _run_query(self, query: str, scope: str, chat_id: int | None = None) -> str:
        # Rate-limiting delay to protect API quotas (RPM)
        min_delay_str = self.env_data.get("TELEGRAM_BOT_MIN_DELAY_SECONDS", "4.0")
        try:
            min_delay = float(min_delay_str)
        except ValueError:
            min_delay = 4.0

        if min_delay > 0:
            now = time.time()
            elapsed = now - self.last_request_time
            if elapsed < min_delay:
                sleep_time = min_delay - elapsed
                logging.info("Respeitando a quota do Gemini (RPM). Aguardando %.2f segundos antes do proximo envio...", sleep_time)
                time.sleep(sleep_time)
        
        self.last_request_time = time.time()
        command = self._build_wrapper_command(query, scope, chat_id)
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
        logging.info("Pergunta processada com scope=%s chat=%s", scope, chat_id)
        return self._extract_answer_body(scope, chat_id)

    def _send_inline_keyboard(self, chat_id: int, text: str, keyboard: list[list[dict]], edit_message_id: int | None = None) -> None:
        payload = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "Markdown",
            "reply_markup": {
                "inline_keyboard": keyboard
            }
        }
        if edit_message_id:
            payload["message_id"] = edit_message_id
            try:
                self._api("editMessageText", payload)
            except Exception as e:
                logging.warning("Falha ao editar mensagem com Markdown chat_id=%s msg_id=%s: %s. Tentando sem Markdown ou enviando nova.", chat_id, edit_message_id, e)
                try:
                    payload_no_md = payload.copy()
                    payload_no_md.pop("parse_mode", None)
                    self._api("editMessageText", payload_no_md)
                except Exception as e_edit_no_md:
                    logging.warning("Falha ao editar sem Markdown chat_id=%s msg_id=%s: %s. Enviando nova mensagem.", chat_id, edit_message_id, e_edit_no_md)
                    payload.pop("message_id", None)
                    try:
                        self._api("sendMessage", payload)
                    except Exception as e2:
                        logging.warning("Falha ao enviar nova mensagem com Markdown chat_id=%s: %s. Tentando sem Markdown.", chat_id, e2)
                        payload.pop("parse_mode", None)
                        try:
                            self._api("sendMessage", payload)
                        except Exception as e3:
                            logging.error("Falha definitiva ao enviar mensagem fallback chat_id=%s: %s", chat_id, e3)
        else:
            try:
                self._api("sendMessage", payload)
            except Exception as e:
                logging.warning("Falha ao enviar mensagem com Markdown chat_id=%s: %s. Tentando sem Markdown.", chat_id, e)
                payload.pop("parse_mode", None)
                try:
                    self._api("sendMessage", payload)
                except Exception as e2:
                    logging.error("Falha definitiva ao enviar mensagem chat_id=%s: %s", chat_id, e2)

    def _is_admin_session(self, chat_id: int) -> bool:
        session = self._get_session(chat_id)
        return isinstance(session, dict) and session.get("acesso") == "admin"

    def _get_main_menu_keyboard(self, acesso: str = "user") -> list[list[dict]]:
        keyboard = [
            [
                {"text": "🔮 Risco Preditivo", "callback_data": "menu_risco"},
                {"text": "🎯 Sentinela (Micronodos)", "callback_data": "menu_sentinela"}
            ],
            [
                {"text": "⚡ Dados Recentes (14d)", "callback_data": "menu_recentes"},
                {"text": "🛣️ Rotas & Ruas Críticas", "callback_data": "menu_rotas"}
            ],
            [
                {"text": "📅 Janelas Temporais", "callback_data": "menu_janelas"},
                {"text": "📊 Contador & Natureza", "callback_data": "menu_contador"}
            ],
            [
                {"text": "👤 Minha Sessão", "callback_data": "menu_sessao"}
            ]
        ]
        if acesso == "admin":
            keyboard.append([{"text": "⚙️ Gerenciar Usuários (Admin)", "callback_data": "menu_admin"}])
        return keyboard

    def _send_main_menu(self, chat_id: int, edit_message_id: int | None = None, message: str | None = None) -> None:
        text = message or (
            "🚨 *PAINEL REPORT PREVIEW*\n\n"
            "Selecione um módulo tático abaixo para navegar de forma ágil ou digite sua pergunta em linguagem natural a qualquer momento.\n\n"
            "💡 _Envie /exit a qualquer momento para encerrar a sessão de forma segura._"
        )
        session = self._get_session(chat_id)
        acesso = session.get("acesso", "user") if isinstance(session, dict) else "user"
        self._send_inline_keyboard(chat_id, text, self._get_main_menu_keyboard(acesso=acesso), edit_message_id=edit_message_id)

    def _send_message_with_menu_button(self, chat_id: int, text: str) -> None:
        keyboard = [[{"text": "🎛️ Painel Principal", "callback_data": "menu_main"}]]
        self._send_inline_keyboard(chat_id, text, keyboard)

    # ─── Admin user management helpers ─────────────────────────────────────────

    def _get_all_users_db(self) -> list[dict]:
        """Return all users from the auth DB as a list of dicts."""
        with sqlite3.connect(self.auth_db_path) as conn:
            rows = conn.execute(
                "SELECT id, username, is_active, acesso, created_at FROM users ORDER BY username"
            ).fetchall()
        return [
            {"id": r[0], "username": r[1], "is_active": bool(r[2]), "acesso": r[3], "created_at": r[4]}
            for r in rows
        ]

    def _show_admin_panel(self, chat_id: int, message_id: int) -> None:
        if not self._is_admin_session(chat_id):
            self._send_message(chat_id, "⛔ Acesso negado.")
            return
        keyboard = [
            [{"text": "📋 Listar Usuários", "callback_data": "admin_list"}],
            [{"text": "➕ Adicionar Usuário", "callback_data": "admin_add"}],
            [{"text": "✅ Ativar Usuário", "callback_data": "admin_activate"}, {"text": "🚫 Desativar Usuário", "callback_data": "admin_deactivate"}],
            [{"text": "🗑️ Excluir Usuário", "callback_data": "admin_delete"}],
            [{"text": "↩️ Voltar", "callback_data": "menu_main"}]
        ]
        self._send_inline_keyboard(
            chat_id,
            "⚙️ *PAINEL ADMINISTRATIVO*\n\nSelecione a operação desejada:",
            keyboard,
            edit_message_id=message_id
        )

    def _show_admin_user_list(self, chat_id: int, message_id: int) -> None:
        if not self._is_admin_session(chat_id):
            self._send_message(chat_id, "⛔ Acesso negado.")
            return
        users = self._get_all_users_db()
        if not users:
            text = "📋 *LISTA DE USUÁRIOS*\n\nNenhum usuário cadastrado."
        else:
            text = "📋 *LISTA DE USUÁRIOS CADASTRADOS*\n\n"
            for u in users:
                status_icon = "✅" if u["is_active"] else "🚫"
                role_icon = "👑" if u["acesso"] == "admin" else "👤"
                text += f"{status_icon} {role_icon} *{u['username']}* ({u['acesso']})\n"
        keyboard = [[{"text": "↩️ Voltar", "callback_data": "menu_admin"}]]
        self._send_inline_keyboard(chat_id, text, keyboard, edit_message_id=message_id)

    def _admin_prompt_target(self, chat_id: int, message_id: int, action: str) -> None:
        """Ask the admin to type the username of the target user."""
        if not self._is_admin_session(chat_id):
            self._send_message(chat_id, "⛔ Acesso negado.")
            return
        labels = {
            "activate": "ativar",
            "deactivate": "desativar",
            "delete": "excluir",
        }
        label = labels.get(action, action)
        session = self._get_session(chat_id)
        session["awaiting_admin_target"] = action
        self._set_session(chat_id, session)
        keyboard = [[{"text": "↩️ Cancelar", "callback_data": "menu_admin"}]]
        self._send_inline_keyboard(
            chat_id,
            f"⚙️ *ADMIN — {label.upper()} USUÁRIO*\n\nDigite o nome de usuário que deseja *{label}*:",
            keyboard,
            edit_message_id=message_id
        )

    def _admin_execute_action(self, chat_id: int, action: str, target_username: str) -> None:
        """Execute activate/deactivate/delete on target_username and send result."""
        if not self._is_admin_session(chat_id):
            self._send_message(chat_id, "⛔ Acesso negado.")
            return

        current_admin = self._get_session(chat_id).get("username", "")
        normalized = target_username.strip().lower()

        if not normalized:
            self._send_message(chat_id, "⚠️ Nome de usuário inválido.")
            return

        # Prevent admin from acting on themselves
        if normalized == (current_admin or "").lower() and action in ("deactivate", "delete"):
            keyboard = [[{"text": "↩️ Voltar", "callback_data": "menu_admin"}]]
            self._send_inline_keyboard(
                chat_id,
                "⛔ *Operação não permitida.*\n\nVocê não pode desativar ou excluir sua própria conta.",
                keyboard
            )
            return

        try:
            with sqlite3.connect(self.auth_db_path) as conn:
                row = conn.execute(
                    "SELECT username, is_active FROM users WHERE lower(username) = ? LIMIT 1",
                    (normalized,)
                ).fetchone()

                if not row:
                    keyboard = [[{"text": "↩️ Voltar", "callback_data": "menu_admin"}]]
                    self._send_inline_keyboard(
                        chat_id,
                        f"⚠️ *Usuário não encontrado:* `{target_username}`",
                        keyboard
                    )
                    return

                real_username = row[0]

                if action == "activate":
                    conn.execute("UPDATE users SET is_active = 1 WHERE lower(username) = ?", (normalized,))
                    conn.commit()
                    self._audit_auth_event("admin_activate", chat_id, username=real_username)
                    keyboard = [[{"text": "↩️ Painel Admin", "callback_data": "menu_admin"}]]
                    self._send_inline_keyboard(
                        chat_id,
                        f"✅ *Usuário `{real_username}` ativado com sucesso.*",
                        keyboard
                    )

                elif action == "deactivate":
                    conn.execute("UPDATE users SET is_active = 0 WHERE lower(username) = ?", (normalized,))
                    conn.commit()
                    self._audit_auth_event("admin_deactivate", chat_id, username=real_username)
                    keyboard = [[{"text": "↩️ Painel Admin", "callback_data": "menu_admin"}]]
                    self._send_inline_keyboard(
                        chat_id,
                        f"🚫 *Usuário `{real_username}` desativado com sucesso.*",
                        keyboard
                    )

                elif action == "delete":
                    # Ask confirmation before deleting
                    session = self._get_session(chat_id)
                    session["awaiting_admin_confirm"] = {"action": "delete", "username": real_username}
                    session.pop("awaiting_admin_target", None)
                    self._set_session(chat_id, session)
                    keyboard = [
                        [{"text": f"⚠️ Confirmar exclusão de '{real_username}'", "callback_data": f"admin_confirm_delete:{real_username}"}],
                        [{"text": "❌ Cancelar", "callback_data": "menu_admin"}]
                    ]
                    self._send_inline_keyboard(
                        chat_id,
                        f"🗑️ *CONFIRMAR EXCLUSÃO*\n\nTem certeza que deseja excluir permanentemente o usuário *`{real_username}`*?\n\n⚠️ _Esta ação não pode ser desfeita._",
                        keyboard
                    )

        except Exception as e:
            logging.exception("Erro ao executar acao admin '%s' para '%s'", action, target_username)
            keyboard = [[{"text": "↩️ Painel Admin", "callback_data": "menu_admin"}]]
            self._send_inline_keyboard(
                chat_id,
                f"❌ *Erro ao executar operação:* {e}",
                keyboard
            )

    def _admin_confirm_delete_user(self, chat_id: int, target_username: str) -> None:
        """Permanently delete a user after admin confirmation."""
        if not self._is_admin_session(chat_id):
            self._send_message(chat_id, "⛔ Acesso negado.")
            return
        try:
            with sqlite3.connect(self.auth_db_path) as conn:
                conn.execute("DELETE FROM users WHERE lower(username) = ?", (target_username.lower(),))
                conn.commit()
            self._audit_auth_event("admin_delete", chat_id, username=target_username)
            session = self._get_session(chat_id)
            session.pop("awaiting_admin_confirm", None)
            self._set_session(chat_id, session)
            keyboard = [[{"text": "↩️ Painel Admin", "callback_data": "menu_admin"}]]
            self._send_inline_keyboard(
                chat_id,
                f"🗑️ *Usuário `{target_username}` excluído permanentemente.*",
                keyboard
            )
            logging.info("Admin excluiu usuario '%s' do banco de dados.", target_username)
        except Exception as e:
            logging.exception("Erro ao excluir usuario '%s'", target_username)
            keyboard = [[{"text": "↩️ Painel Admin", "callback_data": "menu_admin"}]]
            self._send_inline_keyboard(
                chat_id,
                f"❌ *Erro ao excluir usuário:* {e}",
                keyboard
            )

    def _admin_prompt_new_user(self, chat_id: int, message_id: int) -> None:
        """Step 1: ask admin to type the new username."""
        if not self._is_admin_session(chat_id):
            self._send_message(chat_id, "⛔ Acesso negado.")
            return
        session = self._get_session(chat_id)
        session["awaiting_admin_new_user"] = {"step": "username"}
        self._set_session(chat_id, session)
        keyboard = [[{"text": "↩️ Cancelar", "callback_data": "menu_admin"}]]
        self._send_inline_keyboard(
            chat_id,
            "➕ *ADMIN — ADICIONAR USUÁRIO*\n\n*Etapa 1 de 2* — Digite o *nome de usuário* do novo acesso:",
            keyboard,
            edit_message_id=message_id
        )

    def _admin_handle_new_user_input(self, chat_id: int, text: str) -> None:
        """Multi-step state machine for adding a new user (username → password)."""
        if not self._is_admin_session(chat_id):
            self._send_message(chat_id, "⛔ Acesso negado.")
            return

        session = self._get_session(chat_id)
        state = session.get("awaiting_admin_new_user", {})
        step = state.get("step")

        if step == "username":
            candidate = text.strip()
            if not candidate or candidate.startswith("/") or len(candidate) < 3:
                keyboard = [[{"text": "↩️ Cancelar", "callback_data": "menu_admin"}]]
                self._send_inline_keyboard(
                    chat_id,
                    "⚠️ Nome de usuário inválido. Deve ter ao menos 3 caracteres e não pode começar com '/'.",
                    keyboard
                )
                return

            # Check for duplicate
            with sqlite3.connect(self.auth_db_path) as conn:
                exists = conn.execute(
                    "SELECT 1 FROM users WHERE lower(username) = ? LIMIT 1",
                    (candidate.lower(),)
                ).fetchone()
            if exists:
                keyboard = [[{"text": "↩️ Cancelar", "callback_data": "menu_admin"}]]
                self._send_inline_keyboard(
                    chat_id,
                    f"⚠️ *Usuário `{candidate}` já existe no sistema.* Escolha outro nome.",
                    keyboard
                )
                return

            # Advance to password step
            state["step"] = "password"
            state["new_username"] = candidate
            session["awaiting_admin_new_user"] = state
            self._set_session(chat_id, session)
            keyboard = [[{"text": "↩️ Cancelar", "callback_data": "menu_admin"}]]
            self._send_inline_keyboard(
                chat_id,
                f"➕ *ADMIN — ADICIONAR USUÁRIO*\n\n*Etapa 2 de 2* — Usuário: `{candidate}`\n\nAgora envie a *senha* para este acesso:",
                keyboard
            )

        elif step == "password":
            password = text.strip()
            new_username = state.get("new_username", "")

            if not password or len(password) < 4:
                keyboard = [[{"text": "↩️ Cancelar", "callback_data": "menu_admin"}]]
                self._send_inline_keyboard(
                    chat_id,
                    "⚠️ Senha muito curta. Use no mínimo 4 caracteres.",
                    keyboard
                )
                return

            try:
                import secrets as _secrets
                from datetime import datetime as _dt
                salt_hex = _secrets.token_bytes(16).hex()
                password_hash = self._hash_password(password, salt_hex)
                now = _dt.now().isoformat(timespec="seconds")
                acesso = "admin" if new_username.strip().lower() == "boanerges" else "user"

                with sqlite3.connect(self.auth_db_path) as conn:
                    conn.execute(
                        """
                        INSERT INTO users (username, password_salt, password_hash, is_active, acesso, created_at, updated_at)
                        VALUES (?, ?, ?, 1, ?, ?, ?)
                        """,
                        (new_username, salt_hex, password_hash, acesso, now, now),
                    )
                    conn.commit()

                # Delete the password message for security
                msg_ids = session.get("message_ids", [])
                if msg_ids:
                    self._delete_message(chat_id, msg_ids[-1])

                # Clear state
                session.pop("awaiting_admin_new_user", None)
                self._set_session(chat_id, session)

                self._audit_auth_event("admin_add_user", chat_id, username=new_username)
                logging.info("Admin criou usuario '%s' (acesso=%s).", new_username, acesso)

                keyboard = [[{"text": "↩️ Painel Admin", "callback_data": "menu_admin"}]]
                self._send_inline_keyboard(
                    chat_id,
                    f"✅ *Usuário `{new_username}` criado com sucesso!*\n\n🔑 Papel: `{acesso}`\n✅ Status: Ativo",
                    keyboard
                )

            except sqlite3.IntegrityError:
                session.pop("awaiting_admin_new_user", None)
                self._set_session(chat_id, session)
                keyboard = [[{"text": "↩️ Painel Admin", "callback_data": "menu_admin"}]]
                self._send_inline_keyboard(
                    chat_id,
                    f"⚠️ *Usuário `{new_username}` já existe.* Operação cancelada.",
                    keyboard
                )
            except Exception as e:
                logging.exception("Erro ao criar usuario '%s'", new_username)
                session.pop("awaiting_admin_new_user", None)
                self._set_session(chat_id, session)
                keyboard = [[{"text": "↩️ Painel Admin", "callback_data": "menu_admin"}]]
                self._send_inline_keyboard(
                    chat_id,
                    f"❌ *Erro ao criar usuário:* {e}",
                    keyboard
                )
        else:
            # Unknown step — reset state
            session.pop("awaiting_admin_new_user", None)
            self._set_session(chat_id, session)

    # ─── Callback query handler ───────────────────────────────────────────────

    def _handle_callback_query(self, callback_query: dict) -> None:
        query_id = callback_query["id"]
        chat_id = int(callback_query["message"]["chat"]["id"])
        message_id = int(callback_query["message"]["message_id"])
        data = callback_query["data"]
        
        try:
            self._api("answerCallbackQuery", {"callback_query_id": query_id})
        except Exception:
            logging.warning("Nao foi possivel responder callback_query=%s", query_id)

        # Clear any location input wait state if they use inline navigation
        session = self._get_session(chat_id)
        if isinstance(session, dict) and "awaiting_location" in session:
            session.pop("awaiting_location", None)
            self._set_session(chat_id, session)

        if not self._is_authenticated(chat_id):
            self._set_session(
                chat_id,
                {
                    "authenticated": False,
                    "awaiting": None,
                    "failed_attempts": int(session.get("failed_attempts", 0) or 0),
                    "lock_until": int(session.get("lock_until", 0) or 0),
                },
            )
            self._send_message(chat_id, "Sua sessão expirou. Digite /start para iniciar")
            return

        self._touch_authenticated_session(chat_id)

        # Clear admin waiting states if user navigates away via inline button
        if not data.startswith("admin_"):
            changed = False
            for key in ("awaiting_admin_target", "awaiting_admin_confirm", "awaiting_admin_new_user"):
                if key in session:
                    session.pop(key, None)
                    changed = True
            if changed:
                self._set_session(chat_id, session)

        if data == "menu_main":
            self._send_main_menu(chat_id, edit_message_id=message_id)
            
        elif data == "menu_risco":
            keyboard = [
                [
                    {"text": "📈 Risco Capital", "callback_data": "risco_capital"},
                    {"text": "📈 Risco RMF", "callback_data": "risco_rmf"}
                ],
                [
                    {"text": "📈 Risco Interior", "callback_data": "risco_interior"}
                ],
                [
                    {"text": "↩️ Voltar", "callback_data": "menu_main"},
                    {"text": "🏠 Menu Principal", "callback_data": "menu_main"}
                ]
            ]
            self._send_inline_keyboard(
                chat_id, 
                "🔮 *MÓDULO: RISCO PREDITIVO (FORECAST)*\n\nSelecione a região para gerar a análise preditiva baseada nos modelos ST-GCN/ST-GAT:", 
                keyboard, 
                edit_message_id=message_id
            )

        elif data == "menu_sentinela":
            keyboard = [
                [
                    {"text": "📍 Capital", "callback_data": "sentinela_capital"},
                    {"text": "📍 RMF", "callback_data": "sentinela_rmf"}
                ],
                [
                    {"text": "📍 Interior", "callback_data": "sentinela_interior"},
                    {"text": "📍 Geral (Todos)", "callback_data": "sentinela_geral"}
                ],
                [
                    {"text": "↩️ Voltar", "callback_data": "menu_main"},
                    {"text": "🏠 Menu Principal", "callback_data": "menu_main"}
                ]
            ]
            self._send_inline_keyboard(
                chat_id, 
                "🎯 *MÓDULO: SENTINELA (MICRONODOS CRÍTICOS)*\n\nSelecione a região para listar os Top 10 micronodos com maior reincidência espacial (CVLI):", 
                keyboard, 
                edit_message_id=message_id
            )

        elif data == "menu_recentes":
            keyboard = [
                [
                    {"text": "📊 Ranking Recente", "callback_data": "recentes_ranking"},
                    {"text": "🏡 Escolher Bairro/Cidade", "callback_data": "recentes_escolher"}
                ],
                [
                    {"text": "↩️ Voltar", "callback_data": "menu_main"},
                    {"text": "🏠 Menu Principal", "callback_data": "menu_main"}
                ]
            ]
            self._send_inline_keyboard(
                chat_id,
                "⚡ *DADOS RECENTES (14 DIAS)*\n\nSelecione uma opção para visualizar o ranking consolidado ou analisar uma localidade específica (bairro/cidade):",
                keyboard,
                edit_message_id=message_id
            )

        elif data == "recentes_ranking":
            self._show_recent_14d_summary(chat_id, message_id)

        elif data == "recentes_escolher":
            self._prompt_for_location(chat_id, message_id)

        elif data.startswith("recentes_escolher_back:"):
            location_name = data.split(":", 1)[1]
            self._handle_location_input(chat_id, location_name, edit_message_id=message_id)

        elif data == "menu_rotas":
            keyboard = [
                [
                    {"text": "🧭 Caminho do Crime (AIS)", "callback_data": "rotas_caminho"},
                    {"text": "📍 Ranking por Ruas", "callback_data": "rotas_ruas"}
                ],
                [
                    {"text": "↩️ Voltar", "callback_data": "menu_main"},
                    {"text": "🏠 Menu Principal", "callback_data": "menu_main"}
                ]
            ]
            self._send_inline_keyboard(
                chat_id, 
                "🛣️ *MÓDULO: ROTAS E RUAS CRÍTICAS*\n\nSelecione o relatório operacional para visualização de dinâmica migratória ou reincidência por vias:", 
                keyboard, 
                edit_message_id=message_id
            )

        elif data == "menu_janelas":
            keyboard = [
                [
                    {"text": "📅 30 Dias", "callback_data": "janelas_30d"},
                    {"text": "📅 60 Dias", "callback_data": "janelas_60d"},
                    {"text": "📅 90 Dias", "callback_data": "janelas_90d"}
                ],
                [
                    {"text": "↩️ Voltar", "callback_data": "menu_main"},
                    {"text": "🏠 Menu Principal", "callback_data": "menu_main"}
                ]
            ]
            self._send_inline_keyboard(
                chat_id, 
                "📅 *MÓDULO: JANELAS TEMPORAIS HISTÓRICAS*\n\nSelecione o período retroativo para consolidar volumetrias e rankings táticos baseados no pipeline:", 
                keyboard, 
                edit_message_id=message_id
            )

        elif data == "menu_sessao":
            keyboard = [
                [
                    {"text": "⏱️ Tempo Restante", "callback_data": "sessao_tempo"},
                    {"text": "🚪 Encerrar Sessão (/exit)", "callback_data": "sessao_logout"}
                ],
                [
                    {"text": "↩️ Voltar", "callback_data": "menu_main"},
                    {"text": "🏠 Menu Principal", "callback_data": "menu_main"}
                ]
            ]
            self._send_inline_keyboard(
                chat_id, 
                "👤 *MÓDULO: MINHA SESSÃO OPERACIONAL*\n\nGerencie suas credenciais e tempo ativo no gateway:", 
                keyboard, 
                edit_message_id=message_id
            )

        elif data.startswith("risco_"):
            region = data.split("_")[1]
            self._trigger_predictive_risk(chat_id, message_id, region)
            
        elif data.startswith("sentinela_"):
            region = data.split("_")[1]
            self._show_sentinela_micronodes(chat_id, message_id, region)

        elif data == "rotas_caminho":
            keyboard = [
                [
                    {"text": "💀 CVLI (Homicídios)", "callback_data": "rotas_caminho_tipo:cvli"},
                    {"text": "🚗 CVP (Roubos/Patrimoniais)", "callback_data": "rotas_caminho_tipo:cvp"}
                ],
                [
                    {"text": "↩️ Voltar", "callback_data": "menu_rotas"},
                    {"text": "🏠 Menu Principal", "callback_data": "menu_main"}
                ]
            ]
            self._send_inline_keyboard(
                chat_id,
                "🧭 *CAMINHO DO CRIME — SELEÇÃO DE TIPO DE CRIME*\n\nSelecione o tipo de crime para mapear o fluxo migratório de ocorrências:",
                keyboard,
                edit_message_id=message_id
            )

        elif data.startswith("rotas_caminho_tipo:"):
            crime_type = data.split(":")[1]
            crime_label = "CVLI (Homicídios)" if crime_type == "cvli" else "CVP (Roubos/Patrimoniais)"
            keyboard = [
                [
                    {"text": "🏢 Capital", "callback_data": f"rotas_caminho_reg:{crime_type}:capital"},
                    {"text": "🚗 RMF", "callback_data": f"rotas_caminho_reg:{crime_type}:rmf"}
                ],
                [
                    {"text": "🌳 Interior", "callback_data": f"rotas_caminho_reg:{crime_type}:interior"},
                    {"text": "🗺️ Ceará (Geral)", "callback_data": f"rotas_caminho_reg:{crime_type}:geral"}
                ],
                [
                    {"text": "↩️ Voltar", "callback_data": "rotas_caminho"},
                    {"text": "🏠 Menu Principal", "callback_data": "menu_main"}
                ]
            ]
            self._send_inline_keyboard(
                chat_id,
                f"🧭 *CAMINHO DO CRIME — SELEÇÃO DE REGIÃO*\n\nTipo selecionado: *{crime_label}*\n\nSelecione a região de interesse para mapear o fluxo migratório:",
                keyboard,
                edit_message_id=message_id
            )

        elif data.startswith("rotas_caminho_reg:"):
            parts = data.split(":")
            crime_type = parts[1]
            region = parts[2]
            # Bypass temporal window selection, default to 90 days
            self._show_caminho_crime(chat_id, message_id, crime_type=crime_type, region=region, days=90)

        elif data.startswith("rotas_caminho_run:"):
            parts = data.split(":")
            # Support both new f"rotas_caminho_run:{crime_type}:{region}:{days}"
            # and old f"rotas_caminho_run:{region}:{days}" formats
            if len(parts) == 4:
                crime_type = parts[1]
                region = parts[2]
                days = int(parts[3])
                self._show_caminho_crime(chat_id, message_id, crime_type=crime_type, region=region, days=days)
            else:
                region = parts[1]
                days = int(parts[2])
                self._show_caminho_crime(chat_id, message_id, crime_type="cvli", region=region, days=days)

        elif data == "rotas_ruas":
            keyboard = [
                [
                    {"text": "🏢 Capital", "callback_data": "rotas_ruas_reg:capital"},
                    {"text": "🚗 RMF", "callback_data": "rotas_ruas_reg:rmf"}
                ],
                [
                    {"text": "🌳 Interior", "callback_data": "rotas_ruas_reg:interior"},
                    {"text": "🗺️ Ceará (Geral)", "callback_data": "rotas_ruas_reg:geral"}
                ],
                [
                    {"text": "↩️ Voltar", "callback_data": "menu_rotas"},
                    {"text": "🏠 Menu Principal", "callback_data": "menu_main"}
                ]
            ]
            self._send_inline_keyboard(
                chat_id,
                "📍 *RANKING DE RUAS CRÍTICAS — SELEÇÃO DE REGIÃO*\n\nSelecione a região de interesse para mapear a reincidência por vias:",
                keyboard,
                edit_message_id=message_id
            )

        elif data.startswith("rotas_ruas_reg:"):
            region = data.split(":")[1]
            # Bypass temporal window selection, default to 90 days
            self._show_ranking_ruas(chat_id, message_id, region=region, days=90)

        elif data.startswith("rotas_ruas_run:"):
            parts = data.split(":")
            region = parts[1]
            days = int(parts[2])
            self._show_ranking_ruas(chat_id, message_id, region=region, days=days)

        elif data.startswith("janelas_"):
            days = int(data.split("_")[1].replace("d", ""))
            self._show_janela_temporal(chat_id, message_id, days)

        elif data == "menu_contador":
            keyboard = [
                [
                    {"text": "🏙️ Por Cidade", "callback_data": "contador_cidade"},
                    {"text": "🏡 Por Bairro (Fortaleza)", "callback_data": "contador_bairro"}
                ],
                [
                    {"text": "🚔 Por AIS", "callback_data": "contador_ais"},
                    {"text": "🔍 Natureza/Ocorrência", "callback_data": "contador_natureza"}
                ],
                [
                    {"text": "↩️ Voltar", "callback_data": "menu_main"}
                ]
            ]
            self._send_inline_keyboard(
                chat_id,
                "📊 *MÓDULO: CONTADOR & NATUREZA DO CRIME*\n\n"
                "Selecione o escopo para calcular o volume consolidado de CVLI, CVP ou naturezas de crimes dos últimos 90 dias diretamente dos dados do projeto:",
                keyboard,
                edit_message_id=message_id
            )

        elif data == "contador_cidade":
            self._show_contador_cidade(chat_id, message_id)

        elif data == "contador_bairro":
            self._show_contador_bairro(chat_id, message_id)

        elif data == "contador_natureza":
            self._show_contador_natureza(chat_id, message_id)

        elif data == "contador_ais":
            self._show_contador_ais(chat_id, message_id)

        elif "_explicabilidade" in data:
            self._show_explicabilidade(chat_id, message_id, data)

        elif data == "sessao_tempo":
            session = self._get_session(chat_id)
            expiry_seconds = max(0, self.session_ttl_seconds - (self._now() - int(session.get('authenticated_at', 0) or 0)))
            expiry_minutes = max(1, int((expiry_seconds + 59) / 60))
            keyboard = [[{"text": "↩️ Voltar", "callback_data": "menu_sessao"}]]
            self._send_inline_keyboard(
                chat_id,
                f"⏱️ *TEMPO DE SESSÃO OPERACIONAL*\n\nSua sessão expira em aproximadamente *{expiry_minutes} minuto(s)* de inatividade.",
                keyboard,
                edit_message_id=message_id
            )

        elif data == "sessao_logout":
            self._logout_session(chat_id, chat_id, trigger_source="Botão no Menu", trigger_msg_id=message_id)

        # ─── Admin callbacks ────────────────────────────────────────────────────
        elif data == "menu_admin":
            self._show_admin_panel(chat_id, message_id)

        elif data == "admin_list":
            self._show_admin_user_list(chat_id, message_id)

        elif data == "admin_activate":
            self._admin_prompt_target(chat_id, message_id, "activate")

        elif data == "admin_deactivate":
            self._admin_prompt_target(chat_id, message_id, "deactivate")

        elif data == "admin_delete":
            self._admin_prompt_target(chat_id, message_id, "delete")

        elif data.startswith("admin_confirm_delete:"):
            target_username = data.split(":", 1)[1]
            self._admin_confirm_delete_user(chat_id, target_username)

        elif data == "admin_add":
            self._admin_prompt_new_user(chat_id, message_id)

    def _trigger_predictive_risk(self, chat_id: int, message_id: int, region: str) -> None:
        """Exibe o ranking de risco preditivo direto do CSV — sem LLM, sem threading."""
        import csv as _csv

        # Mapeamento region → arquivo CSV e label
        csv_map = {
            "capital":  ("risk_fortaleza_latest.csv",  "FORTALEZA — TOP 10 BAIRROS"),
            "fortaleza":("risk_fortaleza_latest.csv",  "FORTALEZA — TOP 10 BAIRROS"),
            "rmf":      ("risk_rmf_latest.csv",        "RMF — TOP 10 MUNICÍPIOS"),
            "interior": ("risk_interior_latest.csv",   "INTERIOR — TOP 10 MUNICÍPIOS"),
            "geral":    ("risk_snapshot_latest.csv",   "GERAL — TOP 10 LOCALIDADES"),
        }
        csv_file, label = csv_map.get(region.lower(), ("risk_snapshot_latest.csv", f"{region.upper()} — TOP 10"))

        csv_path = self.project_root / "outputs" / "hermes" / csv_file
        if not csv_path.exists():
            self._send_inline_keyboard(
                chat_id,
                f"⚠️ *Arquivo de risco não localizado:* `{csv_file}`\n\nCertifique-se de que o pipeline foi executado na VPS.",
                [[{"text": "↩️ Voltar", "callback_data": "menu_risco"}]],
                edit_message_id=message_id
            )
            return

        try:
            rows = []
            with open(csv_path, "r", encoding="utf-8-sig") as f:
                reader = _csv.DictReader(f)
                for r in reader:
                    rows.append(r)

            # Ordenar por risk_score desc (já vem ordenado, mas garantir)
            rows.sort(key=lambda x: float(x.get("risk_score") or 0), reverse=True)
            top10 = rows[:10]

            if not top10:
                text = f"🔮 *RISCO PREDITIVO — {label}*\n\nNenhum dado disponível."
            else:
                snapshot_at = (top10[0].get("snapshot_generated_at") or top10[0].get("\ufeffsnapshot_generated_at") or "")[:10]
                data_limit  = top10[0].get("data_limit", "")[:10]

                nivel_icon = {
                    "crítico": "🔴", "critico": "🔴",
                    "alto": "🟠", "moderado": "🟡",
                    "baixo": "🟢", "muito baixo": "🟢",
                }

                text = f"🔮 *RISCO PREDITIVO — {label}*\n"
                text += f"📅 Dados até `{data_limit}` | Gerado em `{snapshot_at}`\n"
                text += f"Fonte: Report Preview (ST-GCN/ST-GAT)\n\n"

                for i, r in enumerate(top10, 1):
                    name    = r.get("name", "?")
                    score   = r.get("risk_score", "?")
                    nivel   = r.get("risk_level", "?")
                    trend   = r.get("trend", "")
                    cvli14  = r.get("recent_cvli_14d", "?")
                    conf    = r.get("confidence_pct", "")
                    icon    = nivel_icon.get(nivel.lower().strip(), "⚪")

                    # Rounded score string with %
                    try:
                        score_val = float(score)
                        score_str = f"{int(round(score_val))}%"
                    except:
                        score_str = f"{score}%" if score != "?" else "?"

                    # Rounded confidence string with %
                    try:
                        conf_val = float(conf)
                        conf_str = f"{int(round(conf_val))}%"
                    except:
                        conf_str = f"{conf}%" if conf else ""

                    trend_label = {"up": "Alta 📈", "down": "Queda 📉", "stable": "Estável ➡️"}.get(trend.lower(), "Estável ➡️")

                    text += f"*{i}.* {icon} *{name}*\n"
                    text += f"   Tensão Territorial: `{score_str}` | Nível: *{nivel.capitalize()}*\n"
                    if conf_str:
                        text += f"   Confiança: `{conf_str}` | Tendência: {trend_label}\n"
                    else:
                        text += f"   Tendência: {trend_label}\n"
                    
                    cvli30  = r.get("recent_cvli_30d", "")
                    text += f"   CVLI 14d: *{cvli14}*"
                    if cvli30:
                        text += f" | 30d: *{cvli30}*"
                    text += "\n\n"

            keyboard = [
                [{"text": "📈 Outra Região", "callback_data": "menu_risco"}],
                [{"text": "↩️ Menu Principal", "callback_data": "menu_main"}]
            ]
            self._send_inline_keyboard(chat_id, text, keyboard, edit_message_id=message_id)
            self._log_conversation(chat_id, f"Risco Preditivo ({region})", text)

        except Exception as e:
            logging.exception("Falha ao ler CSV de risco preditivo para %s", region)
            self._send_inline_keyboard(
                chat_id,
                f"❌ *Erro ao processar ranking de risco para {region.upper()}*:\n\n{e}",
                [[{"text": "↩️ Voltar", "callback_data": "menu_risco"}]],
                edit_message_id=message_id
            )

    def _show_sentinela_micronodes(self, chat_id: int, message_id: int, region: str) -> None:
        import csv
        filename = f"top_30_micronodes_{region}.csv" if region != "geral" else "top_30_micronodes.csv"
        path = self.project_root / "outputs" / "hermes" / filename
        if not path.exists():
            path = self.project_root / "outputs" / filename
            
        if not path.exists():
            self._send_inline_keyboard(
                chat_id,
                f"⚠️ *Micronodos de {region.upper()} não localizados.*\nCertifique-se de que os arquivos do pipeline foram gerados na VPS.",
                [[{"text": "↩️ Voltar", "callback_data": "menu_sentinela"}]],
                edit_message_id=message_id
            )
            return

        try:
            rows = []
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    rows.append(r)
            
            top_10 = rows[:10]
            if not top_10:
                text = f"🎯 *SENTINELA - {region.upper()}*\n\nNenhum micronodo crítico encontrado."
            else:
                text = f"🎯 *SENTINELA REPORT PREVIEW — MICRONODOS CRÍTICOS ({region.upper()})*\n"
                text += "Dados históricos consolidados de reincidência espacial (CVLI):\n\n"
                for idx, r in enumerate(top_10, 1):
                    m_id = r.get("micronode_id", "Desconhecido")
                    bairro = r.get("bairro", "Desconhecido")
                    score = r.get("score", "0.0")
                    faction = r.get("faction", "Neutro")
                    streets = r.get("nearby_streets", "")
                    
                    if len(streets) > 60:
                        streets = streets[:57] + "..."
                        
                    text += f"*{idx}. 📍 {m_id}* (Bairro: {bairro})\n"
                    text += f" 🛡️ Score: {score} | Facção: {faction}\n"
                    if streets:
                        text += f" 🛣️ Ruas: {streets}\n"
                    text += "\n"
            
            keyboard = [
                [
                    {"text": "↩️ Voltar", "callback_data": "menu_sentinela"},
                    {"text": "💡 Explicabilidade", "callback_data": f"sentinela_{region}_explicabilidade"}
                ]
            ]
            self._send_inline_keyboard(chat_id, text, keyboard, edit_message_id=message_id)
            self._log_conversation(chat_id, f"Visualizou Sentinela Micronodos (Região: {region.upper()})", text)
        except Exception as e:
            logging.exception("Erro ao ler micronodos de %s", region)
            self._send_inline_keyboard(
                chat_id,
                f"❌ *Erro ao processar micronodos de {region.upper()}*:\n\n{e}",
                [[{"text": "↩️ Voltar", "callback_data": "menu_sentinela"}]],
                edit_message_id=message_id
            )

    def _show_recent_14d_summary(self, chat_id: int, message_id: int) -> None:
        path = self.project_root / "outputs" / "hermes" / "dados_status_enriquecido_14d_summary_latest.md"
        if not path.exists():
            self._send_inline_keyboard(
                chat_id,
                "⚠️ *Resumo tático de 14 dias não localizado.*\nCertifique-se de que os arquivos do pipeline foram gerados na VPS.",
                [[{"text": "↩️ Voltar", "callback_data": "menu_main"}]],
                edit_message_id=message_id
            )
            return

        try:
            content = read_text(path)
            lines = content.splitlines()
            cleaned_lines = []
            for line in lines:
                if line.startswith("#"):
                    cleaned_lines.append(f"⚡ *{line.replace('#', '').strip().upper()}*")
                else:
                    cleaned_lines.append(line)
            
            full_text = "\n".join(cleaned_lines)
            if len(full_text) > 3800:
                full_text = full_text[:3700] + "\n\n...(conteúdo truncado para exibição)..."
                
            keyboard = [
                [
                    {"text": "↩️ Voltar", "callback_data": "menu_recentes"},
                    {"text": "💡 Explicabilidade", "callback_data": "recentes_ranking_explicabilidade"}
                ]
            ]
            self._send_inline_keyboard(chat_id, full_text, keyboard, edit_message_id=message_id)
            self._log_conversation(chat_id, "Visualizou Dados Recentes (14d) - Ranking", full_text)
        except Exception as e:
            logging.exception("Erro ao ler resumo de 14 dias")
            self._send_inline_keyboard(
                chat_id,
                f"❌ *Erro ao processar resumo de 14 dias*:\n\n{e}",
                [[{"text": "↩️ Voltar", "callback_data": "menu_main"}]],
                edit_message_id=message_id
            )

    def _show_caminho_crime(self, chat_id: int, message_id: int, crime_type: str = "cvli", region: str = "geral", days: int = 90) -> None:
        import csv
        from datetime import datetime, timedelta
        path = self.project_root / "outputs" / "hermes" / "caminho_crime.csv"
        if not path.exists():
            self._send_inline_keyboard(
                chat_id,
                "⚠️ *Caminho do crime não localizado.*\nCertifique-se de que os arquivos do pipeline foram gerados na VPS.",
                [[{"text": "↩️ Voltar", "callback_data": "menu_rotas"}, {"text": "🏠 Menu Principal", "callback_data": "menu_main"}]],
                edit_message_id=message_id
            )
            return

        reg_labels = {
            "capital": "Capital",
            "rmf": "RMF",
            "interior": "Interior",
            "geral": "Todo o Ceará"
        }
        reg_label = reg_labels.get(region, "Geral")
        crime_label = "CVLI (Homicídios)" if crime_type == "cvli" else "CVP (Roubos/Patrimoniais)"

        try:
            all_rows = []
            max_dt = None
            
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    if not (r.get("prox_bairro") or r.get("prox_rua")):
                        continue
                    
                    # Filter by crime type
                    r_type = (r.get("tipo") or "cvli").lower().strip()
                    if r_type != crime_type:
                        continue
                    
                    dt_str = r.get("datetime")
                    if dt_str:
                        try:
                            parsed_dt = datetime.strptime(dt_str.split(".")[0], "%Y-%m-%d %H:%M:%S")
                            r["parsed_datetime"] = parsed_dt
                            if max_dt is None or parsed_dt > max_dt:
                                max_dt = parsed_dt
                        except Exception:
                            try:
                                parsed_dt = datetime.strptime(dt_str.split(".")[0], "%Y-%m-%d")
                                r["parsed_datetime"] = parsed_dt
                                if max_dt is None or parsed_dt > max_dt:
                                    max_dt = parsed_dt
                            except Exception:
                                pass
                    all_rows.append(r)

            # Filter by region
            filtered_rows = []
            for r in all_rows:
                risp = (r.get("regiao_risp") or "").upper().strip()
                if region == "capital":
                    if "CAPITAL" not in risp:
                        continue
                elif region == "rmf":
                    if "RMF" not in risp:
                        continue
                elif region == "interior":
                    if risp == "" or "CAPITAL" in risp or "RMF" in risp:
                        continue
                filtered_rows.append(r)

            # Filter by days relative to max_dt
            if max_dt is not None:
                cutoff = max_dt - timedelta(days=days)
                time_filtered = [r for r in filtered_rows if r.get("parsed_datetime") and r["parsed_datetime"] >= cutoff]
            else:
                time_filtered = filtered_rows

            latest_transitions = time_filtered[-10:]
            latest_transitions.reverse()
            
            if not latest_transitions:
                text = f"🧭 *CAMINHO DO CRIME — {reg_label.upper()} ({crime_label.upper()})*\n\nNenhuma transição criminal detectada na janela retroativa de *{days} dias*."
            else:
                text = f"🧭 *CAMINHO DO CRIME — {reg_label.upper()} ({crime_label.upper()})*\n"
                text += f"Últimas transições cronológicas de deslocamento e migração sucessivas (Janela: {days}d):\n\n"
                for idx, r in enumerate(latest_transitions, 1):
                    dt = self._format_date_br(r.get("datetime", "Desconhecida"))
                    cidade = r.get("cidade", "Desconhecida")
                    ais = r.get("ais", "0.0")
                    bairro = r.get("bairro") or "Sem Bairro"
                    rua = r.get("rua") or "Sem Rua"
                    prox_b = r.get("prox_bairro") or "Sem Bairro"
                    prox_r = r.get("prox_rua") or "Sem Rua"
                    dias_val = r.get("dias_para_prox", "0")
                    dist = r.get("distancia_para_prox_km", "0")
                    
                    try:
                        dias_f = f"{float(dias_val):.1f}"
                    except ValueError:
                        dias_f = dias_val
                        
                    try:
                        dist_f = f"{float(dist):.1f}"
                    except ValueError:
                        dist_f = dist
                        
                    try:
                        clean_ais = str(ais).upper().replace("AIS", "").strip()
                        ais_val = float(clean_ais)
                        ais_str = f"{ais_val:.0f}"
                    except ValueError:
                        ais_str = ais
                        
                    display_ais = f"AIS {ais_str}" if not str(ais_str).upper().startswith("AIS") else ais_str
                    text += f"*{idx}. 🛣️ {display_ais} ({cidade})* — _{dt}_\n"
                    text += f"   Origem: {bairro} (Rua: {rua})\n"
                    text += f"   Destino: *{prox_b}* (Rua: {prox_r})\n"
                    text += f"   ⏱️ {dias_f} dias depois | 🗺️ Deslocamento: {dist_f} km\n\n"
            
            keyboard = [
                [
                    {"text": "↩️ Voltar", "callback_data": f"rotas_caminho_tipo:{crime_type}"},
                    {"text": "🏠 Menu Principal", "callback_data": "menu_main"}
                ],
                [
                    {"text": "💡 Explicabilidade", "callback_data": f"rotas_caminho_explicabilidade:{crime_type}:{region}:{days}"}
                ]
            ]
            self._send_inline_keyboard(chat_id, text, keyboard, edit_message_id=message_id)
            self._log_conversation(chat_id, f"Visualizou Caminho do Crime ({crime_type})", text)
        except Exception as e:
            logging.exception("Erro ao processar caminho do crime")
            self._send_inline_keyboard(
                chat_id,
                f"❌ *Erro ao processar caminho do crime*:\n\n{e}",
                [[{"text": "↩️ Voltar", "callback_data": f"rotas_caminho_tipo:{crime_type}"}, {"text": "🏠 Menu Principal", "callback_data": "menu_main"}]],
                edit_message_id=message_id
            )

    def _show_ranking_ruas(self, chat_id: int, message_id: int, region: str = "geral", days: int = 90) -> None:
        """Ranking de ruas com reincidência CVLI, calculado a partir de dados_brutos_Xdias.csv."""
        import csv as _csv
        from collections import defaultdict

        filename = f"dados_brutos_{days}dias.csv"
        path = self.project_root / "outputs" / "hermes" / filename
        if not path.exists():
            path = self.project_root / "outputs" / filename

        # Fallback para 14d: arquivo com nome diferente
        if not path.exists() and days == 14:
            alt = self.project_root / "outputs" / "hermes" / "dados_status_enriquecido_14d_latest.csv"
            if not alt.exists():
                alt = self.project_root / "outputs" / "dados_status_enriquecido_14d_latest.csv"
            if alt.exists():
                path = alt

        reg_labels = {
            "capital": "Capital",
            "rmf": "RMF",
            "interior": "Interior",
            "geral": "Todo o Ceará"
        }
        reg_label = reg_labels.get(region, "Geral")

        if not path.exists():
            self._send_inline_keyboard(
                chat_id,
                f"⚠️ *Dados brutos de {days} dias não localizados.*\nCertifique-se de que os arquivos do pipeline foram gerados na VPS.",
                [[{"text": "↩️ Voltar", "callback_data": "rotas_ruas"}, {"text": "🏠 Menu Principal", "callback_data": "menu_main"}]],
                edit_message_id=message_id
            )
            return

        try:
            rua_counts = defaultdict(lambda: {"cvli": 0, "bairro": "", "cidade": ""})

            with open(path, "r", encoding="utf-8-sig") as f:
                reader = _csv.DictReader(f)
                for r in reader:
                    tipo = (r.get("tipo") or "").strip().lower()
                    if "cvli" not in tipo:
                        continue
                    
                    # Region filter
                    risp = (r.get("regiao_risp") or "").upper().strip()
                    if region == "capital":
                        if "CAPITAL" not in risp:
                            continue
                    elif region == "rmf":
                        if "RMF" not in risp:
                            continue
                    elif region == "interior":
                        if risp == "" or "CAPITAL" in risp or "RMF" in risp:
                            continue
                            
                    rua = (r.get("name") or r.get("rua") or "").strip().upper()
                    if not rua or rua in ("NÃO ESPECIFICADA", "", "SEM NOME", "DESCONHECIDO"):
                        continue
                    bairro = (r.get("bairro") or "").strip().upper()
                    cidade = (r.get("cidade") or "").strip().upper()
                    key = f"{rua}|{bairro}|{cidade}"
                    rua_counts[key]["cvli"] += 1
                    rua_counts[key]["bairro"] = bairro
                    rua_counts[key]["cidade"] = cidade
                    rua_counts[key]["rua"] = rua

            sorted_ruas = sorted(rua_counts.values(), key=lambda x: x["cvli"], reverse=True)
            top10 = sorted_ruas[:10]

            if not top10:
                text = f"🛣️ *RANKING DE RUAS CRÍTICAS — {reg_label.upper()} ({days} DIAS)*\n\nNenhuma rua com CVLI registrado no período nesta região."
            else:
                text = f"🛣️ *RANKING DE RUAS CRÍTICAS — {reg_label.upper()} ({days} DIAS)*\n"
                text += f"Vias urbanas com maior reincidência de homicídios no período nesta região:\n\n"
                for idx, entry in enumerate(top10, 1):
                    text += f"*{idx}. 📍 {entry['rua']}*\n"
                    text += f"   {entry['bairro']}, {entry['cidade']}\n"
                    text += f"   🔥 *{entry['cvli']}* ocorrência(s) de CVLI\n\n"

            keyboard = [
                [
                    {"text": "↩️ Voltar", "callback_data": "rotas_ruas"},
                    {"text": "🏠 Menu Principal", "callback_data": "menu_main"}
                ],
                [
                    {"text": "💡 Explicabilidade", "callback_data": f"rotas_ruas_explicabilidade:{region}:{days}"}
                ]
            ]
            self._send_inline_keyboard(chat_id, text, keyboard, edit_message_id=message_id)
            self._log_conversation(chat_id, f"Visualizou Ranking de Ruas Críticas ({days}d)", text)
        except Exception as e:
            logging.exception("Erro ao processar ranking de ruas (%dd)", days)
            self._send_inline_keyboard(
                chat_id,
                f"❌ *Erro ao processar ranking de ruas ({days}d)*:\n\n{e}",
                [[{"text": "↩️ Voltar", "callback_data": "rotas_ruas"}, {"text": "🏠 Menu Principal", "callback_data": "menu_main"}]],
                edit_message_id=message_id
            )

    def _show_janela_temporal(self, chat_id: int, message_id: int, days: int) -> None:
        import csv
        from collections import Counter
        
        filename = f"dados_brutos_{days}dias.csv"
        path = self.project_root / "outputs" / "hermes" / filename
        if not path.exists():
            self._send_inline_keyboard(
                chat_id,
                f"⚠️ *Dados brutos de {days} dias não localizados.*\nCertifique-se de que os arquivos do pipeline foram gerados.",
                [[{"text": "↩️ Voltar", "callback_data": "menu_janelas"}]],
                edit_message_id=message_id
            )
            return

        try:
            total = 0
            cvli_count = 0
            cvp_count = 0
            cidades = Counter()
            bairros_fortaleza = Counter()
            
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    total += 1
                    tipo = (r.get("tipo") or "").lower().strip()
                    if "cvli" in tipo:
                        cvli_count += 1
                    elif "cvp" in tipo:
                        cvp_count += 1
                        
                    cidade = (r.get("cidade") or "").upper().strip()
                    if cidade:
                        cidades[cidade] += 1
                        
                    if "FORTALEZA" in cidade:
                        bairro = (r.get("bairro") or "").upper().strip()
                        if bairro:
                            bairros_fortaleza[bairro] += 1
            
            text = f"📅 *RESUMO DE JANELA TEMPORAL: ÚLTIMOS {days} DIAS*\n"
            text += f"Volume total de ocorrências registradas: *{total}*\n\n"
            text += f"🔥 *Volume por Tipo de Crime:*\n"
            text += f"• *CVLI* (Homicídios/Letais): {cvli_count} ocorrências\n"
            text += f"• *CVP* (Roubos/Patrimoniais): {cvp_count} ocorrências\n\n"
            
            text += f"📍 *Top 3 Cidades com maior incidência:*\n"
            for idx, (cid, count) in enumerate(cidades.most_common(3), 1):
                text += f"  {idx}. *{cid}*: {count} ocorrências\n"
            text += "\n"
            
            text += f"📍 *Top 3 Bairros Críticos (Fortaleza):*\n"
            for idx, (bai, count) in enumerate(bairros_fortaleza.most_common(3), 1):
                text += f"  {idx}. *{bai}*: {count} ocorrências\n"
                
            keyboard = [
                [
                    {"text": "↩️ Voltar", "callback_data": "menu_janelas"},
                    {"text": "💡 Explicabilidade", "callback_data": f"janelas_{days}d_explicabilidade"}
                ]
            ]
            self._send_inline_keyboard(chat_id, text, keyboard, edit_message_id=message_id)
            self._log_conversation(chat_id, f"Visualizou Janela Temporal (Período: {days} dias)", text)
        except Exception as e:
            logging.exception("Erro ao ler dados brutos de %s dias", days)
            self._send_inline_keyboard(
                chat_id,
                f"❌ *Erro ao processar dados de {days} dias*:\n\n{e}",
                [[{"text": "↩️ Voltar", "callback_data": "menu_janelas"}]],
                edit_message_id=message_id
            )

    def _show_contador_cidade(self, chat_id: int, message_id: int) -> None:
        import csv
        from collections import defaultdict
        
        path = self.project_root / "outputs" / "hermes" / "dados_brutos_90dias.csv"
        if not path.exists():
            path = self.project_root / "outputs" / "dados_brutos_90dias.csv"
            
        if not path.exists():
            self._send_inline_keyboard(
                chat_id,
                "⚠️ *Dados consolidados de 90 dias não localizados.*\nCertifique-se de que os arquivos do pipeline foram gerados na VPS.",
                [[{"text": "↩️ Voltar", "callback_data": "menu_contador"}]],
                edit_message_id=message_id
            )
            return

        try:
            city_counts = defaultdict(lambda: {"cvli": 0, "cvp": 0, "total": 0})
            total_cvli = 0
            total_cvp = 0
            
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    cidade = (r.get("cidade") or "").strip().upper()
                    if not cidade:
                        continue
                    tipo = (r.get("tipo") or "").strip().lower()
                    if "cvli" in tipo:
                       city_counts[cidade]["cvli"] += 1
                       city_counts[cidade]["total"] += 1
                       total_cvli += 1
                    elif "cvp" in tipo:
                       city_counts[cidade]["cvp"] += 1
                       city_counts[cidade]["total"] += 1
                       total_cvp += 1
                       
            sorted_cities = sorted(city_counts.items(), key=lambda x: x[1]["cvli"], reverse=True)
            
            text = "📊 *CONTADOR CVLI & CVP POR CIDADE (90 DIAS)*\n"
            text += f"Total Geral: {total_cvli} CVLIs | {total_cvp} CVPs\n\n"
            text += "Top Cidades ordenadas por volume de CVLI:\n\n"
            
            for idx, (cid, counts) in enumerate(sorted_cities[:10], 1):
                text += f"*{idx}. 🏙️ {cid}*\n"
                text += f"   💀 CVLI: {counts['cvli']} | 🔫 CVP: {counts['cvp']} | 📦 Total: {counts['total']}\n\n"
                
            keyboard = [
                [
                    {"text": "↩️ Voltar", "callback_data": "menu_contador"},
                    {"text": "💡 Explicabilidade", "callback_data": "contador_cidade_explicabilidade"}
                ]
            ]
            self._send_inline_keyboard(chat_id, text, keyboard, edit_message_id=message_id)
            self._log_conversation(chat_id, "Visualizou Contador CVLI/CVP por Cidade", text)
        except Exception as e:
            logging.exception("Erro ao calcular contador por cidade")
            self._send_inline_keyboard(
                chat_id,
                f"❌ *Erro ao processar contador por cidade*:\n\n{e}",
                [[{"text": "↩️ Voltar", "callback_data": "menu_contador"}]],
                edit_message_id=message_id
            )

    def _show_contador_bairro(self, chat_id: int, message_id: int) -> None:
        import csv
        from collections import defaultdict
        
        path = self.project_root / "outputs" / "hermes" / "dados_brutos_90dias.csv"
        if not path.exists():
            path = self.project_root / "outputs" / "dados_brutos_90dias.csv"
            
        if not path.exists():
            self._send_inline_keyboard(
                chat_id,
                "⚠️ *Dados consolidados de 90 dias não localizados.*\nCertifique-se de que os arquivos do pipeline foram gerados na VPS.",
                [[{"text": "↩️ Voltar", "callback_data": "menu_contador"}]],
                edit_message_id=message_id
            )
            return

        try:
            bairro_counts = defaultdict(lambda: {"cvli": 0, "cvp": 0, "total": 0})
            total_cvli = 0
            total_cvp = 0
            
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    cidade = (r.get("cidade") or "").strip().upper()
                    if "FORTALEZA" not in cidade:
                        continue
                    bairro = (r.get("bairro") or "").strip().upper()
                    if not bairro:
                        continue
                    tipo = (r.get("tipo") or "").strip().lower()
                    if "cvli" in tipo:
                       bairro_counts[bairro]["cvli"] += 1
                       bairro_counts[bairro]["total"] += 1
                       total_cvli += 1
                    elif "cvp" in tipo:
                       bairro_counts[bairro]["cvp"] += 1
                       bairro_counts[bairro]["total"] += 1
                       total_cvp += 1
                       
            sorted_bairros = sorted(bairro_counts.items(), key=lambda x: x[1]["cvli"], reverse=True)
            
            text = "📊 *CONTADOR CVLI & CVP POR BAIRRO (FORTALEZA - 90 DIAS)*\n"
            text += f"Total Fortaleza: {total_cvli} CVLIs | {total_cvp} CVPs\n\n"
            text += "Top Bairros ordenados por volume de CVLI:\n\n"
            
            for idx, (bai, counts) in enumerate(sorted_bairros[:10], 1):
                text += f"*{idx}. 🏡 {bai}*\n"
                text += f"   💀 CVLI: {counts['cvli']} | 🔫 CVP: {counts['cvp']} | 📦 Total: {counts['total']}\n\n"
                
            keyboard = [
                [
                    {"text": "↩️ Voltar", "callback_data": "menu_contador"},
                    {"text": "💡 Explicabilidade", "callback_data": "contador_bairro_explicabilidade"}
                ]
            ]
            self._send_inline_keyboard(chat_id, text, keyboard, edit_message_id=message_id)
            self._log_conversation(chat_id, "Visualizou Contador CVLI/CVP por Bairro", text)
        except Exception as e:
            logging.exception("Erro ao calcular contador por bairro")
            self._send_inline_keyboard(
                chat_id,
                f"❌ *Erro ao processar contador por bairro*:\n\n{e}",
                [[{"text": "↩️ Voltar", "callback_data": "menu_contador"}]],
                edit_message_id=message_id
            )

    def _show_contador_natureza(self, chat_id: int, message_id: int) -> None:
        import csv
        from collections import defaultdict
        
        path = self.project_root / "outputs" / "hermes" / "dados_brutos_90dias.csv"
        if not path.exists():
            path = self.project_root / "outputs" / "dados_brutos_90dias.csv"
            
        if not path.exists():
            self._send_inline_keyboard(
                chat_id,
                "⚠️ *Dados consolidados de 90 dias não localizados.*\nCertifique-se de que os arquivos do pipeline foram gerados na VPS.",
                [[{"text": "↩️ Voltar", "callback_data": "menu_contador"}]],
                edit_message_id=message_id
            )
            return

        try:
            nature_counts = defaultdict(int)
            total_events = 0
            
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    nature = (r.get("tipo_evento") or r.get("nature") or r.get("tipo") or "").strip().upper()
                    if not nature:
                        continue
                    nature_counts[nature] += 1
                    total_events += 1
                       
            sorted_natures = sorted(nature_counts.items(), key=lambda x: x[1], reverse=True)
            
            text = "📊 *CONTADOR POR NATUREZA DO CRIME (90 DIAS)*\n"
            text += f"Total Geral: {total_events} ocorrência(s)\n\n"
            text += "Top Naturezas de ocorrências criminais:\n\n"
            
            for idx, (nat, count) in enumerate(sorted_natures[:10], 1):
                text += f"*{idx}. 🔍 {nat}*\n"
                text += f"   📦 Ocorrências: *{count}*\n\n"
                
            keyboard = [
                [
                    {"text": "↩️ Voltar", "callback_data": "menu_contador"},
                    {"text": "💡 Explicabilidade", "callback_data": "contador_natureza_explicabilidade"}
                ]
            ]
            self._send_inline_keyboard(chat_id, text, keyboard, edit_message_id=message_id)
            self._log_conversation(chat_id, "Visualizou Contador por Natureza do Crime (90 Dias)", text)
        except Exception as e:
            logging.exception("Erro ao calcular contador por natureza")
            self._send_inline_keyboard(
                chat_id,
                f"❌ *Erro ao processar contador por natureza*:\n\n{e}",
                [[{"text": "↩️ Voltar", "callback_data": "menu_contador"}]],
                edit_message_id=message_id
            )

    def _show_contador_ais(self, chat_id: int, message_id: int) -> None:
        import csv
        from collections import defaultdict
        
        path = self.project_root / "outputs" / "hermes" / "dados_brutos_90dias.csv"
        if not path.exists():
            path = self.project_root / "outputs" / "dados_brutos_90dias.csv"
            
        if not path.exists():
            self._send_inline_keyboard(
                chat_id,
                "⚠️ *Dados consolidados de 90 dias não localizados.*\nCertifique-se de que os arquivos do pipeline foram gerados na VPS.",
                [[{"text": "↩️ Voltar", "callback_data": "menu_contador"}]],
                edit_message_id=message_id
            )
            return

        try:
            ais_counts = defaultdict(lambda: {"cvli": 0, "cvp": 0, "total": 0})
            total_cvli = 0
            total_cvp = 0
            
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    ais_raw = (r.get("ais") or "").strip()
                    if not ais_raw:
                        continue
                    
                    try:
                        clean_ais = str(ais_raw).upper().replace("AIS", "").strip()
                        if "." in clean_ais:
                            ais_val = float(clean_ais)
                            ais_str = f"{ais_val:.0f}"
                        else:
                            ais_str = str(int(clean_ais))
                    except ValueError:
                        ais_str = clean_ais
                        
                    display_ais = f"AIS {ais_str}" if not str(ais_str).upper().startswith("AIS") else ais_str
                    
                    tipo = (r.get("tipo") or "").strip().lower()
                    if "cvli" in tipo:
                       ais_counts[display_ais]["cvli"] += 1
                       ais_counts[display_ais]["total"] += 1
                       total_cvli += 1
                    elif "cvp" in tipo:
                       ais_counts[display_ais]["cvp"] += 1
                       ais_counts[display_ais]["total"] += 1
                       total_cvp += 1
                       
            sorted_ais = sorted(ais_counts.items(), key=lambda x: x[1]["cvli"], reverse=True)
            
            text = "📊 *CONTADOR CVLI & CVP POR AIS (90 DIAS)*\n"
            text += f"Total Geral: {total_cvli} CVLIs | {total_cvp} CVPs\n\n"
            text += "Áreas de Segurança Integrada (AIS) ordenadas por volume de CVLI:\n\n"
            
            for idx, (ais_name, counts) in enumerate(sorted_ais, 1):
                text += f"*{idx}. 🚔 {ais_name}*\n"
                text += f"   💀 CVLI: {counts['cvli']} | 🔫 CVP: {counts['cvp']} | 📦 Total: {counts['total']}\n\n"
                
            keyboard = [
                [
                    {"text": "↩️ Voltar", "callback_data": "menu_contador"},
                    {"text": "💡 Explicabilidade", "callback_data": "contador_ais_explicabilidade"}
                ]
            ]
            self._send_inline_keyboard(chat_id, text, keyboard, edit_message_id=message_id)
            self._log_conversation(chat_id, "Visualizou Contador CVLI/CVP por AIS", text)
        except Exception as e:
            logging.exception("Erro ao calcular contador por AIS")
            self._send_inline_keyboard(
                chat_id,
                f"❌ *Erro ao processar contador por AIS*:\n\n{e}",
                [[{"text": "↩️ Voltar", "callback_data": "menu_contador"}]],
                edit_message_id=message_id
            )

    def handle_update(self, update: dict) -> None:
        if "callback_query" in update:
            self._handle_callback_query(update["callback_query"])
            return

        message = update.get("message") or update.get("edited_message")
        if not message:
            return
        text = (message.get("text") or "").strip()
        if not text:
            return
        chat_id = int(message["chat"]["id"])
        user_id = int(message.get("from", {}).get("id", chat_id))
        logging.info("Mensagem recebida chat=%s user=%s texto=%s", chat_id, user_id, text)

        # Rastrear message_id do usuario para limpeza posterior no logout/timeout
        msg_id = int(message.get("message_id") or 0)
        if msg_id:
            session = self._get_session(chat_id)
            msg_ids = session.setdefault("message_ids", [])
            if msg_id not in msg_ids:
                msg_ids.append(msg_id)
                self._set_session(chat_id, session)

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

        # Clear any awaiting_location state if a command is run
        if text.startswith("/"):
            session = self._get_session(chat_id)
            if isinstance(session, dict) and "awaiting_location" in session:
                session.pop("awaiting_location", None)
                self._set_session(chat_id, session)

        if text.startswith("/start"):
            if self._is_authenticated(chat_id):
                username = self._get_session(chat_id).get("username", "usuario")
                self._send_main_menu(chat_id, message=f"Bot ativo. Sessão autenticada como *{username}*.\n\nSelecione um módulo operacional abaixo:")
            else:
                self._prompt_for_username(chat_id)
            return

        if text.startswith("/status"):
            if self._is_authenticated(chat_id):
                session = self._get_session(chat_id)
                expiry_seconds = max(0, self.session_ttl_seconds - (self._now() - int(session.get('authenticated_at', 0) or 0)))
                expiry_minutes = max(1, int((expiry_seconds + 59) / 60))
                self._send_message(chat_id, f"Gateway Gemini ativo. Autenticado como {session.get('username', 'usuario')}. Sessão expira em cerca de {expiry_minutes} minuto(s).")
            else:
                self._send_message(chat_id, "Digite /start para iniciar")
            return

        if text.startswith("/logout") or text.startswith("/exit") or text.lower() == "exit":
            self._logout_session(chat_id, user_id, trigger_source=text, trigger_msg_id=msg_id)
            return

        if not self._is_authenticated(chat_id):
            msg_id = int(message.get("message_id") or 0)
            self._handle_auth_message(chat_id, user_id, text, msg_id)
            return

        self._touch_authenticated_session(chat_id)

        # Check session for pending states
        session = self._get_session(chat_id)

        # Admin: awaiting username to act on (activate/deactivate/delete)
        if isinstance(session, dict) and session.get("awaiting_admin_target"):
            action = session.pop("awaiting_admin_target")
            self._set_session(chat_id, session)
            self._admin_execute_action(chat_id, action, text)
            return

        # Admin: multi-step new user creation (username → password)
        if isinstance(session, dict) and session.get("awaiting_admin_new_user"):
            self._admin_handle_new_user_input(chat_id, text)
            return

        # Check if awaiting location input
        if isinstance(session, dict) and session.get("awaiting_location") == "recentes_14d":
            self._handle_location_input(chat_id, text)
            return

        # Free-form typing check: if not in a pending input flow, encourage inline menu only
        keyboard = [[{"text": "🏠 Menu Principal", "callback_data": "menu_main"}]]
        self._send_inline_keyboard(
            chat_id,
            "⚠️ *Entrada de texto livre desativada.*\n\n"
            "Para interagir com o *Report Preview*, utilize as opções e botões dos menus interativos. "
            "Digitações diretas no chat são aceitas apenas quando solicitadas explicitamente pelo sistema "
            "(como ao buscar bairros/cidades ou durante o login).\n\n"
            "Clique no botão abaixo para retornar ao menu principal:",
            keyboard
        )

    def _prompt_for_location(self, chat_id: int, message_id: int) -> None:
        session = self._get_session(chat_id)
        session["awaiting_location"] = "recentes_14d"
        self._set_session(chat_id, session)
        
        text = (
            "🏡 *ESCOLHER BAIRRO OU CIDADE (14 DIAS)*\n\n"
            "Digite o nome do bairro ou da cidade do Ceará que deseja analisar.\n\n"
            "💡 _Exemplos: 'Aldeota', 'Fortaleza', 'Caucaia'._\n"
            "💡 _Você pode enviar /exit para cancelar a qualquer momento ou clicar no botão abaixo._"
        )
        keyboard = [[{"text": "↩️ Cancelar", "callback_data": "menu_recentes"}]]
        self._send_inline_keyboard(chat_id, text, keyboard, edit_message_id=message_id)

    def _normalize_location(self, name: str) -> str:
        import unicodedata
        # Strip common quotation marks, whitespace, and brackets
        cleaned = str(name).strip("'`\"[](){}<> \t\r\n")
        nfkd_form = unicodedata.normalize('NFKD', cleaned)
        return "".join([c for c in nfkd_form if not unicodedata.combining(c)]).strip().upper()

    def _format_date_br(self, date_str: str) -> str:
        if not date_str:
            return ""
        date_str = str(date_str).strip()
        try:
            # Handle time component if present
            parts = date_str.split(" ")
            date_part = parts[0]
            time_part = " " + parts[1].split(".")[0] if len(parts) > 1 else ""
            
            # Skip if already in dd/mm/yyyy
            if "/" in date_part:
                return date_str
                
            # Format yyyy-mm-dd to dd/mm/yyyy
            if "-" in date_part:
                dt_parts = date_part.split("-")
                if len(dt_parts) == 3 and len(dt_parts[0]) == 4:
                    return f"{dt_parts[2]}/{dt_parts[1]}/{dt_parts[0]}{time_part}"
        except Exception:
            pass
        return date_str


    def _load_valid_locations(self) -> None:
        if hasattr(self, "_valid_cities_set"):
            return
        
        self._valid_cities_set = set()
        self._valid_neighborhoods_set = set()
        
        # 1. Load static locations from AISLookup
        try:
            import sys
            scripts_dir = str(self.project_root / "scripts")
            if scripts_dir not in sys.path:
                sys.path.insert(0, scripts_dir)
            from ais_lookup import AISLookup
            lookup = AISLookup(self.project_root)
            
            for city_norm in lookup._cidade_map.keys():
                self._valid_cities_set.add(city_norm)
            for city_norm, bairro_norm in lookup._bairro_map.keys():
                self._valid_neighborhoods_set.add(bairro_norm)
                
            logging.info("Carregados %d cidades e %d bairros estáticos via AISLookup.", len(self._valid_cities_set), len(self._valid_neighborhoods_set))
        except Exception as e:
            logging.error("Erro ao carregar locais validos da AISLookup: %s", e)
            
        # 2. Complement with dynamic locations from dados_brutos_90dias.csv
        path = self.project_root / "outputs" / "hermes" / "dados_brutos_90dias.csv"
        if not path.exists():
            path = self.project_root / "outputs" / "dados_brutos_90dias.csv"
            
        if path.exists():
            try:
                import csv
                with open(path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for r in reader:
                        cidade = r.get("cidade") or ""
                        bairro = r.get("bairro") or ""
                        
                        norm_c = self._normalize_location(cidade)
                        norm_b = self._normalize_location(bairro)
                        
                        if norm_c:
                            self._valid_cities_set.add(norm_c)
                        if norm_b:
                            self._valid_neighborhoods_set.add(norm_b)
                logging.info("Enriquecido localidade com dados dinamicos. Total: %d cidades e %d bairros.", len(self._valid_cities_set), len(self._valid_neighborhoods_set))
            except Exception as e:
                logging.error("Erro ao carregar locais dinamicos do CSV 90d: %s", e)
        else:
            logging.warning("CSV dados_brutos_90dias.csv nao encontrado para carregar locais dinamicos.")

    def _is_valid_location(self, name: str) -> tuple[bool, str, str]:
        self._load_valid_locations()
        norm = self._normalize_location(name)
        if not norm:
            return False, "", ""
            
        # 1. Exact match city
        if norm in self._valid_cities_set:
            return True, "cidade", norm
        # 2. Exact match neighborhood
        if norm in self._valid_neighborhoods_set:
            return True, "bairro", norm
            
        # 3. User input is a substring of target (e.g. "aldeot" -> "ALDEOTA", "barros" -> "BARROSO")
        if len(norm) >= 3:
            matched_bairros = [b for b in self._valid_neighborhoods_set if norm in b]
            matched_cities = [c for c in self._valid_cities_set if norm in c]
            
            if matched_bairros:
                matched_bairros.sort(key=lambda x: len(x) - len(norm))
                return True, "bairro", matched_bairros[0]
            if matched_cities:
                matched_cities.sort(key=lambda x: len(x) - len(norm))
                return True, "cidade", matched_cities[0]
                
        # 4. Target is a full word in user input (e.g. "bairro do barroso" -> "BARROSO")
        if len(norm) >= 3:
            import re
            for b in sorted(self._valid_neighborhoods_set, key=len, reverse=True):
                pattern = r'\b' + re.escape(b) + r'\b'
                if re.search(pattern, norm):
                    return True, "bairro", b
                    
            for c in sorted(self._valid_cities_set, key=len, reverse=True):
                pattern = r'\b' + re.escape(c) + r'\b'
                if re.search(pattern, norm):
                    return True, "cidade", c
                    
        return False, "", ""

    def _handle_location_input(self, chat_id: int, text: str, edit_message_id: int | None = None) -> None:
        is_valid, matched_type, matched_name = self._is_valid_location(text)
        if not is_valid:
            text_err = (
                "⚠️ *Localidade não identificada ou inválida.*\n\n"
                f"O texto '{text}' não correspondeu a nenhuma cidade ou bairro mapeado no Ceará.\n\n"
                "Por favor, digite um local válido (ex: 'Aldeota', 'Fortaleza', 'Caucaia').\n\n"
                "💡 _Ou clique no botão abaixo para cancelar._"
            )
            keyboard = [[{"text": "↩️ Cancelar", "callback_data": "menu_recentes"}]]
            self._send_inline_keyboard(chat_id, text_err, keyboard, edit_message_id=edit_message_id)
            self._log_conversation(chat_id, f"Busca por Localidade (Inválido): {text}", text_err)
            return

        # Clear state
        session = self._get_session(chat_id)
        session.pop("awaiting_location", None)
        self._set_session(chat_id, session)

        # Local deterministic calculation
        try:
            import csv
            csv_path = self.project_root / "outputs" / "hermes" / "dados_status_enriquecido_14d_latest.csv"
            if not csv_path.exists():
                csv_path = self.project_root / "outputs" / "dados_status_enriquecido_14d_latest.csv"

            if not csv_path.exists():
                self._send_inline_keyboard(
                    chat_id,
                    "⚠️ *Dados consolidados de 14 dias não localizados.*\nCertifique-se de que os arquivos do pipeline foram gerados na VPS.",
                    [[{"text": "↩️ Voltar", "callback_data": "menu_recentes"}]],
                    edit_message_id=edit_message_id
                )
                return

            cvli_count = 0
            cvp_count = 0
            events = []

            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row_cidade = self._normalize_location(row.get("cidade") or "")
                    row_bairro = self._normalize_location(row.get("bairro") or "")
                    
                    match = False
                    if matched_type == "cidade" and row_cidade == matched_name:
                        match = True
                    elif matched_type == "bairro" and row_bairro == matched_name:
                        match = True
                        
                    if match:
                        crime_type = (row.get("tipo") or row.get("type") or "").strip().lower()
                        if "cvli" in crime_type:
                            cvli_count += 1
                        elif "cvp" in crime_type:
                            cvp_count += 1
                        
                        dt = row.get("data") or ""
                        hr = row.get("hora") or ""
                        nature = row.get("nature") or row.get("tipo_evento") or "NÃO ESPECIFICADO"
                        street = row.get("name") or row.get("rua") or "NÃO ESPECIFICADA"
                        
                        events.append({
                            "data": dt,
                            "hora": hr,
                            "nature": nature,
                            "street": street,
                            "type": crime_type
                        })

            # Sort events by date and time (most recent first)
            events.sort(key=lambda x: (x["data"], x["hora"]), reverse=True)

            text_res = (
                f"🏡 *ESCOLHER BAIRRO/CIDADE (DADOS RECENTES - 14 DIAS)*\n\n"
                f"📍 *Localidade:* {matched_name} ({matched_type.upper()})\n\n"
                f"📊 *Volumetria no Período:*\n"
                f"• *CVLI* (Homicídios/Letais): *{cvli_count}* ocorrência(s)\n"
                f"• *CVP* (Roubos/Patrimoniais): *{cvp_count}* ocorrência(s)\n"
                f"• *Total Geral*: *{cvli_count + cvp_count}* ocorrência(s)\n\n"
            )

            if events:
                text_res += "🔥 *Últimas Ocorrências Registradas (Máx. 10):*\n"
                for idx, ev in enumerate(events[:10], 1):
                    formatted_dt = self._format_date_br(ev['data'])
                    text_res += f"*{idx}.* `[{formatted_dt}]` {ev['nature']} — _{ev['street']}_\n"
            else:
                text_res += "✅ *Nenhuma ocorrência criminal registrada nesta localidade nos últimos 14 dias.*\n"

            text_res += (
                "\n💡 _Para obter uma análise conceitual de inteligência e explicabilidade sobre esta localidade, clique no botão abaixo._"
            )

            keyboard = [
                [
                    {"text": "↩️ Voltar", "callback_data": "menu_recentes"},
                    {"text": "💡 Explicabilidade", "callback_data": f"recentes_escolher_explicabilidade:{matched_name}"}
                ]
            ]

            self._send_inline_keyboard(chat_id, text_res, keyboard, edit_message_id=edit_message_id)
            self._log_conversation(chat_id, f"Busca por Localidade: {text}", text_res)

        except Exception as e:
            logging.exception("Erro ao processar busca local de 14 dias para %s", matched_name)
            self._send_inline_keyboard(
                chat_id,
                f"❌ *Erro ao processar estatísticas locais para {matched_name}*:\n\n{e}",
                [[{"text": "↩️ Voltar", "callback_data": "menu_recentes"}]],
                edit_message_id=edit_message_id
            )

    def _show_explicabilidade(self, chat_id: int, message_id: int, callback_data: str) -> None:
        query = ""
        back_callback = "menu_main"
        module_name = ""
        
        # 1. ESCOLHER BAIRRO/CIDADE (14d detalhes)
        if callback_data.startswith("recentes_escolher_explicabilidade:"):
            location_name = callback_data.split(":", 1)[1]
            module_name = f"Explicação ({location_name})"
            back_callback = f"recentes_escolher_back:{location_name}"
            
            # Load real occurrences for the location
            local_events = []
            cvli_count = 0
            cvp_count = 0
            try:
                import csv
                csv_path = self.project_root / "outputs" / "hermes" / "dados_status_enriquecido_14d_latest.csv"
                if not csv_path.exists():
                    csv_path = self.project_root / "outputs" / "dados_status_enriquecido_14d_latest.csv"
                if csv_path.exists():
                    loc_norm = self._normalize_location(location_name)
                    with open(csv_path, "r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            row_cidade = self._normalize_location(row.get("cidade") or "")
                            row_bairro = self._normalize_location(row.get("bairro") or "")
                            if row_cidade == loc_norm or row_bairro == loc_norm:
                                crime_type = (row.get("tipo") or row.get("type") or "").strip().lower()
                                if "cvli" in crime_type:
                                    cvli_count += 1
                                elif "cvp" in crime_type:
                                    cvp_count += 1
                                street = row.get("name") or row.get("rua") or "NÃO ESPECIFICADA"
                                nature = row.get("nature") or row.get("tipo_evento") or "NÃO ESPECIFICADO"
                                formatted_dt = self._format_date_br(row.get('data'))
                                local_events.append(f"- [{formatted_dt}] {nature} na rua {street}")
            except Exception as e:
                logging.error("Erro ao carregar dados do local para explicabilidade: %s", e)

            events_summary = "\n".join(local_events[:15]) if local_events else "Nenhuma ocorrência registrada nos últimos 14 dias."
            query = (
                f"Você é um analista sênior de inteligência operacional de segurança pública. Entre IMEDIATAMENTE em modo analítico, técnico e tático, não genérico ou teórico.\n\n"
                f"Forneça uma análise de inteligência criminal extremamente precisa, direta e assertiva (máximo de 15 linhas) "
                f"sobre a dinâmica recente na localidade '{location_name}'. Justifique as métricas reais com base nas ocorrências listadas abaixo:\n\n"
                f"Estatísticas de 14 dias para {location_name}:\n"
                f"- CVLI (Homicídios/Letais): {cvli_count}\n"
                f"- CVP (Roubos/Patrimoniais): {cvp_count}\n\n"
                f"Lista de ocorrências reais:\n{events_summary}\n\n"
                f"Identifique hipóteses táticas plausíveis: controle territorial por facções, padrões de dias/ruas, dinâmicas de atração de roubos ou correlação de migração criminal."
            )

        # 2. DADOS RECENTES (14d resumo/ranking)
        elif callback_data == "recentes_ranking_explicabilidade":
            module_name = "Dados Recentes (14d)"
            back_callback = "recentes_ranking"
            summary_content = ""
            try:
                summary_path = self.project_root / "outputs" / "hermes" / "dados_status_enriquecido_14d_summary_latest.md"
                if summary_path.exists():
                    with open(summary_path, "r", encoding="utf-8") as sf:
                        summary_content = sf.read()
            except Exception as e:
                logging.error("Erro ao carregar resumo de 14 dias para explicabilidade: %s", e)
                
            query = (
                f"Você é um analista sênior de inteligência operacional de segurança pública. Entre IMEDIATAMENTE em modo analítico, técnico e tático, não genérico ou teórico.\n\n"
                f"Forneça uma análise estratégica de dinâmica criminal (máximo de 15 linhas) "
                f"sobre a dinâmica e ranking geral dos últimos 14 dias no Ceará com base no resumo dos dados reais abaixo:\n\n"
                f"Resumo dos dados reais de 14 dias:\n{summary_content[:3000]}\n\n"
                f"Destaque e analise as cidades e bairros que estão no topo de crimes patrimoniais (CVP) ou letais (CVLI). Apresente hipóteses de redes, conflitos operacionais e orientações táticas acionáveis."
            )

        # 3. SENTINELA (Micronodos Críticos)
        elif "sentinela" in callback_data:
            region = "geral"
            parts = callback_data.split("_")
            if len(parts) >= 3:
                region = parts[1]
            module_name = f"Sentinela ({region.capitalize()})"
            back_callback = f"sentinela_{region}"
            
            # Load top micronodes
            micronodes_text = ""
            try:
                import csv
                filename = f"top_30_micronodes_{region}.csv" if region != "geral" else "top_30_micronodes.csv"
                path = self.project_root / "outputs" / "hermes" / filename
                if not path.exists():
                    path = self.project_root / "outputs" / filename
                if path.exists():
                    nodes = []
                    with open(path, "r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        for r in reader:
                            nodes.append(
                                f"- {r.get('bairro') or 'Sem Bairro'} (Score: {r.get('risk_score') or r.get('score') or '?'}) | "
                                f"Facção: {r.get('faction') or r.get('faccao') or 'N/A'} | Ruas: {r.get('ruas') or r.get('streets') or 'Sem Ruas'}"
                            )
                    micronodes_text = "\n".join(nodes[:10])
            except Exception as e:
                logging.error("Erro ao carregar micronodos para explicabilidade: %s", e)
                
            query = (
                f"Você é um analista sênior de inteligência operacional de segurança pública. Entre IMEDIATAMENTE em modo analítico, técnico e tático, não genérico ou teórico.\n\n"
                f"Forneça uma análise tática sobre os micronodos críticos da região {region.upper()} (máximo de 15 linhas) "
                f"com base nos dados consolidados reais listados abaixo:\n\n"
                f"Top 10 Micronodos Críticos ({region.upper()}):\n{micronodes_text}\n\n"
                f"Explique a dinâmica operacional da dominância territorial das facções informadas nos respectivos bairros e eixos viários, justificando o porquê de esses pontos serem focos críticos de reincidência de homicídios (CVLI)."
            )

        # 4. CAMINHO DO CRIME (Vetor de Deslocamento)
        elif "rotas_caminho" in callback_data:
            module_name = "Caminho do Crime"
            crime_type = "cvli"
            region = "geral"
            days = 90
            parts = callback_data.split(":")
            if len(parts) == 4:
                crime_type = parts[1]
                region = parts[2]
                days = int(parts[3])
                back_callback = f"rotas_caminho_run:{crime_type}:{region}:{days}"
            elif len(parts) == 3:
                region = parts[1]
                days = int(parts[2])
                back_callback = f"rotas_caminho_run:{region}:{days}"
            else:
                back_callback = "rotas_caminho"
                
            # Load transitions
            transitions_text = ""
            try:
                import csv
                from datetime import datetime, timedelta
                path = self.project_root / "outputs" / "hermes" / "caminho_crime.csv"
                if path.exists():
                    all_rows = []
                    max_dt = None
                    with open(path, "r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        for r in reader:
                            if not (r.get("prox_bairro") or r.get("prox_rua")):
                                continue
                            r_type = (r.get("tipo") or "cvli").lower().strip()
                            if r_type != crime_type:
                                continue
                            dt_str = r.get("datetime")
                            if dt_str:
                                try:
                                    parsed_dt = datetime.strptime(dt_str.split(".")[0], "%Y-%m-%d %H:%M:%S")
                                    r["parsed_datetime"] = parsed_dt
                                    if max_dt is None or parsed_dt > max_dt:
                                        max_dt = parsed_dt
                                except Exception:
                                    try:
                                        parsed_dt = datetime.strptime(dt_str.split(".")[0], "%Y-%m-%d")
                                        r["parsed_datetime"] = parsed_dt
                                        if max_dt is None or parsed_dt > max_dt:
                                            max_dt = parsed_dt
                                    except Exception:
                                        pass
                            all_rows.append(r)

                    filtered_rows = []
                    for r in all_rows:
                        risp = (r.get("regiao_risp") or "").upper().strip()
                        if region == "capital":
                            if "CAPITAL" not in risp:
                                continue
                        elif region == "rmf":
                            if "RMF" not in risp:
                                continue
                        elif region == "interior":
                            if risp == "" or "CAPITAL" in risp or "RMF" in risp:
                                continue
                        filtered_rows.append(r)

                    if max_dt is not None:
                        cutoff = max_dt - timedelta(days=days)
                        time_filtered = [r for r in filtered_rows if r.get("parsed_datetime") and r["parsed_datetime"] >= cutoff]
                    else:
                        time_filtered = filtered_rows

                    latest_transitions = time_filtered[-10:]
                    latest_transitions.reverse()
                    
                    nodes = []
                    for idx, r in enumerate(latest_transitions, 1):
                        ais_raw = r.get('ais') or ''
                        display_ais = f"AIS {ais_raw}" if not str(ais_raw).upper().startswith("AIS") else ais_raw
                        formatted_dt = self._format_date_br(r.get('datetime'))
                        nodes.append(
                            f"{idx}. {display_ais} ({r.get('cidade')}) em {formatted_dt}: "
                            f"Origem {r.get('bairro')} (Rua {r.get('rua')}) -> Destino {r.get('prox_bairro')} (Rua {r.get('prox_rua')}) | "
                            f"Deslocamento: {r.get('distancia_para_prox_km')} km em {r.get('dias_para_prox')} dias"
                        )
                    transitions_text = "\n".join(nodes)
            except Exception as e:
                logging.error("Erro ao carregar transições para explicabilidade: %s", e)

            crime_label = "CVLI (Homicídios)" if crime_type == "cvli" else "CVP (Roubos)"
            query = (
                f"Você é um analista sênior de inteligência operacional de segurança pública. Entre IMEDIATAMENTE em modo analítico, técnico e tático, não genérico ou teórico.\n\n"
                f"Forneça uma análise de inteligência criminal extremamente assertiva e focada (máximo de 15 linhas) "
                f"sobre as rotas e migrações cronológicas sucessivas de {crime_label} na região {region.upper()} nos últimos {days} dias com base nos dados reais listados abaixo:\n\n"
                f"Rotas de Migração Recentes ({region.upper()}):\n{transitions_text}\n\n"
                f"Identifique o vetor migratório da criminalidade (se os focos estão migrando entre bairros limítrofes, mudando de AIS ou se intensificando na mesma área). Apresente hipóteses operacionais concretas."
            )

        # 5. RANKING POR RUAS (Logradouros Críticos)
        elif "rotas_ruas" in callback_data:
            module_name = "Ranking por Ruas"
            region = "geral"
            days = 90
            parts = callback_data.split(":")
            if len(parts) == 3:
                region = parts[1]
                days = int(parts[2])
                back_callback = f"rotas_ruas_run:{region}:{days}"
            else:
                back_callback = "rotas_ruas"
                
            # Load top streets
            streets_text = ""
            try:
                import csv as _csv
                from collections import defaultdict
                filename = f"dados_brutos_{days}dias.csv"
                path = self.project_root / "outputs" / "hermes" / filename
                if not path.exists():
                    path = self.project_root / "outputs" / filename
                if not path.exists() and days == 14:
                    alt = self.project_root / "outputs" / "hermes" / "dados_status_enriquecido_14d_latest.csv"
                    if not alt.exists():
                        alt = self.project_root / "outputs" / "dados_status_enriquecido_14d_latest.csv"
                    if alt.exists():
                        path = alt
                if path.exists():
                    rua_counts = defaultdict(lambda: {"cvli": 0, "bairro": "", "cidade": "", "rua": ""})
                    with open(path, "r", encoding="utf-8-sig") as f:
                        reader = _csv.DictReader(f)
                        for r in reader:
                            tipo = (r.get("tipo") or "").strip().lower()
                            if "cvli" not in tipo:
                                continue
                            risp = (r.get("regiao_risp") or "").upper().strip()
                            if region == "capital":
                                if "CAPITAL" not in risp:
                                    continue
                            elif region == "rmf":
                                if "RMF" not in risp:
                                    continue
                            elif region == "interior":
                                if risp == "" or "CAPITAL" in risp or "RMF" in risp:
                                    continue
                            rua = (r.get("name") or r.get("rua") or "").strip().upper()
                            if not rua or rua in ("NÃO ESPECIFICADA", "", "SEM NOME", "DESCONHECIDO"):
                                continue
                            bairro = (r.get("bairro") or "").strip().upper()
                            cidade = (r.get("cidade") or "").strip().upper()
                            key = f"{rua}|{bairro}|{cidade}"
                            rua_counts[key]["cvli"] += 1
                            rua_counts[key]["bairro"] = bairro
                            rua_counts[key]["cidade"] = cidade
                            rua_counts[key]["rua"] = rua
                    sorted_ruas = sorted(rua_counts.values(), key=lambda x: x["cvli"], reverse=True)
                    top10 = sorted_ruas[:10]
                    nodes = []
                    for idx, entry in enumerate(top10, 1):
                        nodes.append(f"{idx}. {entry['rua']} em {entry['bairro']}, {entry['cidade']} | Ocorrências CVLI: {entry['cvli']}")
                    streets_text = "\n".join(nodes)
            except Exception as e:
                logging.error("Erro ao carregar ruas críticas para explicabilidade: %s", e)

            query = (
                f"Você é um analista sênior de inteligência operacional de segurança pública. Entre IMEDIATAMENTE em modo analítico, técnico e tático, não genérico ou teórico.\n\n"
                f"Forneça uma análise operacional de vias urbanas críticas (máximo de 15 linhas) "
                f"sobre a recorrência e reincidência de homicídios no ranking de ruas da região {region.upper()} nos últimos {days} dias com base nos dados reais listados abaixo:\n\n"
                f"Ruas Críticas com Maior Reincidência de CVLI:\n{streets_text}\n\n"
                f"Identifique o porquê de estas ruas específicas concentrarem reincidência criminal recorrente (fatores urbanísticos, rotas de fuga, vulnerabilidade, divisão de territórios de facções) e proponha hipóteses para ações de patrulhamento cirúrgico."
            )

        # 6. JANELAS TEMPORAIS HISTÓRICAS
        elif "janelas" in callback_data:
            days = 90
            parts = callback_data.split("_")
            if len(parts) >= 2:
                days_str = parts[1].replace("d", "")
                try:
                    days = int(days_str)
                except:
                    pass
            module_name = f"Janela Temporal ({days}d)"
            back_callback = f"janelas_{days}d"
            
            # Load stats
            total = 0
            cvli_count = 0
            cvp_count = 0
            cidades_summary = ""
            bairros_summary = ""
            try:
                import csv
                from collections import Counter
                filename = f"dados_brutos_{days}dias.csv"
                path = self.project_root / "outputs" / "hermes" / filename
                if not path.exists():
                    path = self.project_root / "outputs" / filename
                if path.exists():
                    cidades = Counter()
                    bairros_fortaleza = Counter()
                    with open(path, "r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        for r in reader:
                            total += 1
                            tipo = (r.get("tipo") or "").lower().strip()
                            if "cvli" in tipo:
                                cvli_count += 1
                            elif "cvp" in tipo:
                                cvp_count += 1
                            cidade = (r.get("cidade") or "").upper().strip()
                            if cidade:
                                cidades[cidade] += 1
                            if "FORTALEZA" in cidade:
                                bairro = (r.get("bairro") or "").upper().strip()
                                if bairro:
                                    bairros_fortaleza[bairro] += 1
                    cidades_summary = "\n".join([f"- {cid}: {count} ocorrências" for cid, count in cidades.most_common(5)])
                    bairros_summary = "\n".join([f"- {bai}: {count} ocorrências" for bai, count in bairros_fortaleza.most_common(5)])
            except Exception as e:
                logging.error("Erro ao carregar dados de janelas temporais para explicabilidade: %s", e)

            query = (
                f"Você é um analista sênior de inteligência operacional de segurança pública. Entre IMEDIATAMENTE em modo analítico, técnico e tático, não genérico ou teórico.\n\n"
                f"Forneça uma análise de inteligência criminal extremamente focada (máximo de 15 linhas) "
                f"sobre a dinâmica acumulada nos últimos {days} dias com base nos dados reais consolidados abaixo:\n\n"
                f"Estatísticas Gerais ({days} dias):\n"
                f"- Total de Ocorrências: {total}\n"
                f"- CVLI (Homicídios/Letais): {cvli_count}\n"
                f"- CVP (Roubos/Patrimoniais): {cvp_count}\n\n"
                f"Top 5 Cidades:\n{cidades_summary}\n\n"
                f"Top 5 Bairros (Fortaleza):\n{bairros_summary}\n\n"
                f"Interprete o equilíbrio tático entre combater roubos (CVP) e homicídios (CVLI), destacando tendências de interiorização, estabilidade na capital ou aquecimento de rotas específicas."
            )

        # 7. CONTADORES (Cidade / Bairro / Natureza)
        elif callback_data == "contador_cidade_explicabilidade":
            module_name = "Contador por Cidade"
            back_callback = "contador_cidade"
            cities_text = ""
            try:
                import csv
                from collections import defaultdict
                path = self.project_root / "outputs" / "hermes" / "dados_brutos_90dias.csv"
                if not path.exists():
                    path = self.project_root / "outputs" / "dados_brutos_90dias.csv"
                if path.exists():
                    city_counts = defaultdict(lambda: {"cvli": 0, "cvp": 0, "total": 0})
                    with open(path, "r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        for r in reader:
                            cidade = (r.get("cidade") or "").strip().upper()
                            if not cidade:
                                continue
                            tipo = (r.get("tipo") or "").strip().lower()
                            if "cvli" in tipo:
                               city_counts[cidade]["cvli"] += 1
                               city_counts[cidade]["total"] += 1
                            elif "cvp" in tipo:
                               city_counts[cidade]["cvp"] += 1
                               city_counts[cidade]["total"] += 1
                    sorted_cities = sorted(city_counts.items(), key=lambda x: x[1]["cvli"], reverse=True)
                    nodes = []
                    for idx, (cid, counts) in enumerate(sorted_cities[:10], 1):
                        nodes.append(f"{idx}. {cid} | CVLI: {counts['cvli']} | CVP: {counts['cvp']} | Total: {counts['total']}")
                    cities_text = "\n".join(nodes)
            except Exception as e:
                logging.error("Erro ao carregar cidades para explicabilidade: %s", e)

            query = (
                f"Você é um analista sênior de inteligência operacional de segurança pública. Entre IMEDIATAMENTE em modo analítico, técnico e tático, não genérico ou teórico.\n\n"
                f"Forneça uma análise de inteligência criminal macrorregional por cidades (máximo de 15 linhas) "
                f"com base nos volumes consolidados de CVLI e CVP reais das 10 principais cidades do Ceará listadas abaixo:\n\n"
                f"Contagem por Cidade (90 dias):\n{cities_text}\n\n"
                f"Justifique taticamente a desproporção entre crimes patrimoniais (CVP) e letais (CVLI) em cada município, relacionando com a interiorização de facções ou presença de eixos logísticos."
            )

        elif callback_data == "contador_bairro_explicabilidade":
            module_name = "Contador por Bairro"
            back_callback = "contador_bairro"
            bairros_text = ""
            try:
                import csv
                from collections import defaultdict
                path = self.project_root / "outputs" / "hermes" / "dados_brutos_90dias.csv"
                if not path.exists():
                    path = self.project_root / "outputs" / "dados_brutos_90dias.csv"
                if path.exists():
                    bairro_counts = defaultdict(lambda: {"cvli": 0, "cvp": 0, "total": 0})
                    with open(path, "r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        for r in reader:
                            cidade = (r.get("cidade") or "").strip().upper()
                            if "FORTALEZA" not in cidade:
                                continue
                            bairro = (r.get("bairro") or "").strip().upper()
                            if not bairro:
                                continue
                            tipo = (r.get("tipo") or "").strip().lower()
                            if "cvli" in tipo:
                               bairro_counts[bairro]["cvli"] += 1
                               bairro_counts[bairro]["total"] += 1
                            elif "cvp" in tipo:
                               bairro_counts[bairro]["cvp"] += 1
                               bairro_counts[bairro]["total"] += 1
                    sorted_bairros = sorted(bairro_counts.items(), key=lambda x: x[1]["cvli"], reverse=True)
                    nodes = []
                    for idx, (bai, counts) in enumerate(sorted_bairros[:10], 1):
                        nodes.append(f"{idx}. {bai} | CVLI: {counts['cvli']} | CVP: {counts['cvp']} | Total: {counts['total']}")
                    bairros_text = "\n".join(nodes)
            except Exception as e:
                logging.error("Erro ao carregar bairros para explicabilidade: %s", e)

            query = (
                f"Você é um analista sênior de inteligência operacional de segurança pública. Entre IMEDIATAMENTE em modo analítico, técnico e tático, não genérico ou teórico.\n\n"
                f"Forneça uma análise de inteligência tática urbana e territorial (máximo de 15 linhas) "
                f"com base nos volumes consolidados de CVLI e CVP dos 10 bairros mais críticos de Fortaleza listados abaixo:\n\n"
                f"Contagem por Bairro em Fortaleza (90 dias):\n{bairros_text}\n\n"
                f"Interprete o porquê de certas regiões terem altíssimos índices de roubo (CVP) com baixa letalidade (CVLI), enquanto outras registram letalidade desenfreada, relacionando com dinâmicas locais."
            )

        elif callback_data == "contador_ais_explicabilidade":
            module_name = "Contador por AIS"
            back_callback = "contador_ais"
            ais_text = ""
            try:
                import csv
                from collections import defaultdict
                path = self.project_root / "outputs" / "hermes" / "dados_brutos_90dias.csv"
                if not path.exists():
                    path = self.project_root / "outputs" / "dados_brutos_90dias.csv"
                if path.exists():
                    ais_counts = defaultdict(lambda: {"cvli": 0, "cvp": 0, "total": 0})
                    with open(path, "r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        for r in reader:
                            ais_raw = (r.get("ais") or "").strip()
                            if not ais_raw:
                                continue
                            try:
                                clean_ais = str(ais_raw).upper().replace("AIS", "").strip()
                                if "." in clean_ais:
                                    ais_val = float(clean_ais)
                                    ais_str = f"{ais_val:.0f}"
                                else:
                                    ais_str = str(int(clean_ais))
                            except ValueError:
                                ais_str = clean_ais
                            display_ais = f"AIS {ais_str}" if not str(ais_str).upper().startswith("AIS") else ais_str
                            
                            tipo = (r.get("tipo") or "").strip().lower()
                            if "cvli" in tipo:
                               ais_counts[display_ais]["cvli"] += 1
                               ais_counts[display_ais]["total"] += 1
                            elif "cvp" in tipo:
                               ais_counts[display_ais]["cvp"] += 1
                               ais_counts[display_ais]["total"] += 1
                    sorted_ais = sorted(ais_counts.items(), key=lambda x: x[1]["cvli"], reverse=True)
                    nodes = []
                    for idx, (ais_name, counts) in enumerate(sorted_ais, 1):
                        nodes.append(f"{idx}. {ais_name} | CVLI: {counts['cvli']} | CVP: {counts['cvp']} | Total: {counts['total']}")
                    ais_text = "\n".join(nodes)
            except Exception as e:
                logging.error("Erro ao carregar AIS para explicabilidade: %s", e)

            query = (
                f"Você é um analista sênior de inteligência operacional de segurança pública. Entre IMEDIATAMENTE em modo analítico, técnico e tático, não genérico ou teórico.\n\n"
                f"Forneça uma análise de inteligência criminal sob o prisma de divisões de Áreas de Segurança Integrada (AIS) (máximo de 15 linhas) "
                f"com base nos volumes consolidados de CVLI e CVP reais das principais AIS listadas abaixo:\n\n"
                f"Contagem por AIS (90 dias):\n{ais_text}\n\n"
                f"Identifique quais AIS concentram a maior carga de violência letal (CVLI) e patrimonial (CVP). Justifique taticamente como a divisão administrativa das AIS impacta a alocação de recursos policiais e a coordenação de operações de saturação de área."
            )

        elif callback_data == "contador_natureza_explicabilidade" or "contador" in callback_data:
            module_name = "Contador por Natureza"
            back_callback = "contador_natureza" if "natureza" in callback_data else "menu_contador"
            nature_text = ""
            try:
                import csv
                from collections import defaultdict
                path = self.project_root / "outputs" / "hermes" / "dados_brutos_90dias.csv"
                if not path.exists():
                    path = self.project_root / "outputs" / "dados_brutos_90dias.csv"
                if path.exists():
                    nature_counts = defaultdict(int)
                    with open(path, "r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        for r in reader:
                            nature = (r.get("tipo_evento") or r.get("nature") or r.get("tipo") or "").strip().upper()
                            if not nature:
                                continue
                            nature_counts[nature] += 1
                    sorted_natures = sorted(nature_counts.items(), key=lambda x: x[1], reverse=True)
                    nodes = []
                    for idx, (nat, count) in enumerate(sorted_natures[:10], 1):
                        nodes.append(f"{idx}. {nat}: {count} ocorrências")
                    nature_text = "\n".join(nodes)
            except Exception as e:
                logging.error("Erro ao carregar naturezas para explicabilidade: %s", e)

            query = (
                f"Você é um analista sênior de inteligência operacional de segurança pública. Entre IMEDIATAMENTE em modo analítico, técnico e tático, não genérico ou teórico.\n\n"
                f"Forneça uma análise tática e comportamental de dinâmicas criminais (máximo de 15 linhas) "
                f"com base na distribuição por naturezas/tipologias de delitos do Ceará nos últimos 90 dias listadas abaixo:\n\n"
                f"Top 10 Naturezas Registradas (90 dias):\n{nature_text}\n\n"
                f"Analise a relação de causa-efeito e coexistência espacial dessas naturezas (por exemplo, a correlação entre roubos patrimoniais comuns, roubos a farmácias e assassinatos). Justifique os padrões observados nos dados."
            )

        # FALLBACK / GERAL
        else:
            query = (
                "Forneça uma explicação conceitual clara e concisa sobre as metodologias de cálculo de estatísticas e inteligência "
                "criminal integradas no Report Preview."
            )
            back_callback = "menu_main"
            module_name = "Report Preview"
            
        loading_text = f"💡 *EXPLICABILIDADE: {module_name.upper()}*\n\nConsultando os dados reais e gerando análise operacional de inteligência. Aguarde alguns instantes..."
        
        try:
            self._api("editMessageText", {
                "chat_id": chat_id,
                "message_id": message_id,
                "text": loading_text,
                "parse_mode": "Markdown"
            })
        except Exception:
            self._send_message(chat_id, loading_text)
            
        def worker() -> None:
            try:
                stop_event = threading.Event()
                def keep_typing():
                    while not stop_event.is_set():
                        self._send_typing(chat_id)
                        stop_event.wait(4)
                
                typer = threading.Thread(target=keep_typing, daemon=True)
                typer.start()
                
                answer = self._run_query(query, "geral", chat_id)
                stop_event.set()
                typer.join(timeout=1)
                
                keyboard = [[{"text": "↩️ Voltar", "callback_data": back_callback}, {"text": "🏠 Menu Principal", "callback_data": "menu_main"}]]
                
                self._send_inline_keyboard(
                    chat_id,
                    f"💡 *EXPLICABILIDADE METODOLÓGICA — {module_name.upper()}*\n\n{answer}",
                    keyboard
                )
                self._log_conversation(chat_id, f"Solicitou Explicabilidade ({module_name})", answer)
            except Exception as e:
                logging.exception("Falha ao gerar explicabilidade para %s", callback_data)
                self._send_inline_keyboard(
                    chat_id,
                    f"❌ *Erro ao carregar a explicabilidade para {module_name}*:\n\n{e}",
                    [[{"text": "↩️ Voltar", "callback_data": back_callback}, {"text": "🏠 Menu Principal", "callback_data": "menu_main"}]]
                )
                
        threading.Thread(target=worker, daemon=True).start()

    def _log_conversation(self, chat_id: int, user_message: str, bot_response: str) -> None:
        try:
            from datetime import datetime
            session = self._get_session(chat_id)
            username = session.get("username", f"user_{chat_id}")
            auth_ts = session.get("authenticated_at", self._now())
            
            auth_time = datetime.fromtimestamp(auth_ts)
            filename = f"session_{auth_time.strftime('%Y%m%d_%H%M%S')}.txt"
            
            user_dir = self.project_root / "outputs" / "users_chat" / username
            user_dir.mkdir(parents=True, exist_ok=True)
            file_path = user_dir / filename
            
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = (
                f"[{current_time}] USUÁRIO: {user_message}\n"
                f"[{current_time}] BOT: {bot_response}\n"
                f"{'-'*50}\n"
            )
            
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(log_entry)
                
            logging.info("Diálogo registrado para o usuário %s em %s", username, file_path.name)
        except Exception as e:
            logging.error("Falha ao registrar diálogo em users_chat: %s", e)

    def _log_system_event(self, chat_id: int, message: str) -> None:
        try:
            from datetime import datetime
            session = self._get_session(chat_id)
            username = session.get("username", f"user_{chat_id}")
            auth_ts = session.get("authenticated_at", self._now())
            
            auth_time = datetime.fromtimestamp(auth_ts)
            filename = f"session_{auth_time.strftime('%Y%m%d_%H%M%S')}.txt"
            
            user_dir = self.project_root / "outputs" / "users_chat" / username
            user_dir.mkdir(parents=True, exist_ok=True)
            file_path = user_dir / filename
            
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = (
                f"[{current_time}] SISTEMA: {message}\n"
                f"{'-'*50}\n"
            )
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(log_entry)
            logging.info("Evento do sistema registrado em log: %s", message)
        except Exception as e:
            logging.error("Falha ao registrar evento do sistema: %s", e)

    def _clear_chat_history(self, chat_id: int) -> None:
        session = self._get_session(chat_id)
        msg_ids = session.get("message_ids", [])
        if msg_ids:
            logging.info("Limpando chat para chat_id=%s. Total de mensagens: %d", chat_id, len(msg_ids))
            for msg_id in reversed(msg_ids):
                self._delete_message(chat_id, msg_id)
            session["message_ids"] = []
            self._set_session(chat_id, session)

    def run(self) -> None:
        while True:
            try:
                self._prune_all_expired_sessions()
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
    parser.add_argument("--hermes-workspace", default="")
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