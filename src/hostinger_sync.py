from __future__ import annotations

import hashlib
import json
import os
import posixpath
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

try:
    from dotenv import dotenv_values
except ImportError:  # pragma: no cover - dependency exists in runtime requirements
    dotenv_values = None


DEFAULT_TARGET_DIR = '/home/reportpreview/apps/report-preview'


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'on'}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class HostingerSyncConfig:
    enabled: bool
    host: str
    user: str
    password: str
    port: int
    target_dir: str
    timeout_seconds: int

    @property
    def is_configured(self) -> bool:
        return self.enabled and bool(self.host and self.user and self.password)


class HostingerSyncManager:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root).resolve()
        self.state_path = self.project_root / 'logs' / 'hostinger_sync_state.json'
        self.config = self._load_config()

    @staticmethod
    def build_risk_fingerprint(artifact: dict) -> str:
        stable_payload = {
            'data_limit': artifact.get('data_limit'),
            'status_enriquecido_14d': {
                'reference_date': (artifact.get('status_enriquecido_14d') or {}).get('reference_date'),
                'row_count': (artifact.get('status_enriquecido_14d') or {}).get('row_count'),
                'window_days': (artifact.get('status_enriquecido_14d') or {}).get('window_days'),
            },
            'rankings': artifact.get('rankings', {}),
        }
        encoded = json.dumps(stable_payload, ensure_ascii=False, sort_keys=True).encode('utf-8')
        return _sha256_bytes(encoded)

    def sync_risk_artifacts(self, artifact: dict) -> dict:
        fingerprint = self.build_risk_fingerprint(artifact)
        relative_paths = [
            'outputs/hermes/risk_snapshot_latest.json',
            'outputs/hermes/risk_snapshot_latest.md',
            'outputs/hermes/risk_brief_latest.md',
            'outputs/hermes/risk_snapshot_latest.csv',
            'outputs/hermes/risk_fortaleza_latest.csv',
            'outputs/hermes/risk_rmf_latest.csv',
            'outputs/hermes/risk_interior_latest.csv',
            'outputs/hermes/dados_status_enriquecido_14d_latest.csv',
            'outputs/hermes/ruas_criticas_latest.csv',
            'outputs/hermes/visible_micronodes.csv',
            'outputs/hermes/top_30_micronodes.csv',
            'outputs/hermes/top_30_micronodes_capital.csv',
            'outputs/hermes/top_30_micronodes_rmf.csv',
            'outputs/hermes/top_30_micronodes_interior.csv',
        ]
        return self._sync_event('risk_outputs', fingerprint, relative_paths)

    def sync_data_merge_artifacts(self) -> dict:
        relative_paths = [
            'data/raw/dados_status_ocorrencias_gerais_ENRIQUECIDO.csv',
            'data/geo_streets_cache.json',
            'VALIDATION_LOG.md',
        ]
        fingerprint = self._build_files_fingerprint(relative_paths)
        return self._sync_event('data_merge', fingerprint, relative_paths)

    def _load_config(self) -> HostingerSyncConfig:
        env_data: dict[str, str] = {}
        if dotenv_values is not None:
            for dotenv_path in (
                self.project_root / '.env.hostinger.example',
                self.project_root / '.env.hostinger',
                self.project_root / '.env',
            ):
                if dotenv_path.exists():
                    env_data.update({
                        key: value
                        for key, value in dotenv_values(dotenv_path).items()
                        if value is not None
                    })
        env_data.update(os.environ)

        host = str(env_data.get('HOSTINGER_SYNC_HOST', '')).strip()
        user = str(env_data.get('HOSTINGER_SYNC_USER', '')).strip()
        password = str(env_data.get('HOSTINGER_SYNC_PASSWORD', '')).strip()

        # Fallback para variáveis já usadas nos scripts de deploy do projeto.
        host_ssh = str(env_data.get('HOST_SSH', '')).strip()
        if (not host or not user) and host_ssh:
            if '@' in host_ssh:
                parsed_user, parsed_host = host_ssh.split('@', 1)
                if not user:
                    user = parsed_user.strip()
                if not host:
                    host = parsed_host.strip()
            elif not host:
                host = host_ssh
        if not password:
            password = str(env_data.get('PASSWORD_VPS_SSH', '')).strip()

        return HostingerSyncConfig(
            enabled=str(env_data.get('HOSTINGER_SYNC_ENABLED', 'false')).strip().lower() in {'1', 'true', 'yes', 'on'},
            host=host,
            user=user,
            password=password,
            port=int(str(env_data.get('HOSTINGER_SYNC_PORT', '22'))),
            target_dir=str(env_data.get('HOSTINGER_SYNC_TARGET_DIR', DEFAULT_TARGET_DIR)).strip() or DEFAULT_TARGET_DIR,
            timeout_seconds=int(str(env_data.get('HOSTINGER_SYNC_TIMEOUT_SECONDS', '30'))),
        )

    def _build_files_fingerprint(self, relative_paths: Iterable[str]) -> str:
        manifest = []
        for relative_path in relative_paths:
            full_path = self.project_root / relative_path
            if not full_path.exists():
                manifest.append({'path': relative_path, 'exists': False})
                continue
            stat = full_path.stat()
            manifest.append({
                'path': relative_path,
                'exists': True,
                'size': stat.st_size,
                'mtime_ns': stat.st_mtime_ns,
            })

        payload = json.dumps(manifest, sort_keys=True).encode('utf-8')
        return _sha256_bytes(payload)

    def _sync_event(self, key: str, fingerprint: str, relative_paths: Iterable[str]) -> dict:
        if not self.config.is_configured:
            reason = []
            if not self.config.enabled:
                reason.append('HOSTINGER_SYNC_ENABLED=false')
            if not self.config.host:
                reason.append('HOSTINGER_SYNC_HOST/HOST_SSH ausente')
            if not self.config.user:
                reason.append('HOSTINGER_SYNC_USER/HOST_SSH ausente')
            if not self.config.password:
                reason.append('HOSTINGER_SYNC_PASSWORD/PASSWORD_VPS_SSH ausente')
            return {'status': 'disabled', 'reason': '; '.join(reason) or 'hostinger sync not configured'}

        state = self._load_state()
        current = state.get(key, {})
        if current.get('fingerprint') == fingerprint:
            return {'status': 'skipped', 'reason': 'unchanged', 'fingerprint': fingerprint}

        uploaded = self._upload_relative_files(relative_paths)
        state[key] = {
            'fingerprint': fingerprint,
            'updated_at': datetime.now().isoformat(timespec='seconds'),
            'uploaded_files': uploaded,
        }
        self._save_state(state)
        return {'status': 'synced', 'fingerprint': fingerprint, 'uploaded_files': uploaded}

    def _load_state(self) -> dict:
        if not self.state_path.exists():
            return {}
        try:
            return json.loads(self.state_path.read_text(encoding='utf-8'))
        except Exception:
            return {}

    def _save_state(self, state: dict) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile('w', delete=False, encoding='utf-8', dir=self.state_path.parent) as handle:
            json.dump(state, handle, indent=2, ensure_ascii=False)
            temp_name = handle.name
        os.replace(temp_name, self.state_path)

    def _upload_relative_files(self, relative_paths: Iterable[str]) -> list[str]:
        try:
            import paramiko
        except ImportError as exc:  # pragma: no cover - surfaced by runtime validation instead
            raise RuntimeError('paramiko is required for Hostinger sync automation') from exc

        uploaded: list[str] = []
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(
            hostname=self.config.host,
            port=self.config.port,
            username=self.config.user,
            password=self.config.password,
            timeout=self.config.timeout_seconds,
            banner_timeout=self.config.timeout_seconds,
            auth_timeout=self.config.timeout_seconds,
        )

        sftp = ssh.open_sftp()
        try:
            self._ensure_remote_dir(ssh, self.config.target_dir)
            for relative_path in relative_paths:
                local_path = self.project_root / relative_path
                if not local_path.exists() or not local_path.is_file():
                    continue

                remote_path = posixpath.join(
                    self.config.target_dir,
                    relative_path.replace('\\', '/'),
                )
                self._ensure_remote_dir(ssh, posixpath.dirname(remote_path))
                sftp.put(str(local_path), remote_path)
                uploaded.append(relative_path)
        finally:
            sftp.close()
            ssh.close()

        return uploaded

    @staticmethod
    def _ensure_remote_dir(ssh_client, remote_dir: str) -> None:
        safe_dir = remote_dir.replace("'", "'\"'\"'")
        command = f"mkdir -p '{safe_dir}'"
        _, stdout, stderr = ssh_client.exec_command(command)
        exit_status = stdout.channel.recv_exit_status()
        if exit_status != 0:
            message = stderr.read().decode('utf-8', errors='ignore').strip()
            raise RuntimeError(f'failed to prepare remote directory {remote_dir}: {message}')
