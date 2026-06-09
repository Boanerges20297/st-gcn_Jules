import tempfile
import unittest
from unittest.mock import patch

from data.raw.inteligencia import import_orcrim_kml as orcrim_module
from src.core.health_monitor import HealthMonitor


class DummyResponse:
    def __init__(self, url, status_code=200, headers=None, text="", history=None, content=b""):
        self.url = url
        self.status_code = status_code
        self.headers = headers or {}
        self.text = text
        self.history = history or []
        self.content = content

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"{self.status_code} error")


class DummyHistoryItem:
    def __init__(self, location):
        self.headers = {"Location": location}


class HealthMonitorCompatibilityTests(unittest.TestCase):
    def test_get_active_alerts_returns_only_unresolved(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            monitor = HealthMonitor(base_dir=temp_dir)
            monitor.add_alert("critical_live", "CRITICAL", "still active", resolved=False)
            monitor.add_alert("resolved_old", "LOW", "already handled", resolved=True)

            active_alerts = monitor.get_active_alerts()

            self.assertEqual(len(active_alerts), 1)
            self.assertEqual(active_alerts[0]["type"], "critical_live")


class OrcrimAuthHandlingTests(unittest.TestCase):
    def test_raise_for_google_auth_failure_on_cookie_mismatch(self):
        response = DummyResponse(
            url="https://accounts.google.com/CookieMismatch",
            status_code=200,
            headers={"Content-Type": "text/html; charset=UTF-8"},
            text="CookieMismatch",
            history=[
                DummyHistoryItem("https://accounts.google.com/CookieMismatch"),
            ],
        )

        with self.assertRaises(orcrim_module.GoogleMapsAuthError):
            orcrim_module._raise_for_google_auth_failure(
                response,
                "https://www.google.com/maps/d/u/0/kml?mid=test",
            )

    def test_refresh_uses_old_local_base_only_as_last_resort(self):
        written_status = {}

        def fake_exists(path):
            if path in (orcrim_module.CURRENT_KML_PATH, orcrim_module.STATIC_KML_PATH):
                return True
            return False

        def fake_write(payload):
            written_status.update(payload)

        with patch.object(orcrim_module, "_log_existing_state"), \
             patch.object(orcrim_module, "_read_update_status", return_value={}), \
             patch.object(orcrim_module, "_write_update_status", side_effect=fake_write), \
             patch.object(orcrim_module, "_resolve_official_url", return_value="https://www.google.com/maps/d/u/0/kml?mid=test"), \
             patch.object(orcrim_module, "_download_official_payload", side_effect=orcrim_module.GoogleMapsAuthError("cookie invalido")), \
             patch.object(orcrim_module, "_download_payload_via_logged_in_chrome", side_effect=orcrim_module.ChromeProfileDownloadTimeout("chrome sem download")), \
             patch.object(orcrim_module.os.path, "exists", side_effect=fake_exists):
            result = orcrim_module.refresh_orcrim_from_official(force=True)

        self.assertFalse(result["updated"])
        self.assertEqual(result["reason"], "fallback_active")
        self.assertTrue(result["fallback_used"])
        self.assertEqual(written_status["status"], "fallback_active")
        self.assertIn("cookie invalido", written_status["last_error"])
        self.assertIn("chrome_profile: chrome sem download", written_status["last_error"])

    def test_refresh_uses_logged_in_chrome_recovery_by_default(self):
        with patch.object(orcrim_module, "_log_existing_state"), \
             patch.object(orcrim_module, "_read_update_status", return_value={}), \
             patch.object(orcrim_module, "_write_update_status"), \
             patch.object(orcrim_module, "_resolve_official_url", return_value="https://www.google.com/maps/d/u/0/kml?mid=test"), \
             patch.object(orcrim_module, "_download_official_payload", side_effect=orcrim_module.GoogleMapsAuthError("cookie invalido")), \
             patch.object(orcrim_module, "_download_payload_via_logged_in_chrome", return_value=(b"<kml></kml>", {"downloaded_via": "chrome_profile"})), \
             patch.object(orcrim_module, "_extract_kml_bytes_from_payload", return_value=b"<kml></kml>"), \
             patch.object(orcrim_module, "_get_content_hash", return_value="hash-1"), \
             patch.object(orcrim_module, "_get_semantic_content_hash", return_value="semantic-1"), \
             patch.object(orcrim_module, "_persist_kml_bytes"), \
             patch.object(orcrim_module, "_generate_intelligence_from_kml"), \
             patch.object(orcrim_module.os.path, "exists", return_value=False):
            result = orcrim_module.refresh_orcrim_from_official(force=True)

        self.assertTrue(result["updated"])
        self.assertEqual(result["reason"], "content_changed")


if __name__ == "__main__":
    unittest.main()
