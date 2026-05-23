import sys
import time
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

# Add project root to sys.path
project_root = Path(r"c:\Users\Boanerges\Desktop\Projetos\Report Preview")
sys.path.insert(0, str(project_root))

from powershell.telegram_gemini_gateway import TelegramGeminiGateway

class TestGatewayLocalMethods(unittest.TestCase):
    def setUp(self):
        # We instantiate a mock instance of the gateway
        self.gateway = TelegramGeminiGateway(
            project_root=project_root,
            hermes_workspace=None,
            gemini_model="gemini-2.5-flash",
            chat_runtime_dir=project_root / "outputs" / "hermes" / "chat"
        )
        # Mock the API calling methods
        self.gateway._send_inline_keyboard = MagicMock()
        self.gateway._send_message = MagicMock()
        self.gateway._run_query = MagicMock(return_value="Mocked Gemini response.")

    def test_handle_location_input_valid_bairro(self):
        chat_id = 12345
        # "Barroso" is a valid neighborhood
        self.gateway._handle_location_input(chat_id, "Barroso")
        
        # Verify that _send_inline_keyboard was called
        self.gateway._send_inline_keyboard.assert_called_once()
        args, kwargs = self.gateway._send_inline_keyboard.call_args
        
        # Check text contents
        text = args[1]
        keyboard = args[2]
        
        self.assertIn("🏡 *ESCOLHER BAIRRO/CIDADE (DADOS RECENTES - 14 DIAS)*", text)
        self.assertIn("BARROSO", text)
        self.assertIn("BAIRRO", text)
        self.assertIn("CVLI", text)
        self.assertIn("CVP", text)
        
        # Check that there is an Explicabilidade button with the correct callback
        self.assertEqual(keyboard[0][1]["text"], "💡 Explicabilidade")
        self.assertEqual(keyboard[0][1]["callback_data"], "recentes_escolher_explicabilidade:BARROSO")

    def test_show_explicabilidade_location(self):
        chat_id = 12345
        message_id = 9999
        
        self.gateway._show_explicabilidade(chat_id, message_id, "recentes_escolher_explicabilidade:BARROSO")
        
        # Wait a brief moment for the background thread to run
        time.sleep(0.5)
        
        # It should trigger a query to Gemini (via _run_query)
        self.gateway._run_query.assert_called_once()
        query_arg = self.gateway._run_query.call_args[0][0]
        self.assertIn("BARROSO", query_arg)
        self.assertIn("10 linhas", query_arg)

    def test_show_contador_natureza(self):
        chat_id = 12345
        message_id = 9999
        self.gateway._show_contador_natureza(chat_id, message_id)
        
        # Verify that _send_inline_keyboard was called
        self.gateway._send_inline_keyboard.assert_called_once()
        args, kwargs = self.gateway._send_inline_keyboard.call_args
        
        text = args[1]
        keyboard = args[2]
        
        self.assertIn("📊 *CONTADOR POR NATUREZA DO CRIME (90 DIAS)*", text)
        self.assertIn("ROUBO A PESSOA", text)
        
        # Check that there is an Explicabilidade button with the correct callback
        self.assertEqual(keyboard[0][1]["text"], "💡 Explicabilidade")
        self.assertEqual(keyboard[0][1]["callback_data"], "contador_natureza_explicabilidade")

    def test_message_id_tracking_via_api(self):
        chat_id = 12345
        # Set session to authenticated so it tracks
        self.gateway._set_session(chat_id, {
            "authenticated": True,
            "username": "test_user",
            "authenticated_at": int(time.time()),
        })
        
        # Mock _api return
        with patch('urllib.request.urlopen') as mock_urlopen:
            mock_res = MagicMock()
            mock_res.read.return_value = b'{"ok": true, "result": {"message_id": 789, "chat": {"id": 12345}}}'
            mock_urlopen.return_value.__enter__.return_value = mock_res
            
            self.gateway._api("sendMessage", {"chat_id": chat_id, "text": "Hello"})
            
            session = self.gateway._get_session(chat_id)
            self.assertIn(789, session.get("message_ids", []))

    def test_logout_session_clears_history(self):
        chat_id = 12345
        # Set session with message_ids
        self.gateway._set_session(chat_id, {
            "authenticated": True,
            "username": "test_user",
            "authenticated_at": int(time.time()),
            "message_ids": [101, 102, 103]
        })
        
        # Mock deleteMessage API calls
        self.gateway._api = MagicMock(return_value={"ok": True})
        
        self.gateway._logout_session(chat_id, 12345, trigger_source="exit command", trigger_msg_id=104)
        
        # Verify that trigger message was tracked before delete
        # It should have called deleteMessage for 101, 102, 103, and 104
        api_calls = [call[0][0] for call in self.gateway._api.call_args_list]
        self.assertIn("deleteMessage", api_calls)
        
        # Session should be reset to unauthenticated
        session = self.gateway._get_session(chat_id)
        self.assertFalse(session.get("authenticated"))
        self.assertEqual(session.get("awaiting"), None)
        self.assertEqual(session.get("message_ids", []), [])

    def test_prune_expired_session(self):
        chat_id = 12345
        # Expired session (e.g. 20 minutes ago)
        self.gateway._set_session(chat_id, {
            "authenticated": True,
            "username": "test_user",
            "authenticated_at": int(time.time()) - 1200,
            "message_ids": [201, 202]
        })
        self.gateway._api = MagicMock(return_value={"ok": True})
        
        # Calling _is_authenticated should trigger prune
        is_auth = self.gateway._is_authenticated(chat_id)
        self.assertFalse(is_auth)
        
        # Session should be reset
        session = self.gateway._get_session(chat_id)
        self.assertFalse(session.get("authenticated"))
        self.assertTrue(session.get("session_expired"))
        self.assertEqual(session.get("message_ids", []), [])

    def test_unauthenticated_message_forces_start(self):
        chat_id = 12345
        self.gateway._set_session(chat_id, {
            "authenticated": False,
            "awaiting": None,
        })
        
        handled = self.gateway._handle_auth_message(chat_id, 12345, "Oi")
        self.assertTrue(handled)
        self.gateway._send_message.assert_called_with(chat_id, "Digite /start para iniciar")
        
        session = self.gateway._get_session(chat_id)
        self.assertEqual(session.get("awaiting"), None)

    def test_expired_session_message_forces_start(self):
        chat_id = 12345
        self.gateway._set_session(chat_id, {
            "authenticated": False,
            "awaiting": None,
            "session_expired": True
        })
        
        handled = self.gateway._handle_auth_message(chat_id, 12345, "Qualquer mensagem")
        self.assertTrue(handled)
        self.gateway._send_message.assert_called_with(chat_id, "Sua sessão expirou. Digite /start para iniciar")
        
        session = self.gateway._get_session(chat_id)
        self.assertEqual(session.get("awaiting"), None)
        self.assertFalse(session.get("session_expired"))

    def test_unauthenticated_status_forces_start(self):
        chat_id = 12345
        self.gateway._set_session(chat_id, {
            "authenticated": False,
            "awaiting": None,
        })
        
        # Simulate /status message in handle_update
        message = {
            "message_id": 1,
            "chat": {"id": chat_id},
            "text": "/status",
            "from": {"id": chat_id}
        }
        
        self.gateway.handle_update({"message": message})
        self.gateway._send_message.assert_called_with(chat_id, "Digite /start para iniciar")

if __name__ == "__main__":
    unittest.main()
