"""边界情况：语音工具的情感写法、普通模式自动语音的权重、带 BOM 的配置文件、接口地址错误提示。"""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config_models import NO_LORA
import plugin as plugin_module


class EdgeCaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="maibot-edge-cases-")
        self.addCleanup(self.temp.cleanup)
        self.p = plugin_module.create_plugin()
        self.p._settings_path = Path(self.temp.name).resolve() / "config.toml"
        self.data = self.p.build_default_config()
        self.data["voice"]["ref_audio_path"] = "/server/ref.wav"
        self.data["behavior"]["cooldown_seconds"] = 0
        self.p.set_plugin_config(self.data)
        self.logger = Mock()
        patcher = patch.object(self.p, "_get_logger", return_value=self.logger)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.p._ctx = SimpleNamespace(send=SimpleNamespace(custom=AsyncMock(return_value={"sent": True}),
                                                           text=AsyncMock()))
        self.client = SimpleNamespace(synthesize=AsyncMock(return_value=b"wav"),
                                      set_gpt_weights=AsyncMock(), set_sovits_weights=AsyncMock(),
                                      probe=AsyncMock(return_value="ok"))
        await self.p.on_load()

    async def asyncTearDown(self):
        await self.p.on_unload()
        self.assertFalse(self.p._busy)

    def configure(self, section, **changes):
        self.data[section].update(changes)
        self.p.set_plugin_config(self.data)

    async def test_tool_emotion_whitespace_case_and_alias_map_to_enabled(self):
        self.configure("role", mode="role", role_id="shu/role.json", emotion="neutral", emotions=["neutral", "happy"])

        def resolve(role_id, emotion, root):
            return {"role_id": role_id, "emotion": emotion,
                    "models": {"base_model": "v5turbo", "gpt_weights": emotion + ".ckpt",
                               "sovits_weights": "base.pth", "lora_weights": NO_LORA},
                    "voice": {"ref_audio_path": emotion + ".wav", "prompt_text": emotion,
                              "prompt_lang": "zh", "text_lang": "zh"}}
        library = SimpleNamespace(resolve=Mock(side_effect=resolve))

        async def paths(cfg):
            return cfg.models.gpt_weights, cfg.models.sovits_weights
        with patch.object(self.p, "_roles", return_value=library), \
                patch.object(self.p, "_client", return_value=self.client), \
                patch.object(self.p, "_resolve_load_paths", side_effect=paths):
            for value in (" happy ", "Happy", "开心"):
                result = await self.p.handle_speak(text="你好", stream_id="s", emotion=value)
                self.assertTrue(result["success"], (value, result))
                self.assertEqual(self.client.synthesize.await_args.args[0]["ref_audio_path"], "happy.wav")
            refused = await self.p.handle_speak(text="你好", stream_id="s", emotion="sad")
            self.assertFalse(refused["success"])
        self.assertEqual(self.p.config.role.emotion, "neutral")

    async def test_original_auto_voice_without_weights_uses_server_current_weights(self):
        self.configure("auto_voice", enabled=True)
        with patch.object(self.p, "_client", return_value=self.client):
            result = await self.p.handle_speak(text="你好", stream_id="s")
        self.assertTrue(result["success"], result)
        self.client.set_gpt_weights.assert_not_awaited()
        self.client.synthesize.assert_awaited_once()

    async def test_original_auto_voice_with_weights_still_loads_once(self):
        self.configure("auto_voice", enabled=True)
        self.configure("models", gpt_weights="/server/a.ckpt", sovits_weights="/server/a.pth")
        with patch.object(self.p, "_client", return_value=self.client):
            for _ in range(2):
                result = await self.p.handle_speak(text="你好", stream_id="s")
                self.assertTrue(result["success"], result)
        self.client.set_gpt_weights.assert_awaited_once_with("/server/a.ckpt")
        self.client.set_sovits_weights.assert_awaited_once_with("/server/a.pth")

    async def test_invalid_base_url_reports_config_error_without_traceback(self):
        self.configure("api", base_url="ftp://127.0.0.1:9880")
        result = await self.p.handle_speak(text="你好", stream_id="s")
        self.assertFalse(result["success"])
        self.assertIn("接口地址", result["content"])
        self.logger.exception.assert_not_called()
        ok, message, _ = await self.p.handle_status(stream_id="s")
        self.assertFalse(ok)
        self.assertIn("接口地址", message)
        self.logger.exception.assert_not_called()

    async def test_status_in_role_mode_does_not_claim_missing_reference(self):
        self.configure("voice", ref_audio_path="")
        self.configure("role", mode="role", role_id="shu/role.json")
        with patch.object(plugin_module, "GPTSoVITSClient", return_value=self.client):
            ok, message, _ = await self.p.handle_status(stream_id="s")
        self.assertTrue(ok, message)
        self.assertIn("由角色JSON提供", message)

    async def test_save_setup_accepts_utf8_bom_config(self):
        await self.p._save_setup({})
        text = self.p._settings_path.read_text(encoding="utf-8")
        self.p._settings_path.write_bytes(b"\xef\xbb\xbf" + text.encode("utf-8"))
        result = await self.p.setup_save_api("http://127.0.0.1:9999", 60)
        self.assertTrue(result["success"], result)
        self.assertEqual(self.p.config.api.base_url, "http://127.0.0.1:9999")
        self.assertFalse(self.p._settings_path.read_bytes().startswith(b"\xef\xbb\xbf"))


if __name__ == "__main__":
    unittest.main()
