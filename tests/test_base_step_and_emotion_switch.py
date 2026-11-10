"""第二步只保存底模、不触碰第三步设置；语音工具指定情感时按该情感切换权重。不访问真实服务。"""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plugin import create_plugin
from config_models import NO_LORA
import setup_service


class BaseStepAndEmotionSwitchTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.p = create_plugin()
        self.p._settings_path = Path(self.temp.name).resolve() / "config.toml"
        cfg = self.p.build_default_config()
        cfg["models"].update(base_model="custom", gpt_weights="old.ckpt", sovits_weights="old.pth", lora_weights="old-lora.pth")
        cfg["voice"]["ref_audio_path"] = "old.wav"
        cfg["behavior"]["cooldown_seconds"] = 0
        self.p.set_plugin_config(cfg)
        await self.p.on_load()
        self.catalog = {"base_models": [{"id": "v5turbo", "gpt_path": "base.ckpt", "sovits_path": "base.pth"}]}

    async def asyncTearDown(self):
        await self.p.on_unload()

    async def test_base_step_saves_only_preset_and_no_lora(self):
        voice = self.p.config.voice.model_dump()
        with patch.object(setup_service, "scan_models", return_value=self.catalog):
            result = await self.p.setup_save_base("C:/models", "base.pth")
        self.assertTrue(result["success"], result)
        self.assertEqual(result["base_model"], "v5turbo")
        self.assertEqual(self.p.config.models.base_model, "v5turbo")
        self.assertEqual(self.p.config.models.gpt_weights, "base.ckpt")
        self.assertEqual(self.p.config.models.sovits_weights, "base.pth")
        self.assertEqual(self.p.config.models.lora_weights, NO_LORA)
        self.assertEqual(self.p.config.voice.model_dump(), voice)
        self.assertEqual(self.p._role_status["status"], "idle")

    async def test_missing_base_does_not_overwrite_config(self):
        before = self.p.config.model_dump()
        with patch.object(setup_service, "scan_models", return_value={"base_models": []}):
            result = await self.p.setup_save_base("C:/models", "C:/missing/none.pth")
        self.assertFalse(result["success"])
        self.assertEqual(self.p.config.model_dump(), before)
        self.assertFalse(self.p._settings_path.exists())

    async def test_enabled_emotion_requests_switch_weights_but_not_saved_default(self):
        cfg = self.p.config.model_dump()
        cfg["role"].update(mode="role", role_id="shu/role.json", emotion="neutral", emotions=["neutral", "happy"])
        self.p.set_plugin_config(cfg)
        llm = SimpleNamespace(generate=AsyncMock(return_value={"success": True, "response": "neutral", "model": "fast"}))
        self.p._ctx = SimpleNamespace(send=SimpleNamespace(custom=AsyncMock(return_value=True)), llm=llm, logger=Mock())
        client = SimpleNamespace(synthesize=AsyncMock(return_value=b"wav"), set_gpt_weights=AsyncMock(), set_sovits_weights=AsyncMock())
        def resolve(role_id, emotion, root):
            return {"role_id": role_id, "emotion": emotion, "models": {"base_model": "v5turbo", "gpt_weights": emotion+".ckpt", "sovits_weights": "base.pth", "lora_weights": NO_LORA},
                    "voice": {"ref_audio_path": emotion+".wav", "prompt_text": emotion, "prompt_lang": "zh", "text_lang": "zh"}}
        library = SimpleNamespace(resolve=Mock(side_effect=resolve))
        async def paths(cfg):return cfg.models.gpt_weights, cfg.models.sovits_weights
        with patch.object(self.p, "_roles", return_value=library), patch.object(self.p, "_client", return_value=client), patch.object(self.p, "_resolve_load_paths", side_effect=paths):
            await self.p._speak("stream", "hello", emotion="happy")
            await self.p._speak("stream", "hello again", emotion="happy")
            self.assertEqual(client.set_gpt_weights.await_count, 1)
            self.assertEqual(client.synthesize.await_args.args[0]["ref_audio_path"], "happy.wav")
            result = await self.p.handle_speak(text="bad emotion", stream_id="stream", emotion="sad")
            self.assertFalse(result["success"])
            await self.p._speak("stream", "default")
        self.assertEqual(client.set_gpt_weights.await_count, 2)
        # 只有未指定情感的一次请求调用 LLM（启用了 2 个情感）。
        llm.generate.assert_awaited_once()
        self.assertEqual(llm.generate.await_args.kwargs["task_name"], "fast_model")
        self.assertEqual(client.synthesize.await_args.args[0]["ref_audio_path"], "neutral.wav")
        self.assertEqual(self.p.config.role.emotion, "neutral")
        self.assertEqual(self.p.config.role.emotions, ["neutral", "happy"])


if __name__ == "__main__":
    unittest.main()
