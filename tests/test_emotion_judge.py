"""多情感角色：调用 MaiBot LLM 判断情感并选择对应模型（LLM 与 GPT-SoVITS 均为模拟）。"""
import asyncio
from datetime import datetime
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config_models import NO_LORA
import emotion_judge
from emotion_judge import build_prompt, parse_emotion
from plugin import create_plugin

ENABLED = ["neutral", "happy", "sad"]


class ParseTests(unittest.TestCase):
    def test_parse_accepts_ids_labels_aliases_and_rejects_others(self):
        cases = {"happy": "happy", " Happy. ": "happy", "\u0060sad\u0060": "sad", "情感：开心": "happy",
                 "<think>可能 sad</think>neutral": "neutral", "这句话很难过": "sad", "我认为是 sad，不是 happy": "sad",
                 "unhappy": "sad", "happiness": None, "生气": None, "": None, None: None, 7: None}
        for response, expected in cases.items():
            with self.subTest(response=response):
                self.assertEqual(parse_emotion(response, ENABLED), expected)

    def test_prompt_lists_only_enabled_emotions_default_and_text(self):
        prompt = build_prompt("今天下雨了", ["neutral", "sad", "自定义"], "neutral")
        for part in ("- neutral：中立", "- sad：难过", "- 自定义", "输出 neutral", "句子：今天下雨了"):
            self.assertIn(part, prompt)
        self.assertNotIn("happy", prompt)


class EmotionJudgeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.p = create_plugin()
        self.p._settings_path = Path(self.temp.name).resolve() / "config.toml"
        self.cfg = self.p.build_default_config()
        self.cfg["behavior"]["cooldown_seconds"] = 0
        self.cfg["role"].update(mode="role", role_id="shu/role.json", emotion="neutral", emotions=list(ENABLED))
        self.p.set_plugin_config(self.cfg)
        self.llm = SimpleNamespace(generate=AsyncMock(return_value={"success": True, "response": "sad", "model_name": "gpt-fast"}),
                                   get_available_models=AsyncMock(return_value=["utils", "fast_model", "embedding", "voice", "replyer"]))
        self.send = SimpleNamespace(custom=AsyncMock(return_value={"sent": True}), text=AsyncMock())
        self.logger = Mock()
        self.p._ctx = SimpleNamespace(send=self.send, llm=self.llm, logger=self.logger)
        await self.p.on_load()
        self.client = SimpleNamespace(synthesize=AsyncMock(return_value=b"RIFF-wav"), set_gpt_weights=AsyncMock(),
                                      set_sovits_weights=AsyncMock(), probe=AsyncMock(return_value="ok"))

        def resolve(role_id, emotion, root):
            return {"role_id": role_id, "emotion": emotion,
                    "models": {"base_model": "v5turbo", "gpt_weights": emotion + ".ckpt", "sovits_weights": "base.pth",
                               "lora_weights": emotion + "-lora.pth"},
                    "voice": {"ref_audio_path": emotion + ".wav", "prompt_text": emotion, "prompt_lang": "zh", "text_lang": "zh"}}
        library = SimpleNamespace(resolve=Mock(side_effect=resolve))

        async def paths(cfg):
            return cfg.models.gpt_weights, cfg.models.lora_weights
        for target, kwargs in (("_roles", {"return_value": library}), ("_client", {"return_value": self.client}),
                               ("_resolve_load_paths", {"side_effect": paths})):
            patcher = patch.object(self.p, target, **kwargs)
            patcher.start()
            self.addCleanup(patcher.stop)

    async def asyncTearDown(self):
        await self.p.on_unload()
        self.assertFalse(self.p._busy)

    def configure(self, section, **values):
        self.cfg[section].update(values)
        self.p.set_plugin_config(self.cfg)

    def used_emotion(self):
        return self.client.synthesize.await_args.args[0]["ref_audio_path"].removesuffix(".wav")

    async def test_command_without_emotion_asks_llm_and_uses_that_lora(self):
        result = await self.p.handle_speak(text="唉，今天的雨下个不停", stream_id="s")
        self.assertTrue(result["success"], result)
        self.llm.generate.assert_awaited_once()
        args, kwargs = self.llm.generate.await_args
        self.assertIn("唉，今天的雨下个不停", args[0])
        self.assertEqual(kwargs, {"task_name": "fast_model"})
        self.assertEqual(self.used_emotion(), "sad")
        self.client.set_sovits_weights.assert_awaited_with("sad-lora.pth")
        self.assertEqual(self.p._emotion_last["source"], "llm")
        self.assertEqual(self.p._emotion_last["model"], "gpt-fast")
        self.assertEqual(self.p.config.role.emotion, "neutral")

    async def test_custom_task_and_cache_for_repeated_sentences(self):
        self.configure("role", llm_task="utils")
        for _ in range(2):
            self.assertTrue((await self.p.handle_speak(text="好耶！", stream_id="s"))["success"])
        self.llm.generate.assert_awaited_once()
        self.assertEqual(self.llm.generate.await_args.kwargs["task_name"], "utils")
        await self.p.handle_speak(text="另一句话", stream_id="s")
        self.assertEqual(self.llm.generate.await_count, 2)

    async def test_explicit_emotion_single_emotion_original_or_disabled_skip_llm(self):
        self.assertTrue((await self.p.handle_speak(text="你好", stream_id="s", emotion="开心"))["success"])
        self.assertEqual(self.used_emotion(), "happy")
        self.configure("role", llm_emotion=False)
        await self.p.handle_speak(text="你好呀", stream_id="s")
        self.assertEqual(self.used_emotion(), "neutral")
        self.configure("role", llm_emotion=True, emotions=["neutral"])
        await self.p.handle_speak(text="你好啊", stream_id="s")
        self.llm.generate.assert_not_awaited()

    async def test_llm_failures_fall_back_to_default_without_blocking(self):
        failures = [{"success": False, "error": "quota"}, {"success": True, "response": "我不知道"},
                    RuntimeError("network"), "not-a-dict"]
        for index, failure in enumerate(failures):
            with self.subTest(failure=failure):
                self.llm.generate.reset_mock(side_effect=True, return_value=True)
                if isinstance(failure, Exception):
                    self.llm.generate.side_effect = failure
                else:
                    self.llm.generate.return_value = failure
                result = await self.p.handle_speak(text=f"第{index}句", stream_id="s")
                self.assertTrue(result["success"], result)
                self.assertEqual(self.used_emotion(), "neutral")
                self.assertEqual(self.p._emotion_last["source"], "llm_failed")
        self.assertEqual(self.logger.warning.call_count, len(failures))

    async def test_llm_timeout_uses_default(self):
        async def slow(*args, **kwargs):
            await asyncio.sleep(10)
        self.llm.generate.side_effect = slow
        with patch.object(emotion_judge, "LLM_TIMEOUT", 0.05):
            result = await self.p.handle_speak(text="慢一点", stream_id="s")
        self.assertTrue(result["success"], result)
        self.assertEqual(self.used_emotion(), "neutral")
        self.assertIn("超时", self.p._emotion_last["detail"])

    async def test_auto_voice_prefers_metadata_then_llm_then_keyword_fallback(self):
        self.configure("auto_voice", enabled=True, probability_percent=100, min_interval_seconds=0)
        synth = AsyncMock(return_value=b"RIFF-wav")
        now = str(datetime.now().timestamp())

        def message(text, msg_id, **extra):
            return {"message_id": msg_id, "timestamp": now, "session_id": "stream", "raw_message": [{"type": "text", "data": text}],
                    "message_info": {"additional_config": {}}, "is_command": False, **extra}
        with patch.object(self.p, "_synthesize_audio", synth):
            await self.p.handle_auto_voice(message=message("今天真开心！", "m1"))
            self.assertEqual(synth.await_args.kwargs["emotion"], "sad")
            self.assertEqual(self.p._auto_stats["last_emotion_source"], "llm")
            await self.p.handle_auto_voice(message=message("无所谓", "m2", metadata={"emotion": "happy"}))
            self.assertEqual(synth.await_args.kwargs["emotion"], "happy")
            self.assertEqual(self.llm.generate.await_count, 1)
            self.llm.generate.side_effect = RuntimeError("down")
            await self.p.handle_auto_voice(message=message("今天真开心呀！", "m3"))
            self.assertEqual(synth.await_args.kwargs["emotion"], "happy")
            self.assertEqual(self.p._auto_stats["last_emotion_source"], "llm_failed")
            self.configure("auto_voice", follow_emotion=False)
            calls = self.llm.generate.await_count
            await self.p.handle_auto_voice(message=message("非常难过", "m4"))
            self.assertEqual(synth.await_args.kwargs["emotion"], "neutral")
            self.assertEqual(self.llm.generate.await_count, calls)

    async def test_setup_apis_save_list_and_test(self):
        tasks = await self.p.setup_llm_tasks()
        self.assertEqual(tasks, {"success": True, "tasks": ["utils", "fast_model", "replyer"], "current": "fast_model"})
        saved = await self.p.setup_save_emotion_judge(False, " utils ")
        self.assertTrue(saved["success"], saved)
        self.assertEqual((self.p.config.role.llm_emotion, self.p.config.role.llm_task), (False, "utils"))
        self.assertIn("已关闭", saved["message"])
        self.assertTrue((await self.p.setup_save_emotion_judge(True, ""))["success"])
        self.assertEqual(self.p.config.role.llm_task, "fast_model")
        self.assertFalse((await self.p.setup_save_emotion_judge(True, "x" * 81))["success"])
        tested = await self.p.setup_test_emotion("下雨天好烦")
        self.assertTrue(tested["success"], tested)
        self.assertEqual((tested["emotion"], tested["label"], tested["model"]), ("sad", "难过", "gpt-fast"))
        self.client.synthesize.assert_not_awaited()
        self.assertFalse((await self.p.setup_test_emotion(""))["success"])
        self.assertFalse((await self.p.setup_test_emotion("长" * 301))["success"])
        self.llm.generate.return_value = {"success": True, "response": "?"}
        self.assertIn("没有已启用的情感", (await self.p.setup_test_emotion("嗯"))["message"])
        await self.p._save_setup({"role": {"emotions": ["neutral"]}})
        self.assertIn("只启用了 1 个情感", (await self.p.setup_test_emotion("嗯"))["message"])
        self.llm.get_available_models.side_effect = RuntimeError("no host")
        self.assertFalse((await self.p.setup_llm_tasks())["success"])

    async def test_check_config_reports_llm_emotion_state_in_step_three(self):
        roles = {"roles": [{"id": "shu/role.json", "emotions": [{"id": item} for item in ENABLED]}], "data_dir": "fixture"}
        with patch.object(self.p, "setup_roles", return_value=roles), \
                patch("setup_service.scan_models", return_value={"base_models": []}):
            def llm_issue():
                return [item for item in result["issues"] if item["field"] == "role.llm_emotion"]
            result = await self.p.setup_check_config()
            self.assertEqual([(item["level"], item["step"]) for item in llm_issue()], [("info", 3)])
            self.p._emotion_last = {"source": "llm_failed", "detail": "quota"}
            result = await self.p.setup_check_config()
            self.assertEqual(llm_issue()[0]["level"], "warning")
            self.assertIn("quota", llm_issue()[0]["message"])
            self.configure("role", llm_emotion=False)
            result = await self.p.setup_check_config()
            self.assertIn("关闭了 LLM", llm_issue()[0]["message"])
        state = await self.p.setup_state()
        self.assertEqual(state["emotion_judge"]["source"], "llm_failed")
        self.assertIn("llm_emotion", state["config"]["role"])


if __name__ == "__main__":
    unittest.main()
