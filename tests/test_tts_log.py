"""每次语音合成在 MaiBot 日志输出开始与结果（LLM、GPT-SoVITS 与发送均为模拟）。"""
from datetime import datetime
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plugin import create_plugin
import setup_service
import tts_log


def wav(seconds=0.5, rate=16000):
    data = io.BytesIO()
    with wave.open(data, "wb") as audio:
        audio.setnchannels(1); audio.setsampwidth(2); audio.setframerate(rate)
        audio.writeframes(b"\0\1" * int(rate * seconds))
    return data.getvalue()


def group_message(text="/tts 你好", group="测试群", group_id="123456"):
    return {"message_info": {"group_info": {"group_id": group_id, "group_name": group},
                             "user_info": {"user_id": "10001", "user_nickname": "小明"}},
            "processed_plain_text": text}


class TraceFormatTests(unittest.TestCase):
    def test_chat_labels_and_preview(self):
        self.assertEqual(tts_log.chat_label(group_message()), "群 测试群（123456）")
        private = {"message_info": {"user_info": {"user_id": "10001", "user_cardname": "", "user_nickname": "小明"}}}
        self.assertEqual(tts_log.chat_label(private), "私聊 小明（10001）")
        outgoing = {"message_info": {"user_info": {"user_id": "bot"}, "additional_config": {"platform_io_target_user_id": "10001"}}}
        self.assertEqual(tts_log.chat_label(outgoing, outgoing=True), "私聊 10001")
        self.assertEqual(tts_log.chat_label(None), "")
        self.assertEqual(tts_log.preview("  第一行\n第二行  "), "第一行 第二行")
        self.assertEqual(tts_log.preview("字" * 50), "字" * 40 + "…")
        self.assertEqual(tts_log.filename("C:\\models\\黍\\shu_sad.ckpt"), "shu_sad.ckpt")
        self.assertEqual(tts_log.audio_info(wav(0.5)), (0.5, len(wav(0.5))))
        self.assertEqual(tts_log.audio_info(b"RIFF-not-wav"), (None, 12))

    def test_failure_before_start_is_info_and_logger_errors_are_swallowed(self):
        logger = Mock()
        trace = tts_log.Trace("tool", stream_id="0123456789abcdef", text="你好")
        tts_log.log_failure(logger, trace, "语音冷却中，请 3 秒后重试")
        logger.warning.assert_not_called()
        self.assertEqual(logger.info.call_args.args[0], "[语音合成] 未执行 · 语音工具 · 聊天 01234567… · 原因：语音冷却中，请 3 秒后重试 · 文本「你好」")
        broken = SimpleNamespace(info=Mock(side_effect=RuntimeError("log down")), warning=Mock(side_effect=RuntimeError("log down")))
        tts_log.log_start(broken, trace)
        tts_log.log_failure(broken, trace, "x")


class TTSLogTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.p = create_plugin()
        self.p._settings_path = Path(self.temp.name).resolve() / "config.toml"
        self.cfg = self.p.build_default_config()
        self.cfg["behavior"]["cooldown_seconds"] = 0
        self.cfg["role"].update(mode="role", role_id="黍/role.json", emotion="neutral", emotions=["neutral", "happy", "sad"])
        self.p.set_plugin_config(self.cfg)
        self.logger = Mock()
        self.llm = SimpleNamespace(generate=AsyncMock(return_value={"success": True, "response": "sad", "model_name": "gpt-fast"}))
        self.send = SimpleNamespace(custom=AsyncMock(return_value={"sent": True}), text=AsyncMock())
        self.p._ctx = SimpleNamespace(send=self.send, llm=self.llm, logger=self.logger)
        await self.p.on_load()
        self.audio = wav(0.5)
        self.client = SimpleNamespace(synthesize=AsyncMock(return_value=self.audio), set_gpt_weights=AsyncMock(),
                                      set_sovits_weights=AsyncMock(), probe=AsyncMock(return_value="ok"))

        def resolve(role_id, emotion, root):
            return {"role_id": role_id, "emotion": emotion,
                    "models": {"base_model": "v5turbo", "gpt_weights": f"C:/data/黍/{emotion}.ckpt", "sovits_weights": "C:/gsv/base.pth",
                               "lora_weights": f"C:/data/黍/{emotion}-lora.pth"},
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

    def configure(self, section, **values):
        self.cfg[section].update(values)
        self.p.set_plugin_config(self.cfg)

    def infos(self):
        return [call.args[0] for call in self.logger.info.call_args_list if str(call.args[0]).startswith("[语音合成]")]

    def warnings(self):
        return [call.args[0] for call in self.logger.warning.call_args_list if str(call.args[0]).startswith("[语音合成]")]

    async def test_command_logs_start_and_full_result(self):
        ok, reply, _ = await self.p.handle_tts(stream_id="stream-1", matched_groups={"text": "唉，今天的雨下个不停"},
                                               message=group_message())
        self.assertTrue(ok, reply)
        start, done = self.infos()
        self.assertEqual(start, "[语音合成] 开始 · 命令 /tts · 群 测试群（123456） · 角色 黍（启用 3 个情感） · 文本 10 字「唉，今天的雨下个不停」")
        self.assertTrue(done.startswith("[语音合成] 完成 · 命令 /tts · 群 测试群（123456） · 角色 黍 · "
                                        "情感 难过（LLM 判断，fast_model/gpt-fast，"), done)
        for part in ("切换模型 GPT sad.ckpt、LoRA sad-lora.pth（", " · 合成 ", "音频 0.5 秒 / ", " · 已发送 · 共 ", "文本「唉，今天的雨下个不停」"):
            self.assertIn(part, done)
        self.assertEqual(self.warnings(), [])

    async def test_tool_reuses_chat_label_cache_and_model_reuse(self):
        await self.p.handle_tts(stream_id="stream-1", matched_groups={"text": "第一句"}, message=group_message())
        self.llm.generate.return_value = {"success": True, "response": "sad", "model_name": "gpt-fast"}
        await self.p.handle_speak(text="第二句", stream_id="stream-1")
        done = self.infos()[-1]
        self.assertIn("[语音合成] 完成 · 语音工具 · 群 测试群（123456）", done)
        self.assertIn("沿用已加载的模型", done)
        await self.p.handle_speak(text="第二句", stream_id="stream-1")
        self.assertIn("LLM 判断，fast_model/gpt-fast，缓存结果", self.infos()[-1])
        await self.p.handle_speak(text="你好", stream_id="0123456789abcdef", emotion="开心")
        self.assertIn("· 聊天 01234567… · 角色 黍 · 情感 开心（指定）", self.infos()[-1])

    async def test_rejection_and_synthesis_failure_levels(self):
        self.configure("behavior", cooldown_seconds=60)
        await self.p.handle_speak(text="第一句", stream_id="s")
        result = await self.p.handle_speak(text="第二句", stream_id="s")
        self.assertFalse(result["success"])
        self.assertTrue(self.infos()[-1].startswith("[语音合成] 未执行 · 语音工具 · 聊天 s · 原因：语音冷却中"), self.infos()[-1])
        self.configure("behavior", cooldown_seconds=0)
        self.client.synthesize.side_effect = RuntimeError("HTTP 500")
        result = await self.p.handle_speak(text="第三句", stream_id="s")
        self.assertFalse(result["success"])
        failure, = self.warnings()
        self.assertIn("[语音合成] 失败 · 语音工具 · 聊天 s · 角色 黍", failure)
        self.assertIn("原因：RuntimeError: HTTP 500", failure)
        self.assertIn("文本「第三句」", failure)

    async def test_llm_failure_is_explained_in_result_line(self):
        self.llm.generate.return_value = {"success": False, "error": "quota"}
        await self.p.handle_speak(text="好吧", stream_id="s")
        self.assertIn("情感 中立（LLM 判断失败，改用默认情感：RuntimeError: MaiBot LLM 调用失败：quota）", self.infos()[-1])

    async def test_original_mode_mentions_server_weights(self):
        self.configure("role", mode="original")
        self.configure("voice", ref_audio_path="/srv/ref.wav")
        await self.p.handle_speak(text="你好", stream_id="s")
        done = self.infos()[-1]
        self.assertIn("· 普通模式 · 使用服务端当前加载的模型 · 合成", done)
        self.assertNotIn("情感", done)

    async def test_auto_voice_success_failure_and_keyword_fallback(self):
        self.configure("auto_voice", enabled=True, probability_percent=100, min_interval_seconds=0)
        now = str(datetime.now().timestamp())

        def message(text, msg_id):
            return {"message_id": msg_id, "timestamp": now, "session_id": "stream-9", "raw_message": [{"type": "text", "data": text}],
                    "message_info": {"group_info": {"group_id": "777", "group_name": "自动群"}, "additional_config": {}}, "is_command": False}
        await self.p.handle_auto_voice(message=message("今天真开心！", "m1"))
        start, done = self.infos()[-2:]
        self.assertEqual(start, "[语音合成] 开始 · 自动语音 · 群 自动群（777） · 角色 黍（启用 3 个情感） · 文本 6 字「今天真开心！」")
        self.assertIn("情感 难过（LLM 判断", done)
        self.assertIn("· 已替换为语音 · 共 ", done)
        self.llm.generate.side_effect = RuntimeError("down")
        await self.p.handle_auto_voice(message=message("今天真开心呀！", "m2"))
        self.assertIn("情感 开心（LLM 判断失败，改用关键词规则：RuntimeError: down）", self.infos()[-1])
        self.client.synthesize.side_effect = RuntimeError("HTTP 500")
        await self.p.handle_auto_voice(message=message("再来一句", "m3"))
        failure, = self.warnings()
        self.assertIn("[语音合成] 失败 · 自动语音 · 群 自动群（777）", failure)
        self.assertIn("原因：RuntimeError: HTTP 500 · 已保留原文字", failure)

    async def test_setup_preview_and_load_jobs_are_logged(self):
        with patch.object(setup_service, "GPTSoVITSClient", return_value=self.client):
            self.p._job = {"id": "j1", "kind": "load_models", "status": "running", "message": "", "audio_bytes": 0, "duration_seconds": 0}
            self.p._busy = True
            await self.p._run_setup_job(self.p.config.model_copy(deep=True), "load_models", "")
            self.p._job = {"id": "j2", "kind": "synthesize", "status": "running", "message": "", "audio_bytes": 0, "duration_seconds": 0}
            self.p._busy = True
            await self.p._run_setup_job(self.p.config.model_copy(deep=True), "synthesize", "试听一下")
        load_start, load_done, preview_start, preview_done = self.infos()
        self.assertEqual(load_start, "[语音合成] 开始 · 设置页加载模型 · 角色 黍（启用 3 个情感）")
        self.assertIn("[语音合成] 完成 · 设置页加载模型 · 角色 黍 · 切换模型 GPT ", load_done)
        self.assertEqual(preview_start, "[语音合成] 开始 · 设置页试听 · 角色 黍（启用 3 个情感） · 文本 4 字「试听一下」")
        for part in ("情感 中立（默认情感）", "使用已加载的模型", "音频 0.5 秒", "仅本地试听，未发送"):
            self.assertIn(part, preview_done)


if __name__ == "__main__":
    unittest.main()
