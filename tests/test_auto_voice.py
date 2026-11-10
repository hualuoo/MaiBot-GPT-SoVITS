"""Real SDK/config tests for the in-place outgoing hook; no host or network needed."""
from __future__ import annotations

import asyncio
import base64
from copy import deepcopy
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from maibot_sdk import MaiBotPlugin
from maibot_sdk.components import collect_components
import auto_voice as module
from config_models import TTSConfig


class Harness(module.AutoVoiceMixin, MaiBotPlugin):
    config_model = TTSConfig

    def __init__(self):
        super().__init__()
        self._active = True
        self._busy = False
        self._tasks = set()
        self._init_auto_voice()

    def _check_access(self, stream_id, cfg):
        if not self._active or not cfg.plugin.enabled or not stream_id:
            raise ValueError("not allowed")
        if cfg.behavior.allowed_stream_ids and stream_id not in cfg.behavior.allowed_stream_ids:
            raise ValueError("stream not allowed")


class AutoVoiceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.p = Harness()
        self.data = self.p.build_default_config()
        self.data["auto_voice"].update(enabled=True, probability_percent=100, min_interval_seconds=0)
        self.p.set_plugin_config(self.data)
        self.audio = b"RIFF\x00WAVE-small-audio"
        self.p._synthesize_audio = AsyncMock(return_value=self.audio)
        self.logger = Mock()
        self.p._get_logger = Mock(return_value=self.logger)
        self.p._ctx = SimpleNamespace(send=SimpleNamespace(text=AsyncMock(), custom=AsyncMock()))
        self.sequence = 0

    def config(self, section, **changes):
        self.data[section].update(changes)
        self.p.set_plugin_config(self.data)

    def message(self, text="普通回复", stream="session-A", **changes):
        self.sequence += 1
        value = {"message_id": str(self.sequence), "session_id": stream,
                 "raw_message": [{"type": "text", "data": text}],
                 "message_info": {"additional_config": {"route": "QQ-group", "account_id": "bot"}},
                 "platform": "qq", "processed_plain_text": text}
        value.update(changes)
        return value

    async def run_hook(self, message=None, **kwargs):
        if message is None:
            message = self.message()
        return await self.p.handle_auto_voice(message=message, hook_name="send_service.before_send", **kwargs)

    @staticmethod
    def voice(result):
        return [seg for seg in result["modified_kwargs"]["message"]["raw_message"] if seg["type"] == "voice"]

    def role(self, **changes):
        self.config("role", mode="role", emotion="neutral",
                    emotions=["neutral", "happy", "angry", "sad", "surprised"], **changes)

    async def test_default_disabled_has_no_behavior_change(self):
        fresh = TTSConfig()
        self.assertFalse(fresh.auto_voice.enabled)
        self.assertEqual(fresh.auto_voice.frequency_mode, "probability")
        self.assertEqual(fresh.auto_voice.probability_percent, 50)
        self.assertEqual(fresh.auto_voice.every_n, 3)
        self.assertEqual(fresh.auto_voice.min_interval_seconds, 5.0)
        self.assertTrue(fresh.auto_voice.follow_emotion)
        self.assertFalse(fresh.auto_voice.keep_text)
        self.config("auto_voice", enabled=False)
        msg = self.message()
        result = await self.run_hook(msg, typing=True)
        self.assertIs(result["modified_kwargs"]["message"], msg)
        self.assertEqual(self.p._auto_stats["last_reason"], "disabled")
        self.p._synthesize_audio.assert_not_called()

    async def test_real_sdk_hook_registration(self):
        hook = next(c for c in collect_components(self.p) if c["type"] == "HOOK_HANDLER")
        self.assertEqual(hook["name"], "gpt_sovits_auto_voice")
        metadata = hook["metadata"]
        self.assertEqual(metadata["hook"], "send_service.before_send")
        self.assertEqual(metadata["mode"], "blocking")
        self.assertEqual(metadata["order"], "late")
        self.assertEqual(metadata["timeout_ms"], 300000)
        self.assertEqual(metadata["error_policy"], "skip")

    async def test_voice_contract_hash_all_kwargs_route_and_no_ctx_send(self):
        msg = self.message("  完整原文。 ")
        before = deepcopy(msg)
        kwargs = dict(typing=True, set_reply=True, reply_message_id="origin",
                      storage_message=False, show_log=False, future_option={"preserve": 1})
        result = await self.run_hook(msg, **kwargs)
        self.assertEqual(result["action"], "continue")
        changed = result["modified_kwargs"]
        self.assertEqual({key: value for key, value in changed.items() if key != "message"},
                         {**kwargs, "set_reply": False, "reply_message_id": None})
        self.assertNotIn("hook_name", changed)
        voice = self.voice(result)[0]
        self.assertEqual(voice["data"], "  完整原文。 ")
        self.assertEqual(base64.b64decode(voice["binary_data_base64"]), self.audio)
        self.assertEqual(voice["hash"], hashlib.sha256(self.audio).hexdigest())
        self.assertEqual(changed["message"]["message_info"]["additional_config"],
                         {"route": "QQ-group", "account_id": "bot", module.MARKER: True})
        self.assertEqual(msg, before)
        self.p._ctx.send.text.assert_not_called()
        self.p._ctx.send.custom.assert_not_called()
        self.assertFalse(self.p._busy)
        self.assertFalse(self.p._tasks)

    async def test_reply_at_and_multiple_text_segments_preserved(self):
        reply = {"type": "reply", "data": {"target_message_id": "earlier"}}
        at = {"type": "at", "data": {"target_user_id": "friend"}}
        msg = self.message(raw_message=[reply, at, {"type": "text", "data": "第一句"},
                                        {"type": "text", "data": "第二句"}])
        result = await self.run_hook(msg, reply_message_id="earlier", set_reply=True)
        segments = result["modified_kwargs"]["message"]["raw_message"]
        self.assertEqual(segments[0], at)
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[1]["data"], "第一句第二句")
        self.assertFalse(result["modified_kwargs"]["set_reply"])
        self.assertIsNone(result["modified_kwargs"]["reply_message_id"])
        self.assertIsNone(result["modified_kwargs"]["message"]["reply_to"])
        self.assertEqual(msg["raw_message"][0], reply)

    async def test_keep_text_appends_one_voice(self):
        self.config("auto_voice", keep_text=True)
        msg = self.message()
        result = await self.run_hook(msg)
        segments = result["modified_kwargs"]["message"]["raw_message"]
        self.assertEqual(segments[:-1], msg["raw_message"])
        self.assertEqual(segments[-1]["type"], "voice")

    async def test_probability_zero_never_and_hundred_always_no_random_needed(self):
        with patch.object(module.random, "random", side_effect=AssertionError("not needed")):
            self.config("auto_voice", probability_percent=0)
            self.assertFalse(self.voice(await self.run_hook()))
            self.config("auto_voice", probability_percent=100)
            self.assertTrue(self.voice(await self.run_hook()))
        self.assertEqual(self.p._auto_stats["considered"], 2)

    async def test_probability_uses_random_strict_threshold(self):
        self.config("auto_voice", probability_percent=50)
        with patch.object(module.random, "random", side_effect=[0.499, 0.5]) as rng:
            self.assertTrue(self.voice(await self.run_hook()))
            self.assertFalse(self.voice(await self.run_hook()))
            self.assertEqual(rng.call_count, 2)

    async def test_every_n_counts_only_eligible_per_stream(self):
        self.config("auto_voice", frequency_mode="every_n", every_n=3)
        for stream in ("A", "B"):
            self.assertFalse(self.voice(await self.run_hook(self.message(stream=stream))))
            self.assertFalse(self.voice(await self.run_hook(self.message(stream=stream))))
            await self.run_hook(self.message(stream=stream, raw_message=[{"type": "image", "data": "x"}]))
            self.assertTrue(self.voice(await self.run_hook(self.message(stream=stream))))
        self.assertEqual(self.p._auto_stats["considered"], 6)
        self.assertEqual(self.p._auto_stats["converted"], 2)

    async def test_interval_filters_before_rng_busy_or_count(self):
        self.config("auto_voice", min_interval_seconds=5)
        with patch.object(module.time, "monotonic", return_value=100):
            await self.run_hook()
        self.p._busy = True
        with patch.object(module.time, "monotonic", return_value=104), \
             patch.object(module.random, "random", side_effect=AssertionError("must not sample")):
            self.assertFalse(self.voice(await self.run_hook()))
        self.assertEqual(self.p._auto_stats["last_reason"], "interval")
        self.assertEqual(self.p._auto_stats["considered"], 1)
        self.p._busy = False
        with patch.object(module.time, "monotonic", return_value=105):
            self.assertTrue(self.voice(await self.run_hook()))

    async def test_busy_skip_without_queue(self):
        for _ in range(2):
            self.p._busy = True
            result = await self.run_hook()
            self.assertFalse(self.voice(result))
            self.assertEqual(self.p._auto_stats["last_reason"], "busy")
        self.p._synthesize_audio.assert_not_called()
        self.assertEqual(self.p._auto_stats["considered"], 0)

    async def test_active_plugin_enabled_and_allowed_access(self):
        self.p._active = False
        await self.run_hook()
        self.assertEqual(self.p._auto_stats["last_reason"], "inactive")
        self.p._active = True
        self.config("plugin", enabled=False)
        await self.run_hook()
        self.assertEqual(self.p._auto_stats["last_reason"], "access")
        self.config("plugin", enabled=True)
        self.config("behavior", allowed_stream_ids=["allowed"])
        await self.run_hook()
        self.assertEqual(self.p._auto_stats["last_reason"], "access")
        self.assertTrue(self.voice(await self.run_hook(self.message(stream="allowed"))))

    async def test_skip_non_text_types_command_and_codeblock(self):
        for kind in ("voice", "image", "emoji", "file", "forward", "command", "dict", "codeblock", "unknown"):
            with self.subTest(kind=kind):
                msg = self.message(raw_message=[{"type": "text", "data": "hi"}, {"type": kind, "data": "x"}])
                await self.run_hook(msg)
                self.assertEqual(self.p._auto_stats["last_reason"], "non_text")
        for text in ("/tts hello", "```py\nx=1\n```", "~~~code~~~"):
            await self.run_hook(self.message(text))
            self.assertEqual(self.p._auto_stats["last_reason"], "command_or_code")
        await self.run_hook(self.message(is_command=True))
        self.assertEqual(self.p._auto_stats["last_reason"], "command")
        self.p._synthesize_audio.assert_not_called()

    async def test_skip_empty_malformed_long_and_tts_error_text(self):
        for raw in ([], [{"type": "reply", "data": {}}], [{"type": "text", "data": {}}],
                    [{"type": "text", "data": " \n"}]):
            await self.run_hook(self.message(raw_message=raw))
            self.assertEqual(self.p._auto_stats["last_reason"], "no_text")
        await self.run_hook(self.message("字" * 301))
        self.assertEqual(self.p._auto_stats["last_reason"], "too_long")
        for text in ("TTS：接口失败", "TTS 执行异常，请查看插件日志。", "语音合成失败"):
            await self.run_hook(self.message(text))
            self.assertEqual(self.p._auto_stats["last_reason"], "tts_error")
        self.p._synthesize_audio.assert_not_called()

    async def test_failure_warns_preserves_original_and_releases_busy_tasks(self):
        self.p._synthesize_audio.side_effect = RuntimeError("upstream failed")
        msg = self.message()
        result = await self.run_hook(msg, typing=False, set_reply=True, reply_message_id="old")
        self.assertIs(result["modified_kwargs"]["message"], msg)
        self.assertEqual(result["action"], "continue")
        self.assertEqual(result["modified_kwargs"]["reply_message_id"], "old")
        self.logger.warning.assert_called_once()
        self.assertEqual(self.p._auto_stats["failed"], 1)
        self.assertEqual(self.p._auto_stats["converted"], 0)
        self.assertFalse(self.p._busy)
        self.assertFalse(self.p._tasks)
        self.p._ctx.send.text.assert_not_called()

    async def test_empty_or_invalid_audio_is_failure(self):
        for audio in (b"", "not bytes", None):
            self.p._synthesize_audio.return_value = audio
            result = await self.run_hook()
            self.assertFalse(self.voice(result))
        self.assertEqual(self.p._auto_stats["failed"], 3)

    async def test_audio_over_configured_or_rpc_limit_falls_back(self):
        self.config("api", max_audio_bytes=1024)
        self.p._synthesize_audio.return_value = b"a" * 1025
        msg = self.message()
        result = await self.run_hook(msg)
        self.assertIs(result["modified_kwargs"]["message"], msg)
        self.assertEqual(self.p._auto_stats["failed"], 1)
        self.config("api", max_audio_bytes=16777216)
        self.p._synthesize_audio.return_value = b"a" * (8388608 + 1)
        result = await self.run_hook()
        self.assertFalse(self.voice(result))
        self.assertEqual(self.p._auto_stats["failed"], 2)
        self.assertFalse(self.p._busy)
        self.assertFalse(self.p._tasks)

    async def test_entire_synthesis_has_configured_capped_timeout(self):
        entered = []
        actual_timeout = asyncio.timeout
        def record_timeout(seconds):
            entered.append(seconds)
            return actual_timeout(seconds)
        self.config("api", timeout=11)
        with patch.object(module.asyncio, "timeout", side_effect=record_timeout):
            await self.run_hook()
        self.assertEqual(entered, [11])
        async def stalled(*args, **kwargs):
            await asyncio.Future()
        self.p._synthesize_audio.side_effect = stalled
        with patch.object(module.asyncio, "timeout", side_effect=lambda _: actual_timeout(0.001)):
            result = await self.run_hook()
        self.assertFalse(self.voice(result))
        self.assertFalse(self.p._busy)
        self.assertFalse(self.p._tasks)
        self.assertEqual(self.p._auto_stats["failed"], 1)

    async def test_cancellation_propagates_and_unload_task_is_tracked(self):
        entered = asyncio.Event()
        async def stalled(*args, **kwargs):
            entered.set()
            await asyncio.Future()
        self.p._synthesize_audio.side_effect = stalled
        task = asyncio.create_task(self.run_hook())
        try:
            await asyncio.wait_for(entered.wait(), 1)
            self.assertTrue(self.p._busy)
            self.assertIn(task, self.p._tasks)
            # This mirrors plugin.on_unload: cancel recorded tasks, do not catch cancellation.
            for tracked in list(self.p._tasks):
                tracked.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        self.assertFalse(self.p._busy)
        self.assertFalse(self.p._tasks)
        self.assertEqual(self.p._auto_stats["failed"], 0)
        self.logger.warning.assert_not_called()

    async def test_concurrent_other_message_skips_busy_not_queued(self):
        entered, release = asyncio.Event(), asyncio.Event()
        async def stalled(*args, **kwargs):
            entered.set()
            await release.wait()
            return self.audio
        self.p._synthesize_audio.side_effect = stalled
        first = asyncio.create_task(self.run_hook())
        try:
            await asyncio.wait_for(entered.wait(), 1)
            result = await self.run_hook()
            self.assertFalse(self.voice(result))
            self.assertEqual(self.p._auto_stats["last_reason"], "busy")
            self.assertEqual(self.p._synthesize_audio.await_count, 1)
        finally:
            release.set()
            await first

    async def test_message_id_dedup_no_cached_audio_or_second_synthesis(self):
        msg = self.message(message_id="same-id")
        result = await self.run_hook(msg)
        self.assertTrue(self.voice(result))
        second = await self.run_hook(deepcopy(msg))
        self.assertFalse(self.voice(second))
        self.assertEqual(self.p._auto_stats["last_reason"], "duplicate")
        self.p._synthesize_audio.assert_awaited_once()
        self.assertEqual(self.p._auto_stats["considered"], 1)
        self.assertNotIn(self.audio, self.p._auto_voice_seen.values())
        self.assertTrue(all(isinstance(v, float) for v in self.p._auto_voice_seen.values()))

    async def test_marker_prevents_hook_recursion_even_with_keep_text(self):
        self.config("auto_voice", keep_text=True)
        first = await self.run_hook()
        converted = first["modified_kwargs"]["message"]
        second = await self.run_hook(converted)
        self.assertEqual(self.p._auto_stats["last_reason"], "already_converted")
        self.assertIs(second["modified_kwargs"]["message"], converted)
        self.p._synthesize_audio.assert_awaited_once()

    async def test_own_speech_suppression_and_expiry(self):
        self.p._auto_voice_suppress[("session-A", "自己的原文")] = 105
        with patch.object(module.time, "monotonic", return_value=100):
            result = await self.run_hook(self.message("自己的原文"))
        self.assertFalse(self.voice(result))
        self.assertEqual(self.p._auto_stats["last_reason"], "own_speech")
        with patch.object(module.time, "monotonic", return_value=106):
            self.assertTrue(self.voice(await self.run_hook(self.message("自己的原文"))))
        self.assertFalse(self.p._auto_voice_suppress)

    async def test_stream_and_seen_caches_max_1024_and_ttl_3600(self):
        self.config("auto_voice", probability_percent=0)
        with patch.object(module.time, "monotonic", return_value=100):
            for index in range(1030):
                await self.run_hook(self.message(stream=f"stream-{index}"))
        self.assertEqual(len(self.p._auto_voice_streams), 1024)
        self.assertEqual(len(self.p._auto_voice_seen), 1024)
        self.assertNotIn("stream-0", self.p._auto_voice_streams)
        with patch.object(module.time, "monotonic", return_value=3700):
            await self.run_hook(self.message(stream="new"))
        self.assertEqual(len(self.p._auto_voice_streams), 1)
        self.assertEqual(len(self.p._auto_voice_seen), 1)

    async def test_dedup_expires_after_hour(self):
        msg = self.message()
        with patch.object(module.time, "monotonic", return_value=100):
            await self.run_hook(msg)
        with patch.object(module.time, "monotonic", return_value=3700):
            self.assertTrue(self.voice(await self.run_hook(msg)))
        self.assertEqual(self.p._synthesize_audio.await_count, 2)

    async def test_emotion_metadata_priority_and_mapping(self):
        self.role()
        for key, value, expected in (("emotion", "开心", "happy"),
                                     ("emotion", "中立", "neutral"),
                                     ("emotion_id", "angry", "angry"),
                                     ("sentiment", "sadness", "sad")):
            with self.subTest(key=key):
                msg = self.message("I am surprised", metadata={key: value})
                self.assertEqual(module.choose_emotion("I am surprised", msg, self.p.config), expected)
        msg = self.message(metadata={"emotion": "happy", "emotion_id": "angry"})
        self.assertEqual(module.choose_emotion("难过", msg, self.p.config), "happy")
        msg = self.message(message_info={"additional_config": {"metadata": {"sentiment": {"label": "positive"}}}})
        self.assertEqual(module.choose_emotion("没情感", msg, self.p.config), "happy")

    async def test_emotion_keywords_chinese_and_english(self):
        self.role()
        for text, expected in (("今天很开心", "happy"), ("我很生气", "angry"),
                               ("真让人难过", "sad"), ("真是令人惊讶", "surprised"),
                               ("I am HAPPY", "happy"), ("I am furious", "angry"),
                               ("I feel sad", "sad"), ("I am astonished", "surprised")):
            self.assertEqual(module.choose_emotion(text, {}, self.p.config), expected, text)

    async def test_emotion_negation_ambiguity_and_default(self):
        self.role()
        for text in ("我不开心", "没有很高兴", "不要生气", "I am not happy",
                     "I am not really happy", "I'm not very sad", "今天开心又难过", "普通问候"):
            self.assertEqual(module.choose_emotion(text, {}, self.p.config), "neutral", text)
        self.assertEqual(module.choose_emotion("不难过了。现在很开心", {}, self.p.config), "happy")

    async def test_emotion_disabled_metadata_unknown_follow_false_original(self):
        self.config("role", mode="role", emotion="neutral", emotions=["neutral", "happy"])
        self.assertEqual(module.choose_emotion("很难过", {}, self.p.config), "neutral")
        self.assertEqual(module.choose_emotion("很开心", {"metadata": {"emotion": "sad"}}, self.p.config), "neutral")
        self.assertEqual(module.choose_emotion("很开心", {"metadata": {"emotion": "not-a-known-emotion"}}, self.p.config), "neutral")
        self.config("auto_voice", follow_emotion=False)
        self.assertEqual(module.choose_emotion("很开心", {}, self.p.config), "neutral")
        self.config("role", mode="original")
        self.assertEqual(module.choose_emotion("很开心", {}, self.p.config), "")

    async def test_emotion_enabled_alias_and_stats_passed_to_synthesizer(self):
        self.config("role", mode="role", emotion="平静", emotions=["平静", "开心"])
        result = await self.run_hook(self.message("I am happy"))
        self.assertTrue(self.voice(result))
        self.assertEqual(self.p._synthesize_audio.await_args.kwargs["emotion"], "开心")
        self.assertEqual(self.p._auto_stats["last_emotion"], "开心")
        self.assertEqual(self.p._auto_stats["last_reason"], "converted")

    async def test_config_snapshot_is_not_changed_by_hook(self):
        old = self.p.config.model_dump()
        await self.run_hook()
        self.assertEqual(self.p.config.model_dump(), old)


if __name__ == "__main__":
    unittest.main()
