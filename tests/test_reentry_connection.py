"""Re-entry connection regressions: real config, temporary progress, mocked transport only."""
from __future__ import annotations

import asyncio
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, Mock, patch

import tomlkit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import plugin as plugin_module
import setup_service as setup_module
from tts_client import TTSError


class ReentryConnectionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="maibot-reentry-")
        self.addCleanup(self.temp.cleanup)
        self.plugin = plugin_module.create_plugin()
        self.plugin._settings_path = Path(self.temp.name) / "config.toml"
        data = self.plugin.build_default_config()
        data["models"].update(base_model="custom", gpt_weights="/server/gpt.ckpt", sovits_weights="/server/sovits.pth")
        data["voice"].update(ref_audio_path="/server/ref.wav", prompt_text="参考文本")
        self.plugin.set_plugin_config(data)
        self.plugin._known_api_url = self.plugin.config.api.base_url.rstrip("/")
        self.plugin._active = True
        self.plugin._settings_path.write_text(tomlkit.dumps(data), encoding="utf-8")
        self.client = Mock()
        self.client.probe = AsyncMock(return_value="POST /tts")
        self.client.set_gpt_weights = AsyncMock()
        self.client.set_sovits_weights = AsyncMock()
        self.client.synthesize = AsyncMock(side_effect=AssertionError("ensure must not synthesize"))
        self.factory = self.enterContext(patch.object(setup_module, "GPTSoVITSClient", return_value=self.client))

    async def asyncTearDown(self):
        self.plugin._active = False
        await self.plugin._cancel_setup_connection()
        for task in list(self.plugin._tasks):
            task.cancel()
        if self.plugin._tasks:
            await asyncio.gather(*self.plugin._tasks, return_exceptions=True)
        self.assertFalse(self.plugin._busy)
        self.assertFalse(self.plugin._tasks)

    async def configured(self):
        await self.plugin._mark_progress(1)

    async def test_state_is_fast_read_only_and_never_probes(self):
        await self.configured()
        before = self.plugin._settings_path.read_bytes()
        start = time.monotonic()
        state = await self.plugin.setup_state()
        self.assertLess(time.monotonic() - start, 1)
        self.assertIn(1, state["progress"]["completed_steps"])
        self.assertEqual(state["connection"]["status"], "idle")
        self.factory.assert_not_called()
        self.assertEqual(before, self.plugin._settings_path.read_bytes())

    async def test_saved_marker_ensures_without_save_or_weight_switch(self):
        await self.configured()
        before = self.plugin.config.model_dump()
        disk = self.plugin._settings_path.read_bytes()
        marker = self.plugin._progress_manager().path.read_bytes()
        result = await self.plugin.setup_ensure_connection()
        self.assertTrue(result["success"], result)
        self.assertEqual(result["connection"]["checked_url"], before["api"]["base_url"].rstrip("/"))
        self.assertEqual(self.plugin.config.model_dump(), before)
        self.assertEqual(disk, self.plugin._settings_path.read_bytes())
        self.assertEqual(marker, self.plugin._progress_manager().path.read_bytes())
        self.client.probe.assert_awaited_once()
        self.client.set_gpt_weights.assert_not_awaited()
        self.client.set_sovits_weights.assert_not_awaited()
        self.client.synthesize.assert_not_awaited()
        self.assertEqual(self.plugin._model_status["status"], "idle")
        self.assertLessEqual(self.factory.call_args.kwargs["timeout"], 4)

    async def test_cached_success_same_url_returns_without_network(self):
        self.assertTrue(self.plugin._mark_setup_connected(self.plugin.config.api.base_url))
        result = await self.plugin.setup_ensure_connection()
        self.assertTrue(result["success"])
        self.factory.assert_not_called()

    async def test_cold_fresh_stays_first_step(self):
        result = await self.plugin.setup_ensure_connection()
        self.assertFalse(result["success"])
        self.assertIn("第一步", result["message"])
        self.factory.assert_not_called()

    async def test_disabled_and_busy_never_probe(self):
        await self.configured()
        self.plugin._active = False
        result = await self.plugin.setup_ensure_connection()
        self.assertFalse(result["success"])
        self.assertIn("启用", result["message"])
        self.plugin._active = True
        self.plugin._busy = True
        result = await self.plugin.setup_ensure_connection()
        self.assertFalse(result["success"])
        self.assertIn("任务", result["message"])
        self.plugin._busy = False
        self.factory.assert_not_called()

    async def test_concurrent_ensure_is_shared_and_waiter_cancel_isolated(self):
        await self.configured()
        started, release = asyncio.Event(), asyncio.Event()
        async def probe():
            started.set()
            await release.wait()
        self.client.probe.side_effect = probe
        first = asyncio.create_task(self.plugin.setup_ensure_connection())
        await started.wait()
        second = asyncio.create_task(self.plugin.setup_ensure_connection())
        await asyncio.sleep(0)
        first.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first
        self.assertFalse(self.plugin._ensure_connection_task.cancelled())
        release.set()
        self.assertTrue((await second)["success"])
        self.client.probe.assert_awaited_once()

    async def test_failure_keeps_specific_error_and_retry_works(self):
        await self.configured()
        self.client.probe.side_effect = TTSError("HTTP 503 服务离线")
        result = await self.plugin.setup_ensure_connection()
        self.assertFalse(result["success"])
        self.assertIn("HTTP 503", result["message"])
        self.assertEqual(result["connection"]["status"], "error")
        self.assertFalse(self.plugin._busy)
        self.client.probe.side_effect = None
        self.assertTrue((await self.plugin.setup_ensure_connection())["success"])
        self.assertEqual(self.client.probe.await_count, 2)

    async def test_expired_success_requires_real_confirmation(self):
        await self.configured()
        self.plugin._mark_setup_connected(self.plugin.config.api.base_url)
        self.plugin._connection_checked_at = time.monotonic() - 301
        self.assertTrue((await self.plugin.setup_ensure_connection())["success"])
        self.client.probe.assert_awaited_once()

    async def test_changed_url_and_aba_never_accept_old_probe(self):
        await self.configured()
        started, release = asyncio.Event(), asyncio.Event()
        async def probe():
            started.set()
            await release.wait()
        self.client.probe.side_effect = probe
        task = asyncio.create_task(self.plugin.setup_ensure_connection())
        await started.wait()
        original = self.plugin.config.api.base_url
        for url in ("http://other.invalid:9880", original):
            old = self.plugin.config.model_copy(deep=True)
            data = old.model_dump()
            data["api"]["base_url"] = url
            self.plugin.set_plugin_config(data)
            self.plugin._invalidate_setup(old, self.plugin.config)
        release.set()
        result = await task
        self.assertFalse(result["success"])
        self.assertNotEqual(self.plugin._connection["status"], "success")
        self.assertIn("旧连接结果", result["message"])

    async def test_unload_cancels_shared_task_and_no_late_success(self):
        await self.configured()
        started = asyncio.Event()
        async def probe():
            started.set()
            await asyncio.Event().wait()
        self.client.probe.side_effect = probe
        waiter = asyncio.create_task(self.plugin.setup_ensure_connection())
        await started.wait()
        self.plugin._active = False
        await self.plugin._cancel_setup_connection()
        with self.assertRaises(asyncio.CancelledError):
            await waiter
        self.assertFalse(self.plugin._busy)
        self.assertIsNone(self.plugin._ensure_connection_task)
        self.assertEqual(self.plugin._connection["status"], "idle")

    async def test_start_job_auto_ensures_saved_configuration(self):
        await self.configured()
        result = await self.plugin.setup_start_job("load_models")
        self.assertTrue(result["success"], result)
        await self.plugin._job_task
        self.assertEqual(self.plugin._job["status"], "success")
        self.client.probe.assert_awaited_once()
        self.client.set_gpt_weights.assert_awaited_once()
        self.client.set_sovits_weights.assert_awaited_once()

    async def test_probe_timeout_is_below_host_budget(self):
        await self.configured()
        self.client.probe.side_effect = lambda: None
        async def slow():
            await asyncio.Event().wait()
        self.client.probe.side_effect = slow
        start = time.monotonic()
        result = await self.plugin.setup_ensure_connection()
        self.assertLess(time.monotonic() - start, 4)
        self.assertFalse(result["success"])
        self.assertIn("超时", result["message"])
        self.assertFalse(self.plugin._busy)


if __name__ == "__main__":
    unittest.main()
