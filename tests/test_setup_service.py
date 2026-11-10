"""设置页后端接口（使用真实 MaiBot SDK，不替换 SDK 或配置）。

在项目根目录运行：
  python -B -m unittest discover -s tests -p test_setup_service.py -v

Only transports, the model scanner, and the host context/logger are mocked.
All configuration writes go to a TemporaryDirectory, never the real config.toml.
"""
from __future__ import annotations

import asyncio
import base64
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch
import wave

import tomlkit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from maibot_sdk import MaiBotPlugin
from maibot_sdk.components import collect_components
from config_models import TTSConfig
import plugin as plugin_module
import setup_service as setup_module
from tts_client import GPTSoVITSClient as RealGPTSoVITSClient, TTSError


API_NAMES = {
    "setup_state", "setup_catalog", "setup_save_api", "setup_test_api", "setup_ensure_connection",
    "setup_save_models", "setup_start_job", "setup_job", "setup_audio",
    "setup_cancel_job", "setup_roles", "setup_set_mode", "setup_prepare_role",
    "setup_save_base", "setup_check_config", "setup_save_parameters", "setup_save_auto_voice",
    "setup_llm_tasks", "setup_save_emotion_judge", "setup_test_emotion",
}
FIELD_NAMES = {
    "plugin": {"enabled", "config_version"},
    "auto_voice": {"enabled", "frequency_mode", "probability_percent", "every_n", "min_interval_seconds", "follow_emotion", "keep_text"},
    "api": {"base_url", "timeout", "max_audio_bytes"},
    "models": {"root", "base_model", "gpt_weights", "sovits_weights", "lora_weights"},
    "role": {"mode", "role_id", "emotion", "emotions", "llm_emotion", "llm_task"},
    "voice": {"ref_audio_path", "prompt_text", "prompt_lang", "text_lang",
              "aux_ref_audio_paths", "text_split_method", "speed_factor", "top_k",
              "top_p", "temperature", "seed", "parallel_infer"},
    "behavior": {"tool_enabled", "max_text_length", "cooldown_seconds",
                 "allowed_stream_ids", "send_text"},
}
FRAME_LIMIT = 512 * 1024


def wav_bytes(frames: int = 3200, rate: int = 16000) -> bytes:
    """A complete, nonempty, mono 16-bit PCM WAV, not a RIFF-looking stub."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes((b"\x01\x00\xff\xff" * ((frames + 1) // 2))[:frames * 2])
    return buffer.getvalue()


class SetupServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="maibot-setup-test-")
        self.addCleanup(self.temporary.cleanup)
        self.settings = Path(self.temporary.name) / "config.toml"
        self.plugin = plugin_module.create_plugin()
        # Use the actual SDK instance/config implementation. Never use a fake plugin.
        data = self.plugin.build_default_config()
        data["models"].update(root="C:/must-not-be-scanned-implicitly",
                              gpt_weights="/server/GPT/角色.ckpt",
                              sovits_weights="/server/SoVITS/角色.pth")
        data["voice"].update(ref_audio_path="/server/references/参考.wav",
                             prompt_text="这是参考音频文本。")
        data["behavior"]["cooldown_seconds"] = 0
        self.plugin.set_plugin_config(data)
        self.plugin._settings_path = self.settings
        self.send = SimpleNamespace(custom=AsyncMock(return_value={"sent": True}),
                                    text=AsyncMock())
        self.plugin._ctx = SimpleNamespace(send=self.send)
        logger_patch = patch.object(self.plugin, "_get_logger", return_value=Mock())
        logger_patch.start()
        self.addCleanup(logger_patch.stop)
        self.audio = wav_bytes()
        self.client = Mock(spec=RealGPTSoVITSClient)
        self.client.probe = AsyncMock(return_value="已确认 POST /tts")
        self.client.set_gpt_weights = AsyncMock(return_value=None)
        self.client.set_sovits_weights = AsyncMock(return_value=None)
        self.client.synthesize = AsyncMock(return_value=self.audio)

        def client_factory(*args, **kwargs):
            # Keep real constructor validation while replacing every network operation.
            RealGPTSoVITSClient(*args, **kwargs)
            return self.client

        client_patch = patch.object(setup_module, "GPTSoVITSClient", autospec=True,
                                    side_effect=client_factory)
        self.factory = client_patch.start()
        self.addCleanup(client_patch.stop)
        scanner_patch = patch.object(setup_module, "scan_models", autospec=True)
        self.scanner = scanner_patch.start()
        # 参考音频扫描同样只看传入的根目录；默认没有参考音频。
        refs_patch = patch.object(setup_module, "scan_reference_audios", autospec=True,
                                  return_value={"reference_audios": [], "reference_dir": "", "reference_note": "无"})
        self.ref_scanner = refs_patch.start()
        self.addCleanup(refs_patch.stop)
        self.addCleanup(scanner_patch.stop)
        await self.plugin.on_load()

    async def asyncTearDown(self):
        await self.plugin.on_unload()
        self.assertFalse(self.plugin._busy, "unload must release the shared busy flag")
        self.assertEqual(self.plugin._tasks, set(), "unload must not leak tasks")

    def configure(self, section, **updates):
        data = self.plugin.config.model_dump(mode="python")
        data[section].update(updates)
        self.plugin.set_plugin_config(data)
        return data

    def write_fixture(self, data=None):
        data = deepcopy(data or self.plugin.config.model_dump(mode="python"))
        document = tomlkit.parse(tomlkit.dumps(data))
        document.add(tomlkit.comment("全局中文注释：必须保留"))
        document["api"]["base_url"].comment("接口行内注释：必须保留")
        document["behavior"].add(tomlkit.comment("使用限制注释：不能覆盖"))
        self.settings.write_text(tomlkit.dumps(document), encoding="utf-8")
        return data

    def storage_snapshot(self):
        return (self.settings.read_bytes() if self.settings.exists() else None,
                deepcopy(self.plugin.config.model_dump(mode="python")),
                deepcopy(self.plugin.get_plugin_config_data()))

    def assert_storage_unchanged(self, snapshot):
        before_bytes, before_config, before_raw = snapshot
        current_bytes = self.settings.read_bytes() if self.settings.exists() else None
        self.assertEqual(current_bytes, before_bytes, "failed operations must not write config.toml")
        self.assertEqual(self.plugin.config.model_dump(mode="python"), before_config)
        self.assertEqual(self.plugin.get_plugin_config_data(), before_raw)
        self.assertEqual(list(self.settings.parent.glob("config.toml.*.tmp")), [])

    def assert_small_response(self, response):
        encoded = json.dumps(response, ensure_ascii=True).encode("utf-8")
        self.assertLess(len(encoded), FRAME_LIMIT, "every API result must fit the WebUI frame budget")

    async def connect(self):
        result = await self.plugin.setup_test_api()
        self.assertIs(result["success"], True, result)
        self.assertEqual(self.plugin._connection["status"], "success")
        return result

    async def finish_job(self, job_id):
        task = self.plugin._job_task
        self.assertIsInstance(task, asyncio.Task)
        await asyncio.wait_for(asyncio.shield(task), 2)
        self.assertNotIn(task, self.plugin._tasks)
        self.assertFalse(self.plugin._busy)
        result = await self.plugin.setup_job(job_id)
        self.assert_small_response(result)
        return result

    async def load_models(self):
        await self.connect()
        result = await self.plugin.setup_start_job("load_models")
        self.assertIs(result["success"], True, result)
        job = await self.finish_job(result["job"]["id"])
        self.assertEqual(job["status"], "success", job)
        return result["job"]["id"]

    async def synthesize(self, text="你好，这是试听。"):
        result = await self.plugin.setup_start_job("synthesize", text)
        self.assertIs(result["success"], True, result)
        job_id = result["job"]["id"]
        job = await self.finish_job(job_id)
        self.assertEqual(job["status"], "success", job)
        return job_id

    async def prepare_preview(self):
        await self.load_models()
        return await self.synthesize()

    async def block_load(self):
        entered, release = asyncio.Event(), asyncio.Event()

        async def set_gpt(_path):
            entered.set()
            await release.wait()

        self.client.set_gpt_weights.side_effect = set_gpt
        await self.connect()
        result = await asyncio.wait_for(self.plugin.setup_start_job("load_models"), 1)
        self.assertIs(result["success"], True, result)
        await asyncio.wait_for(entered.wait(), 2)
        self.assertFalse(self.plugin._job_task.done())
        self.assertTrue(self.plugin._busy)
        return result["job"]["id"], release

    async def test_real_sdk_all_native_fields_have_chinese_labels(self):
        self.assertIsInstance(self.plugin, MaiBotPlugin)
        self.assertIsInstance(self.plugin.config, TTSConfig)
        schema = self.plugin.build_config_schema(plugin_id="gpt_sovits",
                                                 plugin_name="GPT-SoVITS")
        self.assertEqual(set(schema["sections"]), set(FIELD_NAMES))
        for section_name, expected_fields in FIELD_NAMES.items():
            section = schema["sections"][section_name]
            self.assertRegex(section["title"], r"[\u4e00-\u9fff]")
            self.assertEqual(set(section["fields"]), expected_fields)
            typed_section = TTSConfig.model_fields[section_name].annotation
            self.assertEqual(set(section["fields"]), set(typed_section.model_fields))
            for field_name, field in section["fields"].items():
                with self.subTest(section=section_name, field=field_name):
                    label = field.get("label")
                    self.assertIsInstance(label, str)
                    self.assertNotEqual(label, field_name, "do not fall back to an English identifier")
                    self.assertRegex(label, r"[\u4e00-\u9fff]")
                    self.assertNotRegex(label, r"^[A-Za-z0-9_. -]+$")
        self.assertTrue(schema["sections"]["plugin"]["fields"]["config_version"]["hidden"])
        self.assertEqual(schema["sections"]["api"]["fields"]["timeout"]["max"], 240)

    async def test_real_sdk_exposes_every_setup_api_component(self):
        components = collect_components(self.plugin)
        api = [component for component in components if component["type"] == "API"]
        self.assertEqual(len(api), len(API_NAMES))
        self.assertEqual({component["name"] for component in api}, API_NAMES)
        for component in api:
            with self.subTest(api=component["name"]):
                self.assertEqual(component["metadata"]["handler_name"], component["name"])
                self.assertTrue(callable(getattr(self.plugin, component["name"])))
                self.assertRegex(component["metadata"]["description"], r"[\u4e00-\u9fff]")

    async def test_initial_state_does_not_claim_connection_or_model_load(self):
        state = await self.plugin.setup_state()
        self.assertEqual(state["connection"]["status"], "idle")
        self.assertEqual(state["model_status"]["status"], "idle")
        self.assertIn("未知", state["model_status"]["message"])
        self.assertIsNone(state["job"])
        self.assertIsNone(self.plugin._loaded_signature)
        self.factory.assert_not_called()
        self.scanner.assert_not_called()
        self.assertFalse(self.settings.exists())
        state["config"]["api"]["base_url"] = "http://wrong.invalid"
        state["connection"]["status"] = "success"
        self.assertNotEqual(self.plugin.config.api.base_url, "http://wrong.invalid")
        self.assertEqual(self.plugin._connection["status"], "idle")
        self.assert_small_response(state)

    async def test_save_api_preserves_comments_and_latest_behavior_from_disk(self):
        data = self.plugin.config.model_dump(mode="python")
        # A native settings editor may have changed disk since the plugin snapshot.
        data["behavior"].update(tool_enabled=False, max_text_length=777,
                                 cooldown_seconds=17.5,
                                 allowed_stream_ids=["stream", "other"], send_text=True)
        expected = self.write_fixture(data)
        self.plugin._last_sent["stream"] = 123.0
        result = await self.plugin.setup_save_api("  https://tts.example.test:8443/api/  ", 240)
        self.assertIs(result["success"], True, result)
        text = self.settings.read_text(encoding="utf-8")
        for comment in ("全局中文注释：必须保留", "接口行内注释：必须保留", "使用限制注释：不能覆盖"):
            self.assertIn(comment, text)
        saved = tomlkit.parse(text).unwrap()
        self.assertEqual(saved["api"]["base_url"], "https://tts.example.test:8443/api")
        self.assertEqual(saved["api"]["timeout"], 240)
        self.assertEqual(saved["api"]["max_audio_bytes"], expected["api"]["max_audio_bytes"])
        for section in ("plugin", "models", "voice", "behavior"):
            self.assertEqual(saved[section], expected[section])
        self.assertEqual(self.plugin.config.behavior.model_dump(mode="python"), expected["behavior"])
        self.assertEqual(self.plugin.config.api.base_url, saved["api"]["base_url"])
        self.assertEqual(self.plugin._last_sent, {})
        self.assertEqual(self.plugin._connection["status"], "idle")
        self.client.probe.assert_not_awaited()
        self.client.set_gpt_weights.assert_not_awaited()
        self.client.synthesize.assert_not_awaited()
        self.assertEqual(list(self.settings.parent.glob("config.toml.*.tmp")), [])
        self.assert_small_response(result)

    async def test_first_save_creates_complete_valid_config_only_in_temp_directory(self):
        self.assertFalse(self.settings.exists())
        result = await self.plugin.setup_save_api("http://127.0.0.1:9999/", 120)
        self.assertIs(result["success"], True, result)
        saved = tomlkit.parse(self.settings.read_text(encoding="utf-8")).unwrap()
        self.assertEqual(set(saved), set(FIELD_NAMES))
        self.assertEqual(TTSConfig.model_validate(saved).model_dump(mode="python"),
                         self.plugin.config.model_dump(mode="python"))
        self.assertEqual(self.settings.parent, Path(self.temporary.name))
        self.assertEqual(self.plugin._settings_path, self.settings)

    async def test_invalid_api_values_never_change_existing_file_or_config(self):
        self.write_fixture()
        snapshot = self.storage_snapshot()
        invalid = [(None, 180), (123, 180), ("", 180), ("file:///C:/tts", 180),
                   ("http://user:password@localhost:9880", 180),
                   ("http://bad host:9880", 180), ("http://localhost:9880?x=1", 180),
                   ("http://localhost:9880", 0), ("http://localhost:9880", 241),
                   ("http://localhost:9880", float("nan")),
                   ("http://localhost:9880", float("inf")),
                   ("http://localhost:9880", True)]
        for url, timeout in invalid:
            with self.subTest(url=url, timeout=timeout):
                result = await self.plugin.setup_save_api(url, timeout)
                self.assertIs(result["success"], False, result)
                self.assertRegex(result["message"], r"[\u4e00-\u9fff]")
                self.assert_storage_unchanged(snapshot)
        self.client.probe.assert_not_awaited()
        self.client.synthesize.assert_not_awaited()

    async def test_invalid_first_save_never_creates_config(self):
        snapshot = self.storage_snapshot()
        result = await self.plugin.setup_save_api("not a URL")
        self.assertIs(result["success"], False, result)
        self.assert_storage_unchanged(snapshot)
        self.assertFalse(self.settings.exists())

    async def test_failed_atomic_replace_preserves_file_config_and_removes_temp(self):
        self.write_fixture()
        snapshot = self.storage_snapshot()
        with patch.object(setup_module.os, "replace", side_effect=PermissionError("locked config")):
            result = await self.plugin.setup_save_api("http://localhost:9999", 90)
        self.assertIs(result["success"], False, result)
        self.assertIn("locked config", result["message"])
        self.assert_storage_unchanged(snapshot)

    async def test_saving_the_same_models_keeps_loaded_weights(self):
        self.write_fixture()
        await self.connect()
        self.plugin._loaded_signature = self.plugin._model_signature(self.plugin.config)
        signature = self.plugin._loaded_signature
        models = self.plugin.config.models
        result = await self.plugin.setup_save_models(root=models.root, base_model=models.base_model,
            gpt_weights=models.gpt_weights, sovits_weights=models.sovits_weights)
        self.assertTrue(result["success"], result)
        self.assertEqual(self.plugin._loaded_signature, signature)
        self.client.set_gpt_weights.assert_not_awaited()
        self.client.set_sovits_weights.assert_not_awaited()

    async def test_save_models_normalizes_only_targeted_fields_without_loading(self):
        before = self.write_fixture()
        result = await self.plugin.setup_save_models(
            root="  C:/GPT SoVITS  ", base_model="  custom  ",
            gpt_weights="  /remote/model.ckpt  ", sovits_weights="  /remote/model.pth  ",
            ref_audio_path="  /remote/ref.wav  ", prompt_text="  参考文本  ",
            prompt_lang=" EN ", text_lang=" ZH ")
        self.assertIs(result["success"], True, result)
        saved_text = self.settings.read_text(encoding="utf-8")
        saved = tomlkit.parse(saved_text).unwrap()
        self.assertEqual(saved["models"], {"root": "C:/GPT SoVITS", "base_model": "custom",
                                            "gpt_weights": "/remote/model.ckpt",
                                            "sovits_weights": "/remote/model.pth", "lora_weights": before["models"]["lora_weights"]})
        for key, value in {"ref_audio_path": "/remote/ref.wav", "prompt_text": "参考文本",
                           "prompt_lang": "en", "text_lang": "zh"}.items():
            self.assertEqual(saved["voice"][key], value)
        for key in FIELD_NAMES["voice"] - {"ref_audio_path", "prompt_text", "prompt_lang", "text_lang"}:
            self.assertEqual(saved["voice"][key], before["voice"][key])
        self.assertEqual(saved["behavior"], before["behavior"])
        self.assertEqual(saved["api"], before["api"])
        self.assertIn("接口行内注释：必须保留", saved_text)
        self.assertIn("使用限制注释：不能覆盖", saved_text)
        self.factory.assert_not_called()
        self.assertIsNone(self.plugin._loaded_signature)
        self.assertEqual(self.plugin._model_status["status"], "idle")

    async def test_invalid_model_save_never_writes_or_changes_config(self):
        self.write_fixture()
        snapshot = self.storage_snapshot()
        good = {"gpt_weights": "/remote/model.ckpt", "sovits_weights": "/remote/model.pth"}
        for changes in ({"gpt_weights": " "}, {"sovits_weights": ""},
                        {"root": None}, {"prompt_text": 3}, {"prompt_lang": " "},
                        {"text_lang": ""}, {"root": "x" * 1025},
                        {"gpt_weights": "x" * 1025}, {"lora_weights": 3}):
            with self.subTest(changes=changes):
                result = await self.plugin.setup_save_models(**{**good, **changes})
                self.assertIs(result["success"], False, result)
                self.assert_storage_unchanged(snapshot)
        self.factory.assert_not_called()

    async def test_runner_preapplied_self_reload_invalidates_tested_api_and_weights(self):
        for section, key in (("api", "base_url"), ("models", "gpt_weights"),
                             ("models", "sovits_weights"), ("models", "base_model"), ("models", "lora_weights")):
            with self.subTest(section=section, key=key):
                old_job_id = await self.prepare_preview()
                self.assertTrue(self.plugin._preview_audio)
                new = self.plugin.config.model_dump(mode="python")
                if section == "api":
                    new[section][key] = "http://changed.example.test:9881"
                else:
                    new[section][key] += ".changed"
                self.plugin._last_sent["stream"] = 123.0
                # Real Runner order: set_plugin_config THEN on_config_update('self').
                self.plugin.set_plugin_config(new)
                await self.plugin.on_config_update("self", new, "next-version")
                self.assertEqual(self.plugin.config.model_dump(mode="python"), new)
                self.assertIsNone(self.plugin._loaded_signature)
                self.assertEqual(self.plugin._model_status["status"], "idle")
                self.assertEqual(self.plugin._preview_audio, b"")
                self.assertEqual(self.plugin._last_sent, {})
                self.assertEqual(self.plugin._connection["status"],
                                 "idle" if section == "api" else "success")
                with self.assertRaises(TTSError):
                    await self.plugin.setup_audio(old_job_id)

    async def test_unrelated_self_reload_does_not_discard_valid_connection_or_models(self):
        job_id = await self.prepare_preview()
        signature = self.plugin._loaded_signature
        new = self.plugin.config.model_dump(mode="python")
        new["behavior"]["max_text_length"] = 123
        self.plugin.set_plugin_config(new)
        await self.plugin.on_config_update("self", new, "2")
        self.assertEqual(self.plugin._connection["status"], "success")
        self.assertEqual(self.plugin._model_status["status"], "success")
        self.assertEqual(self.plugin._loaded_signature, signature)
        self.assertEqual(base64.b64decode((await self.plugin.setup_audio(job_id))["base64"]), self.audio)

    async def test_connection_success_only_probes_saved_url_with_short_timeout(self):
        self.write_fixture()
        snapshot = self.storage_snapshot()
        result = await self.connect()
        self.assertRegex(result["message"], r"成功")
        self.factory.assert_called_once_with(self.plugin.config.api.base_url, timeout=5)
        self.client.probe.assert_awaited_once_with()
        self.client.set_gpt_weights.assert_not_awaited()
        self.client.set_sovits_weights.assert_not_awaited()
        self.client.synthesize.assert_not_awaited()
        self.assertFalse(self.plugin._busy)
        self.assertEqual(self.plugin._tasks, set())
        self.assert_storage_unchanged(snapshot)

    async def test_connection_failure_and_timeout_set_error_and_release_busy(self):
        self.write_fixture()
        snapshot = self.storage_snapshot()
        errors = [TTSError("HTTP 503 服务不可用"), TTSError("请求超时"),
                  asyncio.TimeoutError("probe timed out")]
        for error in errors:
            with self.subTest(error=type(error).__name__, message=str(error)):
                self.client.probe.side_effect = error
                result = await self.plugin.setup_test_api()
                self.assertIs(result["success"], False, result)
                self.assertIn(str(error), result["message"])
                self.assertEqual(self.plugin._connection["status"], "error")
                self.assertFalse(self.plugin._busy)
                self.assertEqual(self.plugin._tasks, set())
                denied = await self.plugin.setup_start_job("load_models")
                self.assertIs(denied["success"], False, denied)
                self.assertIsNone(self.plugin._job)
                self.assert_storage_unchanged(snapshot)
        self.client.probe.side_effect = None
        await self.connect()

    async def test_catalog_passes_only_explicit_root_and_runs_scanner_off_event_loop(self):
        event_loop_thread = threading.get_ident()
        calls = []
        expected = {"root": "only-explicit", "base_models": [], "gpt_weights": [],
                    "sovits_weights": [], "note": "仅扫描明确指定目录"}

        def scan(root):
            calls.append((root, threading.get_ident()))
            return deepcopy(expected)

        refs = {"reference_audios": [{"path": "C:/chosen GPT-SoVITS/maibot_refs/a.wav", "label": "maibot_refs"}],
                "reference_dir": "C:/chosen GPT-SoVITS/maibot_refs", "reference_note": "找到 1 个"}

        def scan_refs(root):
            calls.append(("refs:" + root, threading.get_ident()))
            return deepcopy(refs)

        self.scanner.side_effect = scan
        self.ref_scanner.side_effect = scan_refs
        snapshot = self.storage_snapshot()
        combined = {**expected, **refs}
        self.assertEqual(await self.plugin.setup_catalog(), combined)
        self.assertEqual(await self.plugin.setup_catalog("C:/chosen GPT-SoVITS"), combined)
        self.assertEqual(await self.plugin.setup_catalog(root=""), combined)
        self.assertEqual([root for root, _ in calls if root.startswith("refs:")],
                         ["refs:", "refs:C:/chosen GPT-SoVITS", "refs:"])
        calls = [call for call in calls if not call[0].startswith("refs:")]
        # 只扫描传入的本机模型目录，不使用配置中保存的目录。
        self.assertEqual([root for root, _ in calls], ["", "C:/chosen GPT-SoVITS", ""])
        self.assertTrue(all(thread != event_loop_thread for _, thread in calls))
        self.assertNotIn(self.plugin.config.models.root, [root for root, _ in calls])
        self.assert_storage_unchanged(snapshot)
        self.factory.assert_not_called()

    async def test_start_requires_successful_connection_and_two_saved_weights(self):
        for kind in ("load_models", "synthesize", "unknown"):
            result = await self.plugin.setup_start_job(kind, "hello")
            self.assertIs(result["success"], False, result)
            self.assertFalse(self.plugin._busy)
            self.assertIsNone(self.plugin._job)
        await self.connect()
        for field in ("gpt_weights", "sovits_weights"):
            old = getattr(self.plugin.config.models, field)
            self.configure("models", **{field: " "})
            result = await self.plugin.setup_start_job("load_models")
            self.assertIs(result["success"], False, result)
            self.assertIn("权重", result["message"])
            self.assertFalse(self.plugin._busy)
            self.assertIsNone(self.plugin._job)
            self.configure("models", **{field: old})
        self.client.set_gpt_weights.assert_not_awaited()
        self.client.synthesize.assert_not_awaited()

    async def test_load_start_returns_before_work_and_switches_gpt_then_sovits_serially(self):
        await self.connect()
        gpt_entered, gpt_release = asyncio.Event(), asyncio.Event()
        sovits_entered, sovits_release = asyncio.Event(), asyncio.Event()
        events = []

        async def gpt(_path):
            events.append("gpt:start")
            gpt_entered.set()
            await gpt_release.wait()
            events.append("gpt:done")

        async def sovits(_path):
            events.append("sovits:start")
            sovits_entered.set()
            await sovits_release.wait()
            events.append("sovits:done")

        self.client.set_gpt_weights.side_effect = gpt
        self.client.set_sovits_weights.side_effect = sovits
        result = await asyncio.wait_for(self.plugin.setup_start_job("load_models"), 1)
        self.assertIs(result["success"], True, result)
        job_id = result["job"]["id"]
        self.assertEqual(result["job"]["status"], "running")
        await asyncio.wait_for(gpt_entered.wait(), 2)
        self.assertFalse(self.plugin._job_task.done())
        self.assertIn(self.plugin._job_task, self.plugin._tasks)
        self.client.set_sovits_weights.assert_not_awaited()
        self.assertEqual(events, ["gpt:start"])
        self.assertIsNone(self.plugin._loaded_signature)
        gpt_release.set()
        await asyncio.wait_for(sovits_entered.wait(), 2)
        self.assertEqual(events, ["gpt:start", "gpt:done", "sovits:start"])
        self.assertIsNone(self.plugin._loaded_signature)
        self.assertEqual((await self.plugin.setup_job(job_id))["status"], "running")
        sovits_release.set()
        job = await self.finish_job(job_id)
        self.assertEqual(events, ["gpt:start", "gpt:done", "sovits:start", "sovits:done"])
        self.assertEqual(job["status"], "success")
        self.assertEqual(self.plugin._model_status["status"], "success")
        self.assertEqual(self.plugin._loaded_signature, self.plugin._model_signature(self.plugin.config))
        self.client.set_gpt_weights.assert_awaited_once_with(self.plugin.config.models.gpt_weights)
        self.client.set_sovits_weights.assert_awaited_once_with(self.plugin.config.models.sovits_weights)
        self.factory.assert_called_with(self.plugin.config.api.base_url,
                                        timeout=self.plugin.config.api.timeout,
                                        max_audio_bytes=self.plugin.config.api.max_audio_bytes)
        result["job"]["status"] = "tampered"
        job["status"] = "tampered"
        self.assertEqual((await self.plugin.setup_job(job_id))["status"], "success")
        self.assert_small_response(result)
        self.client.synthesize.assert_not_awaited()
        self.send.custom.assert_not_awaited()

    async def test_partial_model_failure_marks_error_invalidates_load_and_blocks_preview(self):
        await self.load_models()
        self.write_fixture()
        snapshot = self.storage_snapshot()
        for failing_step in ("gpt", "sovits"):
            with self.subTest(step=failing_step):
                self.client.set_gpt_weights.reset_mock(side_effect=True)
                self.client.set_sovits_weights.reset_mock(side_effect=True)
                method = (self.client.set_gpt_weights if failing_step == "gpt"
                          else self.client.set_sovits_weights)
                method.side_effect = TTSError(f"{failing_step} 权重损坏")
                result = await self.plugin.setup_start_job("load_models")
                self.assertIs(result["success"], True, result)
                job = await self.finish_job(result["job"]["id"])
                self.assertEqual(job["status"], "error")
                self.assertIn(f"{failing_step} 权重损坏", job["message"])
                self.assertRegex(job["message"], "部分切换|原子")
                self.assertEqual(self.plugin._model_status["status"], "error")
                self.assertIsNone(self.plugin._loaded_signature)
                self.assertEqual(self.plugin._preview_audio, b"")
                if failing_step == "gpt":
                    self.client.set_sovits_weights.assert_not_awaited()
                else:
                    self.client.set_gpt_weights.assert_awaited_once()
                    self.client.set_sovits_weights.assert_awaited_once()
                denied = await self.plugin.setup_start_job("synthesize", "不能试听")
                self.assertIs(denied["success"], False, denied)
                self.assertIn("加载", denied["message"])
                with self.assertRaises(TTSError):
                    await self.plugin.setup_audio(result["job"]["id"])
                self.assert_storage_unchanged(snapshot)
        self.client.synthesize.assert_not_awaited()

    async def test_busy_job_rejects_commands_new_jobs_connection_and_both_saves(self):
        self.write_fixture()
        snapshot = self.storage_snapshot()
        job_id, release = await self.block_load()
        task = self.plugin._job_task
        command = await self.plugin.handle_tts(stream_id="stream", text="/tts 不应合成")
        self.assertIs(command[0], False)
        self.assertIn("正在合成", command[1])
        tool = await self.plugin.handle_speak(text="也不应合成", stream_id="stream")
        self.assertIs(tool["success"], False)
        self.assertIn("正在合成", tool["content"])
        responses = [await self.plugin.setup_save_api("http://localhost:9999", 90),
                     await self.plugin.setup_save_models(gpt_weights="/new.ckpt", sovits_weights="/new.pth"),
                     await self.plugin.setup_test_api(),
                     await self.plugin.setup_start_job("load_models"),
                     await self.plugin.setup_start_job("synthesize", "busy")]
        for response in responses:
            self.assertIs(response["success"], False, response)
            self.assertRegex(response["message"], "运行|任务|等待")
        self.assertEqual(self.plugin._job["id"], job_id)
        self.assertIs(self.plugin._job_task, task)
        self.assertTrue(self.plugin._busy)
        self.assert_storage_unchanged(snapshot)
        self.client.synthesize.assert_not_awaited()
        self.client.probe.assert_awaited_once()
        self.send.custom.assert_not_awaited()
        release.set()
        self.assertEqual((await self.finish_job(job_id))["status"], "success")

    async def test_preview_requires_loaded_current_weights_and_reference_audio(self):
        await self.connect()
        denied = await self.plugin.setup_start_job("synthesize", "hello")
        self.assertIs(denied["success"], False)
        self.assertIn("加载", denied["message"])
        self.assertIsNone(self.plugin._job)
        self.configure("voice", ref_audio_path=" ")
        await self.load_models()  # Loading weights alone must not require reference audio.
        previous_job = self.plugin._job["id"]
        denied = await self.plugin.setup_start_job("synthesize", "hello")
        self.assertIs(denied["success"], False)
        self.assertIn("参考音频", denied["message"])
        self.assertEqual(self.plugin._job["id"], previous_job)
        self.assertFalse(self.plugin._busy)
        self.client.synthesize.assert_not_awaited()
        self.configure("voice", ref_audio_path="/remote/only/on/server.wav")
        await self.synthesize()
        payload = self.client.synthesize.await_args.args[0]
        self.assertEqual(payload["ref_audio_path"], "/remote/only/on/server.wav")

    async def test_preview_rejects_invalid_or_over_limit_text_without_starting(self):
        await self.load_models()
        self.configure("behavior", max_text_length=3)
        previous_job = self.plugin._job["id"]
        for text in (None, 123, "", " \t\n", "四个字符"):
            with self.subTest(text=text):
                result = await self.plugin.setup_start_job("synthesize", text)
                self.assertIs(result["success"], False, result)
                self.assertEqual(self.plugin._job["id"], previous_job)
                self.assertFalse(self.plugin._busy)
        self.configure("behavior", max_text_length=2000)
        result = await self.plugin.setup_start_job("synthesize", "字" * 301)
        self.assertIs(result["success"], False, result)
        self.client.synthesize.assert_not_awaited()

    async def test_preview_real_wav_chunked_base64_reconstructs_and_every_response_is_small(self):
        self.audio = wav_bytes(frames=320000)
        self.client.synthesize.return_value = self.audio
        self.configure("voice", prompt_lang="EN", text_lang="ZH",
                       aux_ref_audio_paths=["/remote/extra.wav"], speed_factor=1.25)
        await self.load_models()
        snapshot = self.storage_snapshot()
        entered, release = asyncio.Event(), asyncio.Event()

        async def synthesize(_payload):
            entered.set()
            await release.wait()
            return self.audio

        self.client.synthesize.side_effect = synthesize
        result = await asyncio.wait_for(self.plugin.setup_start_job("synthesize", "  你好\n世界  "), 1)
        self.assertIs(result["success"], True, result)
        job_id = result["job"]["id"]
        self.assertEqual(result["job"]["status"], "running")
        await asyncio.wait_for(entered.wait(), 2)
        self.assertTrue(self.plugin._busy)
        self.assertFalse(self.plugin._job_task.done())
        with self.assertRaises(TTSError):
            await self.plugin.setup_audio(job_id)
        release.set()
        job = await self.finish_job(job_id)
        self.assertEqual(job["status"], "success")
        self.assertEqual(job["audio_bytes"], len(self.audio))
        self.assertEqual(job["duration_seconds"], 20.0)
        self.assertNotIn("base64", job)
        self.assertNotIn("audio", job)
        self.assertGreater(len(base64.b64encode(self.audio)), FRAME_LIMIT,
                           "fixture must actually need chunking")
        chunks, offset = [], 0
        for _ in range(len(self.audio) // 131072 + 2):
            response = await self.plugin.setup_audio(job_id, offset)
            self.assert_small_response(response)
            block = base64.b64decode(response["base64"], validate=True)
            self.assertTrue(block)
            self.assertLessEqual(len(block), 131072)
            self.assertEqual(response["total_bytes"], len(self.audio))
            self.assertEqual(response["next_offset"], offset + len(block))
            self.assertIs(response["done"], response["next_offset"] == len(self.audio))
            chunks.append(block)
            offset = response["next_offset"]
            if response["done"]:
                break
        self.assertGreater(len(chunks), 1)
        self.assertEqual(offset, len(self.audio))
        reconstructed = b"".join(chunks)
        self.assertEqual(reconstructed, self.audio)
        with wave.open(io.BytesIO(reconstructed), "rb") as audio:
            self.assertEqual((audio.getnchannels(), audio.getsampwidth(), audio.getframerate(),
                              audio.getnframes()), (1, 2, 16000, 320000))
            self.assertEqual(len(audio.readframes(audio.getnframes())), 640000)
        end = await self.plugin.setup_audio(job_id, len(self.audio))
        self.assertEqual(end, {"base64": "", "next_offset": len(self.audio),
                               "done": True, "total_bytes": len(self.audio)})
        self.assert_small_response(result)
        self.assert_small_response(await self.plugin.setup_state())
        self.assert_storage_unchanged(snapshot)
        payload = self.client.synthesize.await_args.args[0]
        self.assertEqual(payload["text"], "你好\n世界")
        self.assertEqual(payload["text_lang"], "zh")
        self.assertEqual(payload["prompt_lang"], "en")
        self.assertEqual(payload["ref_audio_path"], self.plugin.config.voice.ref_audio_path)
        self.assertEqual(payload["aux_ref_audio_paths"], ["/remote/extra.wav"])
        self.assertEqual(payload["speed_factor"], 1.25)
        self.assertEqual(payload["media_type"], "wav")
        self.assertEqual(payload["batch_size"], 1)
        for key in ("streaming_mode", "gpt_streaming", "use_cuda_graph", "use_flash_attention"):
            self.assertIs(payload[key], False)
        self.client.synthesize.assert_awaited_once()
        self.client.set_gpt_weights.assert_awaited_once()
        self.client.set_sovits_weights.assert_awaited_once()
        self.send.custom.assert_not_awaited()
        self.send.text.assert_not_awaited()

    async def test_audio_rejects_bad_offsets_but_accepts_end_and_unaligned_byte_offset(self):
        job_id = await self.prepare_preview()
        for offset in (-1, len(self.audio) + 1, True, False, 0.0, "0", None):
            with self.subTest(offset=offset):
                with self.assertRaisesRegex(TTSError, "偏移"):
                    await self.plugin.setup_audio(job_id, offset)
        response = await self.plugin.setup_audio(job_id, 37)
        self.assertEqual(base64.b64decode(response["base64"], validate=True), self.audio[37:37 + 131072])
        response = await self.plugin.setup_audio(job_id, len(self.audio))
        self.assertTrue(response["done"])
        self.assertEqual(response["base64"], "")

    async def test_synthesis_failure_timeout_or_invalid_wav_cannot_expose_audio(self):
        await self.load_models()
        for outcome in (TTSError("参考音频无法读取"), asyncio.TimeoutError("合成超时"), b"not a WAV"):
            with self.subTest(outcome=outcome):
                if isinstance(outcome, Exception):
                    self.client.synthesize.side_effect = outcome
                else:
                    self.client.synthesize.side_effect = None
                    self.client.synthesize.return_value = outcome
                result = await self.plugin.setup_start_job("synthesize", "试听")
                self.assertIs(result["success"], True, result)
                job = await self.finish_job(result["job"]["id"])
                self.assertEqual(job["status"], "error")
                self.assertEqual(job["audio_bytes"], 0)
                self.assertEqual(self.plugin._preview_audio, b"")
                with self.assertRaises(TTSError):
                    await self.plugin.setup_audio(result["job"]["id"])
        self.send.custom.assert_not_awaited()
        self.send.text.assert_not_awaited()

    async def test_cancel_running_load_releases_busy_and_requires_reloading(self):
        job_id, _release = await self.block_load()
        result = await asyncio.wait_for(self.plugin.setup_cancel_job(job_id), 1)
        self.assertIs(result["success"], True, result)
        job = await self.finish_job(job_id)
        self.assertEqual(job["status"], "cancelled")
        self.assertEqual(self.plugin._model_status["status"], "error")
        self.assertIsNone(self.plugin._loaded_signature)
        self.client.set_sovits_weights.assert_not_awaited()
        self.assertFalse((await self.plugin.setup_start_job("synthesize", "不能试听"))["success"])
        self.client.set_gpt_weights.side_effect = None
        await self.load_models()

    async def test_cancel_before_background_coroutine_enters_releases_busy(self):
        await self.connect()
        # These API methods have no intervening await of server work: cancel before
        # the newly created task gets its first event-loop turn.
        result = await self.plugin.setup_start_job("load_models")
        self.assertIs(result["success"], True, result)
        task = self.plugin._job_task
        cancelled = await self.plugin.setup_cancel_job(result["job"]["id"])
        self.assertIs(cancelled["success"], True, cancelled)
        await asyncio.gather(task, return_exceptions=True)
        self.assertTrue(task.cancelled())
        self.assertFalse(self.plugin._busy)
        self.assertEqual(self.plugin._tasks, set())
        self.assertEqual((await self.plugin.setup_job(result["job"]["id"]))["status"], "cancelled")
        self.assertIsNone(self.plugin._loaded_signature)
        self.client.set_gpt_weights.assert_not_awaited()

    async def test_cancel_running_synthesis_discards_audio_and_allows_retry(self):
        await self.load_models()
        entered, release = asyncio.Event(), asyncio.Event()

        async def synthesize(_payload):
            entered.set()
            await release.wait()
            return self.audio

        self.client.synthesize.side_effect = synthesize
        result = await self.plugin.setup_start_job("synthesize", "cancel me")
        self.assertIs(result["success"], True, result)
        job_id = result["job"]["id"]
        await asyncio.wait_for(entered.wait(), 2)
        self.assertIs((await self.plugin.setup_cancel_job(job_id))["success"], True)
        self.assertEqual((await self.finish_job(job_id))["status"], "cancelled")
        self.assertEqual(self.plugin._preview_audio, b"")
        with self.assertRaises(TTSError):
            await self.plugin.setup_audio(job_id)
        self.client.synthesize.side_effect = None
        await self.synthesize("重试成功")

    async def test_unload_cancels_background_task_releases_busy_and_blocks_apis(self):
        _job_id, _release = await self.block_load()
        task = self.plugin._job_task
        await asyncio.wait_for(self.plugin.on_unload(), 2)
        self.assertTrue(task.done())
        self.assertFalse(self.plugin._active)
        self.assertFalse(self.plugin._busy)
        self.assertEqual(self.plugin._tasks, set())
        self.assertIsNone(self.plugin._loaded_signature)
        self.assertEqual(self.plugin._preview_audio, b"")
        self.assertEqual(self.plugin._job["status"], "cancelled")
        # 卸载后只读设置仍可查看，操作性接口继续拒绝。
        state = await self.plugin.setup_state()
        self.assertEqual(state["config"]["models"]["lora_weights"], self.plugin.config.models.lora_weights)
        result = await self.plugin.setup_save_api("http://localhost:9999")
        self.assertIs(result["success"], False)
        self.assertFalse(self.settings.exists())

    async def test_unload_cancels_inflight_connection_probe_and_releases_busy(self):
        entered, release = asyncio.Event(), asyncio.Event()

        async def probe():
            entered.set()
            await release.wait()

        self.client.probe.side_effect = probe
        task = asyncio.create_task(self.plugin.setup_test_api())
        self.addAsyncCleanup(self.cancel_task, task)
        await asyncio.wait_for(entered.wait(), 2)
        self.assertTrue(self.plugin._busy)
        self.assertIn(task, self.plugin._tasks)
        await asyncio.wait_for(self.plugin.on_unload(), 2)
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertFalse(self.plugin._busy)
        self.assertEqual(self.plugin._tasks, set())
        self.assertEqual(self.plugin._connection["status"], "idle")

    async def test_replaced_job_id_is_rejected_by_progress_audio_and_cancel(self):
        old_id = await self.prepare_preview()
        self.assertEqual(base64.b64decode((await self.plugin.setup_audio(old_id))["base64"]), self.audio)
        new_id, release = await self.block_load()
        self.assertNotEqual(old_id, new_id)
        new_task = self.plugin._job_task
        for operation in (self.plugin.setup_job, self.plugin.setup_audio, self.plugin.setup_cancel_job):
            with self.subTest(api=operation.__name__):
                with self.assertRaisesRegex(TTSError, "替代|不存在"):
                    await operation(old_id)
        self.assertIs(self.plugin._job_task, new_task)
        self.assertFalse(new_task.done(), "cancelling an obsolete ID must not cancel its replacement")
        self.assertTrue(self.plugin._busy)
        self.assertEqual((await self.plugin.setup_job(new_id))["status"], "running")
        with self.assertRaises(TTSError):
            await self.plugin.setup_audio(new_id)
        release.set()
        self.assertEqual((await self.finish_job(new_id))["status"], "success")
        with self.assertRaises(TTSError):
            await self.plugin.setup_job(old_id)

    async def test_missing_job_id_is_rejected_and_finished_cancel_is_idempotent(self):
        for operation in (self.plugin.setup_job, self.plugin.setup_audio, self.plugin.setup_cancel_job):
            with self.subTest(api=operation.__name__):
                with self.assertRaisesRegex(TTSError, "不存在|替代"):
                    await operation("missing-job-id")
        job_id = await self.load_models()
        for _ in range(2):
            self.assertIs((await self.plugin.setup_cancel_job(job_id))["success"], True)
        self.assertEqual((await self.plugin.setup_job(job_id))["status"], "success")
        self.assertFalse(self.plugin._busy)

    @staticmethod
    async def cancel_task(task):
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


if __name__ == "__main__":
    unittest.main()
