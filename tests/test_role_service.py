"""角色模式与设置页后端集成（使用真实 SDK），所有文件都在临时目录。

在项目根目录运行：
  python -B -m unittest discover -s tests -p test_role_service.py -v

No SDK/config stubs, model deserialization, network requests, or large weights.
Thread gates test the actual asyncio.to_thread boundary, not a fake job runner.
"""
from __future__ import annotations

import asyncio
import base64
from copy import deepcopy
import hashlib
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
from config_models import NO_LORA, TTSConfig
import model_catalog
import plugin as plugin_module
import role_library
from tts_client import GPTSoVITSClient, TTSError
sys.path.insert(0, str(Path(__file__).resolve().parent))
from role_fixtures import emotion_entry, file_entry, write_manifest

KINDS = {"GPT 权重": "gpt", "SoVITS 权重": "sovits", "参考音频": "ref_audio"}


def tiny_wav() -> bytes:
    stream = io.BytesIO()
    with wave.open(stream, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x01\x00" * 80)
    return stream.getvalue()


class RoleServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="maibot-role-service-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.settings = self.root / "config.toml"
        self.standard = self.root / "GPT-SoVITS"
        # 角色放在 GPT-SoVITS 安装目录下的 maibot_roles；插件 data 只保存状态快照。
        self.data_dir = self.standard / "maibot_roles"
        self.plugin_data = self.root / "data"
        self.role_dir = self.data_dir / "黍"
        self.shared = self.data_dir / "_shared"
        for directory in (self.role_dir, self.shared, self.standard):
            directory.mkdir(parents=True, exist_ok=True)
        self.plugin = plugin_module.create_plugin()
        self.assertIsInstance(self.plugin, MaiBotPlugin)
        values = self.plugin.build_default_config()
        values["models"].update(root=self.standard.as_posix(),
                                base_model="custom", gpt_weights="/server/manual.ckpt",
                                sovits_weights="/server/manual.pth", lora_weights=NO_LORA)
        values["voice"].update(ref_audio_path="/server/manual.wav", prompt_text="手工参考文本。",
                               prompt_lang="ja", text_lang="auto", speed_factor=1.25,
                               aux_ref_audio_paths=["/server/manual-aux.wav"])
        values["behavior"]["cooldown_seconds"] = 0
        self.plugin.set_plugin_config(values)
        self.assertIsInstance(self.plugin.config, TTSConfig)
        self.plugin._settings_path = self.settings
        self.send = SimpleNamespace(custom=AsyncMock(return_value={"sent": True}), text=AsyncMock())
        self.plugin._ctx = SimpleNamespace(send=self.send)
        self.start_patch(patch.object(self.plugin, "_get_logger", return_value=Mock()))
        self.audio = tiny_wav()
        # Keep real constructors/payload handling; replace only network methods.
        self.probe = self.start_patch(patch.object(GPTSoVITSClient, "probe", new_callable=AsyncMock))
        self.probe.return_value = "已确认 POST /tts"
        self.set_gpt = self.start_patch(patch.object(GPTSoVITSClient, "set_gpt_weights", new_callable=AsyncMock))
        self.set_sovits = self.start_patch(patch.object(GPTSoVITSClient, "set_sovits_weights", new_callable=AsyncMock))
        self.synthesize = self.start_patch(patch.object(GPTSoVITSClient, "synthesize", new_callable=AsyncMock))
        self.synthesize.return_value = self.audio
        await self.plugin.on_load()
        await self.plugin._save_setup({})
        self.settings.write_text("# temporary manual configuration\n" + self.settings.read_text(encoding="utf-8"), encoding="utf-8")
        # role.json 格式：每个情感写 gpt、sovits（完整权重或 LoRA）、ref_audio 与 ref_text。
        self.entry = emotion_entry(self.role_dir, "happy", gpt=b"PK-tiny-role-gpt", sovits=b"01-tiny-full-v2",
                                   audio=self.audio, audio_name="happy.wav", ref_text="这株盆栽今天已经完成了它的指标。",
                                   prompt_lang="ZH", text_lang="AUTO")
        self.assets = {key: self.role_dir / self.entry[key]["name"] for key in ("gpt", "sovits", "ref_audio")}
        self.manifest = self.role_dir / "role.json"
        self.role_id = "黍/role.json"
        self.save_manifest()

    async def asyncTearDown(self):
        await self.plugin.on_unload()
        self.assertFalse(self.plugin._busy)
        self.assertEqual(self.plugin._tasks, set())

    def start_patch(self, patcher):
        result = patcher.start()
        self.addCleanup(patcher.stop)
        return result

    def save_manifest(self, payload=None):
        write_manifest(self.role_dir, payload if payload is not None else {"role": "黍", "emotions": {"happy": self.entry}})

    def config_state(self):
        return deepcopy(self.plugin.config.model_dump()), self.settings.read_bytes()

    def assert_config_unchanged(self, before):
        self.assertEqual(self.config_state(), before)

    def assert_no_network(self):
        for method in (self.probe, self.set_gpt, self.set_sovits, self.synthesize):
            method.assert_not_awaited()

    async def finish_job(self, result, status="success"):
        self.assertTrue(result["success"], result)
        task = self.plugin._job_task
        self.assertIsNotNone(task)
        await asyncio.wait_for(asyncio.shield(task), 3)
        # The production done callback owns task-set cleanup.
        await asyncio.sleep(0)
        job = await self.plugin.setup_job(result["job"]["id"])
        self.assertEqual(job["status"], status, job)
        self.assertFalse(self.plugin._busy)
        self.assertNotIn(task, self.plugin._tasks)
        return job

    def blocking_call(self, real_call):
        """A synchronous worker gate, with an asyncio notification (no polling)."""
        entered = asyncio.Event()
        release = threading.Event()
        loop = asyncio.get_running_loop()
        self.addCleanup(release.set)

        def blocked(*args, **kwargs):
            loop.call_soon_threadsafe(entered.set)
            if not release.wait(3):
                raise AssertionError("test worker was not released")
            return real_call(*args, **kwargs)
        return blocked, entered, release

    async def test_setup_roles_only_parses_json_without_resolve_scan_or_hash(self):
        self.save_manifest({"role": "黍", "emotions": {key: self.entry for key in ("happy", "angry", "neutral", "custom")}})
        # Deliberately absent assets are irrelevant to fast list/parse.
        self.assets["ref_audio"].unlink()
        bad = self.data_dir / "坏"
        bad.mkdir()
        (bad / "role.json").write_text("{", encoding="utf-8")
        lib = self.plugin._roles()
        with patch.object(lib, "resolve", side_effect=AssertionError("list must not resolve")), \
             patch.object(lib, "_hash_file", side_effect=AssertionError("list must not hash")), \
             patch.object(role_library, "scan_models", side_effect=AssertionError("list must not scan")):
            result = await asyncio.wait_for(self.plugin.setup_roles(), 1)
        self.assertEqual([r["id"] for r in result["roles"]], [self.role_id])
        self.assertEqual(result["roles"][0]["emotions"], [
            {"id": "happy", "label": "开心"}, {"id": "angry", "label": "生气"},
            {"id": "neutral", "label": "中立"}, {"id": "custom", "label": "custom"}])
        self.assertTrue(any(error.startswith("坏：角色JSON无法解析") for error in result["errors"]), result)
        self.assert_no_network()

    async def test_prepare_returns_immediately_then_real_library_hashes_and_persists(self):
        before = self.config_state()
        lib = self.plugin._roles()
        real_hash = lib._hash_file
        blocked, entered, release = self.blocking_call(real_hash)
        with patch.object(lib, "_hash_file", side_effect=blocked) as hashes:
            result = await asyncio.wait_for(self.plugin.setup_prepare_role(self.role_id, "happy"), 0.5)
            self.assertTrue(result["success"], result)
            self.assertEqual(result["job"]["kind"], "prepare_role")
            self.assertEqual(result["job"]["status"], "running")
            await asyncio.wait_for(entered.wait(), 1)
            self.assertFalse(self.plugin._job_task.done())
            self.assertTrue(self.plugin._busy)
            self.assert_config_unchanged(before)
            release.set()
            await self.finish_job(result)
        self.assertEqual(hashes.call_count, 3)
        profile = self.plugin._role_status["profile"]
        self.assertEqual(self.plugin._role_status["status"], "success")
        self.assertEqual(profile["warnings"], [])
        self.assertEqual([asset["kind"] for asset in profile["assets"]], list(KINDS))
        for asset in profile["assets"]:
            content = self.assets[KINDS[asset["kind"]]].read_bytes()
            self.assertEqual(asset["size"], len(content))
            self.assertEqual(asset["sha256"], hashlib.sha256(content).hexdigest())
            self.assertFalse(asset["reused"])
        cfg = self.plugin.config
        self.assertEqual(cfg.role.model_dump(), {"mode": "role", "role_id": self.role_id, "emotion": "happy",
                                                 "emotions": ["happy"], "llm_emotion": True, "llm_task": "fast_model"})
        self.assertEqual(cfg.models.gpt_weights, self.assets["gpt"].as_posix())
        self.assertEqual(cfg.models.sovits_weights, self.assets["sovits"].as_posix())
        self.assertEqual(cfg.models.base_model, "v2")
        self.assertEqual(cfg.models.lora_weights, NO_LORA)
        self.assertEqual(cfg.voice.ref_audio_path, self.assets["ref_audio"].as_posix())
        self.assertEqual(cfg.voice.prompt_text, self.entry["ref_text"])
        self.assertEqual(cfg.voice.speed_factor, 1.25)
        persisted = tomlkit.parse(self.settings.read_text(encoding="utf-8")).unwrap()
        self.assertEqual(persisted["models"], cfg.models.model_dump())
        self.assertEqual(persisted["voice"], cfg.voice.model_dump())
        self.assertEqual(persisted["role"], cfg.role.model_dump())
        self.assertTrue(self.settings.read_text(encoding="utf-8").startswith("# temporary manual configuration"))
        self.assertIsNone(self.plugin._loaded_signature)
        self.assert_no_network()

    async def test_original_snapshot_manual_then_switch_role_prepare_restore(self):
        manual = {"models": self.plugin.config.models.model_dump(), "voice": self.plugin.config.voice.model_dump()}
        await self.plugin._capture_original()
        snapshot = self.plugin._original_snapshot_path()
        self.assertEqual(json.loads(snapshot.read_text(encoding="utf-8")), manual)
        switched = await self.plugin.setup_set_mode("role")
        self.assertTrue(switched["success"], switched)
        await self.finish_job(await self.plugin.setup_prepare_role(self.role_id, "happy"))
        self.assertNotEqual(self.plugin.config.models.gpt_weights, manual["models"]["gpt_weights"])
        self.assertEqual(json.loads(snapshot.read_text(encoding="utf-8")), manual)
        # An unrelated setting changed in role mode must not be rolled back.
        await self.plugin._save_setup({"behavior": {"send_text": True}})
        self.plugin._preview_audio = self.audio
        self.plugin._loaded_signature = ("previously-loaded",)
        restored = await self.plugin.setup_set_mode("original")
        self.assertTrue(restored["success"], restored)
        self.assertEqual(self.plugin.config.models.model_dump(), manual["models"])
        self.assertEqual(self.plugin.config.voice.model_dump(), manual["voice"])
        self.assertEqual(self.plugin.config.role.mode, "original")
        self.assertTrue(self.plugin.config.behavior.send_text)
        self.assertIsNone(self.plugin._loaded_signature)
        self.assertEqual(self.plugin._preview_audio, b"")
        disk = tomlkit.parse(self.settings.read_text(encoding="utf-8")).unwrap()
        self.assertEqual(disk["models"], manual["models"])
        self.assertEqual(disk["voice"], manual["voice"])
        self.assert_no_network()

    async def test_prepare_lora_role_uses_installed_standard_base_without_copying(self):
        pretrained = self.standard / "GPT_SoVITS" / "pretrained_models"
        gpt = pretrained / "s1v3.ckpt"
        sovits = pretrained / "gsv-v5-pretrained" / "s2Gv5turbo.pth"
        base_content = b"PK-small-known-standard-v5turbo"
        for path, content in ((gpt, b"PK-small-standard-gpt"), (sovits, base_content),
                              (pretrained / "gsv-v5-pretrained" / "vocoder.pth", b"vocoder")):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        self.entry = emotion_entry(self.role_dir, "happy", gpt=b"PK-full-role-gpt", sovits=b"08-small-role-lora",
                                   sovits_name="shu_happy.pth", audio=self.audio, audio_name="happy.wav",
                                   ref_text="这株盆栽今天已经完成了它的指标。")
        self.save_manifest({"role": "黍", "emotions": {"happy": self.entry},
                            "base_model": {"name": "s2Gv5turbo.pth", "size": len(base_content),
                                           "sha256": hashlib.sha256(base_content).hexdigest()}})
        catalog = {"base_models": [{"id": "v5turbo", "gpt_path": gpt.as_posix(), "sovits_path": sovits.as_posix()}],
                   "gpt_weights": [], "sovits_weights": []}
        before = {p.relative_to(self.standard).as_posix(): p.read_bytes() for p in self.standard.rglob("*") if p.is_file()}
        with patch.object(role_library, "scan_models", return_value=catalog) as scan:
            await self.finish_job(await self.plugin.setup_prepare_role(self.role_id, "happy"))
        scan.assert_called_once_with(self.standard.as_posix())
        profile = self.plugin._role_status["profile"]
        self.assertEqual(profile["models"], {"base_model": "v5turbo", "gpt_weights": (self.role_dir / "shu_happy.ckpt").as_posix(),
                                             "sovits_weights": sovits.as_posix(), "lora_weights": (self.role_dir / "shu_happy.pth").as_posix()})
        base = profile["assets"][-1]
        self.assertEqual((base["kind"], base["path"], base["reused"]), ("底模", sovits.as_posix(), True))
        after = {p.relative_to(self.standard).as_posix(): p.read_bytes() for p in self.standard.rglob("*") if p.is_file()}
        self.assertEqual(after, before)
        self.assertFalse((self.role_dir / sovits.name).exists())
        self.assertEqual(self.plugin.config.models.lora_weights, (self.role_dir / "shu_happy.pth").as_posix())
        self.assertIn("v5turbo", self.plugin._role_status["message"])
        self.assert_no_network()

    async def test_prepare_stale_api_or_model_directories_never_overwrites_new_config(self):
        original = {"api": {"base_url": self.plugin.config.api.base_url},
                    "models": {"root": self.plugin.config.models.root, "base_model": self.plugin.config.models.base_model}}
        for section, field in (("api", "base_url"), ("models", "root"), ("models", "base_model")):
            with self.subTest(section=section, field=field):
                # 每个子测试从原始配置开始：角色目录跟随本机模型目录，上一轮改过的目录里没有角色。
                await self.plugin._save_setup(original)
                lib = self.plugin._roles()
                blocked, entered, release = self.blocking_call(lib.resolve)
                with patch.object(lib, "resolve", side_effect=blocked):
                    result = await self.plugin.setup_prepare_role(self.role_id, "happy")
                    await asyncio.wait_for(entered.wait(), 1)
                    value = {"base_url": "http://127.0.0.1:9881", "base_model": "v2"}.get(field) or (self.root / ("changed-" + field)).as_posix()
                    if field == "root":
                        Path(value).mkdir(exist_ok=True)
                    await self.plugin._save_setup({section: {field: value}}, allow_busy=True)
                    changed = self.config_state()
                    release.set()
                    job = await self.finish_job(result, "error")
                self.assertIn("已改变", job["message"])
                self.assert_config_unchanged(changed)
                self.assertEqual(self.plugin.config.role.mode, "original")
                self.assertIsNone(self.plugin._role_status["profile"])
        self.assert_no_network()

    async def test_prepare_cancel_in_worker_does_not_apply_or_write(self):
        before = self.config_state()
        lib = self.plugin._roles()
        blocked, entered, release = self.blocking_call(lib.resolve)
        with patch.object(lib, "resolve", side_effect=blocked):
            result = await self.plugin.setup_prepare_role(self.role_id, "happy")
            await asyncio.wait_for(entered.wait(), 1)
            cancelled = await self.plugin.setup_cancel_job(result["job"]["id"])
            self.assertTrue(cancelled["success"], cancelled)
            release.set()
            await self.finish_job(result, "cancelled")
        self.assert_config_unchanged(before)
        self.assertEqual(self.plugin._role_status["status"], "error")
        self.assertIsNone(self.plugin._role_status["profile"])
        self.assertFalse(self.plugin._original_snapshot_path().exists())
        self.assert_no_network()

    async def test_prepare_cancel_before_task_starts_releases_guard(self):
        before = self.config_state()
        result = await self.plugin.setup_prepare_role(self.role_id, "happy")
        self.assertTrue(result["success"], result)
        task = self.plugin._job_task
        task.cancel()  # No yield since setup_prepare_role: exercise its done callback.
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.sleep(0)
        self.assertEqual(self.plugin._job["status"], "cancelled")
        self.assertFalse(self.plugin._busy)
        self.assertNotIn(task, self.plugin._tasks)
        self.assert_config_unchanged(before)
        self.assert_no_network()

    async def test_prepare_real_hash_error_preserves_manual_config(self):
        self.entry["ref_audio"]["sha256"] = "0" * 64
        self.save_manifest()
        before = self.config_state()
        job = await self.finish_job(await self.plugin.setup_prepare_role(self.role_id, "happy"), "error")
        self.assertIn("SHA256", job["message"])
        self.assert_config_unchanged(before)
        self.assertEqual(self.plugin._role_status["status"], "error")
        self.assertIsNone(self.plugin._role_status["profile"])
        self.assertFalse(self.plugin._original_snapshot_path().exists())
        self.assert_no_network()

    async def test_prepare_invalid_selection_rejected_without_task_or_write(self):
        before = self.config_state()
        for role, emotion in (("", "happy"), ("x" * 241, "happy"), (self.role_id, ""), (self.role_id, "x" * 81)):
            with self.subTest(role=role, emotion=emotion):
                result = await self.plugin.setup_prepare_role(role, emotion)
                self.assertFalse(result["success"], result)
                self.assertIsNone(self.plugin._job_task)
                self.assertFalse(self.plugin._busy)
                self.assert_config_unchanged(before)

    async def test_running_model_job_blocks_mode_change_and_role_prepare(self):
        self.plugin._connection = {"status": "success", "message": "test"}
        entered, release = asyncio.Event(), asyncio.Event()
        self.addCleanup(release.set)

        async def load(_path):
            entered.set()
            await release.wait()
        self.set_gpt.side_effect = load
        result = await self.plugin.setup_start_job("load_models")
        await asyncio.wait_for(entered.wait(), 1)
        self.assertTrue(self.plugin._busy)
        self.assertFalse((await self.plugin.setup_set_mode("role"))["success"])
        self.assertFalse((await self.plugin.setup_prepare_role(self.role_id, "happy"))["success"])
        release.set()
        await self.finish_job(result)
        self.set_gpt.assert_awaited_once_with("/server/manual.ckpt")
        self.set_sovits.assert_awaited_once_with("/server/manual.pth")

    async def test_role_mode_job_guard_requires_matching_success_profile(self):
        await self.plugin._save_setup({"role": {"mode": "role", "role_id": self.role_id, "emotion": "happy"}})
        self.plugin._connection = {"status": "success", "message": "test"}
        self.plugin._loaded_signature = self.plugin._model_signature(self.plugin.config)
        cases = [
            {"status": "idle", "profile": None},
            {"status": "error", "profile": None},
            {"status": "success", "profile": None},
            {"status": "success", "profile": {"role_id": "other/role.json", "emotion": "happy"}},
            {"status": "success", "profile": {"role_id": self.role_id, "emotion": "sad"}},
        ]
        before = self.config_state()
        for status in cases:
            for kind in ("load_models", "synthesize"):
                with self.subTest(status=status, kind=kind):
                    self.plugin._role_status = {"message": "test", **status}
                    result = await self.plugin.setup_start_job(kind, "测试")
                    self.assertFalse(result["success"], result)
                    self.assertIn("角色JSON", result["message"])
                    self.assertFalse(self.plugin._busy)
                    self.assertIsNone(self.plugin._job_task)
                    self.assert_config_unchanged(before)
        self.assert_no_network()
        await self.finish_job(await self.plugin.setup_prepare_role(self.role_id, "happy"))
        self.plugin._connection = {"status": "success", "message": "test"}
        await self.finish_job(await self.plugin.setup_start_job("load_models"))
        self.set_gpt.assert_awaited_once_with(self.assets["gpt"].as_posix())
        self.set_sovits.assert_awaited_once_with(self.assets["sovits"].as_posix())

    async def test_invalid_or_same_mode_never_changes_configuration(self):
        before = self.config_state()
        self.assertFalse((await self.plugin.setup_set_mode("invalid"))["success"])
        self.assertTrue((await self.plugin.setup_set_mode("original"))["success"])
        self.assert_config_unchanged(before)
        self.assertFalse(self.plugin._original_snapshot_path().exists())

    async def test_speak_role_uses_real_profile_models_reference_then_reuses_second_sentence(self):
        # Do not prepare first: stale manual model/ref settings must not win.
        await self.plugin._save_setup({"role": {"mode": "role", "role_id": self.role_id, "emotion": "happy"},
                                       "voice": {"ref_audio_path": ""}})
        before = self.config_state()
        lib = self.plugin._roles()
        events = []

        async def gpt(path):
            events.append(("gpt", path))

        async def sovits(path):
            events.append(("sovits", path))

        async def synthesize(payload):
            events.append(("synthesize", payload["text"]))
            return self.audio
        self.set_gpt.side_effect, self.set_sovits.side_effect = gpt, sovits
        self.synthesize.side_effect = synthesize
        with patch.object(lib, "resolve", wraps=lib.resolve) as resolve:
            first = await self.plugin._speak("stream", "第一句。")
            second = await self.plugin._speak("stream", "第二句。")
        self.assertIn("已发送", first)
        self.assertIn("已发送", second)
        self.assertEqual(resolve.call_count, 2)
        resolve.assert_called_with(self.role_id, "happy", self.standard.as_posix())
        self.set_gpt.assert_awaited_once_with(self.assets["gpt"].as_posix())
        self.set_sovits.assert_awaited_once_with(self.assets["sovits"].as_posix())
        self.assertEqual([item[0] for item in events], ["gpt", "sovits", "synthesize", "synthesize"])
        self.assertEqual(self.synthesize.await_count, 2)
        for index, text in enumerate(("第一句。", "第二句。")):
            payload = self.synthesize.await_args_list[index].args[0]
            self.assertEqual(payload["ref_audio_path"], self.assets["ref_audio"].as_posix())
            self.assertEqual(payload["prompt_text"], self.entry["ref_text"])
            self.assertEqual(payload["prompt_lang"], "zh")
            self.assertEqual(payload["text_lang"], "auto")
            self.assertEqual(payload["speed_factor"], 1.25)
            self.assertEqual(payload["text"], text)
            self.assertEqual(payload["media_type"], "wav")
            self.assertFalse(payload["streaming_mode"])
            sent = self.send.custom.await_args_list[index]
            self.assertEqual(sent.args, ("voice", base64.b64encode(self.audio).decode("ascii"), "stream"))
            self.assertEqual(sent.kwargs, {"processed_plain_text": text})
        self.assertEqual(self.send.custom.await_count, 2)
        self.assertEqual(self.plugin._role_status["status"], "success")
        self.assertIsNotNone(self.plugin._loaded_signature)
        self.assert_config_unchanged(before)
        self.send.text.assert_not_awaited()
        self.probe.assert_not_awaited()

    async def test_speak_original_preserves_manual_payload_and_never_loads_role_models(self):
        before = self.config_state()
        with patch.object(self.plugin, "_roles", side_effect=AssertionError("original must not open role library")), \
             patch.object(self.plugin, "_resolve_load_paths", side_effect=AssertionError("original must not auto load")):
            await self.plugin._speak("stream", "  手工模式。  ")
            await self.plugin._speak("stream", "手工模式第二句。")
        expected = deepcopy(self.plugin.config.voice.model_dump())
        expected.update(text="手工模式。", prompt_lang="ja", text_lang="auto", media_type="wav",
                        streaming_mode=False, batch_size=1, gpt_streaming=False,
                        use_cuda_graph=False, use_flash_attention=False)
        self.assertEqual(self.synthesize.await_args_list[0].args[0], expected)
        self.assertEqual(self.synthesize.await_count, 2)
        self.set_gpt.assert_not_awaited()
        self.set_sovits.assert_not_awaited()
        self.probe.assert_not_awaited()
        self.assertEqual(self.send.custom.await_count, 2)
        self.assertIsNone(self.plugin._loaded_signature)
        self.assert_config_unchanged(before)


if __name__ == "__main__":
    unittest.main()
