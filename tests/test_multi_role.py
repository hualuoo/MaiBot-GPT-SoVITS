"""多情感角色与角色目录（使用真实 SDK）。

角色放在 <GPT-SoVITS 目录>/maibot_roles；插件 data 目录只保存状态快照。
运行：python -B -m unittest discover -s tests -p test_multi_role.py -v
所有写入都在临时目录中进行，不会碰到已安装的插件或模型服务。
"""
from __future__ import annotations

import asyncio
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import tomlkit
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config_models import NO_LORA, RoleConfig
import plugin as plugin_module
sys.path.insert(0, str(Path(__file__).resolve().parent))
from role_fixtures import emotion_entry, write_manifest


class RoleConfigTests(unittest.TestCase):
    def test_emotions_limits_unique_and_default(self):
        self.assertEqual(RoleConfig().emotions, [])
        valid = ["neutral"] + ["e" + str(i) for i in range(49)]
        self.assertEqual(len(RoleConfig(emotions=valid).emotions), 50)
        for values in ({"emotions": valid + ["too-many"]},
                       {"emotions": ["neutral", "x" * 81]},
                       {"emotions": ["neutral", "neutral"]},
                       {"emotions": ["happy"]},
                       {"emotions": ["neutral", 7]}):
            with self.subTest(values=values), self.assertRaises(ValidationError):
                RoleConfig(**values)


class MultiRoleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="maibot-multi-role-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.plugin_root = self.root / "plugins" / "gpt-sovits"
        # 角色固定在 GPT-SoVITS 安装目录下的 maibot_roles；插件 data 只保存状态快照。
        self.gsv = self.root / "GPT-SoVITS"
        self.own = self.gsv / "maibot_roles"
        self.plugin_data = self.plugin_root / "data"
        self.fallback = self.root / "plugins" / "data"
        self.own.mkdir(parents=True)
        self.plugin_data.mkdir(parents=True)
        self.plugin = plugin_module.create_plugin()
        values = self.plugin.build_default_config()
        values["models"].update(root=self.gsv.as_posix(), base_model="custom",
                                 gpt_weights="/manual/gpt.ckpt", sovits_weights="/manual/sovits.pth")
        values["voice"].update(ref_audio_path="/manual/voice.wav", prompt_text="manual",
                                aux_ref_audio_paths=["/manual/aux.wav"])
        self.plugin.set_plugin_config(values)
        self.settings = self.plugin_root / "config.toml"
        self.plugin._settings_path = self.settings
        await self.plugin.on_load()
        await self.plugin._save_setup({})
        self.role_id = "shu/role.json"

    async def asyncTearDown(self):
        await self.plugin.on_unload()
        self.assertFalse(self.plugin._busy)
        self.assertEqual(self.plugin._tasks, set())

    def manifest(self, root=None, emotions=("happy", "angry", "neutral")):
        """在 root/shu/role.json 生成角色：各情感共用 v2 完整 SoVITS 权重，GPT 与参考音频按情感区分。"""
        root = self.own if root is None else root
        role = root / "shu"
        payload = {"role": "shu", "emotions": {emotion: emotion_entry(role, emotion, text_lang="auto") for emotion in emotions}}
        write_manifest(role, payload)
        return payload

    def save(self, payload):
        write_manifest(self.own / "shu", payload)

    def config_state(self):
        return deepcopy(self.plugin.config.model_dump()), self.settings.read_bytes()

    async def finish(self, result, status="success"):
        self.assertTrue(result["success"], result)
        await asyncio.wait_for(asyncio.shield(self.plugin._job_task), 4)
        await asyncio.sleep(0)
        self.assertEqual(self.plugin._job["status"], status, self.plugin._job)
        self.assertFalse(self.plugin._busy)
        return self.plugin._role_status

    async def test_only_maibot_roles_is_scanned_not_plugin_data_or_parent(self):
        self.manifest(self.fallback)
        self.manifest(self.plugin_data)
        result = await self.plugin.setup_roles()
        self.assertEqual(result["data_dir"], self.own.as_posix())
        self.assertEqual(result["scanned_paths"], [self.own.as_posix()])
        self.assertEqual(result["roles"], [])
        self.assertTrue((self.fallback / self.role_id).is_file())
        self.assertTrue((self.plugin_data / self.role_id).is_file())
        self.assertEqual(self.plugin._role_data_dir(), self.own)
        # 插件 data 只保存状态快照。
        self.assertEqual(self.plugin._plugin_data_dir(), self.plugin_data)
        await self.plugin._capture_original()
        self.assertEqual(self.plugin._original_snapshot_path(), self.plugin_data / ".state" / "original-config.json")
        self.assertFalse((self.fallback / ".state").exists())
        self.assertFalse((self.own / ".state").exists())
        self.manifest()
        self.assertEqual([role["id"] for role in (await self.plugin.setup_roles())["roles"]], [self.role_id])

    async def test_role_directory_follows_saved_model_root_and_reports_missing_folder(self):
        await self.plugin._save_setup({"models": {"root": ""}})
        result = await self.plugin.setup_roles()
        self.assertEqual(result["roles"], [])
        self.assertEqual(result["data_dir"], "")
        self.assertIn("第二步", result["message"])
        other = self.root / "其他 GPT-SoVITS"
        other.mkdir()
        await self.plugin._save_setup({"models": {"root": other.as_posix()}})
        result = await self.plugin.setup_roles()
        self.assertEqual(result["data_dir"], (other / "maibot_roles").as_posix())
        self.assertIn("角色文件夹不存在", result["message"])
        self.assertIn("maibot_roles", result["errors"][0])
        self.assertFalse((other / "maibot_roles").exists(), "扫描不应自动创建文件夹")
        await self.plugin._save_setup({"models": {"root": self.gsv.as_posix()}})
        self.manifest()
        result = await self.plugin.setup_roles()
        self.assertEqual([role["id"] for role in result["roles"]], [self.role_id])
        self.assertIn("maibot_roles", result["message"])

    async def test_own_invalid_json_never_falls_back(self):
        self.manifest(self.fallback)
        (self.own / "broken").mkdir()
        (self.own / "broken" / "role.json").write_text("{", encoding="utf-8")
        result = await self.plugin.setup_roles()
        self.assertEqual(result["data_dir"], self.own.as_posix())
        self.assertEqual(result["scanned_paths"], [self.own.as_posix()])
        self.assertEqual(result["roles"], [])
        self.assertTrue(any(item.startswith("broken：角色JSON无法解析") for item in result["errors"]), result["errors"])

    async def test_roles_in_other_directories_are_never_scanned(self):
        external = self.root / "external-data"
        self.manifest(external)
        result = await self.plugin.setup_roles()
        self.assertEqual(result["scanned_paths"], [self.own.as_posix()])
        self.assertEqual(result["roles"], [])
        self.assertTrue((external / self.role_id).exists())

    async def test_empty_directory_diagnostics_only_name_the_roles_folder(self):
        result = await self.plugin.setup_roles()
        self.assertEqual(result["roles"], [])
        self.assertEqual(result["scanned_paths"], [self.own.as_posix()])
        self.assertIn("role.json", result["message"])
        self.assertIn(self.own.as_posix(), result["message"])
        self.assertNotIn(self.fallback.as_posix(), result["message"])

    async def test_three_emotions_resolved_and_default_profile_saved(self):
        self.manifest()
        library = self.plugin._roles()
        with patch.object(library, "resolve", wraps=library.resolve) as resolve:
            status = await self.finish(await self.plugin.setup_prepare_role(self.role_id, "angry", ["happy", "angry", "neutral"]))
        self.assertEqual([call.args[1] for call in resolve.call_args_list], ["happy", "angry", "neutral"])
        self.assertEqual(status["emotions"], ["happy", "angry", "neutral"])
        self.assertEqual(list(status["profiles"]), status["emotions"])
        self.assertEqual(status["profile"], status["profiles"]["angry"])
        self.assertEqual(self.plugin.config.role.emotion, "angry")
        self.assertEqual(self.plugin.config.role.emotions, status["emotions"])
        self.assertEqual(self.plugin.config.models.base_model, "v2")
        self.assertEqual(self.plugin.config.models.lora_weights, NO_LORA)
        self.assertEqual(self.plugin.config.voice.prompt_text, "angry reference")
        self.assertEqual(self.plugin.config.voice.aux_ref_audio_paths, [])
        saved = tomlkit.parse(self.settings.read_text(encoding="utf-8"))
        self.assertEqual(list(saved["role"]["emotions"]), status["emotions"])
        self.assertEqual(saved["voice"]["ref_audio_path"], status["profile"]["voice"]["ref_audio_path"])
        state = await self.plugin.setup_state()
        self.assertEqual(set(state["role_status"]), {"status", "message", "profile", "profiles", "emotions"})

    async def test_any_emotion_failure_checks_all_without_overwriting_config(self):
        payload = self.manifest()
        payload["emotions"]["angry"]["ref_audio"] = {"name": "missing.wav"}
        self.save(payload)
        before = self.config_state()
        library = self.plugin._roles()
        with patch.object(library, "resolve", wraps=library.resolve) as resolve:
            status = await self.finish(await self.plugin.setup_prepare_role(self.role_id, "happy", ["happy", "angry", "neutral"]), "error")
        self.assertEqual(resolve.call_count, 3)
        self.assertIn("angry", status["message"])
        self.assertEqual(self.config_state(), before)
        self.assertEqual(status["profiles"], {})
        self.assertEqual(status["emotions"], [])
        self.assertFalse(self.plugin._original_snapshot_path().exists())

    async def test_invalid_default_duplicates_and_limits_fail_before_job(self):
        self.manifest()
        before = self.config_state()
        for emotion, selected in (("happy", ["angry"]), ("happy", ["happy", "happy"]),
                                  ("happy", []), ("happy", ["happy"] * 51),
                                  ("happy", ["happy", "x" * 81]), ("happy", "happy")):
            with self.subTest(selected=selected):
                result = await self.plugin.setup_prepare_role(self.role_id, emotion, selected)
                self.assertFalse(result["success"], result)
                self.assertFalse(self.plugin._busy)
                self.assertEqual(self.config_state(), before)

    async def test_default_emotion_alone_when_no_emotion_list_is_given(self):
        self.manifest()
        status = await self.finish(await self.plugin.setup_prepare_role(self.role_id, "happy"))
        self.assertEqual(self.plugin.config.role.emotions, ["happy"])
        self.assertEqual(status["profile"]["emotion"], "happy")
        self.assertEqual(list(status["profiles"]), ["happy"])

    async def test_different_base_versions_fail_without_overwriting(self):
        payload = self.manifest()
        # angry 使用另一个版本的完整权重：同一角色启用的情感必须是同一底模版本。
        payload["emotions"]["angry"] = emotion_entry(self.own / "shu", "angry", sovits=b"05-tiny-v2pro-full", sovits_name="other.pth")
        self.save(payload)
        # v2Pro 还需要 GPT-SoVITS 中的说话人模型；装好后才会走到版本一致性检查。
        sv = self.gsv / "GPT_SoVITS" / "pretrained_models" / "sv" / "pretrained_eres2netv2w24s4ep4.ckpt"
        sv.parent.mkdir(parents=True)
        sv.write_bytes(b"sv")
        before = self.config_state()
        status = await self.finish(await self.plugin.setup_prepare_role(self.role_id, "happy", ["happy", "angry"]), "error")
        self.assertIn("版本", status["message"])
        self.assertEqual(self.config_state(), before)

    async def test_selected_step_two_base_mismatch_does_not_change_it(self):
        self.manifest()
        await self.plugin._save_setup({"models": {"base_model": "v3"}})
        before = self.config_state()
        status = await self.finish(await self.plugin.setup_prepare_role(self.role_id, "happy", ["happy", "angry"]), "error")
        self.assertIn("第二步", status["message"])
        self.assertEqual(self.config_state(), before)
        self.assertEqual(self.plugin.config.models.base_model, "v3")

    async def test_busy_guard_blocks_role_prepare(self):
        self.manifest()
        self.plugin._busy = True
        self.assertFalse((await self.plugin.setup_prepare_role(self.role_id, "happy", ["happy"]))["success"])
        self.plugin._busy = False

    async def test_cancel_does_not_apply_config_and_status_shape_survives(self):
        self.manifest()
        before = self.config_state()
        library = self.plugin._roles()
        entered = asyncio.Event()
        release = threading.Event()
        self.addCleanup(release.set)
        loop = asyncio.get_running_loop()
        original = library.resolve
        def blocked(*args):
            loop.call_soon_threadsafe(entered.set)
            if not release.wait(3):
                raise AssertionError("test gate timeout")
            return original(*args)
        with patch.object(library, "resolve", side_effect=blocked):
            result = await self.plugin.setup_prepare_role(self.role_id, "happy", ["happy", "angry"])
            await asyncio.wait_for(entered.wait(), 1)
            self.plugin._job_task.cancel()
            await self.plugin._job_task
            release.set()
            await asyncio.sleep(0)
        self.assertEqual(self.config_state(), before)
        self.assertEqual(self.plugin._job["status"], "cancelled")
        self.assertEqual(self.plugin._role_status["profiles"], {})
        self.assertEqual(self.plugin._role_status["emotions"], [])

    async def test_original_snapshot_is_restored_when_switching_back_from_role_mode(self):
        external = self.root / "external"
        self.manifest(external)
        self.manifest()
        manual = {"models": self.plugin.config.models.model_dump(), "voice": self.plugin.config.voice.model_dump()}
        await self.finish(await self.plugin.setup_prepare_role(self.role_id, "happy", ["happy", "angry"]))
        self.assertTrue((await self.plugin.setup_set_mode("original"))["success"])
        self.assertEqual(self.plugin.config.models.model_dump(), manual["models"])
        self.assertEqual(self.plugin.config.voice.model_dump(), manual["voice"])
        self.assertTrue((self.plugin_data / ".state" / "original-config.json").exists())
        self.assertFalse((external / ".state").exists())
        self.assertFalse((self.own / ".state").exists())


if __name__ == "__main__":
    unittest.main()
