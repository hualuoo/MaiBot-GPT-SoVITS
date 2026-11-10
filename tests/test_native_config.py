"""真实 MaiBot SDK 的原生配置回归；所有文件读写限制在临时目录。"""
from __future__ import annotations

import asyncio
import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

import tomlkit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from maibot_sdk import MaiBotPlugin
from maibot_sdk.config_widgets import ConfigWidgetSnapshot
from config_models import NO_LORA, UNSELECTED, TTSConfig
import native_config as module


CATALOG = {
    "base_models": [{"id": "v3", "label": "三代底模"}],
    "gpt_weights": [{"path": "C:/模型/GPT/s1.ckpt", "label": "GPT"}],
    "sovits_weights": [{"path": "C:/模型/SoVITS/s2.pth", "label": "SoVITS"}],
    "lora_weights": [{"path": "C:/模型/角色/小雨.pth", "label": "角色"}],
    "note": "只读取实际模型文件。",
}


class NativeHarness(module.NativeConfigMixin, MaiBotPlugin):
    config_model = TTSConfig

    def __init__(self, settings: Path):
        super().__init__()
        self._settings_path = settings
        self._active = False
        self._busy = False
        self._connection = {"status": "idle", "message": "尚未测试接口"}
        self._model_status = {"status": "idle", "message": "服务端当前权重未知"}
        self._job = None
        self._init_native()


class NativeConfigTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="native-config-test-")
        self.addCleanup(temporary.cleanup)
        # RoleLibrary拒绝非规范/联接路径；Windows 的 TEMP 可能是 8.3 短路径，先解析成完整路径。
        self.directory = Path(temporary.name).resolve()
        self.settings = self.directory / "config.toml"
        self.data = TTSConfig().model_dump(mode="python")
        self.data["models"].update(root=str(self.directory / "根目录"))
        self.plugin = NativeHarness(self.settings)
        self.plugin.set_plugin_config(self.data)
        self.plugin._active = True
        self.plugin.setup_test_api = AsyncMock(return_value={"success": True})
        self.plugin.setup_start_job = AsyncMock(return_value={"success": True})
        self.plugin.setup_prepare_role = AsyncMock(return_value={"success": True, "job": {"kind": "prepare_role", "status": "running"}})

    def seed(self, catalog=None, *, root=None):
        data = {"version": 1,
                "root": self.data["models"]["root"] if root is None else root,
                "catalog": CATALOG if catalog is None else catalog}
        (self.directory / module.INDEX_FILENAME).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def schema(self, plugin=None):
        return (plugin or self.plugin).get_webui_config_schema(
            plugin_id="local.gpt-sovits-tts", plugin_name="语音合成", plugin_version="测试版",
            plugin_description="原生配置", plugin_author="测试")

    def fields(self, plugin=None):
        sections = self.schema(plugin)["sections"]
        return {**sections["models"]["fields"], **sections["original_weights"]["fields"]}

    @property
    def roles_dir(self) -> Path:
        # 角色放在本机模型目录（GPT-SoVITS 安装目录）下的 maibot_roles。
        return Path(self.data["models"]["root"]) / "maibot_roles"

    def seed_role(self, role_id="黍/role.json", emotions=("happy", "angry", "surprised", "neutral", "sad")):
        path = self.roles_dir / role_id
        path.parent.mkdir(parents=True, exist_ok=True)
        # 资产故意不存在：清单只能读JSON，不能校验或stat/hash这些模型。
        entry = {"gpt": {"name": "不存在.ckpt"}, "sovits": {"name": "不存在.pth"},
                 "ref_audio": {"name": "不存在.wav"}, "ref_text": "参考文本"}
        path.write_text(json.dumps({"role": path.parent.name, "emotions": {emotion: entry for emotion in emotions}},
                                   ensure_ascii=False), encoding="utf-8")
        return path

    async def test_three_step_sections_keep_only_base_fields_and_real_models_save_key(self):
        self.seed()
        sections = self.schema()["sections"]
        base, ordinary, role = sections["models"], sections["original_weights"], sections["role"]
        self.assertEqual(base["title"], "第二步 · 只选择底模")
        self.assertEqual(base["name"], "models")
        self.assertEqual(set(base["fields"]), {"root", "base_model"})
        self.assertEqual({key for key, field in base["fields"].items() if not field.get("hidden")},
                         {"root", "base_model"})
        self.assertNotIn("lora_weights", base["fields"])
        self.assertEqual(ordinary["title"], "第三步 · 普通模式（单一LoRA或不使用）")
        self.assertEqual(ordinary["name"], "models")
        self.assertEqual(set(ordinary["fields"]), {"gpt_weights", "sovits_weights", "lora_weights"})
        self.assertEqual(role["name"], "role")
        self.assertEqual(role["title"], "第三步 · 角色模式与情感")
        self.assertEqual((sections["api"]["order"], base["order"], role["order"], ordinary["order"]), (1, 2, 3, 4))
        self.assertTrue(sections["voice"]["title"].startswith("第三步 · 普通模式参考"))
        # 与原生SectionRenderer相同的name优先寻址，不生成original_weights真配置键。
        config = TTSConfig().model_dump()
        config[ordinary.get("name") or "original_weights"]["lora_weights"] = "saved-role.pth"
        self.assertNotIn("original_weights", config)
        self.assertEqual(TTSConfig.model_validate(config).models.lora_weights, "saved-role.pth")

    async def test_own_json_roles_and_all_emotions_are_selects_even_in_original_mode(self):
        self.seed_role()
        self.data["role"].update(mode="original", role_id="黍/role.json", emotion="sad", emotions=["happy", "sad"])
        self.plugin.set_plugin_config(self.data)
        with patch.object(module, "scan_models", side_effect=AssertionError("schema不能扫描模型")), patch.object(
                module.RoleLibrary, "_hash_file", side_effect=AssertionError("schema不能hash资产")), patch.object(
                module.RoleLibrary, "resolve", side_effect=AssertionError("schema不能解析资产")):
            fields = self.schema()["sections"]["role"]["fields"]
        self.assertEqual(fields["role_id"]["choices"], [module.ROLE_UNSELECTED, "黍/role.json"])
        self.assertEqual(fields["mode"]["choices"], ["original", "role"])
        self.assertEqual(fields["mode"]["default"], "original")
        for name in ("role_id", "emotion", "emotions"):
            self.assertEqual((fields[name]["type"], fields[name]["ui_type"]), ("select", "select"))
            self.assertFalse(fields[name].get("hidden"))
            self.assertNotIn("visible_when", fields[name])
            self.assertTrue(all(isinstance(value, str) and value for value in fields[name]["choices"]))
            self.assertIn("先选择角色和情感并保存", fields[name]["hint"])
            self.assertIn("读取已保存角色", fields[name]["hint"])
        expected = ["happy", "angry", "surprised", "neutral", "sad"]
        self.assertEqual(fields["emotion"]["choices"], expected)
        self.assertEqual(fields["emotions"]["choices"], expected)
        self.assertTrue(fields["emotions"]["multiple"])
        self.assertEqual(fields["emotions"]["default"], ["happy", "sad"])
        self.assertEqual(fields["emotion"]["default"], "sad")

    async def test_unselected_role_previews_only_role_emotions_without_auto_selecting_role(self):
        self.seed_role(emotions=("开心", "生气", "自定义语气"))
        for current in ("", module.ROLE_UNSELECTED):
            self.data["role"]["role_id"] = current
            self.plugin.set_plugin_config(self.data)
            fields = self.schema()["sections"]["role"]["fields"]
            self.assertEqual(fields["role_id"]["default"], current)
            self.assertEqual(fields["emotion"]["choices"], ["开心", "生气", "自定义语气"])
            self.assertEqual(fields["emotions"]["choices"], fields["emotion"]["choices"])
            self.assertIn("唯一角色的预览", fields["emotion"]["hint"])

    async def test_selected_role_gets_its_entire_emotion_set_not_other_roles(self):
        self.seed_role("甲/role.json", ("happy", "angry"))
        self.seed_role("乙/role.json", ("neutral", "surprised", "sad"))
        self.data["role"].update(role_id="乙/role.json")
        self.plugin.set_plugin_config(self.data)
        fields = self.schema()["sections"]["role"]["fields"]
        self.assertEqual(fields["emotion"]["choices"], ["neutral", "surprised", "sad"])
        self.assertEqual(fields["emotions"]["choices"], fields["emotion"]["choices"])

    async def test_role_choices_keep_safe_saved_value_only_and_deduplicate(self):
        self.seed_role()
        for current in ("黍/role.json", "missing/role.json", "黍/其他.json", "../escape.json", "C:/outside.json", "甲/CON.json", "甲/坏\x00.json"):
            self.data["role"]["role_id"] = current
            self.plugin.set_plugin_config(self.data)
            choices = self.schema()["sections"]["role"]["fields"]["role_id"]["choices"]
            self.assertEqual(len(choices), len(set(choices)))
            # 只保留指向“角色文件夹/role.json”的已保存值。
            self.assertEqual(current in choices, current in ("黍/role.json", "missing/role.json"))

    async def test_roles_only_come_from_maibot_roles_not_plugin_data_or_configured_dir(self):
        # 插件 data、上级 data 和其他目录中的 JSON 都不读取；只读 maibot_roles。
        own = self.directory / "plugin"
        own.mkdir()
        for folder in (own / "data", self.directory / "data", self.directory / "外部"):
            path = folder / "不读取" / "role.json"
            path.parent.mkdir(parents=True)
            path.write_text('{"emotions":{"outside":{"gpt":{"name":"a.ckpt"},"sovits":{"name":"b.pth"},"ref_audio":{"name":"c.wav"},"ref_text":"x"}}}', encoding="utf-8")
        self.seed_role("本地/role.json", ("local",))
        self.plugin._settings_path = own / "config.toml"
        self.data["role"]["data_dir"] = str(self.directory / "外部")
        self.plugin.set_plugin_config(self.data)
        fields = self.schema()["sections"]["role"]["fields"]
        self.assertEqual(fields["role_id"]["choices"], [module.ROLE_UNSELECTED, "本地/role.json"])
        self.assertEqual(fields["emotion"]["choices"], ["local"])
        self.assertIn(self.roles_dir.as_posix(), fields["role_id"]["hint"])
        self.assertNotIn((own / "data").as_posix(), fields["role_id"]["hint"])

    async def test_cold_missing_roles_has_absolute_path_errors_and_nonempty_placeholder(self):
        # 冷读取（配置尚未注入）只用 TOML 中保存的本机模型目录。
        self.settings.write_text(tomlkit.dumps(self.data), encoding="utf-8")
        cold = NativeHarness(self.settings)
        self.assertIsNone(cold._ctx)
        fields = self.schema(cold)["sections"]["role"]["fields"]
        self.assertEqual(fields["role_id"]["choices"], [module.ROLE_UNSELECTED])
        self.assertIn(self.roles_dir.as_posix(), fields["role_id"]["hint"])
        self.assertIn("角色文件夹不存在", fields["role_id"]["hint"])
        self.assertIn("未发现可用角色JSON", fields["role_id"]["hint"])
        for name in ("emotion", "emotions"):
            self.assertEqual(fields[name]["choices"], [module.EMOTION_UNSELECTED])
            self.assertEqual(fields[name]["type"], "select")
        # 没有保存本机模型目录时提示先完成第二步。
        empty = dict(self.data, models={**self.data["models"], "root": ""})
        self.settings.write_text(tomlkit.dumps(empty), encoding="utf-8")
        fields = self.schema(NativeHarness(self.settings))["sections"]["role"]["fields"]
        self.assertEqual(fields["role_id"]["choices"], [module.ROLE_UNSELECTED])
        self.assertIn("第二步", fields["role_id"]["hint"])

    async def test_role_json_parse_error_is_specific_and_does_not_fake_success(self):
        path = self.seed_role()
        path.write_text("{错误JSON", encoding="utf-8")
        fields = self.schema()["sections"]["role"]["fields"]
        self.assertEqual(fields["role_id"]["choices"], [module.ROLE_UNSELECTED])
        self.assertIn("角色JSON无法解析", fields["role_id"]["hint"])
        self.assertIn("黍：角色JSON无法解析：role.json", fields["role_id"]["hint"])
        self.assertIn("未发现可用角色JSON", fields["role_id"]["hint"])

    async def test_roles_cache_expires_after_five_seconds_and_refresh_action_invalidates(self):
        self.seed_role()
        with patch.object(module.time, "monotonic", return_value=100.0):
            self.schema()
        self.seed_role("新角色/role.json", ("new",))
        with patch.object(module.time, "monotonic", return_value=104.9), patch.object(
                module.RoleLibrary, "list_roles", side_effect=AssertionError("5秒缓存不应重新打开JSON")):
            self.assertNotIn("新角色/role.json", self.schema()["sections"]["role"]["fields"]["role_id"]["choices"])
        with patch.object(module.time, "monotonic", return_value=105.1):
            self.assertIn("新角色/role.json", self.schema()["sections"]["role"]["fields"]["role_id"]["choices"])
        self.seed_role("刷新/role.json", ("refresh",))
        with patch.object(module, "scan_models", return_value=CATALOG):
            await self.plugin.on_config_widget_action("refresh_models")
        self.assertIn("刷新/role.json", self.schema()["sections"]["role"]["fields"]["role_id"]["choices"])

    async def test_prepare_widget_delegates_saved_role_and_all_emotions_without_hash(self):
        self.data["role"].update(mode="original", role_id="黍/role.json", emotion="sad", emotions=["happy", "angry", "surprised", "neutral", "sad"])
        self.plugin.set_plugin_config(self.data)
        with patch.object(module.RoleLibrary, "resolve", side_effect=AssertionError("按钮不应同步校验模型")), patch.object(
                module.RoleLibrary, "_hash_file", side_effect=AssertionError("按钮不应同步hash")):
            await asyncio.wait_for(self.plugin.on_config_widget_action("prepare_saved_role"), 0.5)
        self.plugin.setup_prepare_role.assert_awaited_once_with("黍/role.json", "sad", ["happy", "angry", "surprised", "neutral", "sad"])
        self.assertFalse(self.plugin._native_action_busy)
        prepare = next(widget for widget in self.plugin.config_widgets if widget.id == "prepare_saved_role")
        self.assertEqual((prepare.kind, prepare.label), ("action", "读取已保存角色"))

    async def test_prepare_widget_empty_emotions_falls_back_to_saved_default(self):
        self.data["role"].update(role_id="黍/role.json", emotion="neutral", emotions=[])
        self.plugin.set_plugin_config(self.data)
        await self.plugin.on_config_widget_action("prepare_saved_role")
        self.plugin.setup_prepare_role.assert_awaited_once_with("黍/role.json", "neutral", ["neutral"])

    async def test_prepare_widget_invalid_unselected_values_raise_before_setup(self):
        for role_id, emotion, emotions in (("", "neutral", []), (module.ROLE_UNSELECTED, "neutral", []),
                                           ("../escape.json", "neutral", []), ("黍/role.json", module.EMOTION_UNSELECTED, [])):
            self.data["role"].update(role_id=role_id, emotion=emotion, emotions=emotions)
            self.plugin.set_plugin_config(self.data)
            with self.assertRaisesRegex(ValueError, "选择.*保存"):
                await self.plugin.on_config_widget_action("prepare_saved_role")
        self.plugin.setup_prepare_role.assert_not_awaited()
        self.assertFalse(self.plugin._native_action_busy)

    async def test_running_prepare_job_is_visible_and_disables_prepare_widget(self):
        self.plugin._job = {"kind": "prepare_role", "status": "running", "message": "正在校验角色所有情感"}
        self.plugin._busy = True
        states = await self.plugin.get_config_widget_state()
        self.assertIn("正在校验角色所有情感", states["model_status"].text)
        self.assertTrue(states["prepare_saved_role"].disabled)

    async def test_real_sdk_schema_uses_sync_instance_hook_and_string_options(self):
        self.seed()
        self.assertFalse(inspect.iscoroutinefunction(self.plugin.get_webui_config_schema))
        with patch.object(module, "scan_models", side_effect=AssertionError("Schema 不能扫描")):
            schema = self.schema()
        self.assertEqual(schema["plugin_id"], "local.gpt-sovits-tts")
        self.assertEqual(len(schema["runtime_widgets"]), 6)
        fields = {**schema["sections"]["models"]["fields"],
                  **schema["sections"]["original_weights"]["fields"]}
        for name in ("base_model", "gpt_weights", "sovits_weights", "lora_weights"):
            self.assertEqual(fields[name]["ui_type"], "select")
            self.assertEqual(fields[name]["type"], "select")
            self.assertTrue(all(isinstance(value, str) and value for value in fields[name]["choices"]))
            self.assertIn("已保存", fields[name]["hint"])
        self.assertIn("C:/模型/GPT/s1.ckpt", fields["gpt_weights"]["choices"])
        self.assertIn("C:/模型/SoVITS/s2.pth", fields["sovits_weights"]["choices"])
        self.assertIn("C:/模型/角色/小雨.pth", fields["lora_weights"]["choices"])
        self.assertNotIn("C:/模型/角色/小雨.pth", fields["sovits_weights"]["choices"])
        self.assertIn("custom", fields["base_model"]["choices"])
        self.assertIn("v3", fields["base_model"]["choices"])
        self.assertNotIn("v5turbo", fields["base_model"]["choices"])

    async def test_cold_instance_reads_toml_and_prebuilt_index_without_context(self):
        self.seed()
        self.data["models"]["gpt_weights"] = "D:/远端/当前.ckpt"
        self.settings.write_text(tomlkit.dumps(self.data), encoding="utf-8")
        cold = NativeHarness(self.settings)
        self.assertIsNone(cold._ctx)
        self.assertIsNone(cold._plugin_config_instance)
        with patch.object(module, "scan_models", side_effect=AssertionError("冷检查不能扫描")):
            fields = self.fields(cold)
        self.assertIn("C:/模型/GPT/s1.ckpt", fields["gpt_weights"]["choices"])
        self.assertIn("D:/远端/当前.ckpt", fields["gpt_weights"]["choices"])
        self.assertIn("已读取本地模型索引", fields["gpt_weights"]["hint"])
        self.assertFalse(cold._active)

    async def test_empty_catalog_has_no_empty_radix_item_and_keeps_placeholders(self):
        self.seed({})
        fields = self.fields()
        self.assertEqual(fields["lora_weights"]["choices"], [NO_LORA])
        self.assertEqual(fields["gpt_weights"]["choices"], [UNSELECTED])
        self.assertEqual(fields["sovits_weights"]["choices"], [UNSELECTED])
        self.assertEqual(fields["gpt_weights"]["default"], "")
        self.assertEqual(fields["gpt_weights"]["ui_type"], "select")

    async def test_current_values_survive_missing_files_and_are_deduplicated(self):
        self.seed()
        self.data["models"].update(base_model="自定义底模", gpt_weights="C:/模型/GPT/s1.ckpt",
                                   sovits_weights="/remote/missing.pth", lora_weights="/remote/role.pth")
        self.plugin.set_plugin_config(self.data)
        fields = self.fields()
        for name in ("base_model", "gpt_weights", "sovits_weights", "lora_weights"):
            self.assertIn(self.data["models"][name], fields[name]["choices"])
            self.assertEqual(len(fields[name]["choices"]), len(set(fields[name]["choices"])))

    async def test_index_is_not_reused_for_different_directories(self):
        self.seed(root="D:/其他目录")
        fields = self.fields()
        self.assertNotIn("C:/模型/GPT/s1.ckpt", fields["gpt_weights"]["choices"])
        self.assertIn("目录已变更", fields["gpt_weights"]["hint"])

    async def test_matching_index_allows_lexical_separator_normalization(self):
        self.seed(root=self.data["models"]["root"].replace("\\", "/"))
        self.assertIn("C:/模型/GPT/s1.ckpt", self.fields()["gpt_weights"]["choices"])

    async def test_no_index_has_clear_hint_and_never_scans(self):
        with patch.object(module, "scan_models", side_effect=AssertionError("不能扫描")):
            fields = self.fields()
        self.assertIn("尚无", fields["gpt_weights"]["hint"])
        self.assertIn("重新进入", fields["gpt_weights"]["hint"])

    async def test_corrupt_oversized_or_bad_index_does_not_break_schema(self):
        index = self.directory / module.INDEX_FILENAME
        for raw in ("{坏的JSON", "[]", '{"version":99}', "x" * (module._INDEX_LIMIT + 1)):
            with self.subTest(raw=raw[:30]):
                index.write_text(raw, encoding="utf-8")
                self.plugin._init_native()
                self.assertIn(NO_LORA, self.fields()["lora_weights"]["choices"])
                self.assertIn("索引", self.fields()["gpt_weights"]["hint"])

    async def test_invalid_config_or_index_paths_do_not_break_cold_schema(self):
        cold = NativeHarness(self.directory / "配置\x00.toml")
        with patch.object(module, "scan_models", side_effect=AssertionError("不能扫描")):
            self.assertIn("models", self.schema(cold)["sections"])
        self.settings.write_text("[models\n无法解析", encoding="utf-8")
        cold = NativeHarness(self.settings)
        self.assertEqual(self.fields(cold)["lora_weights"]["choices"], [NO_LORA])

    async def test_second_schema_uses_memory_index_cache_without_reopening_file(self):
        self.seed()
        self.schema()
        with patch.object(Path, "open", side_effect=AssertionError("热缓存无需读文件")):
            fields = self.fields()
        self.assertIn("C:/模型/GPT/s1.ckpt", fields["gpt_weights"]["choices"])

    async def test_widget_snapshot_is_complete_chinese_and_state_has_no_io(self):
        with patch.object(Path, "open", side_effect=AssertionError("状态不能读文件")), patch.object(
                module, "scan_models", side_effect=AssertionError("状态不能扫描")):
            states = await self.plugin.get_config_widget_state()
        snapshot = ConfigWidgetSnapshot(widgets=self.plugin.config_widgets, states=states)
        self.assertEqual(set(snapshot.states), {widget.id for widget in snapshot.widgets})
        self.assertEqual(len(snapshot.widgets), 6)
        self.assertTrue(all(any("\u4e00" <= char <= "\u9fff" for char in widget.label) for widget in snapshot.widgets))
        load = next(widget for widget in snapshot.widgets if widget.id == "load_saved_models")
        self.assertIn("共享", load.confirmation)
        self.assertIn("未保存", states["api_status"].text)
        for name in ("test_saved_api", "refresh_models", "load_saved_models", "prepare_saved_role"):
            self.assertFalse(states[name].disabled)

    async def test_cold_or_disabled_states_disable_all_actions_without_io(self):
        for active, enabled in ((False, True), (True, False)):
            self.plugin._active = active
            self.data["plugin"]["enabled"] = enabled
            self.plugin.set_plugin_config(self.data)
            with patch.object(Path, "open", side_effect=AssertionError("状态不能读文件")):
                states = await self.plugin.get_config_widget_state()
            for name in ("test_saved_api", "refresh_models", "load_saved_models", "prepare_saved_role"):
                self.assertTrue(states[name].disabled)
            with self.assertRaisesRegex(ValueError, "启用"):
                await self.plugin.on_config_widget_action("test_saved_api")
        self.plugin.setup_test_api.assert_not_awaited()

    async def test_saved_actions_delegate_without_form_draft_arguments(self):
        await self.plugin.on_config_widget_action("test_saved_api")
        self.plugin.setup_test_api.assert_awaited_once_with()
        await self.plugin.on_config_widget_action("load_saved_models")
        self.plugin.setup_start_job.assert_awaited_once_with(kind="load_models")
        self.assertFalse(self.plugin._native_action_busy)

    async def test_busy_and_unknown_action_do_not_invoke_setup(self):
        self.plugin._busy = True
        with self.assertRaisesRegex(ValueError, "正在运行"):
            await self.plugin.on_config_widget_action("test_saved_api")
        states = await self.plugin.get_config_widget_state()
        self.assertTrue(states["refresh_models"].disabled)
        with self.assertRaisesRegex(ValueError, "未知"):
            await self.plugin.on_config_widget_action("api_status")
        self.plugin.setup_test_api.assert_not_awaited()

    async def test_setup_failure_is_value_error_and_persists_as_status(self):
        self.plugin.setup_test_api.return_value = {"success": False, "message": "接口连接失败"}
        with self.assertRaisesRegex(ValueError, "接口连接失败"):
            await self.plugin.on_config_widget_action("test_saved_api")
        self.assertFalse(self.plugin._native_action_busy)
        states = await self.plugin.get_config_widget_state()
        self.assertIn("接口连接失败", states["model_status"].text)
        self.assertEqual(states["model_status"].tone, "error")

    async def test_refresh_runs_scanner_off_loop_persists_small_index_and_reloads_cold(self):
        thread_ids = []
        import threading
        loop_thread = threading.get_ident()
        def scan(root):
            thread_ids.append(threading.get_ident())
            self.assertEqual(root, self.data["models"]["root"])
            return CATALOG
        with patch.object(module, "scan_models", side_effect=scan) as scanner:
            await self.plugin.on_config_widget_action("refresh_models")
        scanner.assert_called_once()
        self.assertNotEqual(thread_ids, [loop_thread])
        index = self.directory / module.INDEX_FILENAME
        saved = json.loads(index.read_text(encoding="utf-8"))
        self.assertEqual(saved["version"], 1)
        self.assertEqual(saved["root"], self.data["models"]["root"])
        self.assertLess(index.stat().st_size, module._INDEX_LIMIT)
        self.assertFalse(list(self.directory.glob("*.tmp")))
        self.settings.write_text(tomlkit.dumps(self.data), encoding="utf-8")
        cold = NativeHarness(self.settings)
        self.assertIn("C:/模型/角色/小雨.pth", self.fields(cold)["lora_weights"]["choices"])
        self.assertIn("重新进入", self.fields()["gpt_weights"]["hint"])

    async def test_refresh_failure_does_not_escape_lifecycle_but_action_raises(self):
        with patch.object(module, "scan_models", side_effect=ValueError("目录不存在")):
            result = await self.plugin.refresh_native_catalog()
            self.assertFalse(result["success"])
            with self.assertRaisesRegex(ValueError, "目录不存在"):
                await self.plugin.on_config_widget_action("refresh_models")
        self.assertFalse(self.plugin._native_refreshing)
        self.assertFalse(self.plugin._native_action_busy)
        self.assertIn("目录不存在", (await self.plugin.get_config_widget_state())["model_status"].text)
        self.assertIn(NO_LORA, self.fields()["lora_weights"]["choices"])

    async def test_persistence_failure_keeps_memory_choices_and_reports_warning(self):
        with patch.object(module, "scan_models", return_value=CATALOG), patch.object(
                module, "_write_index", side_effect=PermissionError("索引不可写")):
            result = await self.plugin.refresh_native_catalog()
        self.assertFalse(result["success"])
        self.assertEqual(self.plugin._native_catalog_tone, "warning")
        self.assertIn("C:/模型/GPT/s1.ckpt", self.fields()["gpt_weights"]["choices"])

    async def test_directory_changes_do_not_expose_old_index_choices(self):
        self.seed()
        self.fields()
        self.data["models"]["root"] = "D:/新目录"
        self.plugin.set_plugin_config(self.data)
        fields = self.fields()
        self.assertNotIn("C:/模型/GPT/s1.ckpt", fields["gpt_weights"]["choices"])
        self.assertIn("目录已变更", fields["gpt_weights"]["hint"])

    async def test_running_model_job_is_visible_in_poll_state(self):
        self.plugin._job = {"kind": "load_models", "status": "running", "message": "正在加载角色权重"}
        self.plugin._busy = True
        states = await self.plugin.get_config_widget_state()
        self.assertIn("正在加载角色权重", states["model_status"].text)
        self.assertTrue(states["load_saved_models"].disabled)


if __name__ == "__main__":
    unittest.main()
