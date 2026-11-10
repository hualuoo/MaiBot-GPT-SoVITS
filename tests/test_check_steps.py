"""配置检查的步骤归属、按需加载提示与 saved_steps（不访问真实服务）。"""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plugin import create_plugin
import setup_service
from wizard_progress import WizardProgress

CATALOG = {"base_models": [{"id": "v5turbo", "gpt_path": "C:/gsv/s1v3.ckpt", "sovits_path": "C:/gsv/v5.pth"}],
           "gpt_weights": [], "sovits_weights": [], "lora_weights": []}


class CheckStepTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.p = create_plugin()
        self.p._settings_path = Path(self.temp.name).resolve() / "config.toml"
        self.cfg = self.p.build_default_config()
        self.cfg["models"].update(root="C:/gsv", base_model="custom", gpt_weights="", sovits_weights="")
        self.p.set_plugin_config(self.cfg)
        await self.p.on_load()
        patcher = patch.object(setup_service, "scan_models", return_value=CATALOG)
        patcher.start()
        self.addCleanup(patcher.stop)

    async def asyncTearDown(self):
        await self.p.on_unload()

    def configure(self, section, **values):
        self.cfg[section].update(values)
        self.p.set_plugin_config(self.cfg)

    async def issues(self):
        result = await self.p.setup_check_config()
        self.assertTrue(result["success"])
        for item in result["issues"]:
            self.assertIn(item["step"], range(1, 7), item)
        return {(item["field"], item["level"]): item["step"] for item in result["issues"]}, result

    async def test_original_mode_issues_point_to_their_steps(self):
        issues, result = await self.issues()
        self.assertFalse(result["ready"])
        self.assertEqual(issues[("models.gpt_weights", "error")], 3)
        self.assertEqual(issues[("models.sovits_weights", "error")], 2)
        # 普通模式的参考音频在第三步选择。
        self.assertEqual(issues[("voice.ref_audio_path", "error")], 3)
        self.assertEqual(issues[("api", "warning")], 1)
        self.assertEqual(issues[("models", "warning")], 4)

    async def test_v5_prompt_text_and_invalid_preset_steps(self):
        self.configure("models", base_model="v4")
        issues, _ = await self.issues()
        self.assertEqual(issues[("models.base_model", "error")], 2)
        self.assertEqual(issues[("voice.prompt_text", "error")], 3)

    async def test_models_not_loaded_is_info_when_plugin_loads_on_demand(self):
        self.configure("models", base_model="v5turbo", gpt_weights="C:/gsv/s1v3.ckpt", sovits_weights="C:/gsv/v5.pth")
        self.configure("auto_voice", enabled=True)
        issues, _ = await self.issues()
        self.assertEqual(issues[("models", "info")], 4)
        self.assertNotIn(("models", "warning"), issues)
        self.configure("auto_voice", enabled=False)
        self.configure("role", mode="role", role_id="shu/role.json", emotion="happy", emotions=["happy"])
        roles = {"roles": [{"id": "shu/role.json", "emotions": [{"id": "happy"}]}], "data_dir": "fixture"}
        with patch.object(self.p, "setup_roles", return_value=roles):
            issues, _ = await self.issues()
        self.assertEqual(issues[("models", "info")], 4)
        self.assertEqual(issues[("role", "warning")], 3)

    async def test_invalid_url_and_disabled_plugin_belong_to_step_one(self):
        self.configure("api", base_url="ftp://127.0.0.1:9880")
        self.configure("plugin", enabled=False)
        issues, _ = await self.issues()
        self.assertEqual(issues[("api.base_url", "error")], 1)
        self.assertEqual(issues[("plugin.enabled", "warning")], 1)


class SavedStepsTests(unittest.TestCase):
    def test_saved_steps_reports_matching_markers_independent_of_order(self):
        with tempfile.TemporaryDirectory() as temp:
            plugin = create_plugin()
            cfg = plugin.config_model.model_validate(plugin.build_default_config())
            progress = WizardProgress(Path(temp) / "config.toml")
            self.assertEqual(progress.compute(cfg)["saved_steps"], [])
            progress.mark(6, cfg)
            progress.mark(4, cfg)
            result = progress.compute(cfg)
            self.assertEqual(result["saved_steps"], [4, 5])
            self.assertNotIn(5, result["completed_steps"])
            self.assertFalse(result["preview_done"])
            changed = cfg.model_copy(deep=True)
            changed.auto_voice.enabled = True
            self.assertEqual(progress.compute(changed)["saved_steps"], [4])


if __name__ == "__main__":
    unittest.main()
