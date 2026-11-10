"""第四步参数与当前配置检查回归。"""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plugin import create_plugin
import setup_service


PARAMS = dict(speed_factor=1.2, top_k=8, top_p=.9, temperature=.8, seed=42,
              text_split_method="cut5", parallel_infer=False, max_text_length=300,
              cooldown_seconds=3, send_text=False)


class ParameterTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.p=create_plugin();self.p._settings_path=Path(self.temp.name).resolve()/"config.toml"
        self.p.set_plugin_config(self.p.build_default_config());await self.p.on_load()

    async def asyncTearDown(self):
        await self.p.on_unload()

    async def test_save_parameters_does_not_change_selected_role_or_assets(self):
        models=self.p.config.models.model_dump();role=self.p.config.role.model_dump()
        ref=self.p.config.voice.ref_audio_path
        result=await self.p.setup_save_parameters(**PARAMS)
        self.assertTrue(result["success"],result)
        self.assertEqual(self.p.config.models.model_dump(),models)
        self.assertEqual(self.p.config.role.model_dump(),role)
        self.assertEqual(self.p.config.voice.ref_audio_path,ref)
        self.assertEqual(self.p.config.voice.speed_factor,1.2)
        self.assertEqual(self.p.config.behavior.cooldown_seconds,3)

    async def test_bad_parameter_does_not_write_or_mutate(self):
        before=self.p.config.model_dump()
        result=await self.p.setup_save_parameters(**{**PARAMS,"top_p":2})
        self.assertFalse(result["success"])
        self.assertEqual(self.p.config.model_dump(),before)
        self.assertFalse(self.p._settings_path.exists())

    async def test_original_requires_reference_and_text_for_v5(self):
        self.p.config.models.base_model="v5turbo"
        with patch.object(setup_service,"scan_models",return_value={"base_models":[{"id":"v5turbo"}]}):
            result=await self.p.setup_check_config()
        self.assertFalse(result["ready"])
        fields={issue["field"] for issue in result["issues"] if issue["level"]=="error"}
        self.assertIn("voice.ref_audio_path",fields)
        self.assertIn("voice.prompt_text",fields)

    async def test_role_empty_manual_reference_is_warning_not_error(self):
        cfg=self.p.config.model_dump();cfg["models"]["base_model"]="v5turbo"
        cfg["role"].update(mode="role",role_id="shu/role.json",emotion="happy",emotions=["happy"])
        self.p.set_plugin_config(cfg)
        roles={"roles":[{"id":"shu/role.json","emotions":[{"id":"happy"}]}],"data_dir":"fixture"}
        with patch.object(setup_service,"scan_models",return_value={"base_models":[{"id":"v5turbo"}]}),patch.object(self.p,"setup_roles",return_value=roles):
            result=await self.p.setup_check_config()
        self.assertTrue(result["ready"],result)
        self.assertTrue(any(issue["field"]=="role" and issue["level"]=="warning" for issue in result["issues"]))
        self.assertFalse(any(issue["field"].startswith("voice.") and issue["level"]=="error" for issue in result["issues"]))


if __name__ == "__main__":
    unittest.main()
