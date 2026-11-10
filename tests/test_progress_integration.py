"""真实配置保存动作与跨实例续配进度集成测试。"""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch, AsyncMock

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from plugin import create_plugin
import setup_service


class ProgressIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.p=create_plugin();self.p._settings_path=Path(self.temp.name).resolve()/"config.toml"
        self.p.set_plugin_config(self.p.build_default_config());await self.p.on_load()

    async def asyncTearDown(self):
        await self.p.on_unload()

    async def test_actions_advance_then_new_instance_resumes(self):
        self.assertEqual((await self.p.setup_state())["progress"]["next_step"],1)
        with patch.object(setup_service,"GPTSoVITSClient") as factory:
            factory.return_value.probe=AsyncMock(return_value="ok")
            await self.p.setup_save_api("http://localhost:9880",180)
            self.assertTrue((await self.p.setup_test_api())["success"])
        self.assertEqual((await self.p.setup_state())["progress"]["next_step"],2)
        catalog={"base_models":[{"id":"v5turbo","gpt_path":"base.ckpt","sovits_path":"base.pth"}]}
        with patch.object(setup_service,"scan_models",return_value=catalog):
            self.assertTrue((await self.p.setup_save_base("C:/models","base.pth"))["success"])
        self.assertEqual((await self.p.setup_state())["progress"]["next_step"],3)
        self.assertTrue((await self.p.setup_save_models(root="C:/models",base_model="v5turbo",
            gpt_weights="base.ckpt",sovits_weights="base.pth",ref_audio_path="ref.wav",prompt_text="参考文本"))["success"])
        self.assertEqual((await self.p.setup_state())["progress"]["next_step"],4)
        result=await self.p.setup_save_parameters(speed_factor=1,top_k=5,top_p=1,temperature=1,
            seed=-1,text_split_method="cut5",parallel_infer=False,max_text_length=300,cooldown_seconds=5,send_text=False)
        self.assertTrue(result["success"])
        self.assertEqual((await self.p.setup_state())["progress"]["next_step"],5)
        fresh=create_plugin();fresh._settings_path=self.p._settings_path;fresh.set_plugin_config(self.p.config.model_dump())
        state=await fresh.setup_state()
        self.assertEqual(state["progress"]["next_step"],5)
        self.assertTrue(state["progress"]["needs_revalidate"])
        self.assertEqual(state["progress"]["completed_steps"],[1,2,3,4])
        self.assertEqual(fresh._connection["status"],"idle")

    async def test_failed_save_does_not_mark_completed(self):
        before=await self.p.setup_state()
        result=await self.p.setup_save_parameters(speed_factor=9,top_k=5,top_p=1,temperature=1,
            seed=-1,text_split_method="cut5",parallel_infer=False,max_text_length=300,cooldown_seconds=5,send_text=False)
        self.assertFalse(result["success"])
        self.assertEqual((await self.p.setup_state())["progress"],before["progress"])
        self.assertFalse((self.p._settings_path.parent/"data/.state/wizard-progress.json").exists())


if __name__=="__main__":unittest.main()
