"""底模/LoRA 独立选择的加载请求回归，无声学模型反序列化。"""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plugin import create_plugin
import setup_service
from config_models import NO_LORA


class LoRALoadingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.lora = self.root / "角色.pth"
        self.lora.write_bytes(b"08\x03\x04" + b"x" * 100)
        self.plugin = create_plugin()
        self.plugin._settings_path = self.root / "config.toml"
        data = self.plugin.build_default_config()
        data["models"].update(base_model="v5turbo", root=str(self.root),
            gpt_weights="/server/gpt.ckpt", sovits_weights="/server/s2Gv5turbo.pth",
            lora_weights=str(self.lora))
        self.plugin.set_plugin_config(data)
        await self.plugin.on_load()
        self.plugin._connection = {"status": "success", "message": "已连接"}
        self.catalog = {"base_models": [{"id": "v5turbo", "gpt_path": "/server/s1v3.ckpt", "sovits_path": "/server/s2Gv5turbo.pth"}],
                        "gpt_weights": [], "sovits_weights": [], "lora_weights": [], "note": "测试"}
        self.client = AsyncMock()
        self.client_patch = patch.object(setup_service, "GPTSoVITSClient", return_value=self.client)
        self.client_patch.start();self.addCleanup(self.client_patch.stop)
        self.scan_patch = patch.object(setup_service, "scan_models", return_value=self.catalog)
        self.scan_patch.start();self.addCleanup(self.scan_patch.stop)

    async def asyncTearDown(self):
        await self.plugin.on_unload()

    async def load(self):
        response = await self.plugin.setup_start_job("load_models")
        self.assertTrue(response["success"], response)
        await self.plugin._job_task
        return await self.plugin.setup_job(response["job"]["id"])

    async def test_lora_path_not_base_is_sent_to_official_setter(self):
        job = await self.load()
        self.assertEqual(job["status"], "success", job)
        self.client.set_gpt_weights.assert_awaited_once_with("/server/gpt.ckpt")
        self.client.set_sovits_weights.assert_awaited_once_with(str(self.lora))

    async def test_no_lora_sentinel_loads_full_base(self):
        self.plugin.config.models.lora_weights = NO_LORA
        job = await self.load()
        self.assertEqual(job["status"], "success", job)
        self.client.set_sovits_weights.assert_awaited_once_with("/server/s2Gv5turbo.pth")

    async def test_wrong_version_rejected_before_any_remote_switch(self):
        self.lora.write_bytes(b"07\x03\x04" + b"x" * 100)
        job = await self.load()
        self.assertEqual(job["status"], "error")
        self.assertIn("不匹配", job["message"])
        self.client.set_gpt_weights.assert_not_awaited()
        self.client.set_sovits_weights.assert_not_awaited()

    async def test_gpt_checkpoint_is_not_lora(self):
        self.plugin.config.models.lora_weights = "/server/黍.ckpt"
        job = await self.load()
        self.assertEqual(job["status"], "error")
        self.assertIn("不是LoRA", job["message"])
        self.client.set_gpt_weights.assert_not_awaited()

    async def test_custom_base_cannot_be_claimed_as_lora_base(self):
        self.plugin.config.models.base_model = "custom"
        job = await self.load()
        self.assertEqual(job["status"], "error")
        self.assertIn("自定义底模", job["message"])
        self.client.set_gpt_weights.assert_not_awaited()

    async def test_preset_can_fill_unselected_paths(self):
        self.plugin.config.models.gpt_weights = ""
        self.plugin.config.models.sovits_weights = ""
        self.plugin.config.models.lora_weights = NO_LORA
        job = await self.load()
        self.assertEqual(job["status"], "success", job)
        self.client.set_gpt_weights.assert_awaited_once_with("/server/s1v3.ckpt")
        self.client.set_sovits_weights.assert_awaited_once_with("/server/s2Gv5turbo.pth")

    async def test_lora_saved_independently_and_signature_invalidated(self):
        old_signature = self.plugin._model_signature(self.plugin.config)
        self.plugin._loaded_signature = old_signature
        result = await self.plugin.setup_save_models(root=str(self.root), base_model="v5turbo",
            gpt_weights="/server/gpt.ckpt", sovits_weights="/server/s2Gv5turbo.pth", lora_weights="")
        self.assertTrue(result["success"], result)
        self.assertEqual(self.plugin.config.models.lora_weights, NO_LORA)
        self.assertIsNone(self.plugin._loaded_signature)
        self.assertNotEqual(old_signature, self.plugin._model_signature(self.plugin.config))


if __name__ == "__main__":
    unittest.main()
