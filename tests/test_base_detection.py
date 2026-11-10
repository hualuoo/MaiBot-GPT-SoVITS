"""第二步：根据所选 SoVITS 权重自动识别底模版本（不访问真实服务、不扫描真实目录）。"""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plugin import create_plugin
from config_models import NO_LORA
import setup_service

CATALOG = {
    "base_models": [
        {"id": "v2", "gpt_path": "C:/gsv/v2.ckpt", "sovits_path": "C:/gsv/v2.pth"},
        {"id": "v2Pro", "gpt_path": "C:/gsv/s1v3.ckpt", "sovits_path": "C:/gsv/v2Pro.pth"},
        {"id": "v5turbo", "gpt_path": "C:/gsv/s1v3.ckpt", "sovits_path": "C:/gsv/v5.pth"},
    ],
    "gpt_weights": [],
    "sovits_weights": [
        {"path": "C:/gsv/v2.pth", "version": "v2", "mode": "full"},
        {"path": "C:/gsv/SoVITS_weights_v2/fine.pth", "version": "v2", "mode": "full"},
        {"path": "C:/gsv/SoVITS_weights/old.pth", "version": "unknown", "mode": "unknown"},
        {"path": "C:/gsv/SoVITS_weights_v4/x.pth", "version": "v4", "mode": "full"},
    ],
    "lora_weights": [{"path": "C:/gsv/SoVITS_weights_v5turbo/lora.pth", "version": "v5turbo"}],
}


class BaseDetectionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.p = create_plugin()
        self.p._settings_path = self.root / "config.toml"
        cfg = self.p.build_default_config()
        cfg["models"].update(root="C:/old", base_model="custom",
                             gpt_weights="C:/my/old.ckpt", sovits_weights="D:/remote/saved.pth",
                             lora_weights="C:/my/old-lora.pth")
        self.p.set_plugin_config(cfg)
        await self.p.on_load()
        patcher = patch.object(setup_service, "scan_models", return_value=CATALOG)
        self.scan = patcher.start()
        self.addCleanup(patcher.stop)

    async def asyncTearDown(self):
        await self.p.on_unload()
        self.assertFalse(self.p._busy)

    async def save(self, sovits, root="C:/gsv"):
        return await self.p.setup_save_base(root, sovits)

    def models(self):
        return self.p.config.models

    async def test_standard_base_pair_is_detected_and_only_root_is_scanned(self):
        result = await self.save("C:/gsv/v5.pth")
        self.assertTrue(result["success"], result)
        self.assertEqual(result["base_model"], "v5turbo")
        self.assertIn("自动识别", result["message"])
        self.scan.assert_called_once_with("C:/gsv")
        m = self.models()
        self.assertEqual((m.root, m.base_model), ("C:/gsv", "v5turbo"))
        self.assertEqual((m.sovits_weights, m.gpt_weights, m.lora_weights), ("C:/gsv/v5.pth", "C:/gsv/s1v3.ckpt", NO_LORA))

    async def test_fine_tuned_full_weight_uses_header_version_and_matching_gpt(self):
        result = await self.save("C:/gsv/SoVITS_weights_v2/fine.pth")
        self.assertTrue(result["success"], result)
        m = self.models()
        self.assertEqual((m.base_model, m.sovits_weights, m.gpt_weights), ("v2", "C:/gsv/SoVITS_weights_v2/fine.pth", "C:/gsv/v2.ckpt"))
        self.assertEqual(m.lora_weights, NO_LORA)

    async def test_unknown_or_uninstalled_versions_fall_back_to_custom_without_touching_gpt(self):
        for path in ("C:/gsv/SoVITS_weights/old.pth", "C:/gsv/SoVITS_weights_v4/x.pth"):
            with self.subTest(path=path):
                result = await self.save(path)
                self.assertTrue(result["success"], result)
                self.assertEqual(result["base_model"], "custom")
                self.assertIn("自定义", result["message"])
                self.assertEqual((self.models().base_model, self.models().sovits_weights), ("custom", path))
                self.assertEqual(self.models().gpt_weights, "C:/my/old.ckpt")

    async def test_same_version_keeps_hand_picked_gpt_and_lora_but_version_change_resets(self):
        await self.p._save_setup({"models": {"base_model": "v5turbo", "sovits_weights": "C:/gsv/v5.pth",
                                             "gpt_weights": "C:/my/shu.ckpt",
                                             "lora_weights": "C:/gsv/SoVITS_weights_v5turbo/lora.pth"}})
        self.assertTrue((await self.save("C:/gsv/v5.pth"))["success"])
        self.assertEqual(self.models().gpt_weights, "C:/my/shu.ckpt")
        self.assertEqual(self.models().lora_weights, "C:/gsv/SoVITS_weights_v5turbo/lora.pth")
        self.assertTrue((await self.save("C:/gsv/v2.pth"))["success"])
        self.assertEqual((self.models().base_model, self.models().gpt_weights, self.models().lora_weights),
                         ("v2", "C:/gsv/v2.ckpt", NO_LORA))

    async def test_preset_gpt_of_another_version_follows_the_new_base(self):
        await self.p._save_setup({"models": {"base_model": "v2Pro", "sovits_weights": "C:/gsv/v2Pro.pth",
                                             "gpt_weights": "C:/gsv/s1v3.ckpt"}})
        self.assertTrue((await self.save("C:/gsv/v2.pth"))["success"])
        self.assertEqual(self.models().gpt_weights, "C:/gsv/v2.ckpt")

    async def test_rejections_do_not_change_saved_config(self):
        before = self.p.config.model_dump()
        for root, path, text in (("C:/gsv", "C:/gsv/SoVITS_weights_v5turbo/lora.pth", "LoRA"),
                                 ("", "C:/gsv/v5.pth", "本机模型目录"),
                                 ("C:/gsv", "", "SoVITS"),
                                 ("C:/gsv", "C:/gsv/model.ckpt", ".pth"),
                                 ("C:/gsv", "C:/elsewhere/not-scanned.pth", "扫描结果")):
            with self.subTest(path=path):
                result = await self.save(path, root)
                self.assertFalse(result["success"], result)
                self.assertIn(text, result["message"])
                self.assertEqual(self.p.config.model_dump(), before)

    async def test_local_file_outside_scan_is_identified_from_header(self):
        full = self.root / "v2pro-full.pth"
        full.write_bytes(b"05" + b"\0" * 64)
        lora = self.root / "v5-lora.pth"
        lora.write_bytes(b"08" + b"\0" * 64)
        result = await self.save(full.as_posix())
        self.assertTrue(result["success"], result)
        self.assertEqual(result["base_model"], "v2Pro")
        denied = await self.save(lora.as_posix())
        self.assertFalse(denied["success"])
        self.assertIn("LoRA", denied["message"])

    async def test_saved_remote_path_keeps_saved_version_when_preset_exists(self):
        await self.p._save_setup({"models": {"base_model": "v2"}})
        result = await self.save("D:/remote/saved.pth")
        self.assertTrue(result["success"], result)
        self.assertEqual(result["base_model"], "v2")
        await self.p._save_setup({"models": {"base_model": "v4"}})
        self.assertEqual((await self.save("D:/remote/saved.pth"))["base_model"], "custom")


if __name__ == "__main__":
    unittest.main()
