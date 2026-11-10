"""Progress regressions using real TTSConfig, never live settings or assets."""
from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config_models import TTSConfig, UNSELECTED
from wizard_progress import WizardProgress


class WizardProgressTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="wizard-progress-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.settings = self.root / "config.toml"
        self.progress = WizardProgress(self.settings)
        self.cfg = TTSConfig()
        # 已保存过本机模型目录的配置（目录本身不会被读取）。
        self.cfg.models.root = "/never-read/models"

    def original(self, preset=False):
        self.cfg.models.root = "/never-read/models"
        self.cfg.models.base_model = "v2" if preset else "custom"
        if not preset:
            self.cfg.models.gpt_weights = "/never-read/gpt.ckpt"
            self.cfg.models.sovits_weights = "/never-read/sovits.pth"
        self.cfg.voice.ref_audio_path = "/never-read/reference.wav"
        self.cfg.voice.prompt_text = "不能出现在进度记录里的参考文本"

    def role(self):
        self.cfg.models.base_model = "v2"
        self.cfg.role.mode = "role"
        self.cfg.role.role_id = "alice.json"
        self.cfg.role.emotion = "neutral"
        self.cfg.role.emotions = ["neutral", "happy"]

    def all_markers(self):
        for step in range(1, 5):
            self.progress.mark(step, self.cfg)

    def test_01_fresh_step1_has_no_filesystem_side_effects(self):
        result = self.progress.compute(self.cfg)
        self.assertEqual(result["next_step"], 1)
        self.assertEqual(result["completed_steps"], [])
        self.assertFalse(result["needs_revalidate"])
        self.assertEqual(list(self.root.iterdir()), [])
        missing = WizardProgress(self.root / "absent" / "config.toml")
        self.assertEqual(missing.compute(self.cfg)["next_step"], 1)
        self.assertFalse((self.root / "absent").exists())

    def test_02_api_marker_is_canonical_and_ignores_timeout(self):
        self.cfg.api.base_url = "HTTP://LOCALHOST:80/tts/"
        self.progress.mark(1, self.cfg)
        self.cfg.api.base_url = "http://localhost/tts"
        self.cfg.api.timeout = 20
        result = self.progress.compute(self.cfg)
        self.assertEqual(result["next_step"], 2)
        self.assertIn(1, result["completed_steps"])
        self.assertTrue(result["needs_revalidate"])
        self.assertFalse(self.progress.compute(self.cfg, {"status": "success"})["needs_revalidate"])
        self.cfg.api.base_url = "http://localhost:9881"
        self.assertEqual(self.progress.compute(self.cfg)["next_step"], 1)

    def test_03_nondefault_api_is_not_a_test(self):
        self.original(preset=True)
        self.cfg.api.base_url = "https://voice.example/api"
        self.assertEqual(self.progress.compute(self.cfg)["next_step"], 1)
        self.assertEqual(self.progress.compute(self.cfg, {"status": "success"})["next_step"], 4)

    def test_04_steps_1_through_5_and_defaults_require_confirmation(self):
        self.assertEqual(self.progress.compute(self.cfg)["next_step"], 1)
        self.progress.mark(1, self.cfg)
        self.assertEqual(self.progress.compute(self.cfg)["next_step"], 2)
        self.cfg.models.base_model = "v2"
        self.progress.mark(2, self.cfg)
        self.assertEqual(self.progress.compute(self.cfg)["next_step"], 3)
        self.original(preset=True)
        self.progress.mark(2, self.cfg)
        self.progress.mark(3, self.cfg)
        self.assertEqual(self.progress.compute(self.cfg)["next_step"], 4)
        self.progress.mark(4, self.cfg)
        self.assertEqual(self.progress.compute(self.cfg)["next_step"], 5)
        self.assertEqual(self.progress.compute(self.cfg)["completed_steps"], [1, 2, 3, 4])

    def test_05_saved_base_is_inferred_without_marker_but_api_needs_a_test(self):
        self.cfg.models.base_model = "v5turbo"
        self.assertIn(2, self.progress.compute(self.cfg)["completed_steps"])
        self.assertEqual(self.progress.compute(self.cfg)["next_step"], 1)
        self.progress.mark(1, self.cfg)
        self.assertEqual(self.progress.compute(self.cfg)["next_step"], 3)
        self.cfg.models.root = ""
        self.assertEqual(self.progress.compute(self.cfg)["next_step"], 2)
        # 已有配置文件但从未测试过接口：仍从第一步开始，不当作已连接。
        saved_root = self.root / "saved"
        saved_root.mkdir()
        saved = WizardProgress(saved_root / "config.toml")
        self.original(preset=True)
        saved._settings_path.write_text("# saved configuration", encoding="utf-8")
        state = saved.compute(self.cfg)
        self.assertEqual(state["next_step"], 1)
        self.assertNotIn(1, state["completed_steps"])
        self.assertFalse(state["needs_revalidate"])
        self.assertEqual(saved.compute(self.cfg, {"status": "success"})["next_step"], 4)

    def test_06_custom_weight_sentinels_are_not_configured(self):
        self.progress.mark(1, self.cfg)
        for invalid in ("", " ", UNSELECTED, "（不使用 LoRA）", "None"):
            self.cfg.models.gpt_weights = invalid
            self.cfg.models.sovits_weights = "/models/valid.pth"
            self.progress.mark(2, self.cfg)
            self.progress.mark(3, self.cfg)
            result = self.progress.compute(self.cfg)
            self.assertNotIn(2, result["completed_steps"])
            self.assertNotIn(3, result["completed_steps"])
            self.assertEqual(result["next_step"], 2)

    def test_07_saved_base_and_original_fingerprints_invalidate(self):
        self.original()
        self.all_markers()
        for section, name, value, missing in (
                ("models", "root", "/new-root", 2),
                ("models", "base_model", "v3", 2),
                ("models", "gpt_weights", "/new.ckpt", 3),
                ("models", "sovits_weights", "/new.pth", 3),
                ("models", "lora_weights", "/new-lora.pth", 3),
                ("voice", "ref_audio_path", "/new.wav", 3),
                ("voice", "prompt_text", "新文本", 3),
                ("voice", "prompt_lang", "ja", 3),
                ("voice", "text_lang", "en", 3)):
            with self.subTest(name=name):
                changed = self.cfg.model_copy(deep=True)
                setattr(getattr(changed, section), name, value)
                result = self.progress.compute(changed)
                self.assertNotIn(missing, result["completed_steps"])
                self.assertEqual(result["next_step"], missing)

    def test_08_role_success_requires_every_selected_profile(self):
        self.role()
        self.progress.mark(1, self.cfg)
        self.assertEqual(self.progress.compute(self.cfg)["next_step"], 3)
        partial = {"status": "success", "profiles": {"neutral": {}}}
        self.assertEqual(self.progress.compute(self.cfg, role_status=partial)["next_step"], 3)
        complete = {"status": "success", "profiles": {"neutral": {}, "happy": {}}}
        self.assertEqual(self.progress.compute(self.cfg, role_status=complete)["next_step"], 4)
        self.cfg.role.role_id = "（请选择角色）"
        self.assertEqual(self.progress.compute(self.cfg, role_status=complete)["next_step"], 3)

    def test_09_role_restart_keeps_prepared_and_revalidates_before_preview(self):
        self.role()
        self.all_markers()
        restarted = WizardProgress(self.settings)
        result = restarted.compute(self.cfg, {"status": "idle"}, {"status": "idle"})
        self.assertEqual(result["next_step"], 5)
        self.assertIn(3, result["completed_steps"])
        self.assertTrue(result["needs_revalidate"])
        self.assertIn("试听前请重新读取角色", result["message"])
        live = {"status": "success", "profiles": {"neutral": {}, "happy": {}}}
        self.assertFalse(restarted.compute(self.cfg, {"status": "success"}, live)["needs_revalidate"])

    def test_10_role_emotions_default_and_mode_changes_invalidate(self):
        self.role()
        self.all_markers()
        reordered = self.cfg.model_copy(deep=True)
        reordered.role.emotions.reverse()
        self.assertIn(3, self.progress.compute(reordered)["completed_steps"])
        for name, value in (("emotion", "happy"), ("emotions", ["neutral"]),
                            ("role_id", "bob.json"), ("mode", "original")):
            changed = self.cfg.model_copy(deep=True)
            setattr(changed.role, name, value)
            self.assertNotIn(3, self.progress.compute(changed)["completed_steps"])
        changed = self.cfg.model_copy(deep=True)
        changed.models.base_model = "v3"
        self.assertNotIn(3, self.progress.compute(changed)["completed_steps"])
        object.__setattr__(self.cfg.role, "emotion", "absent")
        self.progress.mark(3, self.cfg)
        self.assertNotIn(3, self.progress.compute(self.cfg)["completed_steps"])

    def test_11_original_ref_languages_and_v5_required_prompt(self):
        self.original(preset=True)
        self.progress.mark(1, self.cfg)
        self.assertEqual(self.progress.compute(self.cfg)["next_step"], 4)
        for field in ("ref_audio_path", "prompt_lang", "text_lang"):
            changed = self.cfg.model_copy(deep=True)
            object.__setattr__(changed.voice, field, "")
            self.progress.mark(3, changed)
            self.assertNotIn(3, self.progress.compute(changed)["completed_steps"])
        self.cfg.voice.prompt_text = ""
        self.progress.mark(3, self.cfg)
        self.assertIn(3, self.progress.compute(self.cfg)["completed_steps"])
        for version in ("v3", "v4", "v5dev", "v5turbo"):
            self.cfg.models.base_model = version
            self.progress.mark(3, self.cfg)
            self.assertNotIn(3, self.progress.compute(self.cfg)["completed_steps"])
            self.assertEqual(self.progress.compute(self.cfg)["next_step"], 3)

    def test_12_parameters_never_inferred_and_every_parameter_invalidates(self):
        self.original()
        self.cfg.voice.speed_factor = 1.3
        self.progress.mark(1, self.cfg)
        self.assertEqual(self.progress.compute(self.cfg)["next_step"], 4)
        self.progress.mark(4, self.cfg)
        for section, name, value in (
                ("voice", "speed_factor", 1.4), ("voice", "top_k", 9),
                ("voice", "top_p", .8), ("voice", "temperature", .8),
                ("voice", "seed", 7), ("voice", "text_split_method", "cut0"),
                ("voice", "parallel_infer", True), ("behavior", "max_text_length", 100),
                ("behavior", "cooldown_seconds", 12), ("behavior", "send_text", True)):
            with self.subTest(name=name):
                changed = self.cfg.model_copy(deep=True)
                setattr(getattr(changed, section), name, value)
                self.assertEqual(self.progress.compute(changed)["next_step"], 4)

    def test_13_job_priority_keeps_real_completed_steps(self):
        for kind, step in (("prepare_role", 3), ("load", 4), ("synth", 4)):
            result = self.progress.compute(self.cfg, job={"kind": kind, "status": "running"})
            self.assertEqual(result["next_step"], step)
            self.assertEqual(result["completed_steps"], [])
        self.original()
        self.all_markers()
        result = self.progress.compute(self.cfg, job={"kind": "prepare_role", "status": "running"})
        self.assertEqual(result["next_step"], 3)
        self.assertEqual(result["completed_steps"], [1, 2, 3, 4])
        self.assertEqual(self.progress.compute(self.cfg, job={"kind": "prepare_role", "status": "success"})["next_step"], 5)

    def test_14_metadata_corrupt_oversized_and_atomic_failure(self):
        self.progress.mark(1, self.cfg)
        for content in (b"{broken", b"x" * (64 * 1024 + 1), b"[]", b"\xff",
                        b'{"version":1,"steps":{"1":{"fingerprint":"bad","timestamp":1}}}'):
            self.progress.path.write_bytes(content)
            result = self.progress.compute(self.cfg)
            self.assertEqual(result["next_step"], 1)
            self.assertIn("warning", result)
        self.progress.mark(1, self.cfg)
        before = self.progress.path.read_bytes()
        with patch("wizard_progress.os.replace", side_effect=OSError("replace failed")):
            with self.assertRaises(OSError):
                self.progress.mark(4, self.cfg)
        self.assertEqual(self.progress.path.read_bytes(), before)
        self.assertEqual(list(self.progress.path.parent.iterdir()), [self.progress.path])

    def test_16_preview_is_optional_and_auto_voice_is_step_five(self):
        self.original()
        self.all_markers()
        result = self.progress.compute(self.cfg)
        self.assertEqual(result["next_step"], 5)
        self.assertFalse(result["preview_done"])
        self.progress.mark(5, self.cfg)
        result = self.progress.compute(self.cfg)
        self.assertEqual((result["next_step"], result["completed_steps"], result["preview_done"]), (5, [1, 2, 3, 4], True))
        self.progress.mark(6, self.cfg)
        result = WizardProgress(self.settings).compute(self.cfg)
        self.assertEqual(result["next_step"], 5)
        self.assertEqual(result["completed_steps"], [1, 2, 3, 4, 5])
        self.assertEqual(result["saved_steps"], [1, 2, 3, 4, 5])
        self.assertIn("五步配置已完成", result["message"])
        self.cfg.voice.speed_factor = 1.2
        self.assertEqual(self.progress.compute(self.cfg)["next_step"], 4)

    def test_15_only_metadata_written_no_network_scan_or_sensitive_values(self):
        self.original()
        self.settings.write_bytes(b"unmodified-settings")
        asset = self.root / "asset.wav"
        asset.write_bytes(b"unmodified-asset")
        self.cfg.voice.ref_audio_path = str(asset)
        before = self.cfg.model_dump(mode="json")
        self.all_markers()
        document = json.loads(self.progress.path.read_text(encoding="utf-8"))
        serialized = json.dumps(document, ensure_ascii=False)
        for secret in (self.cfg.api.base_url, self.cfg.models.root, str(asset), self.cfg.voice.prompt_text):
            self.assertNotIn(secret, serialized)
        self.assertLess(self.progress.path.stat().st_size, 64 * 1024)
        for marker in document["steps"].values():
            self.assertEqual(set(marker), {"fingerprint", "timestamp"})
        real_open = os.open
        def metadata_only(path, *args, **kwargs):
            self.assertEqual(Path(path), self.progress.path)
            return real_open(path, *args, **kwargs)
        with patch("wizard_progress.os.open", side_effect=metadata_only), \
                patch("socket.socket", side_effect=AssertionError("network forbidden")), \
                patch.object(Path, "iterdir", side_effect=AssertionError("scan forbidden")), \
                patch.object(Path, "glob", side_effect=AssertionError("scan forbidden")):
            self.assertEqual(WizardProgress(self.settings).compute(self.cfg)["next_step"], 5)
        self.assertEqual(self.cfg.model_dump(mode="json"), before)
        self.assertEqual(self.settings.read_bytes(), b"unmodified-settings")
        self.assertEqual(asset.read_bytes(), b"unmodified-asset")
        fake_link = SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0x400)
        original_lstat = Path.lstat
        def lstat(path):
            return fake_link if path == self.root / "data" else original_lstat(path)
        with patch.object(Path, "lstat", lstat):
            self.assertIn("warning", self.progress.compute(self.cfg))
            with self.assertRaises(OSError):
                self.progress.mark(1, self.cfg)


if __name__ == "__main__":
    unittest.main()
