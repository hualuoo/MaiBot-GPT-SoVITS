"""参考音频固定放在 GPT-SoVITS 安装目录下的 maibot_refs：只列文件，不读内容，不创建目录。"""
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import model_catalog
from model_catalog import REFS_DIRNAME, ROLES_DIRNAME, scan_reference_audios


class ReferenceAudioScanTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="maibot-refs-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve() / "GPT-SoVITS"
        self.root.mkdir()
        self.refs = self.root / REFS_DIRNAME

    def test_folder_names(self):
        self.assertEqual(REFS_DIRNAME, "maibot_refs")
        self.assertEqual(ROLES_DIRNAME, "maibot_roles")

    def test_empty_root_and_missing_folder_explain_without_creating(self):
        result = scan_reference_audios("")
        self.assertEqual(result["reference_audios"], [])
        self.assertIn("本机模型目录", result["reference_note"])
        result = scan_reference_audios(self.root.as_posix())
        self.assertEqual(result["reference_audios"], [])
        self.assertEqual(result["reference_dir"], self.refs.as_posix())
        self.assertIn("不存在", result["reference_note"])
        self.assertIn("新建 maibot_refs", result["reference_note"])
        self.assertFalse(self.refs.exists())
        with self.assertRaises(ValueError):
            scan_reference_audios("relative/path")
        with self.assertRaises(ValueError):
            scan_reference_audios(str(self.root / "absent"))

    def test_lists_audio_two_levels_deep_sorted_and_skips_other_files(self):
        (self.refs / "开心" / "更深" / "太深").mkdir(parents=True)
        (self.refs / ".隐藏").mkdir()
        for relative in ("平静.wav", "b.MP3", "开心/开心.flac", "开心/更深/深.ogg", "开心/更深/太深/太深.wav",
                         "说明.txt", "模型.pth", ".隐藏/藏.wav", ".藏.wav"):
            path = self.refs / relative
            path.write_bytes(b"not-real-audio")
        result = scan_reference_audios(self.root.as_posix())
        listed = [(Path(item["path"]).relative_to(self.refs).as_posix(), item["label"]) for item in result["reference_audios"]]
        # 每层按名称排序、深度优先：同一文件夹中的文件与子文件夹交错排列。
        self.assertEqual(listed, [("b.MP3", "maibot_refs"), ("平静.wav", "maibot_refs"),
                                  ("开心/开心.flac", "maibot_refs/开心"), ("开心/更深/深.ogg", "maibot_refs/开心/更深")])
        self.assertTrue(all(Path(item["path"]).is_absolute() for item in result["reference_audios"]))
        self.assertIn("找到 4 个参考音频", result["reference_note"])

    def test_empty_folder_and_limit(self):
        self.refs.mkdir()
        result = scan_reference_audios(self.root.as_posix())
        self.assertEqual(result["reference_audios"], [])
        self.assertIn("还没有参考音频", result["reference_note"])
        for index in range(model_catalog._MAX_REFERENCE_AUDIOS + 5):
            (self.refs / f"{index:04d}.wav").write_bytes(b"x")
        result = scan_reference_audios(self.root.as_posix())
        self.assertEqual(len(result["reference_audios"]), model_catalog._MAX_REFERENCE_AUDIOS)
        self.assertIn("只列出前", result["reference_note"])

    def test_links_are_not_followed(self):
        outside = self.root.parent / "外部"
        outside.mkdir()
        (outside / "外部.wav").write_bytes(b"x")
        self.refs.mkdir()
        try:
            os.symlink(outside, self.refs / "链接", target_is_directory=True)
        except OSError:
            self.skipTest("host does not permit symlink creation")
        self.assertEqual(scan_reference_audios(self.root.as_posix())["reference_audios"], [])


if __name__ == "__main__":
    unittest.main()
