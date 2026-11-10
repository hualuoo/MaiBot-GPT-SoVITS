"""运行：python -B -m unittest discover -s tests -p test_model_catalog.py -v。"""
from contextlib import contextmanager
import hashlib
import json
import sys
from types import SimpleNamespace
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import model_catalog
from model_catalog import scan_models


class ModelCatalogTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    def create(self, relative, root=None, content=None):
        path = (root or self.root) / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content if content is not None else
                         b"PKfixture" if path.suffix.casefold() == ".pth" else b"fixture")
        return path.resolve().as_posix()

    def paths(self, result, kind):
        return [item["path"] for item in result[kind]]

    @contextmanager
    def bounded_pth_reads(self):
        original_open = Path.open
        reads = {}

        @contextmanager
        def guarded(path, mode="r", *args, **kwargs):
            self.assertEqual(path.suffix.casefold(), ".pth", "不得打开 GPT 权重或源码")
            self.assertEqual(mode, "rb")
            self.assertEqual(kwargs.get("buffering"), 0, "禁止隐式预读超出头部")
            with original_open(path, mode, *args, **kwargs) as stream:
                def read(size=-1):
                    self.assertGreater(size, 0)
                    self.assertLessEqual(size, 8192)
                    reads[path] = reads.get(path, 0) + size
                    self.assertLessEqual(reads[path], 8192, "同一文件总读取预算")
                    return stream.read(size)
                yield SimpleNamespace(read=read)

        with patch.object(Path, "open", guarded):
            yield reads

    def test_empty_root_returns_empty_chinese_note_without_access(self):
        with patch("model_catalog.os.scandir", side_effect=AssertionError("不得扫描")):
            for root in ["", " \t\n"]:
                with self.subTest(root=root):
                    result = scan_models(root)
                    self.assertEqual(result["base_models"], [])
                    self.assertEqual(result["gpt_weights"], [])
                    self.assertEqual(result["sovits_weights"], [])
                    self.assertEqual(result["lora_weights"], [])
                    self.assertIn("未配置", result["note"])
                    self.assertIn("未执行扫描", result["note"])

    def test_invalid_roots_raise_chinese_value_error(self):
        file_path = self.root / "not_directory"
        file_path.write_bytes(b"")
        for root in [None, 42, self.root, [], "relative/path", "bad\x00path",
                     str(self.root / "missing"), str(file_path), self.root.anchor]:
            with self.subTest(root=root):
                with self.assertRaisesRegex(ValueError, "[\u4e00-\u9fff]"):
                    scan_models(root)

    def test_empty_directory_has_exact_contract(self):
        result = scan_models(str(self.root))
        self.assertEqual(set(result), {"base_models", "gpt_weights", "sovits_weights", "lora_weights", "note"})
        self.assertEqual(result["base_models"], [])
        self.assertEqual(result["gpt_weights"], [])
        self.assertEqual(result["sovits_weights"], [])
        self.assertEqual(result["lora_weights"], [])
        self.assertIn("未发现", result["note"])
        self.assertIn("不代表服务端当前加载状态", result["note"])

    def test_all_default_pairs_use_real_matching_paths(self):
        # 独立写出本机 config.py / TTS.py 的默认配对，不能从被测常量反推预期。
        pairs = {
            "v1": ("s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt", "s2G488k.pth"),
            "v2": ("gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
                   "gsv-v2final-pretrained/s2G2333k.pth"),
            "v3": ("s1v3.ckpt", "s2Gv3.pth"),
            "v4": ("s1v3.ckpt", "gsv-v4-pretrained/s2Gv4.pth"),
            "v2Pro": ("s1v3.ckpt", "v2Pro/s2Gv2Pro.pth"),
            "v2ProPlus": ("s1v3.ckpt", "v2Pro/s2Gv2ProPlus.pth"),
            "v5dev": ("s1v3.ckpt", "gsv-v5-pretrained/s2Gv5dev.pth"),
            "v5turbo": ("s1v3.ckpt", "gsv-v5-pretrained/s2Gv5turbo.pth"),
        }
        for gpt, sovits in pairs.values():
            self.create("GPT_SoVITS/pretrained_models/" + gpt)
            self.create("GPT_SoVITS/pretrained_models/" + sovits)
        result = scan_models(str(self.root))
        self.assertEqual([item["id"] for item in result["base_models"]], list(pairs))
        for item in result["base_models"]:
            gpt, sovits = pairs[item["id"]]
            self.assertEqual(set(item), {"id", "label", "gpt_path", "sovits_path"})
            self.assertEqual(item["gpt_path"],
                             (self.root / "GPT_SoVITS/pretrained_models" / gpt).as_posix())
            self.assertEqual(item["sovits_path"],
                             (self.root / "GPT_SoVITS/pretrained_models" / sovits).as_posix())
            self.assertIn("底模", item["label"])
        self.assertEqual(len(result["gpt_weights"]), 3)
        self.assertEqual(len(result["sovits_weights"]), 8)

    def test_never_fabricates_missing_or_unrecognized_pairs(self):
        self.create("GPT_SoVITS/pretrained_models/s1v3.ckpt")
        self.create("GPT_SoVITS/pretrained_models/unrecognized/s2Gv4.pth")
        self.create("SoVITS_weights_v3/s2Gv3.pth")
        self.create("GPT_weights_v2/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt")
        result = scan_models(str(self.root))
        self.assertEqual(result["base_models"], [])
        self.assertEqual(len(result["gpt_weights"]), 2)
        self.assertEqual(len(result["sovits_weights"]), 2)

    def test_only_weight_directories_and_extensions_are_scanned(self):
        gpt = self.create("GPT_weights_v2Pro/中文 角色/epoch=20.CKPT")
        sovits = self.create("SoVITS_weights_other/中文 角色/epoch=20.PTH")
        base_gpt = self.create("GPT_SoVITS/pretrained_models/new-version/unknown.ckpt")
        base_sovits = self.create("GPT_SoVITS/pretrained_models/new-version/unknown.pth")
        for relative in ["GPT_weights_v2Pro/no.pth", "SoVITS_weights_other/no.ckpt",
                         "GPT_weights_v2Pro/no.ckpt.bak", "models/no.ckpt",
                         "venv/no.pth", "dataset/GPT_weights/no.ckpt",
                         "GPT_SoVITS/data/no.pth", "root.ckpt"]:
            self.create(relative)
        result = scan_models(str(self.root))
        self.assertEqual(set(self.paths(result, "gpt_weights")), {gpt, base_gpt})
        self.assertEqual(set(self.paths(result, "sovits_weights")), {sovits, base_sovits})
        for kind in ["gpt_weights", "sovits_weights"]:
            for item in result[kind]:
                self.assertEqual(set(item), {"label", "path"} if kind == "gpt_weights"
                                 else {"label", "path", "version", "mode"})
                self.assertIn("中文 角色" if "weights" in item["path"] else "底模目录", item["label"])
                self.assertTrue(Path(item["path"]).is_absolute())

    def test_scandir_does_not_enter_unrelated_or_large_auxiliary_dirs(self):
        self.create("venv/large/no.ckpt")
        self.create("logs/large/no.pth")
        self.create("GPT_SoVITS/large/no.pth")
        for name in ["chinese-hubert-base", "chinese-roberta-wwm-ext-large",
                     "fast_langdetect", "bigvgan", ".cache"]:
            self.create(f"GPT_SoVITS/pretrained_models/{name}/large/no.pth")
        expected = self.create("GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/yes.ckpt")
        original_scandir = os.scandir
        scanned = []

        def guarded(directory):
            path = Path(directory)
            scanned.append(path)
            self.assertNotIn("large", path.parts)
            self.assertNotIn(path.name.casefold(), {"venv", "logs", "chinese-hubert-base",
                                                  "chinese-roberta-wwm-ext-large",
                                                  "fast_langdetect", "bigvgan", ".cache"})
            return original_scandir(directory)

        with patch("model_catalog.os.scandir", side_effect=guarded):
            result = scan_models(str(self.root))
        self.assertEqual(self.paths(result, "gpt_weights"), [expected])
        self.assertEqual(len(scanned), 3)

    def test_each_weight_kind_is_limited_to_200(self):
        for number in range(203):
            self.create(f"GPT_weights_v3/sub/{number:03d}.ckpt")
            self.create(f"SoVITS_weights_v3/sub/{number:03d}.pth")
        result = scan_models(str(self.root))
        self.assertEqual(len(result["gpt_weights"]), 200)
        self.assertEqual(len(result["sovits_weights"]), 200)
        self.assertIn("200", result["note"])
        self.assertEqual(result, scan_models(str(self.root)))

    def test_path_budget_bounds_enumeration_before_collecting_all_entries(self):
        directory = self.root / "GPT_weights_v2"
        directory.mkdir()
        consumed = 0
        original_scandir = os.scandir

        class Entry:
            def __init__(self, index):
                self.path = str(directory / f"missing-{index}.txt")

        def entries():
            nonlocal consumed
            for index in range(10000):
                consumed += 1
                yield Entry(index)

        @contextmanager
        def bounded_scandir(path):
            if Path(path) == directory:
                yield entries()
            else:
                with original_scandir(path) as actual:
                    yield actual

        with patch("model_catalog.os.scandir", side_effect=bounded_scandir):
            result = scan_models(str(self.root))
        # 根目录和 GPT_weights_v2 自身也必须计入 5000 个访问路径。
        self.assertEqual(consumed, 4998)
        self.assertIn("5000", result["note"])
        self.assertIn("结果可能不完整", result["note"])

    def test_path_budget_includes_fixed_pretrained_lookup(self):
        (self.root / "GPT_SoVITS").mkdir()
        (self.root / "GPT_weights").mkdir()
        scanner = model_catalog._Scanner(self.root)
        with patch("model_catalog._MAX_VISITED_PATHS", 4):
            result = scanner.scan()
        self.assertLessEqual(len(scanner.visited), 4)
        self.assertIn("访问上限", result["note"])

    def make_symlink(self, destination, target, is_directory=False):
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            destination.symlink_to(target, target_is_directory=is_directory)
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f"当前系统无法创建符号链接：{exc}")

    def test_symlink_escape_and_cycles_are_never_followed(self):
        outside = tempfile.TemporaryDirectory()
        self.addCleanup(outside.cleanup)
        outside_root = Path(outside.name).resolve()
        self.create("secret.ckpt", root=outside_root)
        weights = self.root / "GPT_weights"
        weights.mkdir()
        self.make_symlink(weights / "outside", outside_root, True)
        self.make_symlink(weights / "loop", weights, True)
        self.make_symlink(weights / "linked.ckpt", outside_root / "secret.ckpt")
        self.make_symlink(self.root / "SoVITS_weights_link", outside_root, True)
        result = scan_models(str(self.root))
        self.assertEqual(result["gpt_weights"], [])
        self.assertEqual(result["sovits_weights"], [])
        self.assertIn("跳过", result["note"])

    def test_root_symlink_is_rejected(self):
        link = self.root / "link"
        self.make_symlink(link, self.root, True)
        with self.assertRaisesRegex(ValueError, "符号链接"):
            scan_models(str(link))

    def test_pretrained_directory_symlink_escape_is_rejected(self):
        outside = tempfile.TemporaryDirectory()
        self.addCleanup(outside.cleanup)
        outside_root = Path(outside.name).resolve()
        self.create("secret.pth", root=outside_root)
        link = self.root / "GPT_SoVITS/pretrained_models"
        self.make_symlink(link, outside_root, True)
        result = scan_models(str(self.root))
        self.assertEqual(result["sovits_weights"], [])
        self.assertEqual(result["base_models"], [])

    def test_overlong_path_is_omitted_without_truncation(self):
        scanner = model_catalog._Scanner(self.root)
        path = self.root / "GPT_weights" / ("角" * 1024 + ".ckpt")
        scanner._record(path, pretrained=False)
        self.assertEqual(scanner.gpt, {})
        self.assertTrue(scanner.skipped_long_paths)
        path = self.root / "GPT_weights" / ("角" * 400 + ".ckpt")
        scanner._record(path, pretrained=False)
        self.assertEqual(next(iter(scanner.gpt.values()))["path"], path.as_posix())

    def test_overlong_path_is_skipped_before_filesystem_stat(self):
        scanner = model_catalog._Scanner(self.root)
        path = self.root / "GPT_weights" / ("角" * 1024 + ".ckpt")
        with patch.object(Path, "lstat", side_effect=AssertionError("不应访问过长路径")):
            self.assertIsNone(scanner._inspect(path))
        self.assertTrue(scanner.skipped_long_paths)

    def test_long_path_note_is_visible(self):
        self.create("GPT_weights/long-name.ckpt")
        with patch("model_catalog._MAX_PATH_CHARS", 1):
            result = scan_models(str(self.root))
        self.assertEqual(result["gpt_weights"], [])
        self.assertIn("已跳过超过", result["note"])

    def test_serialized_size_is_bounded_for_long_chinese_catalogs(self):
        scanner = model_catalog._Scanner(self.root)
        for number in range(200):
            scanner._record(self.root / "GPT_weights" / (f"{number:03d}" + "角" * 850 + ".ckpt"),
                            pretrained=False)
            path = self.root / "SoVITS_weights" / (f"{number:03d}" + "声" * 850 + ".pth")
            with patch("model_catalog._sovits_metadata", return_value=("unknown", "unknown")):
                scanner._record(path, pretrained=False)
        expected = {item["path"] for item in [*scanner.gpt.values(), *scanner.sovits.values()]}
        pair = {"id": "v2", "label": "v2 底模", "gpt_path": "C:/GPT-SoVITS/gpt.ckpt",
                "sovits_path": "C:/GPT-SoVITS/sovits.pth"}
        result = {"base_models": [pair], "gpt_weights": list(scanner.gpt.values()),
                  "sovits_weights": list(scanner.sovits.values()), "lora_weights": [],
                  "note": "模型目录。"}
        model_catalog._Scanner._bound_result(result)
        self.assertLessEqual(len(json.dumps(result, ensure_ascii=False).encode("utf-8")),
                             450 * 1024)
        self.assertLessEqual(len(json.dumps(result, ensure_ascii=True).encode("utf-8")),
                             450 * 1024)
        self.assertEqual(result["base_models"], [pair])
        self.assertIn("大小上限", result["note"])
        self.assertTrue(result["gpt_weights"])
        self.assertTrue(result["sovits_weights"])
        self.assertLess(len(result["gpt_weights"]) + len(result["sovits_weights"]), 400)
        for item in result["gpt_weights"] + result["sovits_weights"]:
            self.assertIn(item["path"], expected)

    def test_scan_errors_are_chinese_value_error(self):
        with patch("model_catalog.os.scandir", side_effect=PermissionError("Access denied")):
            with self.assertRaisesRegex(ValueError, "扫描模型目录失败") as caught:
                scan_models(str(self.root))
        self.assertNotIn("Access denied", str(caught.exception))

    def test_scanner_reads_only_pth_header_without_deserialization(self):
        expected = self.create("GPT_weights/character.ckpt")
        sovits = self.create("SoVITS_weights/character.pth", content=b"PK" + b"x" * 20000)
        self.create("config.py")
        self.create("GPT_SoVITS/TTS_infer_pack/TTS.py")
        forbidden = unittest.mock.Mock(side_effect=AssertionError("不得反序列化"))
        with self.bounded_pth_reads() as reads, \
                patch("pickle.load", forbidden), patch("pickle.loads", forbidden), \
                patch.dict(sys.modules, {"torch": SimpleNamespace(load=forbidden)}), \
                patch("builtins.open", side_effect=AssertionError("只允许受限 Path.open")):
            result = scan_models(str(self.root))
        self.assertEqual(reads, {Path(sovits): 8192})
        self.assertEqual(self.paths(result, "gpt_weights"), [expected])
        forbidden.assert_not_called()


    def test_both_empty_and_omitted_extra_do_not_touch_unconfigured_paths(self):
        with patch("model_catalog.Path.lstat", side_effect=AssertionError("不得访问")), \
                patch("model_catalog.Path.resolve", side_effect=AssertionError("不得解析")), \
                patch("model_catalog.os.scandir", side_effect=AssertionError("不得扫描")):
            for root, extra in [("", ""), (" \t", "\n")]:
                self.assertIn("未执行扫描", scan_models(root, extra)["note"])
        expected = self.create("GPT_weights/only.ckpt")
        original = model_catalog._directory
        checked = []

        def guarded(value, label):
            checked.append(value)
            return original(value, label)

        with patch("model_catalog._directory", side_effect=guarded):
            result = scan_models(str(self.root))
        self.assertEqual(checked, [str(self.root), ""])
        self.assertEqual(self.paths(result, "gpt_weights"), [expected])

    def test_extra_only_lists_desktop_shu_with_exact_labels_and_contract(self):
        gpt = self.create("黍.ckpt")
        sovits = self.create("黍.PTH")
        result = scan_models("", extra_dir=str(self.root))
        self.assertEqual(set(result), {"base_models", "gpt_weights", "sovits_weights", "lora_weights", "note"})
        self.assertEqual(result["base_models"], [])
        self.assertEqual(result["gpt_weights"], [{"label": "补充目录 · 黍.ckpt", "path": gpt}])
        self.assertEqual(result["sovits_weights"], [{"label": "补充目录 · 黍.PTH", "path": sovits,
                                                    "version": "unknown", "mode": "unknown"}])
        self.assertIn("第一层", result["note"])
        self.assertIn("不递归", result["note"])

    def test_extra_scans_top_level_only_without_opening_gpt_or_code(self):
        expected = self.create("top.CKPT")
        for relative in ["sub/hidden.ckpt", "GPT_weights/nested.ckpt", "elsewhere/x.pth",
                         "config.py", "backup.ckpt.bak", "audio.wav"]:
            self.create(relative)
        (self.root / "directory.pth").mkdir()
        original_scandir = os.scandir
        scanned = []

        def guarded(directory):
            scanned.append(Path(directory))
            self.assertEqual(Path(directory), self.root)
            return original_scandir(directory)

        with patch("model_catalog.os.scandir", side_effect=guarded), \
                self.bounded_pth_reads():
            result = scan_models("", str(self.root))
        self.assertEqual(scanned, [self.root])
        self.assertEqual(self.paths(result, "gpt_weights"), [expected])
        self.assertEqual(result["sovits_weights"], [])

    def test_extra_invalid_directories_raise_chinese_value_error(self):
        file_path = self.create("not-a-directory.ckpt")
        for extra in [None, 42, self.root, [], "relative", "bad\x00path",
                      str(self.root / "missing"), file_path, self.root.anchor,
                      "//server/share/models"]:
            with self.subTest(extra=extra):
                with self.assertRaisesRegex(ValueError, "[\u4e00-\u9fff]"):
                    scan_models("", extra)
        with patch("model_catalog.os.scandir", side_effect=PermissionError("Access denied")):
            with self.assertRaisesRegex(ValueError, "扫描模型目录失败") as caught:
                scan_models("", str(self.root))
        self.assertNotIn("Access denied", str(caught.exception))

    def test_extra_symlinks_and_parent_link_paths_are_rejected(self):
        actual = self.root / "actual"
        actual.mkdir()
        self.create("黍.ckpt", root=actual)
        link = self.root / "link"
        self.make_symlink(link, actual, True)
        for root, extra in [("", str(link)), (str(link), "")]:
            with self.subTest(root=root, extra=extra):
                with self.assertRaisesRegex(ValueError, "符号链接|目录联接"):
                    scan_models(root, extra)
        (actual / "sub").mkdir()
        for root, extra in [("", str(link / "sub")), (str(link / "sub"), "")]:
            with self.assertRaisesRegex(ValueError, "符号链接|目录联接"):
                scan_models(root, extra)

    def test_extra_reparse_directory_is_rejected_and_reparse_file_skipped(self):
        from types import SimpleNamespace
        import stat
        expected = self.create("黍.ckpt")
        original_lstat = Path.lstat
        blocked = self.root

        def reparse(path, *args, **kwargs):
            info = original_lstat(path, *args, **kwargs)
            if path == blocked:
                return SimpleNamespace(st_mode=info.st_mode,
                                       st_file_attributes=getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
            return info

        with patch.object(Path, "lstat", reparse):
            with self.assertRaisesRegex(ValueError, "目录联接"):
                scan_models("", str(self.root))
        blocked = Path(expected)
        with patch.object(Path, "lstat", reparse):
            result = scan_models("", str(self.root))
        self.assertEqual(result["gpt_weights"], [])
        self.assertIn("跳过", result["note"])

    def test_extra_escape_is_skipped_before_stat(self):
        scanner = model_catalog._Scanner(None, self.root)
        with patch.object(Path, "lstat", side_effect=AssertionError("不得访问越界路径")):
            self.assertIsNone(scanner._inspect(self.root.parent / "escape.ckpt", boundary=self.root))
        self.assertTrue(scanner.skipped_links)

    def test_root_v5turbo_and_extra_merge_without_vocoder_as_acoustic_weight(self):
        gpt = self.create("GPT_SoVITS/pretrained_models/s1v3.ckpt")
        sovits = self.create("GPT_SoVITS/pretrained_models/gsv-v5-pretrained/s2Gv5turbo.pth")
        self.create("GPT_SoVITS/pretrained_models/gsv-v5-pretrained/vocoder.pth")
        extra = self.root / "desktop"
        shu = self.create("黍.ckpt", root=extra)
        result = scan_models(str(self.root), str(extra))
        self.assertEqual(result["base_models"], [{"id": "v5turbo", "label": "v5turbo 底模",
                                               "gpt_path": gpt, "sovits_path": sovits}])
        self.assertEqual(set(self.paths(result, "gpt_weights")), {gpt, shu})
        self.assertEqual(self.paths(result, "sovits_weights"), [sovits])
        self.assertIn({"label": "补充目录 · 黍.ckpt", "path": shu}, result["gpt_weights"])

    def test_overlap_is_deduplicated_and_directory_enumerated_once(self):
        expected = self.create("GPT_weights/黍.ckpt")
        extra = self.root / "GPT_weights"
        original_scandir = os.scandir
        scanned = []

        def tracked(directory):
            scanned.append(os.path.normcase(str(directory)))
            return original_scandir(directory)

        with patch("model_catalog.os.scandir", side_effect=tracked):
            result = scan_models(str(self.root), str(extra))
        self.assertEqual(result["gpt_weights"], [{"label": "补充目录 · 黍.ckpt", "path": expected}])
        self.assertEqual(scanned.count(os.path.normcase(str(extra))), 1)
        self.assertEqual(result, scan_models(str(self.root), str(extra)))

    @unittest.skipUnless(os.name == "nt", "Windows 路径大小写去重")
    def test_windows_case_variants_of_same_path_are_deduplicated(self):
        path = Path(self.create("GPT_weights/SHU.ckpt"))
        scanner = model_catalog._Scanner(self.root, path.parent)
        scanner._record(path, pretrained=False)
        scanner._record(Path(str(path).swapcase()), pretrained=False, extra=True)
        self.assertEqual(len(scanner.gpt), 1)
        result = scan_models(str(self.root), str(path.parent).swapcase())
        self.assertEqual(len(result["gpt_weights"]), 1)
        self.assertEqual(result["gpt_weights"][0]["path"].casefold(), path.as_posix().casefold())

    def test_mixed_sources_share_the_200_limit_per_kind(self):
        extra = self.root / "desktop"
        for number in range(105):
            self.create(f"GPT_weights/{number:03d}.ckpt")
            self.create(f"SoVITS_weights/{number:03d}.pth")
            self.create(f"extra-{number:03d}.ckpt", root=extra)
            self.create(f"extra-{number:03d}.pth", root=extra)
            self.create(f"SoVITS_weights/lora/{number:03d}.pth", content=b"03fixture")
            self.create(f"lora-{number:03d}.pth", root=extra, content=b"08fixture")
        result = scan_models(str(self.root), str(extra))
        for kind in ["gpt_weights", "sovits_weights", "lora_weights"]:
            self.assertEqual(len(result[kind]), 200)
            self.assertEqual(len(set(self.paths(result, kind))), 200)
            self.assertTrue(any(item["label"].startswith("补充目录 · ") for item in result[kind]))
        self.assertIn("200", result["note"])

    def test_mixed_sources_share_streaming_5000_path_budget(self):
        weights = self.root / "GPT_weights"
        weights.mkdir()
        outside = tempfile.TemporaryDirectory()
        self.addCleanup(outside.cleanup)
        extra = Path(outside.name).resolve()
        consumed = {weights: 0, extra: 0}
        original_scandir = os.scandir

        class Entry:
            def __init__(self, directory, number):
                self.path = str(directory / f"missing-{number}.txt")

        def entries(directory, count):
            for number in range(count):
                consumed[directory] += 1
                yield Entry(directory, number)

        @contextmanager
        def bounded(directory):
            directory = Path(directory)
            if directory in consumed:
                yield entries(directory, 2500 if directory == weights else 10000)
            else:
                with original_scandir(directory) as actual:
                    yield actual

        scanner = model_catalog._Scanner(self.root, extra)
        with patch("model_catalog.os.scandir", side_effect=bounded):
            result = scanner.scan()
        self.assertEqual(consumed, {weights: 2500, extra: 2497})
        self.assertEqual(len(scanner.visited), 5000)
        self.assertIn("访问上限", result["note"])

    def test_extra_overlong_candidate_is_skipped_before_stat(self):
        scanner = model_catalog._Scanner(None, self.root)
        path = self.root / ("黍" * 1024 + ".ckpt")
        with patch.object(Path, "lstat", side_effect=AssertionError("不得访问过长路径")):
            self.assertIsNone(scanner._inspect(path, boundary=self.root))
        self.assertTrue(scanner.skipped_long_paths)
        scanner._record(path, pretrained=False, extra=True)
        self.assertEqual(scanner.gpt, {})

    def test_real_mixed_files_obey_actual_serialized_450kib_budget(self):
        # 保持实际文件路径短于旧版 Windows MAX_PATH；仍以真实 450 KiB 响应限额验收。
        extra = self.root / ("补" * 25) / ("充" * 25)
        expected = set()
        nested = "/".join(["权" * 25, "重" * 25])
        for number in range(100):
            for suffix, directory, content in [("ckpt", "GPT_weights", b"fixture"),
                                                ("pth", "SoVITS_weights", b"PKfixture"),
                                                ("pth", "SoVITS_weights_lora", b"08fixture")]:
                expected.add(self.create(f"{directory}/{nested}/{number:03d}" + "角" * 60 + f".{suffix}",
                                         content=content))
                expected.add(self.create(f"{directory}-{number:03d}" + "黍" * 60 + f".{suffix}",
                                         root=extra, content=content))
        actual_bound = model_catalog._Scanner._bound_result
        unbounded = []

        def tracked_bound(result):
            unbounded.append(len(json.dumps(result, ensure_ascii=True).encode("utf-8")))
            actual_bound(result)

        with patch.object(model_catalog._Scanner, "_bound_result", side_effect=tracked_bound):
            result = scan_models(str(self.root), str(extra))
        self.assertGreater(unbounded[0], 450 * 1024)
        for ensure_ascii in (False, True):
            self.assertLessEqual(len(json.dumps(result, ensure_ascii=ensure_ascii).encode("utf-8")),
                                 450 * 1024)
        self.assertIn("大小上限", result["note"])
        self.assertLess(sum(len(result[kind]) for kind in
                            ("gpt_weights", "sovits_weights", "lora_weights")), 600)
        self.assertTrue(result["gpt_weights"])
        self.assertTrue(result["sovits_weights"])
        self.assertTrue(result["lora_weights"])
        for item in result["gpt_weights"] + result["sovits_weights"] + result["lora_weights"]:
            self.assertTrue({"label", "path"}.issubset(item))
            self.assertLessEqual(len(item["path"]), 1024)
            self.assertIn(item["path"], expected)


    def test_header_versions_classify_full_and_lora_independently_of_name(self):
        expected = {b"00": ("v1", "full"), b"01": ("v2", "full"),
                    b"02": ("v3", "full"), b"03": ("v3", "lora"),
                    b"04": ("v4", "lora"), b"05": ("v2Pro", "full"),
                    b"06": ("v2ProPlus", "full"), b"07": ("v5dev", "lora"),
                    b"08": ("v5turbo", "lora")}
        for head, (version, mode) in expected.items():
            with self.subTest(head=head):
                path = Path(self.create("misleading-v1.pth", content=head + b"fixture"))
                self.assertEqual(model_catalog.inspect_sovits_file(path),
                                 {"version": version, "mode": mode})
        self.create("SoVITS_weights_v4/a/same.pth", content=b"03fixture")
        self.create("SoVITS_weights_v3/b/same.pth", content=b"04fixture")
        self.create("SoVITS_weights_v5/c/same.pth", content=b"05fixture")
        self.create("SoVITS_weights_v1/d/same.pth", content=b"07fixture")
        self.create("SoVITS_weights_v2/e/same.pth", content=b"08fixture")
        result = scan_models(str(self.root))
        self.assertEqual({item["version"] for item in result["lora_weights"]},
                         {"v3", "v4", "v5dev", "v5turbo"})
        self.assertEqual([item["version"] for item in result["sovits_weights"]], ["v2Pro"])
        self.assertEqual(len(set(self.paths(result, "lora_weights"))), 4)
        for item in result["lora_weights"]:
            self.assertEqual(set(item), {"label", "path", "version"})

    def test_all_known_pretrained_hashes_are_full_before_lora_head(self):
        # 独立固定本机 process_ckpt.py 的完整 MD5 表，不能从被测表反推预期。
        expected = {
            "dc3c97e17592963677a4a1681f30c653": "v2",
            "43797be674a37c1c83ee81081941ed0f": "v3",
            "6642b37f3dbb1f76882b69937c95a5f3": "v2",
            "4f26b9476d0c5033e04162c486074374": "v4",
            "c7e9fce2223f3db685cdfa1e6368728a": "v2Pro",
            "66b313e39455b57ab1b0bc0b239c9d0a": "v2ProPlus",
            "c4ce1f839a0d271e41963f9fb16ade76": "v5dev",
            "6bc5cb2a195cc8ec22b2e723cd4d6f47": "v5turbo",
        }
        self.assertEqual(model_catalog._PRETRAINED_HASH_VERSIONS, expected)
        path = Path(self.create("SoVITS_weights/renamed.pth", content=b"08" + b"x" * 10000))
        for digest, version in expected.items():
            with self.subTest(version=version), \
                    patch("model_catalog.hashlib.md5", return_value=SimpleNamespace(hexdigest=lambda: digest)):
                result = scan_models(str(self.root))
                self.assertEqual(result["lora_weights"], [])
                self.assertEqual(result["sovits_weights"][0]["version"], version)
                self.assertEqual(result["sovits_weights"][0]["mode"], "full")
        # 真实计算头部 MD5：尾部不参与，且 hash 优先于 08 LoRA 前缀。
        digest = hashlib.md5(b"08" + b"x" * 8190).hexdigest()
        with patch.dict(model_catalog._PRETRAINED_HASH_VERSIONS, {digest: "v4"}), self.bounded_pth_reads():
            self.assertEqual(model_catalog.inspect_sovits_file(path), {"version": "v4", "mode": "full"})

    def test_unknown_pk_does_not_guess_lora_version_or_size(self):
        path = self.create("SoVITS_weights_v5turbo/v4-lora.pth", content=b"PK" + b"lora_rank" * 2000)
        result = scan_models(str(self.root))
        self.assertEqual(result["lora_weights"], [])
        self.assertEqual(result["sovits_weights"], [{"label": "训练权重 · SoVITS_weights_v5turbo/v4-lora.pth",
                                                   "path": path, "version": "unknown", "mode": "unknown"}])

    def test_bad_or_short_headers_are_not_mislabeled_and_io_error_is_chinese(self):
        for number, header in enumerate([b"", b"0", b"99bad", b"not a checkpoint"]):
            path = Path(self.create(f"SoVITS_weights/{number}.pth", content=header))
            with self.assertRaisesRegex(ValueError, "无法识别"):
                model_catalog.inspect_sovits_file(path)
        result = scan_models(str(self.root))
        self.assertEqual(result["sovits_weights"], [])
        self.assertEqual(result["lora_weights"], [])
        self.assertIn("无法识别", result["note"])
        with patch.object(Path, "open", side_effect=PermissionError("Denied")):
            with self.assertRaisesRegex(ValueError, "扫描模型目录失败"):
                scan_models(str(self.root))

    def test_lora_filtered_from_full_weights_and_default_base_pairs(self):
        self.create("GPT_SoVITS/pretrained_models/s1v3.ckpt")
        lora = self.create("GPT_SoVITS/pretrained_models/gsv-v5-pretrained/s2Gv5turbo.pth", content=b"08fixture")
        full = self.create("SoVITS_weights/full.pth", content=b"06fixture")
        self.create("GPT_weights/wrong.pth", content=b"03fixture")
        self.create("SoVITS_weights/wrong.ckpt")
        result = scan_models(str(self.root))
        self.assertEqual(result["base_models"], [])
        self.assertEqual(self.paths(result, "lora_weights"), [lora])
        self.assertEqual(self.paths(result, "sovits_weights"), [full])
        self.assertFalse(set(self.paths(result, "lora_weights")) & set(self.paths(result, "sovits_weights")))

    def test_mixed_sources_share_lora_limit_and_overlap_reads_once(self):
        extra = self.root / "SoVITS_weights"
        for number in range(205):
            self.create(f"SoVITS_weights/{number:03d}.pth", content=b"08fixture")
        for number in range(5):
            self.create(f"SoVITS_weights/sub/{number:03d}.pth", content=b"07fixture")
        with self.bounded_pth_reads() as reads:
            result = scan_models(str(self.root), str(extra))
        self.assertEqual(len(result["lora_weights"]), 200)
        self.assertEqual(len(set(self.paths(result, "lora_weights"))), 200)
        self.assertEqual(result["sovits_weights"], [])
        self.assertEqual(len(reads), 210)
        self.assertIn("200", result["note"])

    def test_all_four_lists_share_response_budget_including_lora(self):
        pair = {"id": "v4", "label": "v4 底模", "gpt_path": "C:/g.ckpt", "sovits_path": "C:/s.pth"}
        result = {"base_models": [pair], "gpt_weights": [], "sovits_weights": [],
                  "lora_weights": [], "note": "目录。"}
        originals = set()
        for number in range(200):
            for kind in ("gpt_weights", "sovits_weights", "lora_weights"):
                path = f"C:/{kind}/{number:03d}" + "黍" * 850 + ".pth"
                originals.add(path)
                item = {"label": "补充目录 · " + path, "path": path}
                if kind != "gpt_weights":
                    item["version"] = "v5turbo"
                if kind == "sovits_weights":
                    item["mode"] = "full"
                result[kind].append(item)
        model_catalog._Scanner._bound_result(result)
        self.assertEqual(result["base_models"], [pair])
        self.assertIn("大小上限", result["note"])
        for ensure_ascii in (False, True):
            self.assertLessEqual(len(json.dumps(result, ensure_ascii=ensure_ascii).encode()), 450 * 1024)
        for kind in ("gpt_weights", "sovits_weights", "lora_weights"):
            self.assertTrue(result[kind])
            self.assertLess(len(result[kind]), 200)
            self.assertTrue(all(item["path"] in originals for item in result[kind]))


if __name__ == "__main__":
    unittest.main()
