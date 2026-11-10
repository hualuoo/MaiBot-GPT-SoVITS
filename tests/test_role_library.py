"""role.json 角色库：python -B -m unittest discover -s tests -p test_role_library.py -v"""
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import role_library
from role_library import RoleLibrary
from role_fixtures import emotion_entry, file_entry, write_manifest

ROLE_ID = "黍/role.json"


class RoleLibraryTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.gsv = self.root / "GPT-SoVITS"
        self.pretrained = self.gsv / "GPT_SoVITS" / "pretrained_models"
        self.pretrained.mkdir(parents=True)
        self.roles = self.gsv / "maibot_roles"
        self.role = self.roles / "黍"
        self.payload = {"role": "黍", "emotions": {"happy": emotion_entry(self.role, "happy", audio=b"reference-audio",
                                                                           ref_text="这株盆栽今天已经完成了它的指标。")}}
        self.save()
        self.lib = RoleLibrary(self.roles)

    def save(self, payload=None):
        write_manifest(self.role, payload if payload is not None else self.payload)

    @property
    def happy(self):
        return self.payload["emotions"]["happy"]

    def resolve(self, emotion="happy", root=None):
        return self.lib.resolve(ROLE_ID, emotion, self.gsv.as_posix() if root is None else root)

    def install(self, relative, content):
        path = self.pretrained / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return {"name": path.name, "size": len(content), "sha256": hashlib.sha256(content).hexdigest()}, path

    def test_full_weight_role_maps_role_files_and_reports_assets(self):
        result = self.resolve()
        self.assertEqual(set(result), {"role_id", "role_name", "emotion", "emotion_label", "models", "voice", "assets", "warnings"})
        self.assertEqual(result["role_name"], "黍")
        self.assertEqual(result["emotion_label"], "开心")
        self.assertEqual(result["models"], {"base_model": "v2", "gpt_weights": (self.role / "shu_happy.ckpt").as_posix(),
                                            "sovits_weights": (self.role / "shu.pth").as_posix(), "lora_weights": ""})
        self.assertEqual(result["voice"], {"ref_audio_path": (self.role / "happy.wav").as_posix(),
                                           "prompt_text": "这株盆栽今天已经完成了它的指标。", "prompt_lang": "zh", "text_lang": "zh"})
        self.assertEqual([item["kind"] for item in result["assets"]], ["GPT 权重", "SoVITS 权重", "参考音频"])
        for item, key in zip(result["assets"], ("gpt", "sovits", "ref_audio")):
            self.assertEqual(set(item), {"kind", "path", "size", "sha256", "reused"})
            self.assertEqual(item["sha256"], self.happy[key]["sha256"])
            self.assertFalse(item["reused"])
        self.assertEqual(result["warnings"], [])

    def test_optional_languages_and_missing_hash_or_size_only_warn(self):
        self.happy.update(prompt_lang="ja", text_lang="auto")
        del self.happy["ref_audio"]["sha256"]
        del self.happy["gpt"]["size"]
        self.save()
        result = self.resolve()
        self.assertEqual((result["voice"]["prompt_lang"], result["voice"]["text_lang"]), ("ja", "auto"))
        self.assertTrue(any("参考音频" in item and "sha256" in item for item in result["warnings"]))
        self.assertTrue(any("GPT 权重" in item and "size" in item for item in result["warnings"]))

    def test_v2proplus_requires_installed_sv_model_and_base_model_is_found_in_gpt_sovits(self):
        sv = b"sv-model-bytes"
        self.payload["base_model"] = {"name": "pretrained_eres2netv2w24s4ep4.ckpt", "size": len(sv), "sha256": hashlib.sha256(sv).hexdigest()}
        self.payload["emotions"]["happy"] = emotion_entry(self.role, "happy", sovits=b"06-tiny-v2proplus-full", sovits_name="shu_v2ProPlus.pth")
        # 角色文件夹里的副本不会被使用：GPT-SoVITS 只从 pretrained_models/sv 读取。
        file_entry(self.role, "pretrained_eres2netv2w24s4ep4.ckpt", sv)
        self.save()
        with self.assertRaisesRegex(ValueError, r"底模pretrained_eres2netv2w24s4ep4\.ckpt不在 GPT-SoVITS 中：请放到 .*/GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4\.ckpt（角色文件夹里已有"):
            self.resolve()
        _, installed = self.install("sv/pretrained_eres2netv2w24s4ep4.ckpt", sv)
        result = self.resolve()
        self.assertEqual(result["models"]["base_model"], "v2ProPlus")
        self.assertEqual(result["models"]["lora_weights"], "")
        base = result["assets"][-1]
        self.assertEqual((base["kind"], base["path"], base["reused"]), ("底模", installed.as_posix(), True))
        installed.write_bytes(b"different-sv")
        with self.assertRaisesRegex(ValueError, "与 role.json 不一致"):
            self.resolve()
        installed.unlink()
        del self.payload["base_model"]
        self.save()
        # 未声明 base_model 时仍要检查 v2ProPlus 依赖的说话人模型。
        with self.assertRaisesRegex(ValueError, r"v2ProPlus 需要 GPT-SoVITS 中的 sv/pretrained_eres2netv2w24s4ep4\.ckpt"):
            self.resolve()

    def test_lora_role_uses_installed_standard_base_without_copying(self):
        base = b"PK-standard-v5turbo-base"
        spec, installed = self.install("gsv-v5-pretrained/s2Gv5turbo.pth", base)
        self.install("gsv-v5-pretrained/vocoder.pth", b"vocoder")
        self.payload["base_model"] = spec
        self.payload["emotions"]["happy"] = emotion_entry(self.role, "happy", sovits=b"08-tiny-v5turbo-lora", sovits_name="shu_lora.pth")
        self.save()
        catalog = {"base_models": [{"id": "v5turbo", "gpt_path": (self.pretrained / "s1v3.ckpt").as_posix(), "sovits_path": installed.as_posix()}]}
        before = sorted(path.as_posix() for path in self.root.rglob("*"))
        with patch("role_library.scan_models", return_value=catalog) as scan:
            result = self.resolve()
        scan.assert_called_once_with(self.gsv.as_posix())
        self.assertEqual(result["models"], {"base_model": "v5turbo", "gpt_weights": (self.role / "shu_happy.ckpt").as_posix(),
                                            "sovits_weights": installed.as_posix(), "lora_weights": (self.role / "shu_lora.pth").as_posix()})
        self.assertEqual(before, sorted(path.as_posix() for path in self.root.rglob("*")))
        with patch("role_library.scan_models", return_value={"base_models": []}):
            with self.assertRaisesRegex(ValueError, "v5turbo LoRA，需要 GPT-SoVITS 已安装 v5turbo 标准底模"):
                self.resolve()
        (self.pretrained / "gsv-v5-pretrained" / "vocoder.pth").unlink()
        with patch("role_library.scan_models", return_value=catalog):
            with self.assertRaisesRegex(ValueError, "v5turbo 需要 GPT-SoVITS 中的 gsv-v5-pretrained/vocoder.pth"):
                self.resolve()

    def test_vocoder_location_follows_weight_version_and_unknown_base_is_searched(self):
        self.install("gsv-v4-pretrained/vocoder.pth", b"v4-vocoder")
        spec, v5 = self.install("gsv-v5-pretrained/vocoder.pth", b"v5-vocoder")
        self.payload["base_model"] = spec
        self.payload["emotions"]["happy"] = emotion_entry(self.role, "happy", sovits=b"08-lora", sovits_name="lora.pth")
        self.save()
        catalog = {"base_models": [{"id": "v5turbo", "gpt_path": "g", "sovits_path": "s"}]}
        with patch("role_library.scan_models", return_value=catalog):
            self.assertEqual(self.resolve()["assets"][-1]["path"], v5.as_posix())
        custom, path = self.install("自定义/深层/custom-base.pth", b"custom-base")
        self.payload["base_model"] = custom
        self.payload["emotions"]["happy"] = emotion_entry(self.role, "happy")
        self.save()
        self.assertEqual(self.resolve()["assets"][-1]["path"], path.as_posix())
        self.payload["base_model"] = {"name": "missing.pth"}
        self.save()
        with self.assertRaisesRegex(ValueError, "底模missing.pth不在 GPT-SoVITS 中"):
            self.resolve()
        with self.assertRaisesRegex(ValueError, "第二步"):
            self.resolve(root="")

    def test_unrecognized_sovits_header_is_rejected(self):
        self.payload["emotions"]["happy"] = emotion_entry(self.role, "happy", sovits=b"PK-unknown-zip")
        self.save()
        with self.assertRaisesRegex(ValueError, "无法识别 SoVITS 权重"):
            self.resolve()

    def test_asset_size_hash_and_missing_files_are_reported_per_kind(self):
        original = json.loads(json.dumps(self.payload))
        for key, label in (("gpt", "GPT 权重"), ("sovits", "SoVITS 权重"), ("ref_audio", "参考音频")):
            for field, wrong, message in (("size", 1, "大小不匹配"), ("sha256", "0" * 64, "SHA256不匹配")):
                with self.subTest(key=key, field=field):
                    self.payload = json.loads(json.dumps(original))
                    self.happy[key][field] = wrong
                    self.save()
                    with self.assertRaisesRegex(ValueError, label + ".*" + message):
                        self.resolve()
        self.payload = original
        self.save()
        (self.role / "happy.wav").unlink()
        with self.assertRaisesRegex(ValueError, "参考音频（happy.wav）不可用"):
            self.resolve()

    def test_list_only_parses_json_uses_role_name_and_original_emotion_keys(self):
        self.payload["role"] = "黍（显示名）"
        for emotion in ("angry", "surprised", "neutral", "sad", "other"):
            self.payload["emotions"][emotion] = self.happy
        self.save()
        with patch.object(self.lib, "_hash_file", side_effect=AssertionError("不得hash")), \
             patch("role_library.scan_models", side_effect=AssertionError("不得扫描")), \
             patch("role_library.inspect_sovits_file", side_effect=AssertionError("不得读权重")):
            result = self.lib.list_roles()
        self.assertEqual(result["errors"], [])
        item = result["roles"][0]
        self.assertEqual(set(item), {"id", "name", "emotions", "json_path"})
        self.assertEqual((item["id"], item["name"]), (ROLE_ID, "黍（显示名）"))
        self.assertEqual([x["label"] for x in item["emotions"]], ["开心", "生气", "吃惊", "中立", "难过", "other"])

    def test_only_role_json_is_read_and_problems_are_reported_per_folder(self):
        other = self.roles / "没有配置"
        other.mkdir()
        (other / "其他.json").write_text('{"happy": {"GPT": "a.ckpt"}}', encoding="utf-8")
        missing = self.roles / "缺少情感"
        write_manifest(missing, {"happy": {"GPT": "a.ckpt", "SoVITS": "b.pth"}})
        bad = self.roles / "坏"
        bad.mkdir()
        (bad / "role.json").write_text("{", encoding="utf-8")
        for hidden in ("_shared", ".cache"):
            (self.roles / hidden).mkdir()
        result = self.lib.list_roles()
        self.assertEqual([x["id"] for x in result["roles"]], [ROLE_ID])
        errors = " | ".join(result["errors"])
        self.assertIn("没有配置：缺少 role.json", errors)
        self.assertIn("缺少情感：role.json 缺少 emotions", errors)
        self.assertIn("坏：角色JSON无法解析", errors)
        self.assertNotIn("_shared", errors)
        self.assertNotIn(".cache", errors)

    def test_role_id_must_point_to_role_json_and_emotion_must_exist(self):
        for role_id in ("黍/黍.json", "../黍/role.json", "黍/../role.json", "/tmp/role.json", r"C:\role.json",
                        "_shared/role.json", "黍/a/role.json", "黍/role.json:evil", "黍/role\x00.json"):
            with self.subTest(role_id=role_id), self.assertRaises(ValueError):
                self.lib.resolve(role_id, "happy", self.gsv.as_posix())
        for emotion in ("../happy", "happy/other", "..", "CON", "bad\x00", "/happy"):
            with self.subTest(emotion=emotion), self.assertRaises(ValueError):
                self.lib.resolve(ROLE_ID, emotion, self.gsv.as_posix())
        with self.assertRaisesRegex(ValueError, "不存在情感"):
            self.resolve("unknown")
        with self.assertRaisesRegex(ValueError, "不存在"):
            RoleLibrary(self.roles).resolve("缺失/role.json", "happy", self.gsv.as_posix())

    def test_schema_validation_messages(self):
        cases = [
            ({"emotions": {}}, "emotions必须是非空"),
            ({"emotions": []}, "emotions必须是非空"),
            ({"role": "", "emotions": {"happy": self.happy}}, "role必须是"),
            ({"emotions": {"happy": {**self.happy, "ref_text": ""}}}, "ref_text"),
            ({"emotions": {"happy": {k: v for k, v in self.happy.items() if k != "gpt"}}}, "happy.gpt必须是包含 name 的对象"),
            ({"emotions": {"happy": {**self.happy, "sovits": {"name": "../x.pth"}}}}, "不安全路径"),
            ({"emotions": {"happy": {**self.happy, "ref_audio": {"name": "C:/abs.wav"}}}}, "不允许绝对路径"),
            ({"emotions": {"happy": {**self.happy, "gpt": {**self.happy["gpt"], "size": True}}}}, "size必须为"),
            ({"emotions": {"happy": {**self.happy, "gpt": {**self.happy["gpt"], "size": -1}}}}, "size必须为"),
            ({"emotions": {"happy": {**self.happy, "gpt": {**self.happy["gpt"], "sha256": "abc"}}}}, "sha256必须为"),
            ({"base_model": {"name": "sv/x.ckpt"}, "emotions": {"happy": self.happy}}, "只能是文件名"),
            ({"base_model": "x.ckpt", "emotions": {"happy": self.happy}}, "base_model必须是包含 name 的对象"),
            ({"emotions": {"happy": {**self.happy, "text_lang": ""}}}, "text_lang"),
            ({"emotions": {f"e{i}": self.happy for i in range(51)}}, "最多50个情感"),
        ]
        for payload, message in cases:
            with self.subTest(message=message):
                self.save(payload)
                with self.assertRaisesRegex(ValueError, message):
                    self.resolve()
        (self.role / "role.json").write_text('{"emotions": {"happy": {}, "happy": {}}}', encoding="utf-8")
        self.assertIn("重复键", " ".join(self.lib.list_roles()["errors"]))
        (self.role / "role.json").write_bytes(b" " * (1024 * 1024 + 1))
        self.assertIn("1MiB", " ".join(self.lib.list_roles()["errors"]))

    def test_enumeration_and_asset_limits(self):
        with patch("role_library.MAX_ENUMERATED", 1):
            self.assertTrue(self.lib.list_roles()["errors"])
        with patch("role_library.MAX_ROLES", 0):
            self.assertEqual(self.lib.list_roles()["roles"], [])
        with patch("role_library.MAX_ASSET_BYTES", 2):
            with self.assertRaisesRegex(ValueError, "2GiB"):
                self.resolve()
        self.assertTrue(RoleLibrary(self.root / "missing").list_roles()["errors"])

    def test_host_package_import_uses_sibling_model_catalog(self):
        name = "_role_library_host_fixture"
        package = types.ModuleType(name)
        package.__path__ = [str(Path(role_library.__file__).parent)]
        before = set(sys.modules)
        self.addCleanup(lambda: [sys.modules.pop(key, None) for key in list(sys.modules)
                                 if key not in before and key.startswith(name)])
        with patch.dict(sys.modules, {name: package}):
            spec = importlib.util.spec_from_file_location(name + ".role_library", role_library.__file__)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.assertEqual(module.scan_models.__module__, name + ".model_catalog")
            self.assertEqual(module.RoleLibrary(self.roles).list_roles()["errors"], [])

    def test_hash_cache_reuses_and_invalidates_on_size_mtime_and_ctime(self):
        path = self.role / "happy.wav"
        first = self.lib._hash_file(path)
        with patch.object(Path, "open", side_effect=AssertionError("缓存命中不得读文件")):
            self.assertEqual(self.lib._hash_file(path), first)
        old = path.stat()
        path.write_bytes(b"changed-content")
        os.utime(path, ns=(old.st_atime_ns, old.st_mtime_ns + 1000000000))
        self.assertNotEqual(self.lib._hash_file(path)[1], first[1])
        old_key = next(iter(self.lib._hash_cache))
        self.lib._hash_cache = {(*old_key[:3], old_key[3] - 1): "stale"}
        self.assertNotEqual(self.lib._hash_file(path)[1], "stale")
        path.write_bytes(b"longer-content-than-before")
        self.assertEqual(self.lib._hash_file(path)[0], len(b"longer-content-than-before"))

    def test_hash_reads_bounded_chunks(self):
        original_open = Path.open
        reads = []
        test = self

        @contextmanager
        def guarded(path, mode="r", *args, **kwargs):
            with original_open(path, mode, *args, **kwargs) as stream:
                class Stream:
                    def fileno(self):
                        return stream.fileno()

                    def read(self, size=-1):
                        test.assertGreater(size, 0)
                        test.assertLessEqual(size, 1024 * 1024)
                        reads.append(size)
                        return stream.read(size)
                yield Stream()

        with patch.object(Path, "open", guarded):
            self.lib._hash_file(self.role / "happy.wav")
        self.assertTrue(reads)

    def test_symlink_asset_escape_rejected(self):
        outside = self.root / "outside.wav"
        outside.write_bytes(b"reference-audio")
        target = self.role / "happy.wav"
        target.unlink()
        try:
            target.symlink_to(outside)
        except OSError:
            self.skipTest("当前Windows账号没有symlink权限；junction由独立测试覆盖")
        with self.assertRaisesRegex(ValueError, "符号链接|联接"):
            self.resolve()

    @unittest.skipUnless(os.name == "nt", "Windows junction测试")
    def test_junction_directory_escape_rejected(self):
        outside = self.root / "external"
        outside.mkdir()
        (outside / "audio.wav").write_bytes(b"reference-audio")
        junction = self.role / "linked"
        run = subprocess.run(["cmd", "/c", "mklink", "/J", str(junction), str(outside)], capture_output=True)
        if run.returncode:
            self.skipTest("当前系统不能创建junction")
        self.addCleanup(lambda: junction.rmdir() if junction.exists() else None)
        self.happy["ref_audio"] = {"name": "linked/audio.wav"}
        self.save()
        with self.assertRaisesRegex(ValueError, "符号链接|联接"):
            self.resolve()


if __name__ == "__main__":
    unittest.main()
