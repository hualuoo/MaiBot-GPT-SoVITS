"""只读本地角色库：读取 maibot_roles/<角色>/role.json；不复制文件、不联网、不反序列化任何模型权重。

role.json 格式（文件路径都相对 role.json 所在的角色文件夹）：

    {
      "role": "黍",                                   # 可选：显示名，默认用文件夹名
      "base_model": {"name": "...", "size": 1, "sha256": "..."},   # 可选：需要 GPT-SoVITS 已安装的底模文件
      "emotions": {
        "happy": {
          "gpt":       {"name": "shu_happy.ckpt", "size": 1, "sha256": "..."},
          "sovits":    {"name": "shu.pth", "size": 1, "sha256": "..."},
          "ref_audio": {"name": "happy.mp3", "size": 1, "sha256": "..."},
          "ref_text":  "参考音频里说的话",
          "prompt_lang": "zh", "text_lang": "zh"     # 可选，默认 zh
        }
      }
    }

size 与 sha256 可省略（省略时只给出提示）。base_model 只按文件名在 GPT-SoVITS 的
GPT_SoVITS/pretrained_models 中查找：GPT-SoVITS 合成时从那里读取底模，角色文件夹里的副本不会被使用。
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import re
import stat

if __package__:
    from .model_catalog import inspect_sovits_file, scan_models
else:
    from model_catalog import inspect_sovits_file, scan_models

ROLE_FILENAME = "role.json"
MAX_JSON_BYTES = 1024 * 1024
MAX_ROLES = 100
MAX_EMOTIONS = 50
MAX_ASSET_BYTES = 2 * 1024 ** 3
MAX_ENUMERATED = 5000
_HASH_CHUNK = 1024 * 1024
_LABELS = {"happy": "开心", "angry": "生气", "surprised": "吃惊", "neutral": "中立", "sad": "难过"}
_RESERVED = re.compile(r"^(CON|PRN|AUX|NUL|CONIN\$|CONOUT\$|COM[1-9¹²³]|LPT[1-9¹²³])$", re.I)
_VERSIONS = {"v1", "v2", "v3", "v4", "v2Pro", "v2ProPlus", "v5dev", "v5turbo"}
_ASSETS = (("gpt", "GPT 权重"), ("sovits", "SoVITS 权重"), ("ref_audio", "参考音频"))
# GPT-SoVITS 从这些固定位置（相对 GPT_SoVITS/pretrained_models）读取底模；与官方 config.py、TTS.py 默认路径一致。
_SV_MODEL = "sv/pretrained_eres2netv2w24s4ep4.ckpt"
_BASE_LOCATIONS = {
    "pretrained_eres2netv2w24s4ep4.ckpt": (_SV_MODEL,),
    "s1v3.ckpt": ("s1v3.ckpt",),
    "s2Gv3.pth": ("s2Gv3.pth",),
    "s2G488k.pth": ("s2G488k.pth",),
    "s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt": ("s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt",),
    "s2G2333k.pth": ("gsv-v2final-pretrained/s2G2333k.pth",),
    "s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt":
        ("gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",),
    "s2Gv4.pth": ("gsv-v4-pretrained/s2Gv4.pth",),
    "s2Gv2Pro.pth": ("v2Pro/s2Gv2Pro.pth",),
    "s2Gv2ProPlus.pth": ("v2Pro/s2Gv2ProPlus.pth",),
    "s2Gv5dev.pth": ("gsv-v5-pretrained/s2Gv5dev.pth",),
    "s2Gv5turbo.pth": ("gsv-v5-pretrained/s2Gv5turbo.pth",),
    "vocoder.pth": ("gsv-v5-pretrained/vocoder.pth", "gsv-v4-pretrained/vocoder.pth"),
}
# 这些版本在 GPT-SoVITS 中还依赖固定位置的附加模型，缺少时合成会失败（以 / 结尾表示文件夹）。
_RUNTIME_FILES = {
    "v2Pro": (_SV_MODEL,), "v2ProPlus": (_SV_MODEL,),
    "v3": ("models--nvidia--bigvgan_v2_24khz_100band_256x/",),
    "v4": ("gsv-v4-pretrained/vocoder.pth",),
    "v5dev": ("gsv-v5-pretrained/vocoder.pth",), "v5turbo": ("gsv-v5-pretrained/vocoder.pth",),
}
_SEARCH_DEPTH = 3


def _relative(value: str, label: str) -> tuple[str, ...]:
    if not isinstance(value, str) or not value or len(value) > 1024:
        raise ValueError(f"{label}必须是非空的本地相对路径（最长1024字符）")
    windows = PureWindowsPath(value)
    if windows.drive or windows.root or value.startswith(("/", "\\")):
        raise ValueError(f"{label}只能相对角色目录，不允许绝对路径或网络路径")
    parts = value.replace("\\", "/").split("/")
    for part in parts:
        if (not part or part in (".", "..") or part[-1:] in (" ", ".")
                or any(ord(c) < 32 or c in '<>:"|?*' for c in part)
                or _RESERVED.match(part.split(".", 1)[0].rstrip(" ."))):
            raise ValueError(f"{label}包含不安全路径（禁止..、NUL、ADS或Windows保留名）")
    return tuple(parts)


def _plain_id(value: str, label: str) -> str:
    if len(_relative(value, label)) != 1:
        raise ValueError(f"{label}不能包含路径分隔符")
    return value


def _guard(path: Path, *, directory: bool = False) -> Path:
    """检查每一级路径，Windows junction/reparse point 与 symlink 一律拒绝。"""
    path = Path(os.path.abspath(path))
    if path.drive.startswith(("\\\\", "//")):
        raise ValueError("不支持网络目录")
    try:
        for item in (*reversed(path.parents), path):
            info = item.lstat()
            if (stat.S_ISLNK(info.st_mode)
                    or getattr(info, "st_file_attributes", 0) & 0x400):
                raise ValueError(f"拒绝符号链接或目录联接路径：{path}")
        resolved = path.resolve(strict=True)
        if resolved != path:
            raise ValueError(f"路径越界或经过目录联接：{path}")
        info = path.stat()
        valid = stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)
        if not valid:
            raise ValueError(f"路径不是{'目录' if directory else '普通文件'}：{path}")
        return path
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"文件或目录不存在、无法访问：{path}") from exc


def _under(root: Path, relative: str, label: str) -> Path:
    parts = _relative(relative, label)
    base = _guard(root, directory=True)
    path = _guard(base.joinpath(*parts))
    if not path.is_relative_to(base):
        raise ValueError(f"{label}逃逸角色目录")
    return path


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON存在重复键：{key}")
        result[key] = value
    return result


def _text(value, label: str, limit: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit or "\x00" in value:
        raise ValueError(f"{label}必须是1至{limit}字符的非空文本")
    return value


def _file_spec(value, label: str, *, plain: bool = False) -> dict:
    """{"name": 相对路径, "size": 字节数（可选）, "sha256": 64位十六进制（可选）}。"""
    if not isinstance(value, dict):
        raise ValueError(f"{label}必须是包含 name 的对象")
    name = value.get("name")
    parts = _relative(name, label + ".name")
    if plain and len(parts) != 1:
        raise ValueError(f"{label}.name只能是文件名，不能包含文件夹")
    size, sha = value.get("size"), value.get("sha256")
    if size is not None and (type(size) is not int or not 0 <= size <= MAX_ASSET_BYTES):
        raise ValueError(f"{label}.size必须为0至2GiB的整数字节数")
    if sha is not None:
        if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", sha):
            raise ValueError(f"{label}.sha256必须为64位十六进制")
        sha = sha.lower()
    return {"name": name, "size": size, "sha256": sha}


def _parse_manifest(payload) -> dict:
    """校验 role.json 结构（不访问任何资产文件），返回规范化后的清单。"""
    if not isinstance(payload, dict) or not payload:
        raise ValueError("role.json顶层必须是非空JSON对象")
    if "emotions" not in payload:
        raise ValueError("role.json 缺少 emotions，格式见 docs/voice-cloning.md 中的“role.json 格式”")
    role = payload.get("role")
    if role is not None:
        role = _text(role, "role", 80)
    base = payload.get("base_model")
    if base is not None:
        base = _file_spec(base, "base_model", plain=True)
    emotions = payload["emotions"]
    if not isinstance(emotions, dict) or not emotions:
        raise ValueError("emotions必须是非空的情感对象")
    if len(emotions) > MAX_EMOTIONS:
        raise ValueError("每个role.json最多50个情感")
    parsed = {}
    for emotion, fields in emotions.items():
        _plain_id(emotion, "情感ID")
        if len(emotion) > 80:
            raise ValueError(f"情感ID最长80字符：{emotion[:20]}…")
        if not isinstance(fields, dict):
            raise ValueError(f"情感{emotion}必须为JSON对象")
        item = {key: _file_spec(fields.get(key), f"{emotion}.{key}") for key, _ in _ASSETS}
        item["ref_text"] = _text(fields.get("ref_text"), f"{emotion}.ref_text", 4000)
        for key in ("prompt_lang", "text_lang"):
            value = fields.get(key, "zh")
            item[key] = _text(value, f"{emotion}.{key}", 32).strip()
        parsed[emotion] = item
    return {"role": role, "base_model": base, "emotions": parsed}


class RoleLibrary:
    """data_dir（GPT-SoVITS 目录下的 maibot_roles）下每个角色文件夹读取一个 role.json。"""

    def __init__(self, data_dir: Path):
        if not isinstance(data_dir, (str, os.PathLike)):
            raise ValueError("角色库目录必须为Path或本地路径")
        self.data_dir = Path(os.path.abspath(data_dir))
        self._hash_cache: dict[tuple, str] = {}

    def _read_json(self, path: Path) -> dict:
        path = _guard(path)
        try:
            if path.stat().st_size > MAX_JSON_BYTES:
                raise ValueError("role.json超过1MiB上限")
            with path.open("rb") as stream:
                raw = stream.read(MAX_JSON_BYTES + 1)
            if len(raw) > MAX_JSON_BYTES:
                raise ValueError("role.json超过1MiB上限")
            payload = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_unique_object)
        except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
            raise ValueError(f"角色JSON无法解析：{path.name}（{exc}）") from exc
        return _parse_manifest(payload)

    def list_roles(self) -> dict:
        """快速目录：只读有界JSON，不stat/hash角色资产，不调用模型扫描。"""
        roles, errors = [], []
        visited = 0
        limited = False

        def children(directory):
            nonlocal visited, limited
            with os.scandir(directory) as entries:
                for entry in entries:
                    if visited >= MAX_ENUMERATED:
                        limited = True
                        break
                    visited += 1
                    yield entry

        try:
            root = _guard(self.data_dir, directory=True)
            for entry in children(root):
                if entry.name.startswith((".", "_")):
                    continue
                try:
                    if not entry.is_dir(follow_symlinks=False):
                        continue
                    _plain_id(entry.name, "角色文件夹名")
                    role_dir = _guard(Path(entry.path), directory=True)
                    manifest = next((child.name for child in children(role_dir)
                                     if child.name.casefold() == ROLE_FILENAME), None)
                    if manifest is None:
                        errors.append(f"{entry.name}：缺少 {ROLE_FILENAME}（每个角色文件夹放一个 {ROLE_FILENAME}）")
                        continue
                    if len(roles) >= MAX_ROLES:
                        limited = True
                        break
                    path = _under(role_dir, manifest, "角色配置")
                    data = self._read_json(path)
                    roles.append({"id": f"{entry.name}/{manifest}", "name": data["role"] or entry.name,
                                  "emotions": [{"id": key, "label": _LABELS.get(key, key)} for key in data["emotions"]],
                                  "json_path": path.as_posix()})
                except (OSError, ValueError) as exc:
                    errors.append(f"{entry.name}：{exc}")
                if limited:
                    break
        except (OSError, ValueError) as exc:
            errors.append(f"角色库读取失败：{exc}")
        if limited:
            errors.append(f"角色库达到上限（最多{MAX_ROLES}个角色、枚举{MAX_ENUMERATED}项），其余已省略")
        return {"roles": sorted(roles, key=lambda role: role["id"]), "errors": errors}

    def _hash_file(self, path: Path) -> tuple[int, str]:
        path = _guard(path)
        before = path.stat()
        if before.st_size > MAX_ASSET_BYTES:
            raise ValueError(f"资产超过2GiB上限：{path.name}")
        key = (os.path.normcase(str(path)), before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        cached = self._hash_cache.get(key)
        if cached is not None:
            return before.st_size, cached
        digest = hashlib.sha256()
        total = 0
        try:
            with path.open("rb", buffering=0) as stream:
                opened = os.fstat(stream.fileno())
                # Windows的fstat与Path.stat对ctime的语义可能不同，不能跨API比较；
                # 路径ctime仍属于缓存键，并在读取结束后用同一API比较。
                if (opened.st_size, opened.st_mtime_ns, opened.st_ino, opened.st_dev) != (
                        before.st_size, before.st_mtime_ns, before.st_ino, before.st_dev):
                    raise ValueError(f"校验期间文件发生变化：{path.name}")
                while True:
                    block = stream.read(_HASH_CHUNK)
                    if not block:
                        break
                    total += len(block)
                    if total > MAX_ASSET_BYTES:
                        raise ValueError(f"资产超过2GiB上限：{path.name}")
                    digest.update(block)
            after = _guard(path).stat()
        except OSError as exc:
            raise ValueError(f"读取资产失败：{path.name}") from exc
        if (total, after.st_size, after.st_mtime_ns, after.st_ctime_ns, after.st_ino) != (
                before.st_size, before.st_size, before.st_mtime_ns, before.st_ctime_ns, before.st_ino):
            raise ValueError(f"校验期间文件发生变化：{path.name}")
        result = digest.hexdigest()
        # 同一路径保留最新快照，缓存总量有界。
        for old in list(self._hash_cache):
            if old[0] == key[0]:
                del self._hash_cache[old]
        if len(self._hash_cache) >= 512:
            self._hash_cache.pop(next(iter(self._hash_cache)))
        self._hash_cache[key] = result
        return total, result

    def _asset(self, kind: str, path: Path, spec: dict, warnings: list, *, reused: bool = False) -> dict:
        expected_size, expected_hash = spec["size"], spec["sha256"]
        actual_size = _guard(path).stat().st_size
        if actual_size > MAX_ASSET_BYTES:
            raise ValueError(f"{kind}超过2GiB上限")
        if expected_size is not None and expected_size != actual_size:
            raise ValueError(f"{kind}大小不匹配：期望{expected_size}，实际{actual_size}（{path}）")
        size, sha = self._hash_file(path)
        if expected_size is not None and expected_size != size:
            raise ValueError(f"{kind}大小不匹配：期望{expected_size}，实际{size}")
        if expected_hash is not None and expected_hash != sha:
            raise ValueError(f"{kind} SHA256不匹配：{path}")
        if expected_hash is None:
            warnings.append(f"{kind}（{spec['name']}）未提供sha256，无法确认文件与角色配置一致")
        if expected_size is None:
            warnings.append(f"{kind}（{spec['name']}）未提供size，使用实际文件大小")
        return {"kind": kind, "path": path.as_posix(), "size": size, "sha256": sha, "reused": reused}

    @staticmethod
    def _pretrained(model_root: str) -> Path:
        if not isinstance(model_root, str) or not model_root.strip():
            raise ValueError("请先在第二步填写并保存本机模型目录（GPT-SoVITS 安装目录），用于查找底模")
        if "\x00" in model_root or not Path(model_root).is_absolute():
            raise ValueError("本机模型目录必须为本地绝对路径")
        return _guard(Path(model_root), directory=True) / "GPT_SoVITS" / "pretrained_models"

    @staticmethod
    def _base_locations(name: str, version: str) -> tuple[str, ...]:
        locations = _BASE_LOCATIONS.get(name, ())
        if name == "vocoder.pth" and version in ("v4", "v5dev", "v5turbo"):
            folder = "gsv-v4-pretrained/" if version == "v4" else "gsv-v5-pretrained/"
            locations = tuple(item for item in locations if item.startswith(folder))
        return locations

    @staticmethod
    def _search(pretrained: Path, name: str) -> list[Path]:
        """未登记的底模文件名：在 pretrained_models 中有界查找同名文件（不跟随链接）。"""
        found, visited = [], 0

        def walk(directory: Path, depth: int) -> None:
            nonlocal visited
            with os.scandir(directory) as entries:
                for entry in entries:
                    visited += 1
                    if visited > MAX_ENUMERATED:
                        raise ValueError("底模目录枚举超过5000项上限")
                    if entry.is_symlink() or entry.name.startswith("."):
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        if depth < _SEARCH_DEPTH:
                            walk(Path(entry.path), depth + 1)
                    elif entry.name.casefold() == name.casefold():
                        found.append(Path(entry.path))

        if pretrained.is_dir():
            walk(_guard(pretrained, directory=True), 0)
        return sorted(found)

    def _locate_base(self, spec: dict, model_root: str, version: str, role_dir: Path, warnings: list) -> dict:
        name = spec["name"]
        pretrained = self._pretrained(model_root)
        locations = self._base_locations(name, version)
        candidates = ([pretrained.joinpath(*item.split("/")) for item in locations] if locations
                      else self._search(pretrained, name))
        failures = []
        for candidate in candidates:
            if not candidate.exists():
                continue
            try:
                return self._asset("底模", candidate, spec, warnings, reused=True)
            except ValueError as exc:
                failures.append(str(exc))
        target = (pretrained / locations[0]) if locations else pretrained / name
        message = f"底模{name}不在 GPT-SoVITS 中：请放到 {target.as_posix()}"
        if failures:
            message = f"GPT-SoVITS 中的底模{name}与 role.json 不一致：" + "；".join(failures[:2])
        elif (role_dir / name).is_file():
            message += f"（角色文件夹里已有 {name}，复制过去即可；GPT-SoVITS 只从这个位置读取）"
        raise ValueError(message)

    def _check_runtime(self, version: str, model_root: str) -> None:
        if not _RUNTIME_FILES.get(version):
            return
        pretrained = self._pretrained(model_root)
        for relative in _RUNTIME_FILES[version]:
            folder = relative.endswith("/")
            path = pretrained.joinpath(*relative.rstrip("/").split("/"))
            if not (path.is_dir() if folder else path.is_file()):
                raise ValueError(f"{version} 需要 GPT-SoVITS 中的 {relative.rstrip('/')}，但没有找到：{path.as_posix()}")

    def resolve(self, role_id: str, emotion: str, model_root: str) -> dict:
        """完整校验选定角色的一个情感，返回可直接合并到 models/voice 的配置与资产证据。"""
        parts = _relative(role_id, "角色ID")
        if len(parts) != 2 or parts[0].startswith((".", "_")) or parts[1].casefold() != ROLE_FILENAME:
            raise ValueError(f"角色ID必须为“角色文件夹/{ROLE_FILENAME}”，请在第三步重新选择角色")
        _plain_id(emotion, "情感ID")
        path = _under(self.data_dir, role_id, "角色配置")
        manifest = self._read_json(path)
        fields = manifest["emotions"].get(emotion)
        if fields is None:
            raise ValueError(f"角色{parts[0]}不存在情感：{emotion}")
        role_dir = path.parent
        warnings, assets, selected = [], [], {}
        for key, kind in _ASSETS:
            spec = fields[key]
            try:
                selected[key] = _under(role_dir, spec["name"], kind)
                assets.append(self._asset(kind, selected[key], spec, warnings))
            except ValueError as exc:
                raise ValueError(f"{kind}（{spec['name']}）不可用：{exc}") from exc
        try:
            info = inspect_sovits_file(selected["sovits"])
        except (OSError, ValueError) as exc:
            raise ValueError(f"SoVITS 权重头信息无法识别：{exc}") from exc
        version, mode = info.get("version"), info.get("mode")
        if version not in _VERSIONS or mode not in ("full", "lora"):
            raise ValueError(f"无法识别 SoVITS 权重（{fields['sovits']['name']}）的版本，请确认是 GPT-SoVITS 训练导出的权重")
        if manifest["base_model"] is not None:
            assets.append(self._locate_base(manifest["base_model"], model_root, version, role_dir, warnings))
        self._check_runtime(version, model_root)
        if mode == "lora":
            # LoRA 由 GPT-SoVITS 叠加到同版本的标准底模上，底模必须已安装在标准位置。
            catalog = scan_models(model_root)
            preset = next((item for item in catalog.get("base_models", [])
                           if isinstance(item, dict) and item.get("id") == version), None)
            if preset is None:
                raise ValueError(f"SoVITS 权重是 {version} LoRA，需要 GPT-SoVITS 已安装 {version} 标准底模")
            sovits, lora = preset["sovits_path"], selected["sovits"].as_posix()
        else:
            sovits, lora = selected["sovits"].as_posix(), ""
        return {"role_id": "/".join(parts), "role_name": manifest["role"] or parts[0], "emotion": emotion,
                "emotion_label": _LABELS.get(emotion, emotion),
                "models": {"base_model": version, "gpt_weights": selected["gpt"].as_posix(),
                           "sovits_weights": sovits, "lora_weights": lora},
                "voice": {"ref_audio_path": selected["ref_audio"].as_posix(), "prompt_text": fields["ref_text"],
                          "prompt_lang": fields["prompt_lang"], "text_lang": fields["text_lang"]},
                "assets": assets, "warnings": warnings}
