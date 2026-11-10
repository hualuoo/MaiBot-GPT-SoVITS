"""仅扫描明确配置的本地模型目录，不加载模型或执行其源码。"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat


# 参考音频与角色文件统一放在 GPT-SoVITS 安装目录下的指定文件夹：
# 合成时由 GPT-SoVITS 服务直接读取，放在它的目录里才能保证读得到。
REFS_DIRNAME = "maibot_refs"
ROLES_DIRNAME = "maibot_roles"
_AUDIO_SUFFIXES = frozenset({".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus"})
_MAX_REFERENCE_AUDIOS = 200
_MAX_REFERENCE_DEPTH = 2
_MAX_MODELS_PER_KIND = 200
_MAX_VISITED_PATHS = 5000
_MAX_PATH_CHARS = 1024
_MAX_RESULT_BYTES = 450 * 1024
_MAX_HEADER_BYTES = 8192
# 与本机 GPT_SoVITS/process_ckpt.py 的头部 MD5、head2version 一致。
# 只复制常量，不导入/执行 GPT-SoVITS，也不按文件名或文件大小猜版本。
_PRETRAINED_HASH_VERSIONS = {
    "dc3c97e17592963677a4a1681f30c653": "v2",
    "43797be674a37c1c83ee81081941ed0f": "v3",
    "6642b37f3dbb1f76882b69937c95a5f3": "v2",
    "4f26b9476d0c5033e04162c486074374": "v4",
    "c7e9fce2223f3db685cdfa1e6368728a": "v2Pro",
    "66b313e39455b57ab1b0bc0b239c9d0a": "v2ProPlus",
    "c4ce1f839a0d271e41963f9fb16ade76": "v5dev",
    "6bc5cb2a195cc8ec22b2e723cd4d6f47": "v5turbo",
}
_HEADER_VERSIONS = {
    b"00": ("v1", "full"), b"01": ("v2", "full"), b"02": ("v3", "full"),
    b"03": ("v3", "lora"), b"04": ("v4", "lora"),
    b"05": ("v2Pro", "full"), b"06": ("v2ProPlus", "full"),
    b"07": ("v5dev", "lora"), b"08": ("v5turbo", "lora"),
}
_PRETRAINED = "GPT_SoVITS/pretrained_models"
# 已与本机 GPT-SoVITS/config.py:12-31 和
# GPT_SoVITS/TTS_infer_pack/TTS.py:295-356 的真实默认值逐项核对。
# v4 / v2Pro / v2ProPlus / v5 共用 s1v3.ckpt，不虚构同版本 GPT 文件。
_DEFAULT_PAIRS = (
    ("v1", "s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt", "s2G488k.pth"),
    ("v2", "gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
     "gsv-v2final-pretrained/s2G2333k.pth"),
    ("v3", "s1v3.ckpt", "s2Gv3.pth"),
    ("v4", "s1v3.ckpt", "gsv-v4-pretrained/s2Gv4.pth"),
    ("v2Pro", "s1v3.ckpt", "v2Pro/s2Gv2Pro.pth"),
    ("v2ProPlus", "s1v3.ckpt", "v2Pro/s2Gv2ProPlus.pth"),
    ("v5dev", "s1v3.ckpt", "gsv-v5-pretrained/s2Gv5dev.pth"),
    ("v5turbo", "s1v3.ckpt", "gsv-v5-pretrained/s2Gv5turbo.pth"),
)
# 这些目录不是 GPT/SoVITS 推理权重。尤其不递归 BERT/Hubert 下载缓存。
_PRETRAINED_SKIP = frozenset({
    "chinese-hubert-base", "chinese-roberta-wwm-ext-large", "fast_langdetect",
    "bigvgan", "vocoder", "vocos", "sv", ".cache", ".git", "__pycache__",
})


def _sovits_metadata(path: Path) -> tuple[str, str]:
    """仅读取一次至多 8192 字节；MD5 优先，绝不反序列化权重。"""
    with path.open("rb", buffering=0) as stream:
        header = stream.read(_MAX_HEADER_BYTES)
    version = _PRETRAINED_HASH_VERSIONS.get(hashlib.md5(header).hexdigest())
    if version is not None:
        return version, "full"
    if header[:2] in _HEADER_VERSIONS:
        return _HEADER_VERSIONS[header[:2]]
    if header[:2] == b"PK":
        # 早期 GPT-SoVITS 导出的 ZIP 格式权重无法仅凭头部区分版本：记为未知，不猜测是否为 LoRA。
        return "unknown", "unknown"
    raise ValueError("无法识别 SoVITS 权重头，已跳过该文件")


def inspect_sovits_file(path: Path) -> dict[str, str]:
    """供本地加载前校验使用；只读头部，不加载模型。识别失败抛 ValueError。"""
    version, mode = _sovits_metadata(path)
    return {"version": version, "mode": mode}


class _Scanner:
    def __init__(self, root: Path | None, extra_dir: Path | None = None):
        self.root = root
        self.extra_dir = extra_dir
        # 两个来源共享访问、数量和响应预算；重叠目录不重复枚举。
        self.visited: set[str] = set()
        self._children_cache: dict[str, list[Path]] = {}
        self.path_limit = False
        self.model_limits: set[str] = set()
        self.skipped_links = False
        self.skipped_long_paths = False
        self.gpt: dict[str, dict[str, str]] = {}
        self.sovits: dict[str, dict[str, str]] = {}
        self.lora: dict[str, dict[str, str]] = {}
        self.unrecognized_sovits = False
        self._processed_weights: set[str] = set()
        self.pretrained: dict[str, str] = {}
        for directory in (root, extra_dir):
            if directory is not None:
                self._claim(directory)

    def _claim(self, path: Path) -> bool:
        key = os.path.normcase(str(path))
        if key in self.visited:
            return True
        if len(self.visited) >= _MAX_VISITED_PATHS:
            self.path_limit = True
            return False
        self.visited.add(key)
        return True

    def _inspect(self, path: Path, *, boundary: Path | None = None):
        if not self._claim(path):
            return None
        if len(path.as_posix()) > _MAX_PATH_CHARS:
            self.skipped_long_paths = True
            return None
        boundary = boundary if boundary is not None else self.root
        if boundary is None or not path.is_relative_to(boundary):
            self.skipped_links = True
            return None
        try:
            info = path.lstat()
            # Windows 目录联接也是重解析点；不能只靠 is_symlink()。
            if (stat.S_ISLNK(info.st_mode)
                    or getattr(info, "st_file_attributes", 0)
                    & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)):
                self.skipped_links = True
                return None
            resolved = path.resolve(strict=True)
        except FileNotFoundError:
            # 扫描期间被删除的文件不列入目录，不把失效路径返回给界面。
            return None
        if (not resolved.is_relative_to(boundary)
                or resolved != Path(os.path.abspath(path))):
            self.skipped_links = True
            return None
        if len(resolved.as_posix()) > _MAX_PATH_CHARS:
            self.skipped_long_paths = True
            return None
        return resolved, info.st_mode

    def _children(self, directory: Path, *, boundary: Path | None = None) -> list[Path]:
        # 在 scandir 前重新检查目录，拒绝链接、重解析点及扫描期间发生的越界。
        inspected = self._inspect(directory, boundary=boundary)
        if inspected is None or not stat.S_ISDIR(inspected[1]):
            return []
        key = os.path.normcase(str(directory))
        if key in self._children_cache:
            return self._children_cache[key]
        children = []
        if len(self.visited) >= _MAX_VISITED_PATHS:
            self.path_limit = True
            return children
        # 流式读取并在排序前施加路径预算，不能先 list(rglob(...)) 再截断。
        with os.scandir(directory) as entries:
            while len(self.visited) < _MAX_VISITED_PATHS:
                try:
                    entry = next(entries)
                except StopIteration:
                    break
                path = Path(entry.path)
                if not self._claim(path):
                    break
                if path.parent != directory:
                    self.skipped_links = True
                    continue
                children.append(path)
            else:
                self.path_limit = True
        children.sort(key=lambda p: (p.name.casefold(), p.name))
        self._children_cache[key] = children
        return children

    def _record(self, path: Path, *, pretrained: bool, extra: bool = False) -> None:
        kind = path.suffix.casefold()
        if kind not in (".ckpt", ".pth"):
            return
        # 预训练目录里的声码器不是可选择的 SoVITS 声学模型。
        if pretrained and path.name.casefold() == "vocoder.pth":
            return
        relative = path.name if extra else path.relative_to(self.root).as_posix()
        absolute = path.as_posix()
        if len(absolute) > _MAX_PATH_CHARS:
            self.skipped_long_paths = True
            return
        key = os.path.normcase(str(path))
        for existing in (self.gpt, self.sovits, self.lora):
            if key in existing:
                if extra:
                    existing[key]["label"] = f"补充目录 · {path.name}"
                return
        # 超过分类限额或无效头的文件也只读取一次，重叠来源不重读头部。
        if key in self._processed_weights:
            return
        self._processed_weights.add(key)
        version, mode = "unknown", "unknown"
        if kind == ".pth":
            try:
                version, mode = _sovits_metadata(path)
            except ValueError:
                self.unrecognized_sovits = True
                return
        target = self.gpt if kind == ".ckpt" else self.lora if mode == "lora" else self.sovits
        # 重命名为默认底模路径的 LoRA 也不能进入底模配对。
        if pretrained and mode != "lora":
            self.pretrained[relative.casefold()] = absolute
        if len(target) >= _MAX_MODELS_PER_KIND:
            self.model_limits.add("GPT" if kind == ".ckpt" else "LoRA" if mode == "lora" else "SoVITS")
            return
        prefix = "补充目录" if extra else "底模目录" if pretrained else "训练权重"
        item = {"label": f"{prefix} · {relative}", "path": absolute}
        if kind == ".pth":
            item["version"] = version
            if mode != "lora":
                item["mode"] = mode
        target[key] = item

    def _walk(self, directory: Path, *, suffix: str | None = None,
              pretrained: bool = False) -> None:
        pending = [directory]
        scanned = set()
        while pending:
            current = pending.pop()
            key = os.path.normcase(str(current))
            if key in scanned:
                continue
            scanned.add(key)
            subdirectories = []
            for child in self._children(current):
                if pretrained and child.name.casefold() in _PRETRAINED_SKIP:
                    continue
                inspected = self._inspect(child)
                if inspected is None:
                    continue
                path, mode = inspected
                if stat.S_ISDIR(mode):
                    subdirectories.append(path)
                elif stat.S_ISREG(mode) and (suffix is None or path.suffix.casefold() == suffix):
                    self._record(path, pretrained=pretrained)
            pending.extend(reversed(subdirectories))

    def _scan_root(self) -> None:
        if self.root is None:
            return
        roots = self._children(self.root)
        # 直接定位这一个已知子路径，不遍历 GPT_SoVITS 的代码、数据或环境目录。
        for child in roots:
            if child.name.casefold() != "gpt_sovits":
                continue
            inspected = self._inspect(child)
            if inspected is None or not stat.S_ISDIR(inspected[1]):
                continue
            pretrained = self._inspect(inspected[0] / "pretrained_models")
            if pretrained is not None and stat.S_ISDIR(pretrained[1]):
                self._walk(pretrained[0], pretrained=True)
            break
        for child in roots:
            name = child.name.casefold()
            if name.startswith("gpt_weights"):
                suffix = ".ckpt"
            elif name.startswith("sovits_weights"):
                suffix = ".pth"
            else:
                continue
            inspected = self._inspect(child)
            if inspected is not None and stat.S_ISDIR(inspected[1]):
                self._walk(inspected[0], suffix=suffix)

    def _scan_extra(self) -> None:
        if self.extra_dir is None:
            return
        for child in self._children(self.extra_dir, boundary=self.extra_dir):
            # 只查看补充目录第一层候选权重；不进入任何子目录。
            if child.suffix.casefold() not in (".ckpt", ".pth"):
                continue
            inspected = self._inspect(child, boundary=self.extra_dir)
            if inspected is not None and stat.S_ISREG(inspected[1]):
                self._record(inspected[0], pretrained=False, extra=True)

    def scan(self) -> dict:
        self._scan_root()
        self._scan_extra()
        base_models = []
        for version, gpt, sovits in _DEFAULT_PAIRS:
            gpt_path = self.pretrained.get(f"{_PRETRAINED}/{gpt}".casefold())
            sovits_path = self.pretrained.get(f"{_PRETRAINED}/{sovits}".casefold())
            # 缺少任何一侧都不能伪造可用的底模配对。
            if gpt_path is not None and sovits_path is not None:
                base_models.append({"id": version, "label": f"{version} 底模",
                                    "gpt_path": gpt_path, "sovits_path": sovits_path})
        gpt_weights = sorted(self.gpt.values(), key=lambda item: item["label"].casefold())
        sovits_weights = sorted(self.sovits.values(), key=lambda item: item["label"].casefold())
        lora_weights = sorted(self.lora.values(), key=lambda item: item["label"].casefold())
        if gpt_weights or sovits_weights or lora_weights:
            note = "已扫描配置的本地目录，仅展示实际存在的模型文件。"
        else:
            note = "指定目录内未发现模型文件。"
        if self.root is not None:
            note += "根目录只检查 GPT_SoVITS/pretrained_models、GPT_weights* 和 SoVITS_weights*；"
        if self.extra_dir is not None:
            note += "补充目录仅扫描第一层 .ckpt/.pth 文件，不递归子目录；"
        note += "底模配对依据默认配置，不代表服务端当前加载状态。"
        if self.path_limit:
            note += f"已达到 {_MAX_VISITED_PATHS} 个路径的访问上限，结果可能不完整。"
        if self.model_limits:
            note += f"每类最多展示 {_MAX_MODELS_PER_KIND} 个权重，超出部分已省略。"
        if self.skipped_links:
            note += "已跳过符号链接、目录联接或越界路径。"
        if self.skipped_long_paths:
            note += f"已跳过超过 {_MAX_PATH_CHARS} 字符的权重路径。"
        if self.unrecognized_sovits:
            note += "已跳过无法识别或头部不完整的 SoVITS 权重，未猜测其版本或 LoRA 类型。"
        result = {"base_models": base_models[:_MAX_MODELS_PER_KIND],
                  "gpt_weights": gpt_weights, "sovits_weights": sovits_weights,
                  "lora_weights": lora_weights, "note": note}
        self._bound_result(result)
        return result

    @staticmethod
    def _bound_result(result: dict) -> None:
        # 保守采用 ASCII 转义和默认分隔符计费；实际 UTF-8 JSON 只会更小。
        def size(value) -> int:
            return len(json.dumps(value, ensure_ascii=True).encode("utf-8"))

        empty = {"base_models": [], "gpt_weights": [], "sovits_weights": [],
                 "lora_weights": [], "note": result["note"]}
        item_sizes = {key: [size(item) for item in result[key]]
                      for key in ("base_models", "gpt_weights", "sovits_weights", "lora_weights")}
        total = size(empty) + sum(sum(items) + max(0, len(items) - 1) * 2
                                  for items in item_sizes.values())
        if total <= _MAX_RESULT_BYTES:
            return
        suffix = "目录响应已达到大小上限，部分列表项已省略；路径本身未截断。"
        total += size(result["note"] + suffix) - size(result["note"])
        result["note"] += suffix
        while total > _MAX_RESULT_BYTES:
            available = [key for key in ("gpt_weights", "sovits_weights", "lora_weights") if item_sizes[key]]
            if not available:
                available = ["base_models"] if item_sizes["base_models"] else []
            if not available:
                break
            key = max(available, key=lambda name: (len(item_sizes[name]), sum(item_sizes[name])))
            count = len(item_sizes[key])
            total -= item_sizes[key].pop() + (2 if count > 1 else 0)
            result[key].pop()


def _directory(value: str, label: str) -> Path | None:
    if not value.strip():
        return None
    if "\x00" in value:
        raise ValueError(f"{label}不能包含空字符")
    path = Path(value)
    if not path.is_absolute() or path.drive.startswith(("\\\\", "//")):
        raise ValueError(f"请明确填写本地{label}的绝对路径，不支持网络目录")
    if path.parent == path:
        raise ValueError(f"{label}不能是磁盘或文件系统根目录，请指定具体文件夹")
    try:
        info = path.lstat()
    except (OSError, ValueError) as exc:
        raise ValueError(f"{label}不存在或无法访问") from exc
    if (stat.S_ISLNK(info.st_mode)
            or getattr(info, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)):
        raise ValueError(f"{label}不能是符号链接或目录联接")
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"{label}必须是文件夹")
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"{label}无法解析，请检查路径、权限或目录联接") from exc
    if resolved.parent == resolved:
        raise ValueError(f"{label}不能是磁盘或文件系统根目录")
    # 同时拒绝藏在父路径中的链接/联接；abspath 仅作词法规范化，不遍历其他目录。
    if resolved != Path(os.path.abspath(path)):
        raise ValueError(f"{label}不能经过符号链接、目录联接或越界路径")
    return resolved


def scan_models(root: str, extra_dir: str = "") -> dict:
    """返回本地可用底模配对及独立权重；不探测其他目录，不连接远程服务。

    root 保持 GPT-SoVITS 安装根目录语义；extra_dir 为可选的本地绝对目录，
    仅扫描其第一层 .ckpt/.pth 文件，不递归。空字符串表示该来源未配置。
    .pth 仅读取最多 8192 字节以识别完整权重/LoRA；不反序列化或导入模型源码。
    lora_weights 含 label/path/version；sovits_weights 不含已识别的 LoRA，
    未知 PK 权重的 version/mode 均为 unknown，不按文件名或大小推断。
    """
    for value, label in ((root, "GPT-SoVITS 根目录"), (extra_dir, "补充模型目录")):
        if not isinstance(value, str):
            raise ValueError(f"{label}必须是字符串")
    # 两者均为空时在构造 Path、stat 或 scandir 之前返回。
    if not root.strip() and not extra_dir.strip():
        return {"base_models": [], "gpt_weights": [], "sovits_weights": [],
                "lora_weights": [], "note": "尚未配置本地 GPT-SoVITS 根目录或补充模型目录，未执行扫描；远程服务可手动填写权重路径。"}
    root_path = _directory(root, "GPT-SoVITS 根目录")
    extra_path = _directory(extra_dir, "补充模型目录")
    try:
        return _Scanner(root_path, extra_path).scan()
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError("扫描模型目录失败，请检查目录权限、路径和文件状态") from exc


def _linked_or_junction(info) -> bool:
    return bool(stat.S_ISLNK(info.st_mode)
                or getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def scan_reference_audios(root: str) -> dict:
    """列出 GPT-SoVITS 安装目录下 maibot_refs 中的参考音频（含两层子文件夹）。

    只列文件名与路径，不读取音频内容；跳过符号链接、目录联接和隐藏项。
    root 为空或文件夹不存在时返回空列表并说明原因，不创建任何目录。
    """
    if not isinstance(root, str):
        raise ValueError("GPT-SoVITS 根目录必须是字符串")
    empty = {"reference_audios": [], "reference_dir": ""}
    if not root.strip():
        return {**empty, "reference_note": "尚未填写本机模型目录，无法查找参考音频。"}
    root_path = _directory(root, "GPT-SoVITS 根目录")
    folder = root_path / REFS_DIRNAME
    result = {**empty, "reference_dir": folder.as_posix()}
    try:
        info = folder.lstat()
    except FileNotFoundError:
        return {**result, "reference_note": f"参考音频文件夹不存在：{folder.as_posix()}。请在 GPT-SoVITS 安装目录下新建 {REFS_DIRNAME} 文件夹并放入参考音频。"}
    except OSError as exc:
        raise ValueError("参考音频文件夹无法访问，请检查权限") from exc
    if _linked_or_junction(info):
        return {**result, "reference_note": f"{REFS_DIRNAME} 不能是符号链接或目录联接。"}
    if not stat.S_ISDIR(info.st_mode):
        return {**result, "reference_note": f"{folder.as_posix()} 不是文件夹。"}
    items, visited, limited = [], 0, False

    def walk(directory: Path, depth: int) -> None:
        nonlocal visited, limited
        with os.scandir(directory) as entries:
            for entry in sorted(entries, key=lambda item: item.name.casefold()):
                if visited >= _MAX_VISITED_PATHS or len(items) >= _MAX_REFERENCE_AUDIOS:
                    limited = True
                    return
                visited += 1
                if entry.name.startswith("."):
                    continue
                entry_info = entry.stat(follow_symlinks=False)
                if _linked_or_junction(entry_info):
                    continue
                path = Path(entry.path)
                if stat.S_ISDIR(entry_info.st_mode):
                    if depth < _MAX_REFERENCE_DEPTH:
                        walk(path, depth + 1)
                elif stat.S_ISREG(entry_info.st_mode) and path.suffix.lower() in _AUDIO_SUFFIXES:
                    if len(path.as_posix()) <= _MAX_PATH_CHARS:
                        relative = path.parent.relative_to(folder).as_posix()
                        label = REFS_DIRNAME if relative == "." else f"{REFS_DIRNAME}/{relative}"
                        items.append({"path": path.as_posix(), "label": label})

    try:
        walk(folder, 0)
    except OSError as exc:
        raise ValueError("扫描参考音频文件夹失败，请检查权限") from exc
    if items:
        note = f"在 {folder.as_posix()} 中找到 {len(items)} 个参考音频。"
    else:
        note = f"{folder.as_posix()} 中还没有参考音频（支持 wav、mp3、flac、ogg、m4a），放入后点击“刷新参考音频”。"
    if limited:
        note += f"只列出前 {_MAX_REFERENCE_AUDIOS} 个。"
    return {"reference_audios": items, "reference_dir": folder.as_posix(), "reference_note": note}
