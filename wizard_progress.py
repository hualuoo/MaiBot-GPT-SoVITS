"""Cheap, restart-safe wizard progress; never opens model or reference assets."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import threading
import time
import uuid
from urllib.parse import urlsplit, urlunsplit

_MAX_BYTES = 64 * 1024
_PRESETS = frozenset(("v1", "v2", "v3", "v4", "v2Pro", "v2ProPlus", "v5dev", "v5turbo"))
_EMPTY = frozenset(value.casefold() for value in ("", "（请选择模型）", "（不使用 LoRA）", "请选择模型", "请选择角色", "（请选择角色）", "未选择", "none", "null", "undefined"))
_LOCK = threading.RLock()


def _text(value) -> str:
    return str(value or "").strip()


def _selected(value) -> str:
    value = _text(value)
    if value.casefold() in _EMPTY or (value.startswith(("（请选择", "请选择", "(请选择"))):
        return ""
    return value


def _url(value) -> str:
    """Canonical endpoint identity, independent of timeout and trailing slash."""
    value = _text(value)
    try:
        parts = urlsplit(value)
        scheme, host = parts.scheme.lower(), (parts.hostname or "").lower()
        if scheme not in ("http", "https") or not host or parts.username or parts.password:
            return ""
        port = parts.port
        host = "[" + host + "]" if ":" in host else host
        if port is not None and (scheme, port) not in (("http", 80), ("https", 443)):
            host += ":" + str(port)
        return urlunsplit((scheme, host, parts.path.rstrip("/"), parts.query, ""))
    except ValueError:
        return ""


def _linked(info) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


class WizardProgress:
    """Only data/.state/wizard-progress.json is persistent state.

    mark(1..6) records a successful user action, not runtime liveness. compute
    reads bounded metadata every time so independently constructed instances agree.
    Stored fingerprints contain no original URL, filesystem path or prompt text.

    Marker keys: 1 API, 2 base model, 3 voice mode, 4 parameters, 5 optional
    preview, 6 auto voice. compute() reports the five page steps: 1 API, 2 base,
    3 voice mode, 4 parameters & optional preview (marker 4 only), 5 auto voice
    (marker 6).
    """

    def __init__(self, settings_path: Path):
        self._settings_path = Path(os.path.abspath(os.fspath(settings_path)))
        self._root = self._settings_path.parent
        self.path = self._root / "data" / ".state" / "wizard-progress.json"

    def _directories(self, create: bool = False) -> bool:
        # lstat every ancestor: Path.resolve() alone would silently follow junctions.
        directories = list(reversed(self._root.parents)) + [self._root, self._root / "data", self.path.parent]
        for directory in directories:
            try:
                info = directory.lstat()
            except FileNotFoundError:
                if not create or directory not in (self._root / "data", self.path.parent):
                    return False
                try:
                    directory.mkdir()
                except FileExistsError:
                    pass
                info = directory.lstat()
            if _linked(info) or not stat.S_ISDIR(info.st_mode):
                raise OSError("向导进度目录不能是符号链接、目录联接或非目录")
        return True

    def _read(self) -> tuple[dict, str]:
        try:
            if not self._directories():
                return {}, ""
            try:
                info = self.path.lstat()
            except FileNotFoundError:
                return {}, ""
            if _linked(info) or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ValueError("进度文件不是独立普通文件")
            if info.st_size > _MAX_BYTES:
                raise ValueError("进度文件超过64KiB")
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0) | getattr(os, "O_NONBLOCK", 0)
            with os.fdopen(os.open(self.path, flags), "rb") as stream:
                opened = os.fstat(stream.fileno())
                if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1 or opened.st_ino != info.st_ino:
                    raise ValueError("进度文件已被替换")
                content = stream.read(_MAX_BYTES + 1)
            if len(content) > _MAX_BYTES:
                raise ValueError("进度文件超过64KiB")
            document = json.loads(content)
            if not isinstance(document, dict) or document.get("version") != 1 or not isinstance(document.get("steps"), dict):
                raise ValueError("进度格式无效")
            markers = {}
            for key, marker in document["steps"].items():
                if key not in ("1", "2", "3", "4", "5", "6") or not isinstance(marker, dict):
                    raise ValueError("进度步骤无效")
                fingerprint, timestamp = marker.get("fingerprint"), marker.get("timestamp")
                if not isinstance(fingerprint, str) or not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
                    raise ValueError("进度指纹无效")
                if type(timestamp) not in (int, float) or not 0 < timestamp < 1e12:
                    raise ValueError("进度时间无效")
                markers[key] = {"fingerprint": fingerprint, "timestamp": timestamp}
            if not self._directories():
                raise OSError("进度目录已改变")
            return markers, ""
        except (OSError, ValueError, TypeError, RecursionError):
            return {}, "向导进度记录不可读或已损坏；请重新确认未完成的步骤。"

    @staticmethod
    def _values(step: int, cfg):
        models, role, voice, behavior = cfg.models, cfg.role, cfg.voice, cfg.behavior
        if step == 1:
            return _url(cfg.api.base_url)
        if step == 2:
            return [_text(models.root), _text(models.base_model)]
        if step == 3:
            if role.mode == "role":
                selected = list(role.emotions) or [role.emotion]
                return ["role", _text(role.role_id), _text(role.emotion), sorted(selected),
                        _text(models.base_model), _text(models.root)]
            return [role.mode, _text(models.base_model), _selected(models.gpt_weights), _selected(models.sovits_weights),
                    _selected(models.lora_weights), _text(voice.ref_audio_path),
                    voice.prompt_text, _text(voice.prompt_lang), _text(voice.text_lang),
                    list(voice.aux_ref_audio_paths)]
        if step == 4:
            return [voice.speed_factor, voice.top_k, voice.top_p, voice.temperature, voice.seed,
                    voice.text_split_method, voice.parallel_infer, behavior.max_text_length,
                    behavior.cooldown_seconds, behavior.send_text]
        if step == 5:
            return [WizardProgress._values(1, cfg), WizardProgress._values(2, cfg), WizardProgress._values(3, cfg), WizardProgress._values(4, cfg)]
        if step == 6:
            return cfg.auto_voice.model_dump(mode="json")
        raise ValueError("只有向导第1至6步可以保存完成标记")

    @classmethod
    def _fingerprint(cls, step: int, cfg) -> str:
        data = json.dumps(cls._values(step, cfg), ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    def mark(self, step: int, cfg) -> None:
        """Synchronously record a successful action using the final saved config."""
        if type(step) is not int or step not in (1, 2, 3, 4, 5, 6):
            raise ValueError("只有向导第1至6步可以保存完成标记")
        fingerprint = self._fingerprint(step, cfg)
        with _LOCK:
            if not self._directories(create=True):
                raise OSError("插件目录不存在，无法保存向导进度")
            markers, _ = self._read()
            # Never replace a link, even when _read deliberately ignored it.
            try:
                info = self.path.lstat()
            except FileNotFoundError:
                pass
            else:
                if _linked(info) or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    raise OSError("拒绝替换链接或非普通向导进度文件")
            markers[str(step)] = {"fingerprint": fingerprint, "timestamp": time.time()}
            content = json.dumps({"version": 1, "steps": markers}, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
            if len(content) > _MAX_BYTES:
                raise ValueError("向导进度超过64KiB")
            temporary = self.path.with_name(".wizard-progress-" + uuid.uuid4().hex + ".tmp")
            try:
                flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
                with os.fdopen(os.open(temporary, flags, 0o600), "wb") as stream:
                    stream.write(content)
                    stream.flush()
                    os.fsync(stream.fileno())
                if not self._directories():
                    raise OSError("进度目录已改变")
                os.replace(temporary, self.path)
            finally:
                # Do not follow a concurrently replaced directory during cleanup.
                if self._directories():
                    temporary.unlink(missing_ok=True)

    def compute(self, cfg, connection: dict = None, role_status: dict = None, job: dict = None) -> dict:
        """Infer navigation only: no probes, scans, asset reads, or config mutation."""
        markers, warning = self._read()
        matches = {step: markers.get(str(step), {}).get("fingerprint") == self._fingerprint(step, cfg)
                   for step in range(1, 7)}
        connection, role_status, job = connection or {}, role_status or {}, job or {}
        api_live = connection.get("status") == "success"
        api_complete = bool(_url(cfg.api.base_url)) and (matches[1] or api_live)
        models, role, voice = cfg.models, cfg.role, cfg.voice
        explicit_weights = bool(_selected(models.gpt_weights) and _selected(models.sovits_weights))
        preset = _text(models.base_model) in _PRESETS and bool(_selected(models.root))
        base_valid = explicit_weights if models.base_model == "custom" else preset
        # 没有第二、三步记录时按已保存的配置推断；有记录时必须与记录一致。
        base_complete = bool(base_valid and (matches[2] or "2" not in markers))
        role_mode = role.mode == "role"
        role_live = False
        if role_mode:
            selected = list(role.emotions) or [role.emotion]
            role_valid = bool(_selected(role.role_id)) and bool(selected) and all(_selected(item) for item in selected)
            role_valid = role_valid and role.emotion in selected and len(set(selected)) == len(selected)
            profiles = role_status.get("profiles")
            role_live = (role_status.get("status") == "success" and isinstance(profiles, dict)
                         and all(item in profiles and isinstance(profiles[item], dict) for item in selected))
            voice_complete = bool(role_valid and (role_live or matches[3]))
        else:
            # v3/v4/v5 require the reference transcript; v1/v2 allow text-free refs.
            requires_prompt = _text(models.base_model) in ("v3", "v4", "v5dev", "v5turbo")
            prompt_valid = bool(_text(voice.prompt_text)) or not requires_prompt
            voice_valid = (role.mode == "original" and (explicit_weights or preset)
                           and bool(_selected(voice.ref_audio_path)) and prompt_valid
                           and bool(_selected(voice.prompt_lang)) and bool(_selected(voice.text_lang)))
            voice_complete = bool(voice_valid and (matches[3] or "3" not in markers))
        completed = [step for step, done in enumerate(
            (api_complete, base_complete, voice_complete, matches[4]), start=1) if done]
        needs_revalidate = bool((matches[1] and not api_live)
                                or (role_mode and matches[3] and not role_live))
        next_step = next((step for step in range(1, 5) if step not in completed), 5)
        messages = {1: "请保存接口地址并测试连接。", 2: "请选择底模；自定义模式需要填写GPT与SoVITS权重。",
                    3: "请选择并读取全部角色语气。" if role_mode else "请确认普通模式权重、参考音频、文本与语言。",
                    4: "请确认并保存推理参数（默认参数也需要保存一次）；试听可选。",
                    5: "五步配置已完成，自动语音按已保存设置工作。" if matches[6]
                    else "请配置自动语音回复；可保持关闭，仅使用命令和语音工具。"}
        # marker 5 记录可选的试听；第五步“自动语音回复”对应 marker 6。
        if next_step == 5 and matches[6]:
            completed.append(5)
        message = messages[next_step]
        if job.get("status") in ("running", "pending", "queued"):
            if job.get("kind") == "prepare_role":
                next_step, message = 3, "正在读取角色并校验所选语气，请等待任务完成。"
            elif job.get("kind") in ("load", "synth", "synthesize", "load_models", "synthesis"):
                next_step, message = 4, "模型加载或试听任务正在进行，请等待任务完成。"
        if needs_revalidate:
            hints = []
            if matches[1] and not api_live:
                hints.append("接口曾测试通过，本次运行尚未重新确认连接")
            if role_mode and matches[3] and not role_live:
                hints.append("角色曾准备完成，试听前请重新读取角色以校验资源")
            message += " " + "；".join(hints) + "。"
        # saved_steps：当前配置与该步最近一次成功操作的指纹一致（不受前面步骤是否完成影响）。
        result = {"next_step": next_step, "message": message, "completed_steps": completed,
                  "saved_steps": [step for step, marker in ((1, 1), (2, 2), (3, 3), (4, 4), (5, 6)) if matches[marker]],
                  "preview_done": matches[5], "needs_revalidate": needs_revalidate}
        if warning:
            result["warning"] = warning
        return result
