"""每次语音合成在 MaiBot 日志里输出“开始”和“结果”两行。

插件日志会经 MaiBot 的插件运行时转发到主程序，终端和 WebUI「日志」页都能看到。
一次合成的信息（来源、聊天、角色与情感、模型切换、各阶段耗时、音频大小）先记录在
Trace 上，结束时汇总成一行；当前 Trace 通过 contextvars 传递，不改变合成函数签名。
"""
from __future__ import annotations

from collections import OrderedDict
import contextvars
from dataclasses import dataclass, field
import io
import re
import time
import wave

PREFIX = "[语音合成]"
TEXT_PREVIEW = 40
DETAIL_LIMIT = 120
CHAT_LABEL_LIMIT = 256
SOURCE_LABELS = {"command": "命令 /tts", "tool": "语音工具", "auto": "自动语音",
                 "preview": "设置页试听", "load": "设置页加载模型"}
EMOTION_SOURCE_LABELS = {"llm": "LLM 判断", "requested": "指定", "metadata": "消息情感标记",
                         "keyword": "关键词规则", "default": "默认情感"}
EMOTION_LABELS = {"happy": "开心", "angry": "生气", "surprised": "吃惊", "neutral": "中立", "sad": "难过"}

_current: contextvars.ContextVar = contextvars.ContextVar("gpt_sovits_tts_trace", default=None)


def current() -> "Trace | None":
    return _current.get()


def activate(trace: "Trace"):
    return _current.set(trace)


def deactivate(token) -> None:
    _current.reset(token)


def preview(text, limit: int = TEXT_PREVIEW) -> str:
    text = re.sub(r"\s+", " ", text if isinstance(text, str) else "").strip()
    return text if len(text) <= limit else text[:limit] + "…"


def short(text, limit: int = DETAIL_LIMIT) -> str:
    return preview(str(text or ""), limit)


def filename(path) -> str:
    value = str(path or "").replace("\\", "/").rstrip("/")
    return value.rsplit("/", 1)[-1] or value


def seconds(value) -> str:
    return f"{value:.1f} 秒"


def size(count: int) -> str:
    return f"{count / 1024 / 1024:.1f} MB" if count >= 1024 * 1024 else f"{max(1, round(count / 1024))} KB"


def audio_info(data) -> tuple[float | None, int]:
    if not isinstance(data, (bytes, bytearray)) or not data:
        return None, 0
    try:
        with wave.open(io.BytesIO(bytes(data)), "rb") as audio:
            rate = audio.getframerate()
            return (audio.getnframes() / rate if rate else None), len(data)
    except Exception:
        return None, len(data)


def emotion_name(emotion_id: str) -> str:
    return EMOTION_LABELS.get(emotion_id, emotion_id)


def role_name(role_id: str) -> str:
    """role_id 形如 黍/黍.json，只显示角色目录名。"""
    return str(role_id or "").replace("\\", "/").split("/", 1)[0]


def chat_label(message, outgoing: bool = False) -> str:
    """从 MaiBot 消息字典取出“群 名称（群号）”或“私聊 昵称（QQ）”；取不到时返回空字符串。"""
    if not isinstance(message, dict):
        return ""
    info = message.get("message_info") if isinstance(message.get("message_info"), dict) else {}
    group = info.get("group_info") if isinstance(info.get("group_info"), dict) else {}
    group_id, group_name = str(group.get("group_id") or "").strip(), str(group.get("group_name") or "").strip()
    if group_id or group_name:
        return "群 " + (f"{group_name}（{group_id}）" if group_id and group_name else group_name or group_id)
    if outgoing:
        # 机器人自己发出的私聊消息里 user_info 是机器人，只能拿到目标 QQ。
        extra = info.get("additional_config") if isinstance(info.get("additional_config"), dict) else {}
        target = str(extra.get("platform_io_target_user_id") or "").strip()
        return f"私聊 {target}" if target else ""
    user = info.get("user_info") if isinstance(info.get("user_info"), dict) else {}
    user_id = str(user.get("user_id") or "").strip()
    nickname = str(user.get("user_cardname") or user.get("user_nickname") or "").strip()
    if user_id or nickname:
        return "私聊 " + (f"{nickname}（{user_id}）" if user_id and nickname else nickname or user_id)
    return ""


def remember_chat(owner, stream_id: str, label: str) -> None:
    """记住聊天流对应的群名/私聊对象，供拿不到消息的语音工具调用显示。"""
    if not stream_id or not label:
        return
    labels = getattr(owner, "_tts_chat_labels", None)
    if labels is None:
        labels = OrderedDict()
        setattr(owner, "_tts_chat_labels", labels)
    labels[stream_id] = label
    labels.move_to_end(stream_id)
    while len(labels) > CHAT_LABEL_LIMIT:
        labels.popitem(last=False)


def known_chat(owner, stream_id: str) -> str:
    labels = getattr(owner, "_tts_chat_labels", None) or {}
    return labels.get(stream_id, "")


@dataclass
class Trace:
    source: str
    stream_id: str = ""
    chat: str = ""
    text: str = ""
    started: float = field(default_factory=time.monotonic)
    begun: bool = False
    mode: str = ""
    role: str = ""
    emotion_count: int = 0
    emotion: str = ""
    emotion_source: str = ""
    llm_task: str = ""
    llm_model: str = ""
    llm_seconds: float | None = None
    llm_cached: bool = False
    llm_error: str = ""
    fallback_source: str = ""
    switched: list = field(default_factory=list)
    switch_seconds: float | None = None
    weights_note: str = ""
    synth_seconds: float | None = None
    audio: bytes = b""
    outcome: str = ""

    def set_role(self, cfg) -> None:
        self.mode = cfg.role.mode
        if cfg.role.mode == "role":
            self.role = role_name(cfg.role.role_id)
            self.emotion_count = len(cfg.role.emotions or [cfg.role.emotion])

    def set_emotion(self, emotion: str, source: str, *, task: str = "", model: str = "",
                    elapsed: float | None = None, cached: bool = False, error: str = "") -> None:
        if source == "llm_failed":
            # 记下回退到的是哪种结果：自动语音会先按情感标记/关键词选好，再请 LLM 判断。
            same = emotion == self.emotion and self.emotion_source not in ("", "llm", "llm_failed")
            self.fallback_source = self.emotion_source if same else "default"
        self.emotion, self.emotion_source = emotion, source
        self.llm_task, self.llm_model, self.llm_seconds = task, model, elapsed
        self.llm_cached, self.llm_error = cached, error

    def record_switch(self, gpt_path: str, acoustic_path: str, lora: bool, elapsed: float) -> None:
        self.switched = [("GPT", filename(gpt_path)), ("LoRA" if lora else "SoVITS", filename(acoustic_path))]
        self.switch_seconds = elapsed

    # ---------- 文本 ----------
    def _where(self) -> str:
        if self.chat:
            return self.chat
        return f"聊天 {self.stream_id[:8]}…" if len(self.stream_id) > 8 else (f"聊天 {self.stream_id}" if self.stream_id else "")

    def _head(self) -> list[str]:
        parts = [SOURCE_LABELS.get(self.source, self.source)]
        if self._where():
            parts.append(self._where())
        if self.mode == "role":
            parts.append(f"角色 {self.role or '未选择'}")
        elif self.mode == "original":
            parts.append("普通模式")
        return parts

    def _text_part(self, with_count: bool = False) -> str:
        if not self.text:
            return ""
        count = len(self.text.strip())
        return ("文本 " + (f"{count} 字" if with_count else "")).rstrip() + f"「{preview(self.text)}」"

    def _emotion_part(self) -> str:
        if not self.emotion:
            return ""
        name = emotion_name(self.emotion)
        if self.emotion_source == "llm":
            detail = [part for part in ("/".join(item for item in (self.llm_task, self.llm_model) if item),
                                        "缓存结果" if self.llm_cached else
                                        (seconds(self.llm_seconds) if self.llm_seconds is not None else "")) if part]
            return f"情感 {name}（LLM 判断" + ("，" + "，".join(detail) if detail else "") + "）"
        if self.emotion_source == "llm_failed":
            fallback = EMOTION_SOURCE_LABELS.get(self.fallback_source, "默认情感")
            return f"情感 {name}（LLM 判断失败，改用{fallback}：{short(self.llm_error, 80)}）"
        return f"情感 {name}（{EMOTION_SOURCE_LABELS.get(self.emotion_source, '默认情感')}）"

    def _model_part(self) -> str:
        if self.switched:
            files = "、".join(f"{kind} {name}" for kind, name in self.switched)
            return f"切换模型 {files}" + (f"（{seconds(self.switch_seconds)}）" if self.switch_seconds is not None else "")
        return self.weights_note

    def start_message(self) -> str:
        parts = ["开始", *self._head()]
        if self.mode == "role" and self.emotion_count > 1:
            parts[-1] += f"（启用 {self.emotion_count} 个情感）"
        if self._text_part(True):
            parts.append(self._text_part(True))
        return PREFIX + " " + " · ".join(parts)

    def finish_message(self, status: str, reason: str = "") -> str:
        parts = [status, *self._head()]
        for part in (self._emotion_part(), self._model_part()):
            if part:
                parts.append(part)
        if self.synth_seconds is not None:
            parts.append(f"合成 {seconds(self.synth_seconds)}")
        duration, count = audio_info(self.audio)
        if count:
            parts.append("音频 " + (f"{seconds(duration)} / " if duration is not None else "") + size(count))
        if reason:
            parts.append("原因：" + short(reason))
        if self.outcome:
            parts.append(self.outcome)
        if self.begun:
            parts.append(f"共 {seconds(time.monotonic() - self.started)}")
        if self._text_part():
            parts.append(self._text_part())
        return PREFIX + " " + " · ".join(parts)


def _emit(method, message: str) -> None:
    try:
        method(message)
    except Exception:
        pass  # 日志不可用时不能影响合成


def log_start(logger, trace: Trace) -> None:
    trace.begun = True
    _emit(logger.info, trace.start_message())


def log_success(logger, trace: Trace) -> None:
    _emit(logger.info, trace.finish_message("完成"))


def log_failure(logger, trace: Trace, reason: str) -> None:
    """合成开始后失败记为警告；开始前就被拒绝（冷却、忙碌、参数错误）记为普通信息。"""
    if trace.begun:
        _emit(logger.warning, trace.finish_message("失败", reason))
    else:
        _emit(logger.info, trace.finish_message("未执行", reason))


def log_cancelled(logger, trace: Trace) -> None:
    if trace.begun:
        _emit(logger.info, trace.finish_message("已取消"))


def reason(exc: BaseException) -> str:
    text = str(exc).strip()
    if isinstance(exc, TimeoutError) and not text:
        return "超时"
    return text if type(exc).__name__ == "TTSError" and text else f"{type(exc).__name__}: {text}".rstrip(": ")
