"""In-place outgoing TTS hook; never sends another message.

Rule-based emotion selection is deterministic metadata/keyword matching; for
multi-emotion roles the plugin may additionally ask MaiBot's LLM (emotion_judge.py).
Ambiguous, negated, or unavailable emotions use the configured default.
Each conversion writes a start and a result line to the MaiBot log (tts_log.py).
Only SDK APIs are imported; raw_message follows the real host message_utils
contract (text.data and voice.binary_data_base64/hash).
"""
from __future__ import annotations

import asyncio
import base64
from collections import OrderedDict
import hashlib
import random
import re
import time

from maibot_sdk import HookHandler
from maibot_sdk.types import ErrorPolicy, HookMode, HookOrder

if __package__:
    from . import tts_log
else:
    import tts_log

CACHE_LIMIT = 1024
CACHE_TTL = 3600.0
MARKER = "gpt_sovits_auto_voice"

_ALIASES = {
    "neutral": {"neutral", "normal", "default", "平静", "普通", "中性", "中立", "默认"},
    "happy": {"happy", "joy", "joyful", "positive", "cheerful", "excited", "开心", "高兴", "快乐", "喜悦", "兴奋"},
    "angry": {"angry", "anger", "mad", "furious", "生气", "愤怒", "恼火"},
    "sad": {"sad", "sadness", "unhappy", "negative", "悲伤", "难过", "伤心", "低落"},
    "surprised": {"surprised", "surprise", "amazed", "astonished", "惊讶", "吃惊", "惊喜"},
}
_KEYWORDS = {
    "happy": re.compile(r"开心|高兴|快乐|太棒了|真棒|\b(?:happy|glad|delighted|joyful)\b", re.I),
    "angry": re.compile(r"生气|愤怒|恼火|气死我了|\b(?:angry|furious|outraged)\b", re.I),
    "sad": re.compile(r"难过|伤心|悲伤|失落|\b(?:sad|heartbroken|sorrowful)\b", re.I),
    "surprised": re.compile(r"惊讶|吃惊|没想到|不可思议|\b(?:surprised|astonished|amazed)\b", re.I),
}
_ERROR_TEXT = re.compile(r"^\s*(?:TTS\s*[:：]|TTS\s*(?:执行|异常|错误|失败)|语音(?:合成|发送)(?:失败|异常))", re.I)


def _canonical_emotion(value: str) -> str:
    normalized = value.strip().casefold()
    return next((key for key, aliases in _ALIASES.items() if normalized in aliases), normalized)


def _enabled_emotion(value, enabled):
    if isinstance(value, dict):
        value = next((value[key] for key in ("emotion_id", "id", "name", "label", "value") if key in value), "")
    if not isinstance(value, str) or not value.strip():
        return None
    value = value.strip()
    for item in enabled:
        if item.casefold() == value.casefold():
            return item
    canonical = _canonical_emotion(value)
    return next((item for item in enabled if _canonical_emotion(item) == canonical), None)


def _negated(text: str, start: int) -> bool:
    # Keep a small, clause-local window: no sentiment or semantic model is used.
    prefix = re.split(r"[，。！？,;.!?\n]", text[max(0, start - 40):start])[-1]
    return bool(re.search(r"(?:不|没|未|别|非|无)(?:[\u4e00-\u9fff]{0,5})$", prefix) or
                re.search(r"\b(?:not|never|no|hardly|without|\w+n't)(?:\W+\w+){0,3}\W*$", prefix, re.I))


def classify_emotion(text, message, cfg) -> tuple[str, str]:
    """Rule-based emotion: (emotion, source) where source is metadata/keyword/default.

    Metadata keys emotion/emotion_id/sentiment are checked in that order.
    Existing but unknown/disabled metadata does not fall through to keywords.
    Multiple keyword categories, negation, or no match means the default.
    This is rule matching; multi-emotion roles may additionally ask MaiBot's LLM
    (see emotion_judge.py) and use this result only as the fallback.
    """
    if cfg.role.mode != "role":
        return "", "default"
    default = cfg.role.emotion
    if not cfg.auto_voice.follow_emotion:
        return default, "default"
    enabled = cfg.role.emotions or [default]
    message = message if isinstance(message, dict) else {}
    info = message.get("message_info") or {}
    additional = info.get("additional_config", {}) if isinstance(info, dict) else {}
    sources = [message.get("metadata", {}), additional, message]
    if isinstance(additional, dict):
        sources.insert(1, additional.get("metadata", {}))
    for key in ("emotion", "emotion_id", "sentiment"):
        for source in sources:
            if isinstance(source, dict) and key in source and source[key] not in (None, ""):
                return _enabled_emotion(source[key], enabled) or default, "metadata"
    matches = {key for key, pattern in _KEYWORDS.items()
               if any(not _negated(text, match.start()) for match in pattern.finditer(text))}
    if len(matches) != 1:
        return default, "default"
    selected = _enabled_emotion(matches.pop(), enabled)
    return (selected, "keyword") if selected else (default, "default")


def choose_emotion(text, message, cfg) -> str:
    """Compatibility wrapper: rule-based emotion only."""
    return classify_emotion(text, message, cfg)[0]


def _without_voice_reply(message, send_kwargs):
    """语音不与指定消息引用混发；同时清理已存在引用和Host稍后补引用的开关。"""
    segments = message.get("raw_message", [])
    cleaned = dict(message)
    cleaned["raw_message"] = [segment for segment in segments if segment.get("type") != "reply"]
    cleaned["reply_to"] = None
    modified = dict(send_kwargs, message=cleaned, set_reply=False, reply_message_id=None)
    if "reply_message" in modified:
        modified["reply_message"] = None
    return modified


def _has_voice_reply(message, send_kwargs):
    if not isinstance(message, dict) or str(message.get("platform", "")).lower() != "qq":
        return False
    segments = message.get("raw_message")
    if not isinstance(segments, list) or not all(isinstance(segment, dict) for segment in segments):
        return False
    return any(segment.get("type") == "voice" for segment in segments) and (
        bool(send_kwargs.get("set_reply")) or bool(send_kwargs.get("reply_message_id"))
        or bool(message.get("reply_to")) or any(segment.get("type") == "reply" for segment in segments))


class AutoVoiceMixin:
    """Requires pure _synthesize_audio(cfg,text,emotion), _check_access and lifecycle flags."""

    def _init_auto_voice(self):
        self._auto_voice_streams = OrderedDict()
        self._auto_voice_seen = OrderedDict()
        self._auto_voice_suppress = {}
        self._auto_stats = {"considered": 0, "converted": 0, "skipped": 0, "failed": 0,
                            "last_reason": "not_started", "last_emotion": "", "last_emotion_source": ""}

    def _auto_voice_prune(self, now):
        for key, state in list(self._auto_voice_streams.items()):
            if now - state["touched"] >= CACHE_TTL:
                self._auto_voice_streams.pop(key, None)
        for key, expiry in list(self._auto_voice_seen.items()):
            if expiry <= now:
                self._auto_voice_seen.pop(key, None)
        for key, expiry in list(self._auto_voice_suppress.items()):
            if expiry <= now:
                self._auto_voice_suppress.pop(key, None)
        for cache in (self._auto_voice_streams, self._auto_voice_seen):
            while len(cache) > CACHE_LIMIT:
                cache.popitem(last=False)
        while len(self._auto_voice_suppress) > CACHE_LIMIT:
            self._auto_voice_suppress.pop(next(iter(self._auto_voice_suppress)))

    def _auto_voice_skip(self, reason, original):
        self._auto_stats["skipped"] += 1
        self._auto_stats["last_reason"] = reason
        return {"action": "continue", "modified_kwargs": original}

    @HookHandler("send_service.before_send", name="gpt_sovits_auto_voice",
                 description="按频率将普通文字就地转为语音，失败保留原文；关键词情感非AI语义识别",
                 mode=HookMode.BLOCKING, order=HookOrder.LATE, timeout_ms=300000, error_policy=ErrorPolicy.SKIP)
    async def handle_auto_voice(self, message=None, hook_name="", **kwargs):
        # modified_kwargs REPLACES the full host argument dictionary. Preserve all
        # current/future send kwargs, but hook_name is injected invocation metadata.
        original = dict(kwargs, message=message)
        # QQ 语音不能附带引用：已有语音也要去掉引用，与自动转换开关无关。
        if _has_voice_reply(message, original):
            return {"action": "continue", "modified_kwargs": _without_voice_reply(message, original)}
        trace = None
        try:
            cfg = self.config.model_copy(deep=True)
            if not cfg.auto_voice.enabled:
                return self._auto_voice_skip("disabled", original)
            if not self._active:
                return self._auto_voice_skip("inactive", original)
            if not isinstance(message, dict):
                return self._auto_voice_skip("invalid_message", original)
            stream_id = str(message.get("session_id") or "")
            try:
                self._check_access(stream_id, cfg)
            except asyncio.CancelledError:
                raise
            except Exception:
                return self._auto_voice_skip("access", original)
            now = time.monotonic()
            self._auto_voice_prune(now)
            info = message.get("message_info") or {}
            additional = (info.get("additional_config") or {}) if isinstance(info, dict) else {}
            if isinstance(additional, dict) and additional.get(MARKER):
                return self._auto_voice_skip("already_converted", original)
            message_id = str(message.get("message_id") or "")
            if message_id:
                seen_key = (stream_id, message_id)
                if seen_key in self._auto_voice_seen:
                    return self._auto_voice_skip("duplicate", original)
                self._auto_voice_seen[seen_key] = now + CACHE_TTL
                while len(self._auto_voice_seen) > CACHE_LIMIT:
                    self._auto_voice_seen.popitem(last=False)
            segments = message.get("raw_message")
            if not isinstance(segments, list) or not segments:
                return self._auto_voice_skip("no_text", original)
            if message.get("is_command"):
                return self._auto_voice_skip("command", original)
            if any(not isinstance(seg, dict) or seg.get("type") not in {"text", "reply", "at"} for seg in segments):
                return self._auto_voice_skip("non_text", original)
            texts = [seg.get("data") for seg in segments if seg.get("type") == "text"]
            if not texts or any(not isinstance(text, str) for text in texts):
                return self._auto_voice_skip("no_text", original)
            text = "".join(texts)
            if not text.strip():
                return self._auto_voice_skip("no_text", original)
            if len(text) > cfg.behavior.max_text_length:
                return self._auto_voice_skip("too_long", original)
            if text.lstrip().startswith("/") or "```" in text or "~~~" in text:
                return self._auto_voice_skip("command_or_code", original)
            if _ERROR_TEXT.search(text):
                return self._auto_voice_skip("tts_error", original)
            if self._auto_voice_suppress.get((stream_id, text.strip()), 0) > now:
                return self._auto_voice_skip("own_speech", original)
            state = self._auto_voice_streams.get(stream_id)
            # Interval is checked before frequency and busy; no inference queue.
            if state and now - state["last_converted"] < cfg.auto_voice.min_interval_seconds:
                return self._auto_voice_skip("interval", original)
            if self._busy:
                return self._auto_voice_skip("busy", original)
            if state is None:
                state = {"count": 0, "last_converted": float("-inf"), "touched": now}
                self._auto_voice_streams[stream_id] = state
            state["touched"] = now
            self._auto_voice_streams.move_to_end(stream_id)
            while len(self._auto_voice_streams) > CACHE_LIMIT:
                self._auto_voice_streams.popitem(last=False)
            self._auto_stats["considered"] += 1
            state["count"] += 1
            if cfg.auto_voice.frequency_mode == "every_n":
                selected = state["count"] % cfg.auto_voice.every_n == 0
            else:
                percent = cfg.auto_voice.probability_percent
                selected = percent >= 100 or (percent > 0 and random.random() < percent / 100)
            if not selected:
                return self._auto_voice_skip("frequency", original)
            emotion, emotion_source = classify_emotion(text, message, cfg)
            self._auto_stats["last_emotion"] = emotion
            self._auto_stats["last_emotion_source"] = emotion_source
            # All checks and the busy acquisition are synchronous: never wait for
            # another synthesis or setup task to finish and never recurse through send.
            self._busy = True
            task = asyncio.current_task()
            if task:
                self._tasks.add(task)
            chat = tts_log.chat_label(message, outgoing=True) or tts_log.known_chat(self, stream_id)
            trace = tts_log.Trace("auto", stream_id=stream_id, chat=chat, text=text)
            trace.set_role(cfg)
            if cfg.role.mode == "role":
                trace.set_emotion(emotion, emotion_source)
            token = tts_log.activate(trace)
            try:
                tts_log.log_start(self._get_logger(), trace)
                # 多情感角色：消息没有显式情感标记时请 MaiBot 的 LLM 判断，失败回退到上面的规则结果。
                judge = getattr(self, "_judge_emotion", None)
                if judge is not None and emotion_source != "metadata" and cfg.auto_voice.follow_emotion \
                        and self._llm_emotion_active(cfg):
                    emotion, emotion_source = await judge(text, cfg, emotion)
                    self._auto_stats["last_emotion"] = emotion
                    self._auto_stats["last_emotion_source"] = emotion_source
                started = time.monotonic()
                async with asyncio.timeout(min(cfg.api.timeout, 240)):
                    audio = await self._synthesize_audio(cfg, text, emotion=emotion)
                if trace.synth_seconds is None:
                    trace.synth_seconds = time.monotonic() - started
                trace.audio = audio if isinstance(audio, bytes) else b""
                if not isinstance(audio, bytes) or not audio:
                    raise ValueError("合成未返回有效音频")
                if len(audio) > min(cfg.api.max_audio_bytes, 8388608):
                    raise ValueError("音频超过配置或RPC安全大小上限")
                voice = {"type": "voice", "data": text,
                         "binary_data_base64": base64.b64encode(audio).decode("ascii"),
                         "hash": hashlib.sha256(audio).hexdigest()}
                # Build a fresh shallow tree. Original message and reply/at
                # segments stay untouched on BOTH success and failure.
                changed = dict(message)
                if cfg.auto_voice.keep_text:
                    changed["raw_message"] = [*segments, voice]
                else:
                    output = []
                    inserted = False
                    for seg in segments:
                        if seg["type"] == "text":
                            if not inserted:
                                output.append(voice)
                                inserted = True
                        else:
                            output.append(seg)
                    changed["raw_message"] = output
                changed_info = dict(info) if isinstance(info, dict) else {}
                changed_additional = dict(additional) if isinstance(additional, dict) else {}
                changed_additional[MARKER] = True
                changed_info["additional_config"] = changed_additional
                changed["message_info"] = changed_info
                state["last_converted"] = state["touched"] = time.monotonic()
                self._auto_stats["converted"] += 1
                self._auto_stats["last_reason"] = "converted"
                trace.outcome = "已转为语音（保留文字）" if cfg.auto_voice.keep_text else "已替换为语音"
                tts_log.log_success(self._get_logger(), trace)
                return {"action": "continue", "modified_kwargs": _without_voice_reply(changed, original)}
            finally:
                tts_log.deactivate(token)
                self._busy = False
                if task:
                    self._tasks.discard(task)
        except asyncio.CancelledError:
            if trace is not None:
                tts_log.log_cancelled(self._get_logger(), trace)
            raise
        except Exception as exc:
            self._auto_stats["failed"] += 1
            self._auto_stats["last_reason"] = "failed"
            if trace is not None and trace.begun:
                trace.outcome = "已保留原文字"
                tts_log.log_failure(self._get_logger(), trace, tts_log.reason(exc))
            else:
                self._get_logger().warning(f"自动语音转换失败，保留原文字：{type(exc).__name__}: {exc}")
            return {"action": "continue", "modified_kwargs": original}
