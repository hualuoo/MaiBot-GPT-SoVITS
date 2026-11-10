"""多情感角色：调用 MaiBot 的 LLM 判断句子情感，再选用对应情感的模型。

只在第三步“已启用的情感”中选择；LLM 不可用、超时或回答无法识别时回退到
调用方给出的情感（关键词规则或默认情感），从不阻塞合成。
"""
from __future__ import annotations

import asyncio
from collections import OrderedDict
import re
import time

from maibot_sdk import API

if __package__:
    from . import tts_log
    from .auto_voice import _ALIASES, _canonical_emotion, _enabled_emotion
    from .tts_client import TTSError
else:
    import tts_log
    from auto_voice import _ALIASES, _canonical_emotion, _enabled_emotion
    from tts_client import TTSError

DEFAULT_TASK = "fast_model"
LLM_TIMEOUT = 15.0
CACHE_LIMIT = 256
CACHE_TTL = 600.0
MAX_TEXT = 300
LABELS = {"happy": "开心", "angry": "生气", "surprised": "吃惊", "neutral": "中立", "sad": "难过"}
# 这些 MaiBot 任务不是文本对话模型，不能用来判断情感。
NON_TEXT_TASKS = frozenset({"embedding", "image_embedding", "voice", "web_search"})


def emotion_label(emotion_id: str) -> str:
    return LABELS.get(emotion_id, emotion_id)


def enabled_emotions(cfg) -> list[str]:
    return list(cfg.role.emotions or [cfg.role.emotion])


def llm_emotion_active(cfg) -> bool:
    """角色模式、开启 LLM 判断且启用了 2 个以上情感时才调用 LLM。"""
    return cfg.role.mode == "role" and bool(cfg.role.llm_emotion) and len(enabled_emotions(cfg)) > 1


def build_prompt(text: str, enabled: list[str], default: str) -> str:
    options = "\n".join(f"- {item}：{emotion_label(item)}" if emotion_label(item) != item else f"- {item}"
                        for item in enabled)
    return ("你是语音合成系统的情感分类器。请判断下面这句话最适合用哪一种情感朗读。\n"
            "可选情感（只能从中选择一个，原样输出它的 ID）：\n"
            f"{options}\n"
            f"如果情感不明显或无法判断，输出 {default}。\n"
            "只输出一个情感 ID，不要解释，也不要输出其他内容。\n\n"
            f"句子：{text}")


def _names(emotion_id: str) -> set[str]:
    names = {emotion_id, emotion_label(emotion_id)}
    names.update(_ALIASES.get(_canonical_emotion(emotion_id), ()))
    return {name for name in names if name}


def parse_emotion(response, enabled: list[str]) -> str | None:
    """从 LLM 回答中取出一个已启用的情感；先整句精确匹配，再取最早出现的情感名或别名。"""
    if not isinstance(response, str):
        return None
    text = re.sub(r"<think>.*?</think>", "", response, flags=re.S | re.I).strip()
    if not text:
        return None
    exact = _enabled_emotion(text.strip("`'\"“”‘’「」『』 。.!！：:"), enabled)
    if exact:
        return exact
    best = None
    for item in enabled:
        for name in _names(item):
            if name.isascii():
                match = re.search(r"(?<![a-z])" + re.escape(name) + r"(?![a-z])", text, re.I)
            else:
                index = text.find(name)
                match = None if index < 0 else index
            position = match.start() if hasattr(match, "start") else match
            if position is not None and (best is None or position < best[0]):
                best = (position, item)
    return best[1] if best else None


class EmotionJudgeMixin:
    """需要插件提供 ctx、config、_save_setup、_require_setup 与 _get_logger。"""

    def _init_emotion_judge(self) -> None:
        self._emotion_cache: OrderedDict = OrderedDict()
        self._emotion_last = {"emotion": "", "source": "", "model": "", "detail": ""}

    def _llm_emotion_active(self, cfg) -> bool:
        return llm_emotion_active(cfg)

    async def _ask_llm_emotion(self, text: str, cfg, *, use_cache: bool = True) -> tuple[str, str, bool]:
        """返回 (情感, 模型, 是否命中缓存)；LLM 失败或回答无法识别时抛出异常。"""
        enabled = enabled_emotions(cfg)
        default = cfg.role.emotion
        task = (cfg.role.llm_task or "").strip() or DEFAULT_TASK
        text = text.strip()[:MAX_TEXT]
        key = (text, tuple(enabled), default, task)
        now = time.monotonic()
        cached = self._emotion_cache.get(key) if use_cache else None
        if cached and cached[2] > now:
            self._emotion_cache.move_to_end(key)
            return cached[0], cached[1], True
        # 温度、最大长度沿用 MaiBot 对该模型任务的配置：部分推理模型不接受自定义温度。
        result = await asyncio.wait_for(
            self.ctx.llm.generate(build_prompt(text, enabled, default), task_name=task),
            timeout=LLM_TIMEOUT)
        if not isinstance(result, dict) or not result.get("success"):
            detail = result.get("error") if isinstance(result, dict) else ""
            raise RuntimeError(f"MaiBot LLM 调用失败：{detail or '未返回成功结果'}")
        emotion = parse_emotion(result.get("response"), enabled)
        if emotion is None:
            raise ValueError("LLM 回答中没有已启用的情感：" + str(result.get("response", ""))[:60])
        model = str(result.get("model") or result.get("model_name") or task)
        self._emotion_cache[key] = (emotion, model, now + CACHE_TTL)
        self._emotion_cache.move_to_end(key)
        for stale in [item for item, value in self._emotion_cache.items() if value[2] <= now]:
            self._emotion_cache.pop(stale, None)
        while len(self._emotion_cache) > CACHE_LIMIT:
            self._emotion_cache.popitem(last=False)
        return emotion, model, False

    async def _judge_emotion(self, text: str, cfg, fallback: str) -> tuple[str, str]:
        """多情感时请求 LLM 判断；任何失败都回退到 fallback。返回 (情感, 来源)。"""
        if not llm_emotion_active(cfg):
            return fallback, "default"
        trace = tts_log.current()
        task = (cfg.role.llm_task or "").strip() or DEFAULT_TASK
        started = time.monotonic()
        try:
            emotion, model, cached = await self._ask_llm_emotion(text, cfg)
            self._emotion_last = {"emotion": emotion, "source": "llm", "model": model, "detail": ""}
            if trace is not None:
                trace.set_emotion(emotion, "llm", task=task, model=model, elapsed=time.monotonic() - started, cached=cached)
            return emotion, "llm"
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            detail = "LLM 判断超时" if isinstance(exc, asyncio.TimeoutError) else f"{type(exc).__name__}: {exc}"
            self._emotion_last = {"emotion": fallback, "source": "llm_failed", "model": "", "detail": detail[:200]}
            if trace is not None:
                trace.set_emotion(fallback, "llm_failed", task=task, elapsed=time.monotonic() - started, error=detail)
            try:
                self._get_logger().warning(f"情感判断失败，使用 {fallback}：{detail}")
            except Exception:
                pass  # 日志不可用时也不能影响合成
            return fallback, "llm_failed"

    @API("setup_llm_tasks", description="读取 MaiBot 可用于情感判断的 LLM 模型任务")
    async def setup_llm_tasks(self) -> dict:
        current = self.config.role.llm_task
        try:
            tasks = await asyncio.wait_for(self.ctx.llm.get_available_models(), timeout=10)
            names = [str(task) for task in tasks if isinstance(task, str) and task and task not in NON_TEXT_TASKS]
            return {"success": True, "tasks": names, "current": current}
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return {"success": False, "tasks": [], "current": current, "message": f"无法读取 MaiBot 模型任务：{exc}"}

    @API("setup_save_emotion_judge", description="第三步保存多情感时的 LLM 情感判断设置")
    async def setup_save_emotion_judge(self, llm_emotion: bool, llm_task: str) -> dict:
        try:
            if not isinstance(llm_emotion, bool) or not isinstance(llm_task, str):
                raise TTSError("情感判断设置格式无效")
            task = llm_task.strip() or DEFAULT_TASK
            if len(task) > 80:
                raise TTSError("模型任务名最多 80 个字符")
            await self._save_setup({"role": {"llm_emotion": llm_emotion, "llm_task": task}})
            self._emotion_cache.clear()
            if not llm_emotion:
                message = "已关闭 LLM 情感判断：合成时使用默认情感（自动语音可按关键词规则选择）"
            elif len(enabled_emotions(self.config)) < 2:
                message = f"已保存：使用 {task} 判断情感；当前只启用了 1 个情感，暂不会调用 LLM"
            else:
                message = f"已保存：多情感时使用 MaiBot 的 {task} 判断每句话的情感"
            return {"success": True, "message": message}
        except (TTSError, ValueError, OSError) as exc:
            return {"success": False, "message": f"情感判断设置未保存：{exc}"}

    @API("setup_test_emotion", description="用 MaiBot LLM 试判断一句话的情感，不合成语音")
    async def setup_test_emotion(self, text: str) -> dict:
        try:
            self._require_setup()
            cfg = self.config.model_copy(deep=True)
            if cfg.role.mode != "role":
                raise TTSError("请先切换到角色模式")
            if len(enabled_emotions(cfg)) < 2:
                raise TTSError("只启用了 1 个情感，不需要判断；请至少勾选 2 个情感并读取角色配置")
            if not isinstance(text, str) or not text.strip():
                raise TTSError("请输入要判断的文字")
            if len(text) > MAX_TEXT:
                raise TTSError(f"试判断文字最多 {MAX_TEXT} 字")
            emotion, model, _ = await self._ask_llm_emotion(text, cfg, use_cache=False)
            label = emotion_label(emotion)
            return {"success": True, "emotion": emotion, "label": label, "model": model,
                    "message": f"判断结果：{label}（{emotion}） · 模型：{model}"}
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            return {"success": False, "message": f"情感判断超时（{LLM_TIMEOUT:.0f} 秒），请检查 MaiBot 模型配置"}
        except Exception as exc:
            return {"success": False, "message": f"情感判断失败：{exc}"}
