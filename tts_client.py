"""Small, bounded, non-streaming client for GPT-SoVITS api_v2.py."""
from __future__ import annotations

import asyncio
import io
import json
import math
import struct
import wave
from urllib.parse import urlsplit, urlunsplit

import aiohttp


class TTSError(Exception):
    """Synthesis/probe failure; status_code is set for HTTP errors."""

    def __init__(self, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class GPTSoVITSClient:
    def __init__(self, base_url, timeout=180, max_audio_bytes=8388608):
        """Validate configuration. Invalid configuration raises ValueError."""
        try:
            if not isinstance(base_url, str) or not base_url or any(
                c.isspace() or ord(c) < 32 or c == "\\" for c in base_url
            ):
                raise ValueError("服务地址无效")
            parts = urlsplit(base_url)
            if (parts.scheme.lower() not in ("http", "https") or not parts.hostname
                    or parts.username is not None or parts.password is not None
                    or "?" in base_url or "#" in base_url):
                raise ValueError("服务地址必须使用 HTTP 或 HTTPS，且不能包含账号密码、查询参数或片段")
            _ = parts.port
            if isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
                raise ValueError("超时秒数必须是有限正数")
            if (isinstance(max_audio_bytes, bool) or not isinstance(max_audio_bytes, int)
                    or max_audio_bytes <= 0):
                raise ValueError("音频响应字节上限必须是正整数")
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("客户端配置无效") from exc
        self.base_url = urlunsplit((parts.scheme.lower(), parts.netloc,
                                   parts.path.rstrip("/"), "", ""))
        self.timeout = timeout
        self.max_audio_bytes = max_audio_bytes

    async def _request(self, method: str, path: str, *, limit: int, payload=None, params=None):
        try:
            # 每次使用独立会话；GPT-SoVITS 的 GET 管理接口不能自动重试。
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout), trust_env=False
            ) as session:
                # aiohttp 3.10+ 会对 GET 首次断线也重试；目前没有公开关闭参数。
                # 老版本没有该属性，也没有这项自动重试。
                if hasattr(session, "_retry_connection"):
                    session._retry_connection = False
                async with session.request(
                    method, self.base_url + path, json=payload, params=params,
                    allow_redirects=False,
                ) as response:
                    if response.content_length is not None and response.content_length > limit:
                        raise TTSError("响应超过字节上限")
                    body = bytearray()
                    async for chunk in response.content.iter_chunked(65536):
                        if len(body) + len(chunk) > limit:
                            raise TTSError("响应超过字节上限")
                        body.extend(chunk)
                    data = bytes(body)
                    if not 200 <= response.status < 300:
                        detail = self._error_detail(data)
                        raise TTSError(f"HTTP {response.status} 请求失败：{detail}",
                                       status_code=response.status)
                    return data, response.content_type
        except TTSError:
            raise
        except asyncio.TimeoutError as exc:
            raise TTSError("请求超时") from exc
        except aiohttp.ClientError as exc:
            raise TTSError(f"连接或 HTTP 传输错误：{exc}") from exc
        except (TypeError, ValueError, OverflowError) as exc:
            raise TTSError(f"请求参数或 JSON 无效：{exc}") from exc

    @staticmethod
    def _error_detail(data: bytes) -> str:
        try:
            return str(json.loads(data))[:512]
        except (ValueError, UnicodeError, RecursionError):
            return data[:512].decode("utf-8", errors="replace")

    async def synthesize(self, payload: dict) -> bytes:
        """POST /tts, return a complete nonempty PCM WAV or raise TTSError.

        payload is copied; media_type and streaming_mode are always overridden.
        Cancellation propagates normally. No retries, proxies, or redirects.
        """
        if not isinstance(payload, dict):
            raise TTSError("合成参数必须是字典")
        request = dict(payload)
        request.update(media_type="wav", streaming_mode=False)
        data, content_type = await self._request(
            "POST", "/tts", limit=self.max_audio_bytes, payload=request
        )
        if content_type == "application/json" or content_type.endswith("+json"):
            raise TTSError("服务端返回 JSON 错误：" + self._error_detail(data))
        if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
            raise TTSError("WAV 的 RIFF/WAVE 文件头无效：" + self._error_detail(data))
        if int.from_bytes(data[4:8], "little") + 8 != len(data):
            raise TTSError("WAV 的 RIFF 长度无效")
        try:
            with wave.open(io.BytesIO(data), "rb") as audio:
                count = audio.getnframes()
                frame_size = audio.getnchannels() * audio.getsampwidth()
                if (audio.getcomptype() != "NONE" or count <= 0
                        or frame_size <= 0 or audio.getframerate() <= 0):
                    raise TTSError("WAV 必须包含非空 PCM 音频")
                pcm = audio.readframes(count + 1)
                if len(pcm) != count * frame_size:
                    raise TTSError("WAV 的 PCM 数据不完整或未对齐")
        except (wave.Error, EOFError, OSError, ValueError, RuntimeError, struct.error) as exc:
            raise TTSError(f"PCM WAV 音频无效：{exc}") from exc
        return data

    async def set_gpt_weights(self, path: str) -> None:
        """切换服务端 GPT 权重；路径不要求存在于客户端机器。"""
        await self._set_weights("/set_gpt_weights", path)

    async def set_sovits_weights(self, path: str) -> None:
        """切换服务端 SoVITS 权重；请求仅发送一次，不自动重试。"""
        await self._set_weights("/set_sovits_weights", path)

    async def _set_weights(self, endpoint: str, path: str) -> None:
        if not isinstance(path, str) or not path.strip() or "\x00" in path:
            raise TTSError("权重路径必须是非空字符串，且不能包含空字符")
        # params 交由 aiohttp 编码，保留中文、空格、反斜杠及 &/+/#/?/%。
        # 管理接口独立限制为 64 KiB，不受音频响应大小设置影响。
        data, _ = await self._request(
            "GET", endpoint, limit=65536, params={"weights_path": path}
        )
        try:
            document = json.loads(data)
        except (ValueError, UnicodeError, RecursionError) as exc:
            raise TTSError("模型切换失败：服务端未返回有效 JSON") from exc
        if not isinstance(document, dict) or document.get("message") != "success":
            raise TTSError("模型切换失败：服务端未确认成功；" + self._error_detail(data))

    async def probe(self) -> str:
        """Read GET /openapi.json only; return the /tts URL when advertised.

        OpenAPI responses are separately limited to 1 MiB. No model-changing
        endpoints or synthesis requests are made by this method.
        """
        data, _ = await self._request("GET", "/openapi.json", limit=1048576)
        try:
            document = json.loads(data)
        except (ValueError, UnicodeError, RecursionError) as exc:
            raise TTSError("OpenAPI JSON 无效") from exc
        paths = document.get("paths") if isinstance(document, dict) else None
        if (not isinstance(paths, dict) or not isinstance(paths.get("/tts"), dict)
                or not isinstance(paths["/tts"].get("post"), dict)):
            raise TTSError("OpenAPI 未声明 POST /tts 接口")
        return self.base_url + "/tts"
