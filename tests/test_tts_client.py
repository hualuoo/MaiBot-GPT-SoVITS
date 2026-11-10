"""Run: python -B -m unittest discover -s tests -p test_tts_client.py -v"""
import asyncio
import io
import json
import os
import socket
import struct
import unittest
import wave
from unittest.mock import patch

from aiohttp import web
from tts_client import GPTSoVITSClient, TTSError


def wav_bytes(pcm=b"\x00\x00\x01\x00"):
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(32000)
        audio.writeframes(pcm)
    return output.getvalue()


class ClientTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.audio = wav_bytes()
        self.calls = []
        self.body = self.audio
        self.status = 200
        self.content_type = "audio/wav"
        self.delay = 0
        self.chunked = False
        self.openapi = {"openapi": "3.1.0", "paths": {"/tts": {"post": {}}}}
        self.openapi_raw = None
        self.weights_body = b'{"message":"success"}'
        self.weights_requests = []
        self.weights_headers = {}
        self.weights_disconnect = False
        app = web.Application()
        app.router.add_post("/tts", self.handle_tts)
        app.router.add_get("/openapi.json", self.handle_probe)
        app.router.add_get("/set_gpt_weights", self.handle_weights)
        app.router.add_get("/set_sovits_weights", self.handle_weights)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        self.addAsyncCleanup(self.runner.cleanup)
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        sock.listen(128)
        sock.setblocking(False)
        self.url = f"http://127.0.0.1:{sock.getsockname()[1]}"
        self.site = web.SockSite(self.runner, sock)
        await self.site.start()
        self.client = GPTSoVITSClient(self.url)

    async def handle_tts(self, request):
        self.calls.append((request.method, request.path, await request.json()))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.chunked:
            response = web.StreamResponse(status=self.status,
                                          headers={"Content-Type": self.content_type})
            await response.prepare(request)
            for start in range(0, len(self.body), 7):
                await response.write(self.body[start:start + 7])
            await response.write_eof()
            return response
        return web.Response(body=self.body, status=self.status, content_type=self.content_type)

    async def handle_probe(self, request):
        self.calls.append((request.method, request.path, None))
        if self.openapi_raw is not None:
            return web.Response(body=self.openapi_raw, status=self.status)
        return web.json_response(self.openapi, status=self.status)

    async def handle_weights(self, request):
        self.calls.append((request.method, request.path, dict(request.query)))
        self.weights_requests.append((request.raw_path, await request.read()))
        if self.weights_disconnect:
            request.transport.close()
            return web.Response()
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.chunked:
            response = web.StreamResponse(status=self.status,
                                          headers={"Content-Type": self.content_type})
            await response.prepare(request)
            for start in range(0, len(self.weights_body), 4096):
                await response.write(self.weights_body[start:start + 4096])
            await response.write_eof()
            return response
        return web.Response(body=self.weights_body, status=self.status,
                            content_type=self.content_type, headers=self.weights_headers)

    async def test_success_forces_wav_and_nonstreaming_without_mutation(self):
        payload = {"text": "你好", "media_type": "mp3", "streaming_mode": True}
        self.assertEqual(await self.client.synthesize(payload), self.audio)
        self.assertEqual(payload["media_type"], "mp3")
        self.assertIs(payload["streaming_mode"], True)
        self.assertEqual(self.calls, [("POST", "/tts", {
            "text": "你好", "media_type": "wav", "streaming_mode": False})])

    async def test_chunked_success_and_exact_limit(self):
        self.chunked = True
        client = GPTSoVITSClient(self.url, max_audio_bytes=len(self.audio))
        self.assertEqual(await client.synthesize({}), self.audio)

    async def test_content_length_limit(self):
        client = GPTSoVITSClient(self.url, max_audio_bytes=len(self.audio) - 1)
        with self.assertRaisesRegex(TTSError, "字节上限"):
            await client.synthesize({})

    async def test_chunked_total_limit(self):
        self.chunked = True
        client = GPTSoVITSClient(self.url, max_audio_bytes=len(self.audio) - 1)
        with self.assertRaisesRegex(TTSError, "字节上限"):
            await client.synthesize({})

    async def test_http_json_error_no_retry(self):
        self.status = 400
        self.body = b'{"message":"tts failed","Exception":"bad reference"}'
        self.content_type = "application/json"
        with self.assertRaisesRegex(TTSError, "HTTP 400.*bad reference") as caught:
            await self.client.synthesize({})
        self.assertEqual(caught.exception.status_code, 400)
        self.assertEqual(len(self.calls), 1)

    async def test_http_plain_error(self):
        self.status = 503
        self.body = b"unavailable"
        with self.assertRaisesRegex(TTSError, "HTTP 503.*unavailable"):
            await self.client.synthesize({})

    async def test_json_success_status_is_error(self):
        for content_type, body in [("application/json", b'{"message":"error"}'),
                                   ("application/problem+json", b"not json"),
                                   ("audio/wav", b'{"message":"error"}')]:
            with self.subTest(content_type=content_type, body=body):
                self.content_type, self.body = content_type, body
                with self.assertRaises(TTSError):
                    await self.client.synthesize({})

    async def test_invalid_wav_variants(self):
        truncated = bytearray(self.audio[:-2])
        struct.pack_into("<I", truncated, 4, len(truncated) - 8)
        not_pcm = bytearray(self.audio)
        struct.pack_into("<H", not_pcm, 20, 3)
        malformed = b"RIFF" + struct.pack("<I", 4) + b"WAVE"
        unaligned = wav_bytes(b"\x00\x00\x01")
        bad_chunk = bytearray(self.audio)
        bad_chunk[12:16] = b"JUNK"
        struct.pack_into("<I", bad_chunk, 16, 0x7fffffff)
        for body in [b"", b"RIFFxxxxNOPE", malformed, wav_bytes(b""),
                     self.audio[:-1], bytes(truncated), bytes(not_pcm),
                     bytes(bad_chunk), unaligned]:
            with self.subTest(body=body):
                self.body = body
                with self.assertRaises(TTSError):
                    await self.client.synthesize({})

    async def test_timeout_no_retry(self):
        self.delay = 0.6
        with self.assertRaisesRegex(TTSError, "请求超时"):
            await GPTSoVITSClient(self.url, timeout=0.3).synthesize({})
        self.assertEqual(len(self.calls), 1)

    async def test_connection_failure(self):
        await self.site.stop()
        with self.assertRaisesRegex(TTSError, "连接或 HTTP 传输"):
            await self.client.synthesize({})

    async def test_bad_payload_and_json_serialization(self):
        for payload in [None, [], {"bad": object()}]:
            with self.subTest(payload=payload):
                with self.assertRaises(TTSError):
                    await self.client.synthesize(payload)
        self.assertEqual(self.calls, [])

    async def test_environment_proxies_are_ignored(self):
        with patch.dict(os.environ, {"HTTP_PROXY": "http://127.0.0.1:1",
                                     "HTTPS_PROXY": "http://127.0.0.1:1",
                                     "ALL_PROXY": "http://127.0.0.1:1",
                                     "NO_PROXY": "", "no_proxy": ""}):
            self.assertEqual(await self.client.synthesize({}), self.audio)
            self.assertEqual(await self.client.probe(), self.url + "/tts")

    async def test_probe_is_read_only(self):
        self.assertEqual(await self.client.probe(), self.url + "/tts")
        self.assertEqual(self.calls, [("GET", "/openapi.json", None)])

    async def test_probe_invalid_json(self):
        self.openapi_raw = b"not JSON"
        with self.assertRaisesRegex(TTSError, "OpenAPI JSON 无效"):
            await self.client.probe()

    async def test_probe_missing_tts_and_invalid_structure(self):
        for document in [{}, [], {"paths": []}, {"paths": {"/tts": None}},
                         {"paths": {"/tts": {"get": {}}}}]:
            self.openapi = document
            with self.subTest(document=document):
                with self.assertRaisesRegex(TTSError, "未声明"):
                    await self.client.probe()

    async def test_probe_http_error(self):
        self.status = 500
        with self.assertRaisesRegex(TTSError, "HTTP 500"):
            await self.client.probe()

    async def test_probe_response_limit(self):
        self.openapi_raw = b" " * (1048576 + 1)
        with self.assertRaisesRegex(TTSError, "字节上限"):
            await self.client.probe()

    async def test_redirect_not_followed(self):
        self.status = 302
        with self.assertRaisesRegex(TTSError, "HTTP 302"):
            await self.client.synthesize({})
        self.assertEqual(len(self.calls), 1)

    async def test_cancellation_propagates(self):
        self.delay = 0.15
        task = asyncio.create_task(self.client.synthesize({}))
        await asyncio.sleep(0.03)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

    async def test_set_weights_get_params_round_trip_and_none(self):
        paths = [r"C:\模型 目录\角色&特别+#?=%2F.ckpt",
                 "/remote/模型 name+A&B#Q?part=%20.pth"]
        for method_name, endpoint in [("set_gpt_weights", "/set_gpt_weights"),
                                      ("set_sovits_weights", "/set_sovits_weights")]:
            for path in paths:
                with self.subTest(method=method_name, path=path):
                    self.calls.clear()
                    self.weights_requests.clear()
                    self.assertIsNone(await getattr(self.client, method_name)(path))
                    self.assertEqual(self.calls, [("GET", endpoint, {"weights_path": path})])
                    raw_path, body = self.weights_requests[0]
                    self.assertEqual(body, b"")
                    self.assertTrue(raw_path.isascii())
                    self.assertIn("%26", raw_path)
                    self.assertIn("%2B", raw_path)
                    self.assertIn("%23", raw_path)
                    self.assertIn("%25", raw_path)

    async def test_set_weights_requires_exact_json_success(self):
        for method_name in ["set_gpt_weights", "set_sovits_weights"]:
            for document in [{}, [], None, "success", True, {"message": "failed"},
                             {"message": "SUCCESS"}, {"message": True},
                             {"message": "success "}, {"message": {"status": "success"}}]:
                with self.subTest(method=method_name, document=document):
                    self.calls.clear()
                    self.weights_body = json.dumps(document).encode()
                    with self.assertRaisesRegex(TTSError, "模型切换失败"):
                        await getattr(self.client, method_name)("remote/path")
                    self.assertEqual(len(self.calls), 1)

    async def test_set_weights_rejects_invalid_json_once(self):
        for method_name in ["set_gpt_weights", "set_sovits_weights"]:
            for body in [b"", b"not JSON", b'{"message":', b"\xff"]:
                with self.subTest(method=method_name, body=body):
                    self.calls.clear()
                    self.weights_body = body
                    with self.assertRaisesRegex(TTSError, "有效 JSON"):
                        await getattr(self.client, method_name)("remote/path")
                    self.assertEqual(len(self.calls), 1)

    async def test_set_weights_deep_json_is_wrapped_as_tts_error(self):
        self.weights_body = b"[" * 2000 + b"]" * 2000
        for method_name in ["set_gpt_weights", "set_sovits_weights"]:
            with self.subTest(method=method_name):
                self.calls.clear()
                with self.assertRaisesRegex(TTSError, "模型切换失败"):
                    await getattr(self.client, method_name)("remote/path")
                self.assertEqual(len(self.calls), 1)

    async def test_set_weights_http_error_preserves_status_and_no_retry(self):
        self.status = 400
        self.weights_body = b'{"message":"failed","Exception":"bad weights"}'
        for method_name in ["set_gpt_weights", "set_sovits_weights"]:
            with self.subTest(method=method_name):
                self.calls.clear()
                with self.assertRaisesRegex(TTSError, "HTTP 400.*bad weights") as caught:
                    await getattr(self.client, method_name)("remote/path")
                self.assertEqual(caught.exception.status_code, 400)
                self.assertEqual(len(self.calls), 1)

    async def test_set_weights_content_length_limit(self):
        self.weights_body = b" " * 65537
        for method_name in ["set_gpt_weights", "set_sovits_weights"]:
            with self.subTest(method=method_name):
                self.calls.clear()
                with self.assertRaisesRegex(TTSError, "字节上限"):
                    await getattr(self.client, method_name)("remote/path")
                self.assertEqual(len(self.calls), 1)

    async def test_set_weights_chunked_limit(self):
        self.chunked = True
        self.weights_body = b" " * 65537
        for method_name in ["set_gpt_weights", "set_sovits_weights"]:
            with self.subTest(method=method_name):
                self.calls.clear()
                with self.assertRaisesRegex(TTSError, "字节上限"):
                    await getattr(self.client, method_name)("remote/path")
                self.assertEqual(len(self.calls), 1)

    async def test_set_weights_exact_independent_limit_and_success_content_type(self):
        self.weights_body = b'{"message":"success"}' + b" " * (65536 - 21)
        client = GPTSoVITSClient(self.url, max_audio_bytes=1)
        for chunked in [False, True]:
            self.chunked = chunked
            for method_name in ["set_gpt_weights", "set_sovits_weights"]:
                with self.subTest(chunked=chunked, method=method_name):
                    self.assertIsNone(await getattr(client, method_name)("remote/path"))

    async def test_set_weights_invalid_path_does_not_send(self):
        for method_name in ["set_gpt_weights", "set_sovits_weights"]:
            for path in [None, [], {}, 42, "", " \t", "bad\x00path"]:
                with self.subTest(method=method_name, path=path):
                    with self.assertRaisesRegex(TTSError, "权重路径"):
                        await getattr(self.client, method_name)(path)
        self.assertEqual(self.calls, [])

    async def test_set_weights_disconnect_is_never_retried(self):
        self.weights_disconnect = True
        for method_name in ["set_gpt_weights", "set_sovits_weights"]:
            with self.subTest(method=method_name):
                self.calls.clear()
                with self.assertRaisesRegex(TTSError, "连接或 HTTP 传输"):
                    await getattr(self.client, method_name)("remote/path")
                self.assertEqual(len(self.calls), 1)

    async def test_set_weights_connection_failure(self):
        await self.site.stop()
        for method_name in ["set_gpt_weights", "set_sovits_weights"]:
            with self.subTest(method=method_name):
                with self.assertRaisesRegex(TTSError, "连接或 HTTP 传输"):
                    await getattr(self.client, method_name)("remote/path")
        self.assertEqual(self.calls, [])

    async def test_set_weights_timeout_no_retry(self):
        self.delay = 0.6
        client = GPTSoVITSClient(self.url, timeout=0.1)
        for method_name in ["set_gpt_weights", "set_sovits_weights"]:
            with self.subTest(method=method_name):
                self.calls.clear()
                with self.assertRaisesRegex(TTSError, "请求超时"):
                    await getattr(client, method_name)("remote/path")
                self.assertEqual(len(self.calls), 1)

    async def test_set_weights_redirect_is_not_followed(self):
        self.status = 302
        self.weights_headers = {"Location": self.url + "/set_gpt_weights?weights_path=changed"}
        for method_name in ["set_gpt_weights", "set_sovits_weights"]:
            with self.subTest(method=method_name):
                self.calls.clear()
                with self.assertRaisesRegex(TTSError, "HTTP 302"):
                    await getattr(self.client, method_name)("remote/path")
                self.assertEqual(len(self.calls), 1)

    async def test_set_weights_cancellation_propagates(self):
        self.delay = 0.15
        for method_name in ["set_gpt_weights", "set_sovits_weights"]:
            with self.subTest(method=method_name):
                task = asyncio.create_task(getattr(self.client, method_name)("remote/path"))
                await asyncio.sleep(0.03)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task

    async def test_set_weights_environment_proxies_are_ignored(self):
        with patch.dict(os.environ, {"HTTP_PROXY": "http://127.0.0.1:1",
                                     "HTTPS_PROXY": "http://127.0.0.1:1",
                                     "ALL_PROXY": "http://127.0.0.1:1",
                                     "NO_PROXY": "", "no_proxy": ""}):
            await self.client.set_gpt_weights("remote/gpt.ckpt")
            await self.client.set_sovits_weights("remote/sovits.pth")
        self.assertEqual(len(self.calls), 2)

    def test_configuration_validation(self):
        for url in ["", "ftp://host", "http:///tts", "http://user@host",
                    "http://user:pass@host", "http://host?", "http://host#",
                    "http://host?x=1", "http://host#fragment", "http://host:bad",
                    "http://host:99999", "http://ho st", "http://host\n", None,
                    "http://host\\other"]:
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    GPTSoVITSClient(url)
        for timeout in [0, -1, float("inf"), float("nan"), True, "180", None]:
            with self.subTest(timeout=timeout):
                with self.assertRaises(ValueError):
                    GPTSoVITSClient(self.url, timeout=timeout)
        for limit in [0, -1, True, 1.5, "100", None]:
            with self.subTest(limit=limit):
                with self.assertRaises(ValueError):
                    GPTSoVITSClient(self.url, max_audio_bytes=limit)
        self.assertEqual(GPTSoVITSClient("https://localhost:9880/prefix/").base_url,
                         "https://localhost:9880/prefix")


if __name__ == "__main__":
    unittest.main()
