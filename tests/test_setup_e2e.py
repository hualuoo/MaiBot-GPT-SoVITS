"""真实 aiohttp 协议链路：设置→连接→双权重→试听→分块音频（无模型下载）。"""
import asyncio
import base64
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import wave

from aiohttp import web

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plugin import create_plugin


class SetupEndToEndTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.calls = []
        output = io.BytesIO()
        with wave.open(output, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(32000)
            wav.writeframes(b"\0\1" * 200000)
        self.audio = output.getvalue()
        self.sovits_fails = False
        app = web.Application()
        async def openapi(request):
            self.calls.append("probe")
            return web.json_response({"paths": {"/tts": {"post": {}}}})
        async def weights(request):
            self.calls.append((request.path, request.query["weights_path"]))
            if self.sovits_fails and request.path == "/set_sovits_weights":
                return web.json_response({"message": "测试加载失败"}, status=400)
            return web.json_response({"message": "success"})
        async def tts(request):
            self.calls.append(("tts", await request.json()))
            return web.Response(body=self.audio, content_type="audio/wav")
        app.router.add_get("/openapi.json", openapi)
        app.router.add_get("/set_gpt_weights", weights)
        app.router.add_get("/set_sovits_weights", weights)
        app.router.add_post("/tts", tts)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        self.url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
        self.temp = tempfile.TemporaryDirectory()
        self.plugin = create_plugin()
        self.plugin.set_plugin_config(self.plugin.build_default_config())
        self.plugin._settings_path = Path(self.temp.name) / "config.toml"
        await self.plugin.on_load()

    async def asyncTearDown(self):
        await self.plugin.on_unload()
        await self.runner.cleanup()
        self.temp.cleanup()

    async def configure(self):
        result = await self.plugin.setup_save_api(base_url=self.url, timeout=10)
        self.assertTrue(result["success"], result)
        result = await self.plugin.setup_test_api()
        self.assertTrue(result["success"], result)
        result = await self.plugin.setup_save_models(root="", base_model="custom",
            gpt_weights="C:/权重/声音 & GPT.ckpt", sovits_weights="C:/权重/声音 # SoVITS.pth",
            ref_audio_path="C:/参考.wav", prompt_text="参考文本", prompt_lang="ZH", text_lang="ZH")
        self.assertTrue(result["success"], result)

    async def finish_job(self, kind, text=""):
        result = await self.plugin.setup_start_job(kind, text)
        self.assertTrue(result["success"], result)
        self.assertEqual(result["job"]["status"], "running")
        await asyncio.wait_for(self.plugin._job_task, 5)
        return await self.plugin.setup_job(result["job"]["id"])

    async def test_complete_http_setup_and_preview(self):
        await self.configure()
        loaded = await self.finish_job("load_models")
        self.assertEqual(loaded["status"], "success", loaded)
        job = await self.finish_job("synthesize", "你好，测试语音。")
        self.assertEqual(job["status"], "success", job)
        data = bytearray()
        offset = 0
        chunks = 0
        while True:
            part = await self.plugin.setup_audio(job["id"], offset)
            self.assertLess(len(json.dumps(part).encode()), 524288)
            data.extend(base64.b64decode(part["base64"], validate=True))
            self.assertGreater(part["next_offset"], offset)
            offset = part["next_offset"]
            chunks += 1
            if part["done"]:
                break
        self.assertGreater(chunks, 1)
        self.assertEqual(bytes(data), self.audio)
        self.assertEqual(self.calls[:3], ["probe",
            ("/set_gpt_weights", "C:/权重/声音 & GPT.ckpt"),
            ("/set_sovits_weights", "C:/权重/声音 # SoVITS.pth")])
        self.assertEqual(self.calls[3][1]["text"], "你好，测试语音。")
        self.assertFalse(self.calls[3][1]["streaming_mode"])
        self.assertTrue(self.plugin._settings_path.exists())
        self.assertFalse(self.plugin._busy)

    async def test_partial_http_load_is_not_ready(self):
        await self.configure()
        self.sovits_fails = True
        job = await self.finish_job("load_models")
        self.assertEqual(job["status"], "error")
        self.assertIn("部分", job["message"])
        result = await self.plugin.setup_start_job("synthesize", "不应合成")
        self.assertFalse(result["success"])
        self.assertEqual(len(self.calls), 3)
        self.assertIsNone(self.plugin._loaded_signature)


if __name__ == "__main__":
    unittest.main()
