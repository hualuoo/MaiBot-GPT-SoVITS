"""QQ 语音不能附带引用：Hook → MaiBot 发送预处理 → 适配器编码，全程不发送真实消息。"""
import base64
from copy import deepcopy
from datetime import datetime
import importlib
import io
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import AsyncMock, patch
import wave

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from plugin import create_plugin
from host_paths import MAIBOT_ROOT, SKIP_REASON, installed_plugin, use_maibot

SNOWLUMA = installed_plugin("maibot-team.snowluma-adapter")


class VoiceQuoteRemovalTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.p=create_plugin();self.p._settings_path=Path(self.temp.name).resolve()/"config.toml"
        config=self.p.build_default_config();config["auto_voice"].update(enabled=True,probability_percent=100,min_interval_seconds=0)
        self.p.set_plugin_config(config);await self.p.on_load()
        data=io.BytesIO()
        with wave.open(data,"wb") as f:
            f.setnchannels(1);f.setsampwidth(2);f.setframerate(16000);f.writeframes(b"\0\1"*1600)
        self.audio=data.getvalue()
        self.synth=AsyncMock(return_value=self.audio)
        patcher=patch.object(self.p,"_synthesize_audio",self.synth);patcher.start();self.addCleanup(patcher.stop)
        self.reply={"type":"reply","data":{"target_message_id":"quoted-123","target_message_content":"原消息"}}
        self.message={"message_id":"reply-text-1","timestamp":str(datetime.now().timestamp()),"platform":"qq","account_id":"bot","scope":"group",
            "message_info":{"user_info":{"user_id":"bot","user_nickname":"麦麦"},"group_info":{"group_id":"group","group_name":"test"},
                "additional_config":{"platform_io_target_group_id":"group"}},
            "raw_message":[self.reply,{"type":"text","data":"今天真开心！"}],"session_id":"stream",
            "processed_plain_text":"今天真开心！","is_mentioned":False,"is_at":False,"is_emoji":False,"is_picture":False,
            "is_command":False,"is_notify":False,"reply_to":"quoted-123"}
        self.kwargs={"typing":False,"set_reply":True,"reply_message_id":"quoted-123","storage_message":True,"show_log":False}

    async def asyncTearDown(self):await self.p.on_unload()

    async def result(self):
        return await self.p.handle_auto_voice(message=self.message,hook_name="send_service.before_send",**self.kwargs)

    async def test_converted_voice_clears_all_quote_routes_without_mutating_input(self):
        original=deepcopy(self.message)
        changed=(await self.result())["modified_kwargs"]
        self.assertFalse(changed["set_reply"])
        self.assertIsNone(changed["reply_message_id"])
        self.assertIsNone(changed["message"]["reply_to"])
        self.assertEqual([s["type"] for s in changed["message"]["raw_message"]],["voice"])
        self.assertEqual(self.message,original)
        self.assertEqual(changed["message"]["message_id"],original["message_id"])
        self.assertEqual(changed["message"]["message_info"]["additional_config"]["platform_io_target_group_id"],"group")

    async def test_failed_synthesis_keeps_original_text_and_quote(self):
        self.synth.side_effect=RuntimeError("test synthesis failed")
        original=deepcopy(self.message)
        result=await self.result()
        self.assertEqual(result["action"],"continue")
        self.assertEqual(result["modified_kwargs"]["message"],original)
        self.assertTrue(result["modified_kwargs"]["set_reply"])
        self.assertEqual(result["modified_kwargs"]["reply_message_id"],"quoted-123")
        self.assertFalse(self.p._busy)

    async def test_unselected_text_keeps_original_quote(self):
        self.p.config.auto_voice.probability_percent=0
        result=await self.result()
        self.assertEqual(result["modified_kwargs"]["message"],self.message)
        self.assertTrue(result["modified_kwargs"]["set_reply"])
        self.synth.assert_not_awaited()

    async def test_keep_text_still_removes_incompatible_quote(self):
        self.p.config.auto_voice.keep_text=True
        changed=(await self.result())["modified_kwargs"]
        self.assertEqual([s["type"] for s in changed["message"]["raw_message"]],["text","voice"])
        self.assertFalse(changed["set_reply"])
        self.assertIsNone(changed["message"]["reply_to"])

    async def test_existing_voice_quote_is_sanitized_when_automatic_conversion_disabled(self):
        self.p.config.auto_voice.enabled=False
        voice={"type":"voice","data":"已有语音","binary_data_base64":base64.b64encode(self.audio).decode()}
        self.message["raw_message"]=[self.reply,voice]
        original=deepcopy(self.message)
        changed=(await self.result())["modified_kwargs"]
        self.assertEqual(changed["message"]["raw_message"],[voice])
        self.assertFalse(changed["set_reply"])
        self.assertIsNone(changed["reply_message_id"])
        self.assertEqual(self.message,original)
        self.synth.assert_not_awaited()

    @unittest.skipIf(MAIBOT_ROOT is None or SNOWLUMA is None, SKIP_REASON + "，并安装 SnowLuma 适配器插件")
    async def test_host_preparation_and_snowluma_napcat_encoding_cannot_add_quote_back(self):
        use_maibot()
        from src.plugin_runtime.hook_payloads import deserialize_session_message,serialize_session_message
        from src.services.send_service import _prepare_message_for_platform_io
        adapter="voice_quote_adapter_fixture"
        if adapter not in sys.modules:
            package=ModuleType(adapter);package.__path__=[str(SNOWLUMA)]
            sys.modules[adapter]=package
        profile=importlib.import_module(adapter+".profile")
        codec=importlib.import_module(adapter+".codecs.outbound.message_codec").QQOutboundCodec
        changed=(await self.result())["modified_kwargs"]
        message=deserialize_session_message(changed["message"])
        await _prepare_message_for_platform_io(message,typing=False,set_reply=changed["set_reply"],reply_message_id=changed["reply_message_id"])
        outgoing=serialize_session_message(message)
        self.assertEqual([s["type"] for s in outgoing["raw_message"]],["voice"])
        for target_profile in (profile.SNOWLUMA_PROFILE,profile.NAPCAT_PROFILE):
            actions=codec(profile.ProfileState(target_profile)).build_outbound_actions(outgoing,{"group_id":"group"})
            self.assertEqual(len(actions),1)
            name,params=actions[0]
            self.assertEqual([segment["type"] for segment in params["message"]],["record"])
            self.assertEqual(params["message"][0]["data"]["file"],"base64://"+base64.b64encode(self.audio).decode())
            self.assertNotIn("reply",str(params["message"]))


if __name__=="__main__":unittest.main()
