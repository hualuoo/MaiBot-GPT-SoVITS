"""自动语音 Hook 与 MaiBot 消息序列化/语音编码的集成，不发送平台消息。"""
import base64
from copy import deepcopy
from datetime import datetime
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, patch
import wave

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from plugin import create_plugin
from host_paths import MAIBOT_ROOT, SKIP_REASON, use_maibot


class AutoVoiceHostTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.p=create_plugin();self.p._settings_path=Path(self.temp.name).resolve()/"config.toml"
        cfg=self.p.build_default_config();cfg["auto_voice"].update(enabled=True,probability_percent=100,min_interval_seconds=0)
        cfg["role"].update(mode="role",role_id="shu/role.json",emotion="neutral",emotions=["neutral","happy"])
        self.p.set_plugin_config(cfg);await self.p.on_load()
        data=io.BytesIO()
        with wave.open(data,"wb") as f:f.setnchannels(1);f.setsampwidth(2);f.setframerate(16000);f.writeframes(b"\0\1"*1600)
        self.audio=data.getvalue()
        self.synth=AsyncMock(return_value=self.audio)
        self.patch=patch.object(self.p,"_synthesize_audio",self.synth);self.patch.start();self.addCleanup(self.patch.stop)
        self.message={"message_id":"auto-text-1","timestamp":str(datetime.now().timestamp()),"received_at":str(datetime.now().timestamp()),
            "platform":"qq","account_id":"bot","scope":"group",
            "message_info":{"user_info":{"user_id":"bot","user_nickname":"麦麦","user_cardname":None},
                "group_info":{"group_id":"group","group_name":"test"},"additional_config":{"platform_io_target_group_id":"group"}},
            "raw_message":[{"type":"text","data":"今天真开心！"}],"session_id":"stream","processed_plain_text":"今天真开心！",
            "is_mentioned":False,"is_at":False,"is_emoji":False,"is_picture":False,"is_command":False,"is_notify":False,"reply_to":None}

    async def asyncTearDown(self):await self.p.on_unload()

    @unittest.skipIf(MAIBOT_ROOT is None, SKIP_REASON)
    async def test_hook_message_roundtrips_through_host_voice_model(self):
        use_maibot()
        from src.plugin_runtime.hook_payloads import deserialize_session_message,serialize_session_message
        before=deepcopy(self.message)
        kwargs={"typing":True,"set_reply":True,"reply_message_id":"original-reply","storage_message":True,"show_log":False}
        result=await self.p.handle_auto_voice(message=self.message,hook_name="send_service.before_send",**kwargs)
        self.assertEqual(result["action"],"continue")
        changed=result["modified_kwargs"]
        self.assertFalse(changed["set_reply"])
        self.assertIsNone(changed["reply_message_id"])
        self.assertIsNone(changed["message"]["reply_to"])
        for key,value in kwargs.items():
            if key not in ("set_reply", "reply_message_id"):self.assertEqual(changed[key],value)
        msg=deserialize_session_message(changed["message"])
        recovered=serialize_session_message(msg)
        voice=recovered["raw_message"][0]
        self.assertEqual(voice["type"],"voice")
        self.assertEqual(base64.b64decode(voice["binary_data_base64"]),self.audio)
        self.assertEqual(recovered["session_id"],"stream")
        self.assertEqual(recovered["message_info"]["additional_config"]["platform_io_target_group_id"],"group")
        self.assertEqual(self.message,before)
        self.assertEqual(self.synth.await_args.kwargs["emotion"],"happy")

    async def test_failures_preserve_original_and_no_new_send_is_invoked(self):
        self.synth.side_effect=RuntimeError("模型尚未就绪")
        before=deepcopy(self.message)
        result=await self.p.handle_auto_voice(message=self.message,typing=False,set_reply=False,storage_message=True,show_log=False)
        self.assertEqual(result["action"],"continue")
        self.assertEqual(result["modified_kwargs"]["message"],before)
        self.assertFalse(self.p._busy)
        self.assertEqual(self.p._auto_stats["failed"],1)

    async def test_save_auto_voice_only_updates_seventh_config_section(self):
        self.p.config.auto_voice.enabled=False
        other={key:value for key,value in self.p.config.model_dump().items() if key!="auto_voice"}
        result=await self.p.setup_save_auto_voice(True,"every_n",70,3,.5,True,False)
        self.assertTrue(result["success"],result)
        self.assertEqual({key:value for key,value in self.p.config.model_dump().items() if key!="auto_voice"},other)
        self.assertEqual(self.p.config.auto_voice.every_n,3)
        self.synth.assert_not_awaited()


if __name__=="__main__":unittest.main()
