"""插件与 MaiBot 原生 LLM 语义分句协同工作；不访问真实模型和平台。"""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from plugin import create_plugin
from maibot_sdk.components import collect_components
from host_paths import MAIBOT_ROOT, SKIP_REASON, use_maibot


@unittest.skipIf(MAIBOT_ROOT is None, SKIP_REASON)
class NativeSemanticSplitTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.p=create_plugin();cfg=self.p.build_default_config()
        cfg["auto_voice"].update(enabled=True,probability_percent=100,min_interval_seconds=0)
        self.p.set_plugin_config(cfg);await self.p.on_load()
        self.synth=AsyncMock(return_value=b"RIFF-fixture-audio")
        p=patch.object(self.p,"_synthesize_audio",self.synth);p.start();self.addCleanup(p.stop)
        use_maibot()
        from src.maisaka.builtin_tool.context import BuiltinToolRuntimeContext
        self.ctx=object.__new__(BuiltinToolRuntimeContext);self.ctx.runtime=SimpleNamespace(session_id="fixture")
        self.text="原来还惦记着这一锅。好，微辣小锅，吃完就睡。"
        owner=self
        class Manager:
            async def invoke_hook(self,hook_name,**kwargs):
                for component in collect_components(owner.p):
                    if component["type"]=="HOOK_HANDLER" and component["metadata"]["hook"]==hook_name:
                        result=await getattr(owner.p,component["metadata"]["handler_name"])(hook_name=hook_name,**kwargs)
                        kwargs=result.get("modified_kwargs",kwargs)
                return SimpleNamespace(kwargs=kwargs)
        self.ctx.get_runtime_manager=lambda:Manager()

    async def asyncTearDown(self):await self.p.on_unload()

    async def test_plugin_does_not_override_maisaka_splitter_options(self):
        from src.maisaka.builtin_tool.reply import _invoke_before_post_process_hook
        hooks=[c["metadata"]["hook"] for c in collect_components(self.p) if c["type"]=="HOOK_HANDLER"]
        self.assertEqual(hooks,["send_service.before_send"])
        text,options=await _invoke_before_post_process_hook(self.ctx,self.text,"target",{})
        self.assertEqual(text,self.text)
        self.assertEqual(options,{"skip_post_process":False,"enable_splitter":True,"enable_chinese_typo":True})

    async def test_native_llm_split_preserves_model_semantic_boundaries_before_tts(self):
        from src.chat.utils import utils
        from src.chat.utils import llm_sentence_splitter
        from src.plugin_runtime.host.message_utils import PluginMessageUtils
        response='["原来还惦记着这一锅。", "好，微辣小锅，吃完就睡。"]'
        generate=AsyncMock(return_value=SimpleNamespace(response=response))
        with patch.object(utils.global_config.response_splitter,"mode","llm"), \
             patch.object(utils.global_config.response_splitter,"enable",True), \
             patch.object(utils.global_config.chinese_typo,"enable",False), \
             patch.object(llm_sentence_splitter,"LLMServiceClient",return_value=SimpleNamespace(generate_response=generate)) as factory, \
             patch.object(utils,"_split_into_sentence_segments",side_effect=AssertionError("LLM模式不能随机切句")):
            items=await self.ctx.post_process_reply_message_items_async(self.text)
        self.assertEqual(len(items),2)
        texts=[]
        for i,item in enumerate(items):
            message={"session_id":"fixture","message_id":str(i),"platform":"qq", "message_info":{"additional_config":{}},
                     "raw_message":PluginMessageUtils._message_sequence_to_dict(item.sequence)}
            result=await self.p.handle_auto_voice(message=message,set_reply=True,reply_message_id="target")
            segments=result["modified_kwargs"]["message"]["raw_message"]
            self.assertEqual([s["type"] for s in segments],["voice"])
            self.assertFalse(result["modified_kwargs"]["set_reply"])
            texts.append(segments[0]["data"])
        self.assertEqual(texts,["原来还惦记着这一锅。","好，微辣小锅，吃完就睡。"])
        self.assertEqual(self.synth.await_count,2)
        generate.assert_awaited_once()
        factory.assert_called_once_with(task_name="fast_model",request_type="response.splitter")

    async def test_llm_no_split_marker_keeps_complete_response(self):
        from src.chat.utils import llm_sentence_splitter
        generate=AsyncMock(return_value=SimpleNamespace(response='"no_split"'))
        with patch.object(llm_sentence_splitter,"LLMServiceClient",return_value=SimpleNamespace(generate_response=generate)):
            result=await llm_sentence_splitter.split_text_with_llm(self.text)
        self.assertEqual(result,[(self.text,"")])

    async def test_llm_rewrite_rejected_and_default_falls_back_unsplit(self):
        from src.chat.utils import utils
        from src.chat.utils import llm_sentence_splitter
        generate=AsyncMock(return_value=SimpleNamespace(response='["原来还惦记着这一锅。", "好。"]'))
        with patch.object(utils.global_config.response_splitter,"mode","llm"), \
             patch.object(utils.global_config.response_splitter,"enable",True), \
             patch.object(utils.global_config.chinese_typo,"enable",False), \
             patch.object(llm_sentence_splitter,"LLMServiceClient",return_value=SimpleNamespace(generate_response=generate)):
            items=await self.ctx.post_process_reply_message_items_async(self.text)
        self.assertEqual(len(items),1)
        self.assertEqual(items[0].sequence.components[0].text,self.text)


if __name__=="__main__":unittest.main()
