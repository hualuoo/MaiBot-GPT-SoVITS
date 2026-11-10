"""Real SDK tests; run with maibot/python.exe -B -m unittest discover -s tests -p test_plugin.py -v.

Only client and _ctx are mocked; neither MaiBot SDK nor config models are stubbed.
"""
from __future__ import annotations
import asyncio
import base64
import inspect
from pathlib import Path
import re
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from maibot_sdk import MaiBotPlugin
from maibot_sdk.components import collect_components
import plugin as module


class PluginTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.plugin = module.create_plugin()
        self.data = self.plugin.build_default_config()
        self.data['voice']['ref_audio_path'] = '/server/references/voice.wav'
        self.data['behavior']['cooldown_seconds'] = 0
        self.plugin.set_plugin_config(self.data)
        self.send = SimpleNamespace(custom=AsyncMock(return_value={'sent': True}), text=AsyncMock())
        self.plugin._ctx = SimpleNamespace(send=self.send)
        self.logger = Mock()
        p = patch.object(self.plugin, '_get_logger', return_value=self.logger)
        p.start()
        self.addCleanup(p.stop)
        self.audio = b'RIFF\x00\x01\x80\xffWAVE-test-audio'
        self.client = SimpleNamespace(synthesize=AsyncMock(return_value=self.audio), probe=AsyncMock(return_value='API reachable'))
        p = patch.object(module, 'GPTSoVITSClient', return_value=self.client)
        self.factory = p.start()
        self.addCleanup(p.stop)
        await self.plugin.on_load()

    async def asyncTearDown(self):
        await self.plugin.on_unload()
        self.assertFalse(self.plugin._busy)
        self.assertEqual(self.plugin._tasks, set())

    def configure(self, section, **changes):
        self.data[section].update(changes)
        self.plugin.set_plugin_config(self.data)

    @staticmethod
    async def cancel_task(task):
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    async def blocked_request(self):
        entered, release = asyncio.Event(), asyncio.Event()
        async def synthesize(payload):
            entered.set()
            await release.wait()
            return self.audio
        self.client.synthesize.side_effect = synthesize
        task = asyncio.create_task(self.plugin._speak('stream', 'first'))
        self.addAsyncCleanup(self.cancel_task, task)
        await asyncio.wait_for(entered.wait(), 2)
        return task, release

    async def test_create_plugin_real_sdk_and_independent_state(self):
        self.assertIsInstance(self.plugin, MaiBotPlugin)
        self.assertIsInstance(self.plugin.config, module.TTSConfig)
        other = module.create_plugin()
        self.assertIsNot(other, self.plugin)
        self.assertIsNot(other._tasks, self.plugin._tasks)
        self.assertIsNot(other._last_sent, self.plugin._last_sent)
        self.assertFalse(other._active)
        self.factory.assert_not_called()

    async def test_real_sdk_collect_components_build_schema(self):
        components = {c['name']: c for c in collect_components(self.plugin)}
        self.assertEqual({name for name, c in components.items() if c['type'] not in ('API', 'HOOK_HANDLER')},
                         {'gpt_sovits_tts', 'gpt_sovits_tts_status', 'gpt_sovits_speak'})
        self.assertEqual(len([c for c in components.values() if c['type'] == 'API']), 20)
        command = components['gpt_sovits_tts']
        self.assertEqual(command['type'], 'COMMAND')
        self.assertEqual(command['metadata']['handler_name'], 'handle_tts')
        self.assertEqual(command['metadata']['timeout_ms'], 300000)
        self.assertEqual(command['metadata']['command_pattern'], module.TTS_PATTERN)
        tool = components['gpt_sovits_speak']
        self.assertEqual(tool['type'], 'TOOL')
        self.assertTrue(tool['metadata']['parameters_raw']['text']['required'])
        schema = self.plugin.build_config_schema(plugin_id='gpt_sovits', plugin_name='GPT-SoVITS')
        self.assertEqual(schema['plugin_id'], 'gpt_sovits')
        self.assertEqual(set(schema['sections']), {'plugin', 'api', 'models', 'role', 'voice', 'behavior', 'auto_voice'})
        api = schema['sections']['api']['fields']
        self.assertEqual(api['timeout']['max'], 240)
        self.assertEqual(api['max_audio_bytes']['default'], 8388608)
        voice = schema['sections']['voice']['fields']
        self.assertIn('cut5', voice['text_split_method']['choices'])
        self.assertEqual(voice['aux_ref_audio_paths']['item_type'], 'string')

    async def test_runner_self_scope_reload_resets_cooldown(self):
        # Local Runner runner_main.py:2450-2460 applies config BEFORE callback;
        # scope is self, not plugin. Keep this regression strict (no xfail).
        self.assertIn('self', inspect.getdoc(MaiBotPlugin.on_config_update))
        self.plugin._last_sent['stream'] = 123.0
        self.data['voice']['speed_factor'] = 1.5
        self.plugin.set_plugin_config(self.data)
        await self.plugin.on_config_update('self', self.data, '2')
        self.assertEqual(self.plugin.config.voice.speed_factor, 1.5)
        self.assertEqual(self.plugin._last_sent, {})

    async def test_global_config_notifications_do_not_replace_config(self):
        self.plugin._last_sent['stream'] = 123.0
        for scope in ('bot', 'model'):
            await self.plugin.on_config_update(scope, {'unrelated': True}, '2')
        self.assertEqual(self.plugin.config.voice.ref_audio_path, self.data['voice']['ref_audio_path'])
        self.assertEqual(self.plugin._last_sent, {'stream': 123.0})

    async def test_command_parameters_multiline_and_missing_argument(self):
        cases = [
            ({'matched_groups': {'text': '  one\ntwo  '}}, 'one\ntwo'),
            ({'text': '/tts first\nsecond'}, 'first\nsecond'),
            ({'matched_groups': None, 'text': '/tts\t你好'}, '你好'),
            ({'matched_groups': 'invalid', 'text': '/tts fallback'}, 'fallback'),
        ]
        for kwargs, expected in cases:
            with self.subTest(kwargs=kwargs):
                result = await self.plugin.handle_tts(stream_id='stream', **kwargs)
                self.assertEqual((result[0], result[2]), (True, True))
                self.assertEqual(self.client.synthesize.await_args.args[0]['text'], expected)
        before = self.client.synthesize.await_count
        for raw in ('/tts', '/tts   ', '/tts_status', '/ttsx nope'):
            with self.subTest(raw=raw):
                result = await self.plugin.handle_tts(stream_id='stream', text=raw)
                self.assertFalse(result[0])
                self.assertIn('请输入', result[1])
        self.assertEqual(self.client.synthesize.await_count, before)
        self.assertIsNone(re.match(module.TTS_PATTERN, '/tts_status'))

    async def test_success_exact_base64_voice_cpu_payload(self):
        self.configure('voice', text_lang='ZH', prompt_lang='EN')
        result = await self.plugin.handle_speak(text='  hello\nworld  ', stream_id='stream')
        self.assertTrue(result['success'])
        self.assertIn('已发送', result['content'])
        self.send.custom.assert_awaited_once_with('voice', base64.b64encode(self.audio).decode('ascii'), 'stream', processed_plain_text='hello\nworld')
        self.assertEqual(base64.b64decode(self.send.custom.await_args.args[1], validate=True), self.audio)
        payload = self.client.synthesize.await_args.args[0]
        self.assertEqual(payload['text_lang'], 'zh')
        self.assertEqual(payload['prompt_lang'], 'en')
        self.assertEqual(payload['media_type'], 'wav')
        self.assertEqual(payload['batch_size'], 1)
        for flag in ('streaming_mode', 'gpt_streaming', 'use_cuda_graph', 'use_flash_attention'):
            self.assertIs(payload[flag], False)
        self.factory.assert_called_once_with(self.data['api']['base_url'], timeout=180, max_audio_bytes=8388608)
        self.send.text.assert_not_awaited()
        self.assertIn('stream', self.plugin._last_sent)

    async def test_send_failure_no_success_or_cooldown(self):
        for response in ({'sent': False}, {}, False, None):
            with self.subTest(response=response):
                self.send.custom.return_value = response
                result = await self.plugin.handle_speak(text='hello', stream_id='stream')
                self.assertFalse(result['success'])
                self.assertIn('发送失败', result['content'])
                self.assertFalse(self.plugin._busy)
                self.assertNotIn('stream', self.plugin._last_sent)

    async def test_send_exception_releases_busy_and_allows_retry(self):
        self.send.custom.side_effect = RuntimeError('adapter offline')
        result = await self.plugin.handle_speak(text='hello', stream_id='stream')
        self.assertFalse(result['success'])
        self.assertFalse(self.plugin._busy)
        self.assertEqual(self.plugin._tasks, set())
        self.assertEqual(self.plugin._last_sent, {})
        self.send.custom.side_effect = None
        self.assertTrue((await self.plugin.handle_speak(text='retry', stream_id='stream'))['success'])

    async def test_whitespace_length_reference_validation(self):
        self.configure('behavior', max_text_length=3)
        for text, error in ((' \t\n', '请输入'), (None, '请输入'), ('四个字符', '超过 3')):
            with self.subTest(text=text):
                with self.assertRaisesRegex(module.TTSError, error):
                    await self.plugin._speak('stream', text)
        self.configure('voice', ref_audio_path=' \t ')
        with self.assertRaisesRegex(module.TTSError, 'ref_audio_path'):
            await self.plugin._speak('stream', '好')
        self.factory.assert_not_called()
        self.assertFalse(self.plugin._busy)
        self.configure('voice', ref_audio_path='/remote/only.wav')
        await self.plugin._speak('stream', '  正好三  ')
        self.assertEqual(self.client.synthesize.await_args.args[0]['text'], '正好三')

    async def test_disabled_tool_allows_command(self):
        self.configure('behavior', tool_enabled=False)
        result = await self.plugin.handle_speak(text='hello', stream_id='stream')
        self.assertFalse(result['success'])
        self.assertIn('已关闭', result['content'])
        self.factory.assert_not_called()
        self.assertTrue((await self.plugin.handle_tts(stream_id='stream', text='/tts hello'))[0])

    async def test_access_disabled_missing_stream_allowlist(self):
        with self.assertRaisesRegex(module.TTSError, '聊天流 ID'):
            await self.plugin._speak('', 'hello')
        self.configure('behavior', allowed_stream_ids=['allowed'])
        with self.assertRaisesRegex(module.TTSError, '未获准'):
            await self.plugin._speak('stream', 'hello')
        self.configure('plugin', enabled=False)
        with self.assertRaisesRegex(module.TTSError, '未启用'):
            await self.plugin._speak('allowed', 'hello')
        self.factory.assert_not_called()

    async def test_concurrent_busy_no_second_synthesis(self):
        task, release = await self.blocked_request()
        self.assertTrue(self.plugin._busy)
        self.assertIn(task, self.plugin._tasks)
        result = await self.plugin.handle_speak(text='second', stream_id='other')
        self.assertFalse(result['success'])
        self.assertIn('正在合成', result['content'])
        self.assertEqual(self.client.synthesize.await_count, 1)
        release.set()
        await asyncio.wait_for(task, 2)
        self.assertFalse(self.plugin._busy)
        self.assertEqual(self.plugin._tasks, set())

    async def test_cooldown_per_stream_expiry_pruning(self):
        self.configure('behavior', cooldown_seconds=10)
        clock = SimpleNamespace(monotonic=Mock(return_value=100.0))
        # Do not patch global time.monotonic used by the event loop.
        with patch.object(module, 'time', clock):
            await self.plugin._speak('stream', 'first')
            clock.monotonic.return_value = 105.0
            with self.assertRaisesRegex(module.TTSError, '冷却'):
                await self.plugin._speak('stream', 'second')
            await self.plugin._speak('other', 'independent')
            self.assertEqual(self.client.synthesize.await_count, 2)
            clock.monotonic.return_value = 110.0
            await self.plugin._speak('stream', 'expired')
            self.assertEqual(self.plugin._last_sent['stream'], 110.0)
            clock.monotonic.return_value = 121.0
            await self.plugin._speak('third', 'prune')
            self.assertEqual(self.plugin._last_sent, {'third': 121.0})

    async def test_client_errors_release_busy(self):
        for error in (module.TTSError('unavailable'), RuntimeError('broken transport')):
            with self.subTest(error=type(error).__name__):
                self.client.synthesize.side_effect = error
                result = await self.plugin.handle_speak(text='hello', stream_id='stream')
                self.assertFalse(result['success'])
                self.assertFalse(self.plugin._busy)
                self.assertEqual(self.plugin._tasks, set())
                self.assertEqual(self.plugin._last_sent, {})
                self.client.synthesize.side_effect = None
                self.assertTrue((await self.plugin.handle_speak(text='retry', stream_id='stream'))['success'])
                self.plugin._last_sent.clear()

    async def test_cancel_propagates_releases_busy(self):
        task, release = await self.blocked_request()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertFalse(self.plugin._busy)
        self.assertEqual(self.plugin._tasks, set())
        self.send.custom.assert_not_awaited()
        self.assertEqual(self.plugin._last_sent, {})
        self.client.synthesize.side_effect = None
        await self.plugin._speak('stream', 'retry')

    async def test_unload_cancels_inflight_and_blocks_new_requests(self):
        task, release = await self.blocked_request()
        self.plugin._last_sent['old'] = 1
        await asyncio.wait_for(self.plugin.on_unload(), 2)
        self.assertTrue(task.cancelled())
        self.assertFalse(self.plugin._active)
        self.assertFalse(self.plugin._busy)
        self.assertEqual(self.plugin._tasks, set())
        self.assertEqual(self.plugin._last_sent, {})
        self.send.custom.assert_not_awaited()
        with self.assertRaisesRegex(module.TTSError, '未启用'):
            await self.plugin._speak('stream', 'after unload')
        await self.plugin.on_unload()

    async def test_inflight_configuration_snapshot(self):
        task, release = await self.blocked_request()
        self.configure('behavior', send_text=True)
        self.configure('voice', speed_factor=2)
        release.set()
        await asyncio.wait_for(task, 2)
        self.assertEqual(self.client.synthesize.await_args.args[0]['speed_factor'], 1)
        self.send.text.assert_not_awaited()

    async def test_optional_text_failure_keeps_voice_success(self):
        self.configure('behavior', send_text=True)
        self.send.text.side_effect = RuntimeError('text failed')
        result = await self.plugin.handle_speak(text='hello', stream_id='stream')
        self.assertTrue(result['success'])
        self.send.text.assert_awaited_once_with('hello', 'stream')
        self.logger.warning.assert_called_once()
        self.assertIn('stream', self.plugin._last_sent)

    async def test_status_without_synthesis(self):
        result = await self.plugin.handle_status(stream_id='stream')
        self.assertTrue(result[0])
        self.assertIn('已配置', result[1])
        self.factory.assert_called_once_with(self.data['api']['base_url'], timeout=5)
        self.client.probe.assert_awaited_once()
        self.client.synthesize.assert_not_awaited()
        self.send.custom.assert_not_awaited()
        self.send.text.assert_awaited_once()


if __name__ == '__main__':
    unittest.main()
