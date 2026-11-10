# 配置项

配置保存在插件目录的 `config.toml`，完整示例见 [config.example.toml](../config.example.toml)。通常通过设置页修改即可。

| 配置 | 默认值 | 说明 |
|---|---|---|
| `api.base_url` | `http://127.0.0.1:9880` | `api_v2.py` 地址 |
| `api.timeout` | 180 | 单次请求超时（秒），上限 240 |
| `api.max_audio_bytes` | 8388608 | 音频大小上限 |
| `models.root` | 空 | 本机模型目录（GPT-SoVITS 安装目录），在第二步填写并扫描底模与训练权重 |
| `models.sovits_weights` | 空 | 第二步选择的 SoVITS 底模或完整权重 |
| `models.base_model` | `custom` | 由第二步自动识别；`custom` 表示自定义权重 |
| `role.mode` | `original` | `original` 普通模式 / `role` 角色模式 |
| `role.role_id` / `emotion` / `emotions` | — | 角色清单、默认情感、启用的情感 |
| `role.llm_emotion` | true | 启用多个情感时，合成前调用 MaiBot 的 LLM 判断情感 |
| `role.llm_task` | `fast_model` | 判断情感使用的 MaiBot 模型任务 |
| `voice.ref_audio_path` / `prompt_text` | 空 | 普通模式的参考音频和对应文本 |
| `voice.prompt_lang` / `text_lang` | `zh` | 语言代码（zh/en/ja/yue/ko/auto 等） |
| `voice.speed_factor` | 1.0 | 语速 0.5～2.0 |
| `voice.text_split_method` | `cut5` | 文本切分方式 |
| `behavior.tool_enabled` | true | 是否允许大模型调用语音工具 |
| `behavior.max_text_length` | 300 | 超过时拒绝，不会截断 |
| `behavior.cooldown_seconds` | 5 | 同一聊天流两次语音之间的间隔 |
| `behavior.allowed_stream_ids` | `[]` | 允许的聊天流 ID（不是 QQ 群号），空表示全部 |
| `behavior.send_text` | false | 语音发送后是否再发一条原文 |
| `auto_voice.*` | 关闭 | 第五步的自动语音设置 |

Windows 路径请使用正斜杠，例如 `C:/GPT-SoVITS/ref.wav`。
