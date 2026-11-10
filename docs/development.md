# 开发

```text
plugin.py            插件入口：命令、工具、合成调度
auto_voice.py        自动语音 Hook 与情感规则
emotion_judge.py     多情感时调用 MaiBot LLM 判断情感
tts_log.py           每次合成的“开始/结果”日志
maibot-patch/        对 MaiBot 本体的可选修改（日志天蓝色），不进入发行包
setup_service.py     设置页后端 API（接口、底模、参数、任务、试听）
role_service.py      角色模式 API（读取 GPT-SoVITS 目录下 maibot_roles 中的角色，不提供上传）
role_library.py      role.json 解析与资源校验
model_catalog.py     本地模型与 maibot_refs 参考音频扫描（只读权重文件头，不反序列化）
native_config.py     原生配置页适配
wizard_progress.py   设置进度记录
tts_client.py        GPT-SoVITS HTTP 客户端
config_models.py     配置模型
webui.json / webui/  设置页声明与页面
assets/              插件图标 maibot-microphone-chibi-256.png（_manifest.json 的 display.icon）
docs/                使用文档与截图（docs/images/），随发行包附带
scripts/             文档截图脚本 capture_screenshots.py
tests/               单元测试与集成测试（host_paths.py 负责定位 MaiBot 源码）
```

以下命令中的 `python` 指 MaiBot 所用的 Python 环境，`<MaiBot 目录>` 是包含 `bot.py` 的文件夹。

运行测试（不需要真实模型，也不会发送真实消息）：

```powershell
python -B -m unittest discover -s tests
node --test tests/test_setup_ui.mjs
python -B tests/validate_host.py <MaiBot 目录>
# 可选：需要 Playwright + Chrome，以及在 8001 端口运行的 MaiBot WebUI（地址可用 MAIBOT_WEBUI 修改）
python -B tests/browser_setup_page_check.py
```

需要 MaiBot 源码的测试会读取环境变量 `MAIBOT_ROOT`，未设置时使用与本项目并列的 `MaiBot` 文件夹，找不到时这些测试会跳过。部分集成测试会导入 MaiBot 源码，运行时可能在当前目录生成 `logs/`（已加入 `.gitignore`）。

生成发行包并用 MaiBot 自身的校验逻辑检查：

```powershell
python -B build_release.py --verify-host <MaiBot 目录>
```

重新生成 README 与文档中的截图（需要 Playwright + Chrome、在 8001 端口运行的 MaiBot WebUI，以及放有角色文件夹的本机 GPT-SoVITS；不会加载模型或合成语音，页面上的路径统一显示为 `D:/GPT-SoVITS`）：

```powershell
python -B scripts/capture_screenshots.py --gsv <GPT-SoVITS 目录> --role 黍 --maibot <MaiBot 目录>
```
