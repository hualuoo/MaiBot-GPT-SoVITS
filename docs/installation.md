# 安装

## 环境要求

| 项目 | 要求 |
|---|---|
| 插件 ID | `local.gpt-sovits-tts` |
| MaiBot | 1.3.6 ～ 1.4.x |
| Python | 3.12 及以上（使用 MaiBot 的运行环境） |
| 插件 SDK | maibot-plugin-sdk 2.20+（验证版本 2.21.0） |
| Python 依赖 | aiohttp、tomlkit（MaiBot 环境通常已自带） |
| GPT-SoVITS | 需运行 `api_v2.py`（不是 Gradio WebUI，也不是旧版 `api.py`） |

GPT-SoVITS 的部署方法见 [部署 GPT-SoVITS](gpt-sovits.md)。

## 方式一：在 MaiBot WebUI 导入 ZIP（推荐）

1. 获取发行包 `MaiBot-GPT-SoVITS-<版本>.zip`（或在源码目录运行 `python build_release.py` 生成，产物在 `dist/`）。
2. 打开 MaiBot WebUI → **插件管理** → **从 ZIP 安装**，选择该文件。
3. 插件需要被加载后才能配置：若安装提示“请重启麦麦使其生效”，请重启 MaiBot。首次加载时会根据默认值自动生成 `config.toml`。
4. 打开顶部工作区 **GPT-SoVITS 语音设置**，按五个步骤完成配置。

安装并加载后，可以在 MaiBot 的「插件配置」中看到插件图标、版本和简介：

![MaiBot「插件配置」中的插件条目：图标、版本与简介](images/plugin-list.png)

> 插件在 `_manifest.json` 中申请了 `llm.generate` 与 `llm.get_available_models` 能力，用于多情感时调用 MaiBot 的 LLM；不使用多情感角色时不会调用。

> **更新已安装的插件**：MaiBot 的 ZIP 安装不会覆盖已有插件，必须先卸载旧版本。**卸载会直接删除整个插件目录（包括 `config.toml` 和设置进度），不会进回收站**：卸载前请先备份 `config.toml`。角色文件和参考音频放在 GPT-SoVITS 目录下的 `maibot_roles`、`maibot_refs`，卸载插件不受影响。更推荐用方式二更新，配置会保留。

## 方式二：用 install.ps1 安装或更新（保留配置）

在源码目录运行，`-MaiBotPath` 填你的 MaiBot 根目录（包含 `bot.py` 的文件夹）：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File ./install.ps1 -MaiBotPath "D:/MaiBot" -Force
```

脚本会按插件 ID 找到已安装的目录（包括通过 ZIP 安装的 `local_gpt-sovits-tts`），只覆盖程序文件，**保留 `config.toml` 和 `data/`**；完成后在 MaiBot 中重载插件即可。首次安装可省略 `-Force`。

## 依赖

依赖要装在 **MaiBot 运行的 Python 环境**里（不是 GPT-SoVITS 的环境）：

```shell
python -m pip install -r requirements.txt
```
