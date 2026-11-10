# 部署 GPT-SoVITS

插件通过 GPT-SoVITS 的 `api_v2.py` 接口合成语音，GPT-SoVITS 运行在你自己的电脑（或局域网内的另一台电脑）上，支持 NVIDIA 显卡（CUDA）、只用 CPU，以及 macOS（MPS）。

设置页第一步下方也有同样的教程（“还没有部署 GPT-SoVITS？查看部署教程”），连接测试失败时会自动展开。

## 1. 获取 GPT-SoVITS

**Windows 整合包（推荐）**：自带运行环境和全部底模，解压即用。下载最新的 `GPT-SoVITS-v5-20261005.7z`（RTX 50 系列显卡下载 `GPT-SoVITS-v5-20261005nvidia50x0.7z`），解压到不含中文和空格的目录。

Hugging Face 下载：[GPT-SoVITS-v5-20261005.7z](https://huggingface.co/lj1995/GPT-SoVITS-windows-package/resolve/main/GPT-SoVITS-v5-20261005.7z) · [nvidia50x0 版](https://huggingface.co/lj1995/GPT-SoVITS-windows-package/resolve/main/GPT-SoVITS-v5-20261005nvidia50x0.7z) · [全部版本](https://huggingface.co/lj1995/GPT-SoVITS-windows-package/tree/main)

**源码安装**（Windows / Linux / macOS，需要 Conda）：

```shell
git clone https://github.com/RVC-Boss/GPT-SoVITS
cd GPT-SoVITS
conda create -n GPTSoVits python=3.10
conda activate GPTSoVits
pwsh -F install.ps1 --Device CU128 --Source HF        # Windows
# bash install.sh --device CU128 --source HF           # Linux / macOS
```

没有 NVIDIA 显卡时 Device 填 `CPU`（macOS 可填 `MPS`），较旧显卡也可用 `CU126`。安装脚本会从 Hugging Face 下载 v1～v4、v2Pro 底模；v5 底模需另行下载（见下文“底模选择与下载”）。完整教程见 [GitHub](https://github.com/RVC-Boss/GPT-SoVITS)。

## 2. 启动 API 服务

插件需要的是 `api_v2.py` 接口，不是 `go-webui.bat` 打开的网页界面。在 GPT-SoVITS 根目录运行：

```shell
python api_v2.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS/configs/tts_infer.yaml
```

整合包用户可以在根目录新建 `go-api.bat`：

```bat
@echo off
cd /d "%~dp0"
set "PATH=%~dp0runtime;%PATH%"
runtime\python.exe api_v2.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS/configs/tts_infer.yaml
pause
```

浏览器访问 `http://127.0.0.1:9880/docs` 能看到接口文档即表示服务已就绪，然后在设置页第一步填写 `http://127.0.0.1:9880` 并测试连接。

- 9880 是 API 端口；9872、9874 是 Gradio 网页界面，不能填在插件里。
- 只有 CPU：把 `GPT_SoVITS/configs/tts_infer.yaml` 的 `custom` 段改为 `device: cpu`、`is_half: false`；CPU 合成较慢，建议用 v2 / v2Pro，并调大请求超时。
- GPT-SoVITS 在其他机器：用 `-a 0.0.0.0` 启动、在防火墙放行端口，插件填写那台机器的局域网地址，不要暴露到公网；GPT-SoVITS 在 Docker 中时，容器内的 127.0.0.1 指容器本身。
- 所有模型和参考音频都必须放在 **GPT-SoVITS 服务端能读到的位置**：参考音频放在 GPT-SoVITS 安装目录下的 `maibot_refs`，角色文件夹放在 `maibot_roles`（见 [音色克隆](voice-cloning.md)）。插件需要扫描这两个文件夹，所以插件和 GPT-SoVITS 要在同一台电脑上，或能用同一路径访问。
- 切换权重会影响整个 GPT-SoVITS 服务，同一服务的其他调用方也会受影响。

## 底模选择与下载

设置页第二步下方也有这份说明（“该选哪个底模？底模下载地址”），扫描不到任何 SoVITS 权重时会自动展开。

- 有 NVIDIA 显卡、想要最像参考音频：**v5turbo**（不训练也有更高的音色相似度，48 kHz）。
- 只有 CPU 或想要更快：**v2** / **v2Pro**（32 kHz）。
- 自己训练、训练素材音质一般：**v2Pro** / **v2ProPlus**。
- 已经有某个版本训练好的模型：选与它相同版本的底模。

| 版本 | 特点 | 需要的文件（`GPT_SoVITS/pretrained_models/` 下） |
|---|---|---|
| v5turbo | 最新（2026-10），默认 4 步采样，较快；48 kHz；参考文本必填 | `s1v3.ckpt`、`gsv-v5-pretrained/s2Gv5turbo.pth`、`gsv-v5-pretrained/vocoder.pth` |
| v5dev | 同代，默认 32 步采样并启用 CFG，明显更慢；参考文本必填 | `s1v3.ckpt`、`gsv-v5-pretrained/s2Gv5dev.pth`、`gsv-v5-pretrained/vocoder.pth` |
| v2ProPlus | v2Pro 的加大版本，效果更好、略慢 | `s1v3.ckpt`、`v2Pro/s2Gv2ProPlus.pth`、`sv/pretrained_eres2netv2w24s4ep4.ckpt` |
| v2Pro | 显存略高于 v2，效果超过 v4，保留 v2 的速度；32 kHz | `s1v3.ckpt`、`v2Pro/s2Gv2Pro.pth`、`sv/pretrained_eres2netv2w24s4ep4.ckpt` |
| v4 | 修复 v3 金属音，48 kHz；参考文本必填 | `s1v3.ckpt`、`gsv-v4-pretrained/s2Gv4.pth`、`gsv-v4-pretrained/vocoder.pth` |
| v3 | 24 kHz，已被 v4 / v5 取代；参考文本必填 | `s1v3.ckpt`、`s2Gv3.pth`、`models--nvidia--bigvgan_v2_24khz_100band_256x/` |
| v2 | 中日英韩粤，速度快，CPU 友好；32 kHz | `gsv-v2final-pretrained/` 下的 `s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt`、`s2G2333k.pth` |
| v1 | 最早版本，不推荐 | `s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt`、`s2G488k.pth` |

所有版本还需要 `chinese-hubert-base/`、`chinese-roberta-wwm-ext-large/`；中文需要 G2PW 模型（解压为 `G2PWModel` 放到 `GPT_SoVITS/text/`）。v3 / v4 / v5 训练出的 SoVITS 是 LoRA，要配合同版本标准底模在第三步使用；v1 / v2 / v2Pro 训练出的是完整权重，可直接在第二步选择。

下载地址（使用整合包时底模已经齐全，无需下载）：

- 全部底模：[Hugging Face lj1995/GPT-SoVITS](https://huggingface.co/lj1995/GPT-SoVITS/tree/main)；v5 在 [gsv-v5-pretrained](https://huggingface.co/lj1995/GPT-SoVITS/tree/main/gsv-v5-pretrained)（`s2Gv5turbo.pth` / `s2Gv5dev.pth` 各约 731 MB，`vocoder.pth` 约 55 MB，另需 `s1v3.ckpt` 约 148 MB）。
- v5 逐个文件直接下载：[s2Gv5turbo.pth](https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/gsv-v5-pretrained/s2Gv5turbo.pth) · [vocoder.pth](https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/gsv-v5-pretrained/vocoder.pth) · [s1v3.ckpt](https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/s1v3.ckpt)
- v1～v4、v2Pro 打包（约 4.3 GB，不含 v5）：[pretrained_models.zip](https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/pretrained_models.zip)
- G2PW 中文模型：[G2PWModel.zip](https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/G2PWModel.zip)
- [各版本特性（GitHub Wiki）](https://github.com/RVC-Boss/GPT-SoVITS/wiki/GPT%E2%80%90SoVITS%E2%80%90features-(%E5%90%84%E7%89%88%E6%9C%AC%E7%89%B9%E6%80%A7))

放好后在第二步点击“扫描本机目录”并选择对应的 SoVITS 文件即可，版本自动识别；插件通过接口切换模型，不需要重启 API。
