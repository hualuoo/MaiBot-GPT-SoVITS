# 音色克隆

插件提供两种克隆音色的方式：

- **普通模式**：一段 3～10 秒的参考音频加上它对应的文字，即可用标准底模（或自己训练的权重）合成出相近的音色，不需要训练。
- **角色模式**：把训练好的角色打包成一个文件夹，用 `role.json` 写明每种情感使用的 GPT 权重、SoVITS 权重和参考音频；启用多个情感时可以配合 [情感分类](emotion-classification.md) 自动切换语气。

## 参考音频与角色文件放在哪里

合成时由 GPT-SoVITS 服务直接读取参考音频和模型文件，所以它们统一放在 **GPT-SoVITS 安装目录（第二步的本机模型目录）下的指定文件夹**，不要放到插件的 `data` 目录（插件不会在那里查找，而且在 MaiBot 中卸载插件时插件目录会被删除）。设置页第三步下方也有这份说明（“参考音频和角色文件放在哪里？”），没有可选的参考音频或角色时会自动展开。

```text
<GPT-SoVITS 安装目录>/
  maibot_refs/            # 普通模式的参考音频（可再建子文件夹，最多两层）
    平静.wav
    开心/开心.wav
  maibot_roles/           # 角色模式的角色文件夹
    黍/
      role.json
      shu_happy.ckpt
      shu_happy.pth
      happy.wav
```

- **普通模式**：把参考音频复制到 `maibot_refs`（3～10 秒、单人、清晰、没有背景音乐；支持 wav、mp3、flac、ogg、m4a），在第三步点击“刷新参考音频”后从下拉框选择，填写参考音频对应的文字（v3、v4、v5 底模必填，要与音频内容一字不差）、参考音频语言和合成文字语言，再点击“仅保存设置”。
- **角色模式**：把整个角色文件夹复制到 `maibot_roles`，在第三步点击“刷新角色列表”，选择角色和情感后点击“读取角色配置”。
- 文件夹不存在时请自己新建；设置页不提供网页上传。

## 角色模式

### 目录结构

角色文件只从 **GPT-SoVITS 安装目录（第二步的本机模型目录）下的 `maibot_roles`** 读取，不会读取插件 `data/`、上级目录或其他目录。每个角色一个文件夹，文件夹里放一个 **`role.json`**：

```text
<GPT-SoVITS 安装目录>/maibot_roles/
  黍/
    role.json                            # 角色配置（固定文件名）
    shu_happy.ckpt / shu_sad.ckpt …      # 各情感的 GPT 权重
    shu_v2ProPlus.pth                    # SoVITS 权重（完整权重或 LoRA）
    happy.mp3 / sad.mp3 …                # 各情感的参考音频
```

请自行把整个角色文件夹复制到这里，然后在第三步点击“刷新角色列表”。文件夹不存在时请自己新建；设置页不提供网页上传。插件自己的 `data/.state/` 只保存设置进度与普通模式配置快照。

### role.json 格式

```json
{
  "role": "黍",
  "base_model": {
    "name": "pretrained_eres2netv2w24s4ep4.ckpt",
    "size": 107528697,
    "sha256": "4f5a0bf73c61eb41b174e1bb54e7ee3c83233892be8e0af1f187024e8e581a35"
  },
  "emotions": {
    "happy": {
      "gpt": {"name": "shu_happy.ckpt", "size": 155313312, "sha256": "<64 位十六进制>"},
      "sovits": {"name": "shu_v2ProPlus.pth", "size": 172755827, "sha256": "<64 位十六进制>"},
      "ref_audio": {"name": "happy.mp3", "size": 250819, "sha256": "<64 位十六进制>"},
      "ref_text": "一千年前，我那弟弟刚来大荒城，创造了这种游戏来哄那些孤单的孩子。"
    },
    "sad": {
      "gpt": {"name": "shu_sad.ckpt"},
      "sovits": {"name": "shu_v2ProPlus.pth"},
      "ref_audio": {"name": "sad.mp3"},
      "ref_text": "……眼前的惨剧，又是何时种下的因呢。"
    }
  }
}
```

| 字段 | 必填 | 说明 |
|---|---|---|
| `role` | 否 | 角色显示名；不填时使用文件夹名 |
| `emotions` | 是 | 情感 ID → 该情感的配置（1～50 个）。`happy/angry/sad/surprised/neutral` 显示为开心/生气/难过/吃惊/中立，也可以用其他自定义 ID |
| `emotions.*.gpt` | 是 | 该情感的 GPT `.ckpt` 权重 |
| `emotions.*.sovits` | 是 | 该情感的 SoVITS `.pth` 权重：**完整权重**直接加载；**LoRA** 会叠加到 GPT-SoVITS 中同版本的标准底模上（需已安装）。版本从文件头自动识别 |
| `emotions.*.ref_audio` / `ref_text` | 是 | 参考音频（3～10 秒）及其逐字文本（1～4000 字符） |
| `emotions.*.prompt_lang` / `text_lang` | 否 | 参考音频语言、合成文字语言，默认 `zh` |
| `base_model` | 否 | 需要已安装在 GPT-SoVITS 中的底模文件（只写文件名），插件按文件名在 `GPT_SoVITS/pretrained_models` 的标准位置查找并校验，例如 `pretrained_eres2netv2w24s4ep4.ckpt` → `sv/` |

- 文件对象都写成 `{"name": ..., "size": ..., "sha256": ...}`。`name` 是相对 `role.json` 所在文件夹的路径，不允许绝对路径或 `..`；`size`（字节数）和 `sha256` 可省略，填写后会逐一校验，不一致就拒绝读取。
- `base_model` 文件必须放在 GPT-SoVITS 的标准位置：GPT-SoVITS 合成时只从那里读取，角色文件夹里的副本不会被使用。缺少时，第三步会写明应该放到哪个路径。
- 不写 `base_model` 也会检查该版本运行所需的文件：v2Pro / v2ProPlus 需要 `sv/pretrained_eres2netv2w24s4ep4.ckpt`，v4 / v5 需要对应的 `vocoder.pth`，v3 需要 BigVGAN 文件夹。

读取角色时，会逐一校验所有勾选情感的文件、大小、摘要和版本；任何一项失败都不会修改现有配置。所选情感必须使用同一个底模版本，并且要与第二步选择的版本一致（例如 v2ProPlus 角色在第二步选择 `v2Pro/s2Gv2ProPlus.pth`）。

> GPT-SoVITS 官方 API 加载 LoRA 时，会使用服务端标准目录中对应版本的底模，因此 LoRA 必须配合“标准底模”使用，不能叠加到任意自定义底模上。
