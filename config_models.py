"""强类型配置：与适配器等原生设置页一致，字段显式声明中文 label。"""
from typing import Annotated, Literal

from pydantic import model_validator

from maibot_sdk import Field, PluginConfigBase

NO_LORA = "（不使用 LoRA）"
UNSELECTED = "（请选择模型）"


def CN(label: str, **kwargs):
    extra = {"label": label, **kwargs.pop("json_schema_extra", {})}
    return Field(**kwargs, json_schema_extra=extra)


class PluginSection(PluginConfigBase):
    __ui_label__ = "插件设置"
    __ui_order__ = 0
    enabled: bool = CN("启用插件", default=True, description="是否启用语音合成和设置向导")
    config_version: str = CN("配置版本", default="1.0.0", description="插件配置结构版本", json_schema_extra={"hidden": True})


class APIConfig(PluginConfigBase):
    __ui_label__ = "第一步 · 接口配置与测试"
    __ui_order__ = 1
    base_url: str = CN("GPT-SoVITS 接口地址", default="http://127.0.0.1:9880", max_length=2048, description="填写 api_v2.py HTTP 服务地址，不是 Gradio WebUI 地址；请在语音设置页测试连接")
    timeout: float = CN("请求超时（秒）", default=180, ge=1, le=240, description="CPU 推理可适当增加；模型与语音请求上限240秒")
    max_audio_bytes: int = CN("音频大小上限（字节）", default=8388608, ge=1024, le=16777216, description="还需满足 MaiBot RPC 帧大小限制；试听使用分块读取")


class ModelConfig(PluginConfigBase):
    __ui_label__ = "底模与普通模式权重"
    __ui_order__ = 2
    root: str = CN("本机模型目录", default="", max_length=1024, description="GPT-SoVITS 安装目录，用于扫描底模和训练权重（GPT_SoVITS/pretrained_models、GPT_weights*、SoVITS_weights*）")
    base_model: str = CN("底模版本", default="custom", max_length=80, description="由设置页根据所选 SoVITS 权重自动识别；custom 表示自定义权重", json_schema_extra={"x-widget": "select"})
    gpt_weights: str = CN("GPT 权重路径", default="", max_length=1024, description="服务端可读的 .ckpt 权重路径；请在语音设置页点击加载并测试")
    sovits_weights: str = CN("SoVITS 底模 / 完整权重", default="", max_length=1024, description="选择完整底模或非LoRA微调权重；LoRA请在下一项单独选择。未收录路径可在语音设置页手填。")
    lora_weights: str = CN("LoRA 角色权重（可选）", default=NO_LORA, max_length=1024, description="默认不使用LoRA。需要 .pth 角色文件；GPT 的 .ckpt 不是LoRA。空列表表示本机尚未发现对应文件。", json_schema_extra={"x-widget": "select"})


class RoleConfig(PluginConfigBase):
    __ui_label__ = "第三步 · 模式与角色情感"
    __ui_order__ = 2
    mode: Literal["original", "role"] = CN("使用模式", default="original", description="original为原版手动模式；role自动读取 GPT-SoVITS 目录下 maibot_roles 中的角色JSON")
    role_id: str = CN("角色清单", default="", max_length=240, description="在语音设置页角色模式选择 maibot_roles 中的角色JSON")
    emotion: str = CN("默认试听语气", default="neutral", min_length=1, max_length=80, description="对应JSON顶层键，也是所选语气中的默认试听项")
    emotions: list[Annotated[str, Field(strict=True, min_length=1, max_length=80)]] = CN("启用的角色语气", default_factory=list, max_length=50, description="最多50项，每项最多80字符；空列表表示只启用默认语气")
    llm_emotion: bool = CN("多情感时用 LLM 判断情感", default=True, description="角色模式且启用 2 个以上情感时，合成前调用 MaiBot 的 LLM 判断每句话的情感，再选用该情感的 GPT/LoRA/参考音频；失败或超时使用默认情感")
    llm_task: str = CN("情感判断模型任务", default="fast_model", min_length=1, max_length=80, description="MaiBot 模型任务名（如 fast_model、utils）；fast_model 留空时 MaiBot 会自动使用 utils")

    @model_validator(mode="after")
    def validate_emotions(self):
        if len(set(self.emotions)) != len(self.emotions):
            raise ValueError("角色语气不能重复")
        if self.emotions and self.emotion not in self.emotions:
            raise ValueError("默认试听语气必须在所选语气中")
        return self


class VoiceConfig(PluginConfigBase):
    __ui_label__ = "参考音频与推理参数"
    __ui_order__ = 3
    ref_audio_path: str = CN("参考音频路径", default="", max_length=1024, description="放在 GPT-SoVITS 安装目录下的 maibot_refs，在设置页第三步选择；3~10秒")
    prompt_text: str = CN("参考音频文本", default="", max_length=4000, description="与参考音频逐字对应的文本")
    prompt_lang: str = CN("参考音频语言", default="zh", min_length=1, max_length=32, description="例如 zh/en/ja/yue/ko")
    text_lang: str = CN("合成文本语言", default="zh", min_length=1, max_length=32, description="例如 zh/en/ja/auto")
    aux_ref_audio_paths: list[str] = CN("辅助参考音频", default_factory=list, description="可选的服务端辅助参考音频路径")
    text_split_method: Literal["cut0", "cut1", "cut2", "cut3", "cut4", "cut5"] = CN("文本切分方式", default="cut5", description="cut0不切、cut1四句、cut2约50字、cut3中文句号、cut4英文句号、cut5标点切分")
    speed_factor: float = CN("语速", default=1.0, ge=0.5, le=2.0, description="数值越大语速越快")
    top_k: int = CN("候选采样数量", default=5, ge=1, le=100, description="GPT 采样参数 top_k")
    top_p: float = CN("累计采样概率", default=1.0, gt=0, le=1, description="GPT 采样参数 top_p")
    temperature: float = CN("采样温度", default=1.0, gt=0, le=2, description="GPT 采样温度")
    seed: int = CN("随机种子", default=-1, ge=-1, le=2147483647, description="-1为随机，固定数字用于复现")
    parallel_infer: bool = CN("并行推理", default=False, description="CPU建议关闭；不是HTTP并发开关")


class BehaviorConfig(PluginConfigBase):
    __ui_label__ = "使用限制"
    __ui_order__ = 4
    tool_enabled: bool = CN("启用模型语音工具", default=True, description="关闭后仍可使用 /tts 命令")
    max_text_length: int = CN("最大合成字数", default=300, ge=1, le=2000, description="超过时拒绝，不静默截断")
    cooldown_seconds: float = CN("语音冷却（秒）", default=5, ge=0, le=3600, description="同一聊天流两次成功发送之间的间隔")
    allowed_stream_ids: list[str] = CN("允许的聊天流", default_factory=list, description="空列表允许全部；填写 MaiBot 聊天流ID而非QQ群号")
    send_text: bool = CN("同时发送原文", default=False, description="语音发送成功后额外发送文字")


class AutoVoiceConfig(PluginConfigBase):
    __ui_label__ = "第五步 · 自动语音回复"
    __ui_order__ = 6
    enabled: bool = CN("启用自动文字转语音", default=False, description="在机器人普通文字发出前按频率转为语音，失败保留原文字；默认关闭")
    frequency_mode: Literal["probability", "every_n"] = CN("转换频率模式", default="probability", description="MaiBot完成语义分句后，probability按概率、every_n每N条文字转一条；插件不覆盖分段")
    probability_percent: int = CN("转换概率（%）", default=50, ge=0, le=100, description="0从不转换，100每条合格文字都尝试转换")
    every_n: int = CN("每N条转一条", default=3, ge=1, le=1000, description="只在按条数模式生效，按聊天流统计实际语义分句后的消息")
    min_interval_seconds: float = CN("自动语音最短间隔（秒）", default=5, ge=0, le=3600, description="间隔内保留原文字，不排队积压推理")
    follow_emotion: bool = CN("角色模式按情感选择模型", default=True, description="优先消息情感标记，再按关键词推断；仅在已启用情感中选择，不明确时使用默认")
    keep_text: bool = CN("自动语音同时保留文字", default=False, description="关闭时语音替换文字；开启时同条回复保留原文字和语音")


class TTSConfig(PluginConfigBase):
    plugin: PluginSection = Field(default_factory=PluginSection)
    api: APIConfig = Field(default_factory=APIConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    role: RoleConfig = Field(default_factory=RoleConfig)
    voice: VoiceConfig = Field(default_factory=VoiceConfig)
    behavior: BehaviorConfig = Field(default_factory=BehaviorConfig)
    auto_voice: AutoVoiceConfig = Field(default_factory=AutoVoiceConfig)
