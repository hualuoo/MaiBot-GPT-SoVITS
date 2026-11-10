"""用 MaiBot 源码校验插件：Manifest、导入、生命周期、配置、组件、页面声明与插件图标。

用法（项目根目录）：python -B tests/validate_host.py [MaiBot 目录]
不传目录时使用环境变量 MAIBOT_ROOT，或与本项目并列的 MaiBot 文件夹。
"""
import asyncio
import importlib.util
import sys
import tempfile
import tomllib
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from host_paths import MAIBOT_ROOT, SKIP_REASON

host = Path(sys.argv[1]) if len(sys.argv) > 1 else MAIBOT_ROOT
if host is None or not (host / "bot.py").is_file():
    sys.exit(SKIP_REASON)
sys.path.insert(0, str(host))
from src.plugin_runtime.runner.manifest_validator import PluginManifest
from src.plugin_runtime.runner.plugin_loader import PluginLoader
from maibot_sdk.components import collect_components

manifest = PluginManifest.model_validate_json((root / "_manifest.json").read_text(encoding="utf-8"))
spec = importlib.util.spec_from_file_location("host_contract_tts", root / "plugin.py",
                                             submodule_search_locations=[str(root)])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
instance = module.create_plugin()
PluginLoader._validate_sdk_plugin_contract(manifest.id, instance)
instance.set_plugin_config(tomllib.loads((root / "config.example.toml").read_text(encoding="utf-8")))
# 模型索引是运行时生成的；在临时目录里扫描一个空的模型目录，不写入源码目录。
index_dir = tempfile.TemporaryDirectory()
instance._settings_path = Path(index_dir.name) / "config.toml"
models_dir = Path(index_dir.name).resolve() / "GPT-SoVITS"
models_dir.mkdir()
instance.config.models.root = models_dir.as_posix()
refreshed = asyncio.run(instance.refresh_native_catalog())
assert refreshed["success"], refreshed
schema = instance.get_webui_config_schema()
assert len(schema["runtime_widgets"]) == 6
base_section = schema["sections"]["models"]
assert set(base_section["fields"]) == {"root", "base_model"}
assert schema["sections"]["original_weights"]["name"] == "models"
model_fields = {**base_section["fields"], **schema["sections"]["original_weights"]["fields"]}
for key in ("base_model", "gpt_weights", "sovits_weights", "lora_weights"):
    assert model_fields[key]["ui_type"] == "select" and model_fields[key]["choices"]
components = collect_components(instance)
assert schema and len(components) == 24
from src.plugin_runtime.webui_schema import load_webui_extension
extension = load_webui_extension(str(root))
assert extension and extension.pages[0].mode == "custom"
registered = {c["name"] for c in components if c["type"] == "API"}
page = extension.pages[0]
assert all(b.api in registered for b in [*page.queries.values(), *page.actions.values()])
page.queries["catalog"].validate_args({"root": instance.config.models.root})
page.actions["save_base"].validate_args({"root": instance.config.models.root,
                                        "sovits_weights": instance.config.models.sovits_weights})
models = instance.config.models.model_dump()
page.actions["save_models"].validate_args({**models,
    **{key: getattr(instance.config.voice, key) for key in ("ref_audio_path", "prompt_text", "prompt_lang", "text_lang")}})
assert all(field["label"] != key for section in schema["sections"].values()
           for key, field in section["fields"].items())
# 插件图标：用 MaiBot 图标接口自身的规则检查（相对路径、允许的格式、文件存在且不越界、不超过 512 KB）。
import json
from src.webui.routers.plugin.icon_routes import _MAX_ICON_BYTES, _get_local_icon_path, _validate_local_icon_path
from src.webui.routers.plugin.support import resolve_plugin_file_path
icon_path = _get_local_icon_path(json.loads((root / "_manifest.json").read_text(encoding="utf-8")))
assert icon_path == manifest.display.icon.value, icon_path
_validate_local_icon_path(icon_path)
icon_file = resolve_plugin_file_path(root, icon_path, allow_missing=False)
assert icon_file.is_file() and icon_file.stat().st_size <= _MAX_ICON_BYTES
print(f"PASS: Host manifest v2, package import, lifecycle, TOML, Chinese schema, {len(components)} components, custom page and API bindings, plugin icon")
print("Plugin ID:", manifest.id)
print("Components:", len(components))
