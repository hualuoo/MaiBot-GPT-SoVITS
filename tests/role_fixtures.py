"""测试用 role.json 角色夹具：按插件采用的 role.json 格式在临时目录生成角色文件。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def file_entry(directory: Path, name: str, content: bytes) -> dict:
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {"name": name, "size": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def emotion_entry(role_dir: Path, emotion: str, *, gpt=b"PK-tiny-gpt", sovits=b"01-tiny-v2-full",
                  audio=None, gpt_name=None, sovits_name="shu.pth", audio_name=None, ref_text=None, **extra) -> dict:
    entry = {
        "gpt": file_entry(role_dir, gpt_name or f"shu_{emotion}.ckpt", gpt),
        "sovits": file_entry(role_dir, sovits_name, sovits),
        "ref_audio": file_entry(role_dir, audio_name or f"{emotion}.wav", audio if audio is not None else b"ref-" + emotion.encode()),
        "ref_text": ref_text or f"{emotion} reference",
    }
    entry.update(extra)
    return entry


def write_manifest(role_dir: Path, payload: dict) -> Path:
    role_dir.mkdir(parents=True, exist_ok=True)
    path = role_dir / "role.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def write_role(roles_dir: Path, folder: str = "shu", emotions=("happy",), *, role=None, base_model=None, **entry_options) -> dict:
    """生成 roles_dir/folder/role.json，返回写入的清单。"""
    role_dir = roles_dir / folder
    payload = {"role": role or folder, "emotions": {emotion: emotion_entry(role_dir, emotion, **entry_options) for emotion in emotions}}
    if base_model is not None:
        payload["base_model"] = base_model
    write_manifest(role_dir, payload)
    return payload
