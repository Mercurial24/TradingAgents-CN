"""
轻量提示词管理器：从 tradingagents/prompts/ 目录加载 YAML，支持安全变量渲染。

用法:
    from tradingagents.config.prompt_manager import prompt_manager

    # 仅加载模板（保留 {tool_names} 等占位符，交给 ChatPromptTemplate）
    template = prompt_manager.get("analysts/fundamentals", "system_prompt")

    # 渲染已有变量，未提供的占位符原样保留
    text = prompt_manager.render(
        "analysts/fundamentals",
        "system_message",
        company_name="贵州茅台",
        ticker="600519",
    )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger("tradingagents.config.prompt_manager")


class _SafeDict(dict):
    """format_map 时缺失键保留为 {key}，便于与 ChatPromptTemplate.partial 混用。"""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


class PromptManager:
    """从仓库根目录 prompts/ 加载并缓存 YAML 提示词。"""

    def __init__(self, prompts_dir: Optional[Path] = None):
        if prompts_dir is None:
            # tradingagents/config/prompt_manager.py -> tradingagents/prompts
            prompts_dir = Path(__file__).resolve().parents[1] / "prompts"
        self.prompts_dir = Path(prompts_dir)
        self._cache: Dict[str, Dict[str, Any]] = {}

    def clear_cache(self) -> None:
        self._cache.clear()

    def _load_file(self, relative_path: str) -> Dict[str, Any]:
        """加载 prompts/{relative_path}.yaml，带缓存。"""
        key = relative_path.replace("\\", "/").removesuffix(".yaml").removesuffix(".yml")
        if key in self._cache:
            return self._cache[key]

        file_path = self.prompts_dir / f"{key}.yaml"
        if not file_path.exists():
            raise FileNotFoundError(f"提示词文件不存在: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        if not isinstance(data, dict):
            raise ValueError(f"提示词文件格式错误（应为 mapping）: {file_path}")

        self._cache[key] = data
        logger.debug("已加载提示词文件: %s (%d keys)", file_path, len(data))
        return data

    def get(self, relative_path: str, key: str) -> str:
        """获取原始模板字符串，不做变量替换。"""
        data = self._load_file(relative_path)
        if key not in data:
            raise KeyError(f"提示词键不存在: {relative_path} -> {key}")
        value = data[key]
        if value is None:
            return ""
        if not isinstance(value, str):
            raise TypeError(f"提示词必须是字符串: {relative_path}.{key} ({type(value)})")
        return value

    def render(self, relative_path: str, key: str, **kwargs: Any) -> str:
        """加载模板并用 kwargs 做安全 format（缺失占位符保留）。"""
        template = self.get(relative_path, key)
        if not kwargs:
            return template
        try:
            return template.format_map(_SafeDict(**{k: v if v is not None else "" for k, v in kwargs.items()}))
        except Exception as e:
            logger.error("渲染提示词失败 %s.%s: %s", relative_path, key, e)
            raise

    def has(self, relative_path: str, key: str) -> bool:
        try:
            data = self._load_file(relative_path)
            return key in data
        except FileNotFoundError:
            return False


# 全局单例
prompt_manager = PromptManager()


def get_prompt(relative_path: str, key: str, **kwargs: Any) -> str:
    """便捷函数：有 kwargs 则 render，否则 get。"""
    if kwargs:
        return prompt_manager.render(relative_path, key, **kwargs)
    return prompt_manager.get(relative_path, key)
