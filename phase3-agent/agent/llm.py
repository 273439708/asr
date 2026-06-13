"""LLM 接入：OpenAI 兼容接口（deepseek 云 API），密钥/base_url/模型来自 .env。
绑定座舱工具供 function calling。"""

import os
import threading

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from .tools import ALL_TOOLS

load_dotenv()

_llm = None
_lock = threading.Lock()


def get_llm_with_tools():
    """返回已 bind_tools 的 ChatOpenAI 单例。temperature=0 让意图→工具映射稳定。"""
    global _llm
    with _lock:
        if _llm is None:
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError(
                    "缺少 OPENAI_API_KEY。复制 .env.example 为 .env 并填入 deepseek 密钥。")
            base = ChatOpenAI(
                model=os.environ.get("COCKPIT_MODEL", "deepseek-chat"),
                base_url=os.environ.get("OPENAI_BASE_URL"),
                temperature=0,
                timeout=60,
                max_retries=1,
            )
            _llm = base.bind_tools(ALL_TOOLS)
        return _llm
