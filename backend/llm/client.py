"""
LLM 客户端 - 对接任意 OpenAI 兼容接口

默认指向 Agnes AI 中转站（agnes-3.0-flash），也可通过环境变量切回
阿里云百炼（DashScope）通义千问。直接复用 openai SDK，支持原生
Function Calling。

环境变量：
    LLM_API_KEY        必填，未配置时 LLM 不可用（自动降级为抽取式回答）
    LLM_BASE_URL       可选，默认 Agnes AI 中转站地址
    LLM_MODEL          可选，默认 agnes-3.0-flash
    LLM_TIMEOUT        可选，请求超时秒数，默认 120
    LLM_MAX_TOKENS     可选，单次最大输出 token，默认 4000
                       （思考型模型会先消耗 token 输出思维链，再给出正文，
                        该值需同时覆盖「思考 + 正文」两部分）
    兼容旧变量：DASHSCOPE_API_KEY / OPENAI_API_KEY 仍可作为 api_key 来源。
"""
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

DEFAULT_BASE_URL = "https://apihub.agnes-ai.cn/v1"
DEFAULT_MODEL = "agnes-3.0-flash"


class LLMUnavailableError(RuntimeError):
    """LLM 未配置或调用失败"""


def to_openai_tools(descriptors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """把项目内的简化工具描述转为 OpenAI Function Calling 格式"""
    return [
        {
            "type": "function",
            "function": {
                "name": d["name"],
                "description": d.get("description", ""),
                "parameters": d.get(
                    "parameters", {"type": "object", "properties": {}}
                ),
            },
        }
        for d in descriptors
    ]


class LLMClient:
    """通义千问（OpenAI 兼容模式）客户端封装"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        self.api_key = (
            api_key
            or os.getenv("LLM_API_KEY")
            or os.getenv("DASHSCOPE_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        self.base_url = base_url or os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL)
        self.model = model or os.getenv("LLM_MODEL", DEFAULT_MODEL)
        self.timeout = float(timeout or os.getenv("LLM_TIMEOUT", 120))
        self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", 4000))
        self._client = None

        if self.api_key:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    timeout=self.timeout,
                    max_retries=1,  # 失败快速返回，避免拖垮接口响应
                )
                print(f"[LLM] 已启用 {self.model} @ {self.base_url}")
            except ImportError:
                print("[LLM] 未安装 openai 包，LLM 功能不可用（pip install openai）")
        else:
            print("[LLM] 未配置 LLM_API_KEY，将降级为抽取式回答")

    @property
    def available(self) -> bool:
        return self._client is not None

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
    ):
        """发起一次对话补全，返回 message 对象（可能携带 tool_calls）"""
        if not self._client:
            raise LLMUnavailableError(
                "未配置 LLM_API_KEY，无法调用大模型"
            )

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        return self._client.chat.completions.create(**kwargs).choices[0].message
