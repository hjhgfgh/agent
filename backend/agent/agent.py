"""
Agent模块 - Function Calling工具定义与调度
"""
import json
import re
import math
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from rag.pipeline import RAGPipeline, SimpleVectorStore
from cache.redis_cache import RedisCache
from llm.client import LLMClient, to_openai_tools


# 工具函数定义
class Tools:
    """可用的工具函数集合"""

    # 允许在计算器中出现的函数/常量名（长名在前，避免 log10 被 log 误截断）
    _ALLOWED_NAMES = (
        'log10', 'sqrt', 'pow', 'round', 'abs', 'sin', 'cos', 'tan',
        'log', 'math.', 'pi', 'e',
    )
    _ALLOWED_CHARS = set("0123456789+-*/().,% ")

    @staticmethod
    def calculator(expression: str) -> str:
        """
        计算器工具 - 执行数学表达式
        Args:
            expression: 数学表达式字符串，如 "2+3*4" 或 "sqrt(16)"
        Returns:
            计算结果
        """
        expr = (expression or "").strip()
        if not expr:
            return "错误：表达式为空"

        # 先剔除白名单函数名，剩余字符必须全部是数字/运算符，防止任意代码执行
        probe = expr.lower()
        for name in Tools._ALLOWED_NAMES:
            probe = probe.replace(name, '')
        if not set(probe) <= Tools._ALLOWED_CHARS:
            return "错误：表达式中包含不允许的字符"

        try:
            result = eval(expr, {"__builtins__": {}}, {
                "math": math, "sqrt": math.sqrt, "pow": pow, "abs": abs,
                "round": round, "log": math.log, "log10": math.log10,
                "sin": math.sin, "cos": math.cos, "tan": math.tan,
                "pi": math.pi, "e": math.e,
            })
            return f"{result}"
        except Exception as e:
            return f"计算错误: {str(e)}"

    @staticmethod
    def get_current_time() -> str:
        """获取当前时间"""
        return datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")

    @staticmethod
    def count_words(text: str) -> str:
        """统计文本字数"""
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_words = len(re.findall(r'[a-zA-Z]+', text))
        return f"中文字符: {chinese_chars}, 英文单词: {english_words}"

    @staticmethod
    def extract_keywords(text: str, top_k: int = 5) -> str:
        """简易关键词提取（基于词频）"""
        # 移除标点符号
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)
        words = text.split()
        word_freq = {}
        for word in words:
            if len(word) >= 2:
                word_freq[word] = word_freq.get(word, 0) + 1
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return ", ".join([w[0] for w in sorted_words]) if sorted_words else "未提取到关键词"


# Function Calling定义（用于LLM识别）
FUNCTION_DESCRIPTORS = [
    {
        "name": "calculator",
        "description": "执行数学计算，支持加减乘除、幂运算、开方等",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "数学表达式，例如 '2+3*4' 或 'sqrt(16)'"
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "get_current_time",
        "description": "获取当前日期和时间",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "count_words",
        "description": "统计给定文本的中文字数和英文单词数",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "要统计的文本内容"
                }
            },
            "required": ["text"]
        }
    },
    {
        "name": "extract_keywords",
        "description": "从文本中提取关键词",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "要提取关键词的文本"
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回的关键词数量，默认为5"
                }
            },
            "required": ["text"]
        }
    }
]


SYSTEM_PROMPT = """你是"AI技术文档助手"，一位严谨、专业的技术文档问答专家。

回答要求：
1. 必须严格依据【参考资料】作答，不得编造资料中不存在的内容。
2. 若参考资料不足以回答问题，请明确说明"根据现有文档无法回答该问题"，并指出缺少哪方面信息，不要强行作答。
3. 回答要条理清晰，使用 Markdown 组织内容（小标题、有序/无序列表、代码块）。涉及代码时给出完整可运行的示例。
4. 引用资料中的关键结论时，在句末标注来源，格式为（来源：文件名 第N页）。文件名必须原样复制参考资料中"来源："后的内容，不要缩写、改写或使用"文档名"之类的占位符。
5. 若问题需要精确计算或当前时间，必须先调用相应工具获取结果，不要自行口算或猜测时间。
6. 不要使用 LaTeX 数学公式（如 $...$ 或 $$...$$），数学表达式请直接用普通文本或代码块表示，以保证前端正常显示。
"""


class AIAgent:
    """AI Agent - 集成 RAG 检索 + 大模型生成 + Function Calling"""

    MAX_TOOL_ROUNDS = 3  # 工具调用最大轮次，防止无限循环

    def __init__(self, rag_pipeline: RAGPipeline, redis_cache: RedisCache,
                 llm: Optional[LLMClient] = None):
        self.rag = rag_pipeline
        self.cache = redis_cache
        self.llm = llm or LLMClient()
        self.tools = {
            'calculator': Tools.calculator,
            'get_current_time': Tools.get_current_time,
            'count_words': Tools.count_words,
            'extract_keywords': Tools.extract_keywords
        }
        self.openai_tools = to_openai_tools(FUNCTION_DESCRIPTORS)

    # ===== 工具执行 =====

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """调用指定的工具函数（带结果缓存，时间类工具除外）"""
        if tool_name not in self.tools:
            return f"错误：未知的工具 '{tool_name}'"

        # 时间类工具结果具有时效性，不能缓存
        cacheable = tool_name != 'get_current_time'
        params_key = json.dumps(arguments, ensure_ascii=False, sort_keys=True)

        if cacheable:
            try:
                cached = self.cache.get_tool_result(tool_name, params_key)
                if cached is not None:
                    return cached
            except Exception:
                pass  # Redis 异常不影响工具执行

        try:
            result = str(self.tools[tool_name](**arguments))
        except TypeError as e:
            return f"错误：工具 '{tool_name}' 的参数不正确（{e}）"
        except Exception as e:
            return f"工具调用失败: {str(e)}"

        if cacheable:
            try:
                self.cache.set_tool_result(tool_name, params_key, result)
            except Exception:
                pass
        return result

    # ===== 消息构造 =====

    def _build_messages(self, user_question: str, context: str,
                        history: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """构造送往大模型的消息：系统提示 + 历史对话 + 参考资料 + 本次问题"""
        reference = context.strip() or "（未检索到相关文档内容）"
        messages: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]

        for h in (history or []):
            role, content = h.get('role'), h.get('content')
            if role in ('user', 'assistant') and content:
                messages.append({"role": role, "content": content})

        messages.append({
            "role": "user",
            "content": f"【参考资料】\n{reference}\n\n【用户问题】\n{user_question}",
        })
        return messages

    def _fallback_answer(self, context: str) -> str:
        """大模型不可用时的降级：返回检索原文并明确标注原因"""
        if not context.strip():
            return ("未检索到相关文档内容，且大模型未启用（缺少 LLM_API_KEY），"
                    "暂时无法生成回答。请在 backend/.env 中配置 API Key 后重试。")
        return ("⚠️ 大模型未启用（缺少 LLM_API_KEY），无法生成归纳性回答。\n"
                "以下为检索到的原文片段：\n\n" + context[:800])

    # ===== 主流程 =====

    def process_with_tools(self, user_question: str,
                           history: Optional[List[Dict[str, Any]]] = None,
                           top_k: int = 5) -> Dict[str, Any]:
        """完整流程：缓存 → RAG 检索 → 大模型生成（含工具调用循环）→ 写缓存"""
        # 1. 缓存命中直接返回
        try:
            cached = self.cache.get_cached_entry(user_question)
        except Exception:
            cached = None
        if cached:
            return {
                'response': cached.get('answer', ''),
                'source': 'cache',
                'sources': cached.get('source_docs', []),
                'tools_used': cached.get('tools_used', []),
                'has_tool_call': cached.get('has_tool_call', False),
            }

        # 2. RAG 检索
        _, context, sources = self.rag.retrieve_and_format(user_question, top_k=top_k)

        result: Dict[str, Any] = {
            'response': '',
            'source': 'rag',
            'sources': sources,
            'tools_used': [],
            'has_tool_call': False,
        }

        # 3. 大模型生成
        if not self.llm.available:
            result['source'] = 'fallback'
            result['response'] = self._fallback_answer(context)
            return result

        messages = self._build_messages(user_question, context, history)
        tools_used: List[str] = []

        try:
            for _ in range(self.MAX_TOOL_ROUNDS):
                msg = self.llm.chat(messages, tools=self.openai_tools)

                # 没有工具调用 => 得到最终回答
                if not msg.tool_calls:
                    result['response'] = (msg.content or '').strip()
                    break

                # 把 assistant 的工具调用请求回填到对话中
                messages.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                })

                # 依次执行工具并把结果回传
                for tc in msg.tool_calls:
                    name = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments or '{}')
                    except json.JSONDecodeError:
                        args = {}
                    if not isinstance(args, dict):
                        args = {}

                    output = self.call_tool(name, args)
                    tools_used.append(name)
                    print(f"[AGENT] 工具调用 {name}({args}) -> {output[:80]}")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": output,
                    })
            else:
                # 工具轮次用尽仍未收敛，强制让模型给出最终回答
                messages.append({
                    "role": "user",
                    "content": "请基于以上信息直接给出最终回答，不要再调用工具。",
                })
                final = self.llm.chat(messages, tools=None)
                result['response'] = (final.content or '').strip()

            if tools_used:
                result['tools_used'] = tools_used
                result['has_tool_call'] = True

            if not result['response']:
                result['response'] = '抱歉，模型未返回有效内容，请重试或换一种问法。'

        except LLMUnavailableError as e:
            print(f"[WARN] {e}")
            result['source'] = 'fallback'
            result['response'] = self._fallback_answer(context)
        except Exception as e:
            # 大模型异常不应让整个问答 500，降级返回检索原文
            print(f"[ERROR] 大模型调用失败: {type(e).__name__}: {e}")
            result['source'] = 'fallback'
            result['response'] = (
                f"⚠️ 大模型调用失败（{type(e).__name__}），已降级返回检索到的原文片段：\n\n"
                + (context[:800] if context.strip() else "未检索到相关内容。")
            )

        # 4. 写缓存（降级结果不缓存，避免污染后续回答）
        if result['source'] != 'fallback':
            try:
                self.cache.set_cached_answer(
                    user_question,
                    result['response'],
                    sources,
                    tools_used=result['tools_used'],
                    has_tool_call=result['has_tool_call'],
                )
            except Exception as e:
                print(f"[WARN] 问答缓存写入失败: {e}")

        return result
