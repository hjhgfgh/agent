"""
修复RedisCache中的import问题
"""
import json
import time
import hashlib  # 添加缺失的导入
from typing import List, Optional, Dict, Any
from datetime import timedelta

import redis
from dotenv import load_dotenv
import os

load_dotenv()


class RedisCache:
    """Redis缓存管理器"""

    def __init__(self, host: str = None, port: int = None, db: int = None):
        config = {
            'host': host or os.getenv('REDIS_HOST', 'localhost'),
            'port': port or int(os.getenv('REDIS_PORT', 6379)),
            'db': db or int(os.getenv('REDIS_DB', 0)),
            'decode_responses': True
        }
        self.client = redis.Redis(**config)
        # TTL设置
        self.CONVERSATION_TTL = timedelta(hours=2)  # 对话历史过期时间
        self.CACHE_TTL = timedelta(hours=1)         # 问答缓存过期时间
        self.TOOL_RESULT_TTL = timedelta(minutes=30)  # 工具调用结果缓存

    def ping(self) -> bool:
        """测试连接"""
        try:
            return self.client.ping()
        except redis.ConnectionError:
            return False

    # ===== 会话管理 =====

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """添加消息到会话历史（使用Redis List）"""
        key = f"conversation:{session_id}"
        message = json.dumps({
            'role': role,
            'content': content,
            'timestamp': time.time()
        }, ensure_ascii=False)
        self.client.rpush(key, message)
        self.client.expire(key, self.CONVERSATION_TTL)
        # 限制会话历史长度（保留最近20条）
        while self.client.llen(key) > 20:
            self.client.lpop(key)

    def get_session_history(self, session_id: str, limit: int = 10) -> List[Dict[str, str]]:
        """获取会话历史"""
        key = f"conversation:{session_id}"
        messages = self.client.lrange(key, -limit, -1)
        return [json.loads(m) for m in messages] if messages else []

    def clear_session(self, session_id: str) -> None:
        """清空会话历史"""
        self.client.delete(f"conversation:{session_id}")

    def session_exists(self, session_id: str) -> bool:
        """检查会话是否存在"""
        return self.client.exists(f"conversation:{session_id}") > 0

    # ===== 问答缓存 =====

    def get_cached_entry(self, question: str) -> Optional[Dict[str, Any]]:
        """获取缓存的完整问答记录（含来源、工具调用信息）"""
        key = f"qa_cache:{hashlib.md5(question.encode()).hexdigest()}"
        cached = self.client.get(key)
        if not cached:
            return None
        # 更新命中次数
        self.client.incr(f"qa_hit:{key}")
        try:
            data = json.loads(cached)
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, dict) else None

    def get_cached_answer(self, question: str) -> Optional[str]:
        """获取缓存的答案文本（基于问题哈希）"""
        entry = self.get_cached_entry(question)
        return entry.get('answer') if entry else None

    def set_cached_answer(self, question: str, answer: str, source_docs: List[str] = None,
                          tools_used: List[str] = None, has_tool_call: bool = False) -> None:
        """缓存问答对（同时保存来源与工具调用信息，便于命中时原样返回）"""
        key = f"qa_cache:{hashlib.md5(question.encode()).hexdigest()}"
        data = {
            'answer': answer,
            'source_docs': source_docs or [],
            'tools_used': tools_used or [],
            'has_tool_call': has_tool_call,
            'created_at': time.time()
        }
        self.client.setex(key, self.CACHE_TTL, json.dumps(data, ensure_ascii=False))

    def increment_hit(self, question: str) -> int:
        """增加缓存命中率计数"""
        key = f"qa_hit:{hashlib.md5(question.encode()).hexdigest()}"
        return self.client.incr(key)

    # ===== 工具结果缓存 =====

    def set_tool_result(self, tool_name: str, params: str, result: str) -> None:
        """缓存工具调用结果"""
        cache_key = f"tool:{tool_name}:{hashlib.md5(params.encode()).hexdigest()}"
        self.client.setex(cache_key, self.TOOL_RESULT_TTL, result)

    def get_tool_result(self, tool_name: str, params: str) -> Optional[str]:
        """获取工具调用缓存"""
        cache_key = f"tool:{tool_name}:{hashlib.md5(params.encode()).hexdigest()}"
        return self.client.get(cache_key)

    # ===== 通用操作 =====

    def set(self, key: str, value: str, ttl: Optional[timedelta] = None) -> None:
        """设置键值对"""
        if ttl:
            self.client.setex(key, ttl, value)
        else:
            self.client.set(key, value)

    def get(self, key: str) -> Optional[str]:
        """获取键值"""
        return self.client.get(key)

    def delete(self, key: str) -> None:
        """删除键"""
        self.client.delete(key)

    def keys(self, pattern: str) -> List[str]:
        """模糊查找key"""
        return self.client.keys(pattern)

    def clear_qa_cache(self) -> int:
        """清空全部问答缓存，返回清理条数

        文档被删除后，此前缓存的答案可能引用了已删除的切片。
        若不清掉，用户再次提问会命中缓存，依然拿到"基于已删文档"的旧答案。
        """
        try:
            keys = self.client.keys("qa_cache:*")
            if keys:
                self.client.delete(*keys)
            return len(keys)
        except redis.RedisError as e:
            print(f"[WARN] 清理问答缓存失败: {e}")
            return 0


if __name__ == "__main__":
    cache = RedisCache()
    print(f"Redis连接状态: {'正常' if cache.ping() else '异常'}")
    # 测试会话管理
    test_session = "test_session_001"
    cache.add_message(test_session, "user", "你好")
    cache.add_message(test_session, "assistant", "你好！有什么可以帮你的？")
    history = cache.get_session_history(test_session)
    print(f"会话历史: {history}")
