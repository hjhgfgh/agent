"""
数据库模型定义
使用SQLite作为默认数据库（无需密码配置），也可切换为MySQL
"""
from contextlib import contextmanager

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, func, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

from dotenv import load_dotenv
load_dotenv()

# 优先使用环境变量中的数据库配置，否则默认使用SQLite
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./tech_doc_assistant.db")
engine = create_engine(
    DATABASE_URL,
    echo=False,
    pool_pre_ping=True,   # 连接被 MySQL 掐断后自动重建，避免 "server has gone away"
    pool_recycle=3600,    # 回收超过 1 小时的连接，规避 MySQL wait_timeout
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Document(Base):
    """文档表 - 存储已上传的PDF文档信息"""
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    original_name = Column(String(255), nullable=False)
    content_preview = Column(Text, nullable=True)
    chunk_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=func.now())


class Conversation(Base):
    """对话历史表 - 存储用户与AI的对话记录"""
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(64), nullable=False, index=True)
    role = Column(String(10), nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    related_chunks = Column(Text, nullable=True)  # 相关的文档片段ID
    created_at = Column(DateTime, default=func.now())


class CacheEntry(Base):
    """缓存表 - 存储高频问答对，加速重复查询"""
    __tablename__ = "cache_entries"

    id = Column(Integer, primary_key=True, index=True)
    question_hash = Column(String(64), unique=True, nullable=False)
    question = Column(String(1000), nullable=False)
    answer = Column(Text, nullable=False)
    source_docs = Column(String(500), nullable=True)
    hit_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


def init_db():
    """初始化数据库表"""
    Base.metadata.create_all(bind=engine)
    print("数据库表初始化完成")


def get_db():
    """获取数据库会话（仅供 FastAPI 的 Depends 使用）

    ⚠️ 不要写成 `db = next(get_db())`。
    get_db() 是生成器，`next()` 只取到会话就丢弃了生成器，
    `finally` 永远不会执行，连接不会归还连接池（连接泄漏）。
    连接池默认只有 pool_size(5) + max_overflow(10) = 15 条，
    泄漏满之后所有 DB 操作都会阻塞，表现为接口长时间无响应。
    内部直接取会话请改用下面的 db_session()。
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def db_session():
    """数据库会话上下文管理器（内部代码统一用它取会话）

    用法：
        with db_session() as db:
            db.query(Document).all()
    无论正常返回还是抛异常，连接都会归还连接池。
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


if __name__ == "__main__":
    init_db()
    print(f"数据库连接: {DATABASE_URL}")
