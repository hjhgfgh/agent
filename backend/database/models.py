"""
数据库模型定义
使用SQLite作为默认数据库（无需密码配置），也可切换为MySQL
"""
from contextlib import contextmanager

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Boolean, func, text
from sqlalchemy import inspect as sa_inspect
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
    """文档表 - 存储知识库里的PDF文档信息

    文档分两类，生命周期完全不同，靠 is_builtin 区分：

      内置文档（is_builtin=True）
        随镜像发布（backend/builtin_docs/），如 VueJS 官方文档。
        它不在 upload_data 卷里，所以卷被清空、换台服务器重新部署后依然存在，
        开箱即有内容可问。接口层拒绝删除，避免误操作把知识库清空。

      临时文档（is_builtin=False，来自访客上传）
        只属于上传者的那一次网页会话：关掉网页后，下次打开页面会被自动回收。
        这是刻意的设计 —— 访客不该有能力把文件永久留在别人的知识库里。
    """
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    original_name = Column(String(255), nullable=False)
    content_preview = Column(Text, nullable=True)
    chunk_count = Column(Integer, default=0)
    # 注意这两个字段是后加的，老库需要 _ensure_columns() 补列，详见该函数注释
    is_builtin = Column(Boolean, nullable=False, default=False, server_default=text("0"))
    client_id = Column(String(64), nullable=True, index=True)
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


def _ensure_columns():
    """给已存在的旧表补上后加的字段（轻量迁移）。

    为什么必须有它：
      database/init.sql 只在 MySQL 数据卷【首次初始化】时执行一次，之后无论
      怎么改脚本都不会再跑；而 Base.metadata.create_all() 也只建"缺失的表"，
      对已存在的表不会加列。结果就是老部署升级后，代码读 is_builtin /
      client_id 时直接报 Unknown column，整个文档列表 500。

    只在列不存在时才 ALTER，所以本函数可以随便重复执行。
    """
    wanted = [
        ("is_builtin", "BOOLEAN NOT NULL DEFAULT 0"),
        ("client_id", "VARCHAR(64)"),
    ]
    try:
        inspector = sa_inspect(engine)
        if "documents" not in inspector.get_table_names():
            return
        existing = {c["name"] for c in inspector.get_columns("documents")}
        missing = [(n, ddl) for n, ddl in wanted if n not in existing]
        if not missing:
            return

        # MySQL 与 SQLite 的 ALTER TABLE ADD COLUMN 语法在这几条上是通用的
        with engine.begin() as conn:
            for name, ddl in missing:
                conn.execute(text(f"ALTER TABLE documents ADD COLUMN {name} {ddl}"))
                print(f"[DB] documents 表补列: {name}")
    except Exception as e:
        print(f"[WARN] 补列检查失败（若为全新库可忽略）: {e}")


def init_db():
    """初始化数据库表"""
    Base.metadata.create_all(bind=engine)
    _ensure_columns()
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
