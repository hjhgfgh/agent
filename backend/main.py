"""
主应用入口 - FastAPI + RAG + Agent
"""
import os
import uuid
import time
import hashlib
import secrets
import sys
import mimetypes
import traceback
from typing import List, Optional
from contextlib import asynccontextmanager

# 强制stdout使用UTF-8编码
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from starlette.concurrency import run_in_threadpool
from sqlalchemy import text

# 导入自定义模块
from rag.pipeline import (
    DocumentProcessor, SimpleVectorStore, RAGPipeline, compute_file_hash,
)
from agent.agent import AIAgent, FUNCTION_DESCRIPTORS
from cache.redis_cache import RedisCache
from llm.client import LLMClient
from database.models import init_db, Document, Conversation, CacheEntry
from database.models import db_session

load_dotenv()

# ===== 字体 MIME 类型补丁 =====
# 必须在 StaticFiles 挂载之前执行。
#
# python:3.11-slim 镜像里没有 /etc/mime.types，Python 的 mimetypes 认不出
# .woff2，Starlette 的 StaticFiles 会退化成 text/plain 返回。
# 而浏览器对网页字体有强制 MIME 校验，类型不对就直接拒绝加载 ——
# 表现是"所有图标变成空白方块"，且控制台只有一行很不起眼的警告，
# 在服务器上几乎无法定位。这里显式注册，不依赖系统文件，
# Windows 本地开发时同样受益。
mimetypes.add_type("font/woff2", ".woff2")
mimetypes.add_type("font/woff", ".woff")
mimetypes.add_type("font/ttf", ".ttf")
mimetypes.add_type("application/vnd.ms-fontobject", ".eot")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 所有数据目录都以本文件位置为基准，不再依赖启动时的 cwd。
# 否则换个目录启动，上传的文件与索引就会落到别处，
# 表现为"重启后文档全没了""索引凭空损坏"这类怪问题。
UPLOAD_DIR = os.path.join(BASE_DIR, "documents")
INDEX_DIR = os.path.join(BASE_DIR, "faiss_index")
FRONTEND_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "frontend"))

# ===== 访问控制配置（公网部署必读）=====
# ACCESS_CODE 为空 = 完全不鉴权，任何拿到链接的人都能直接调 /chat 消耗你的
# 大模型额度。本地开发图省事可以留空，部署到服务器时务必设置。
ACCESS_CODE = os.getenv("ACCESS_CODE", "").strip()

# READ_ONLY=true 时只允许提问，禁止上传与删除文档。
# 公开演示场景下，访客不该有能力往你的知识库里塞文件、或删掉你准备好的演示文档。
READ_ONLY = os.getenv("READ_ONLY", "false").strip().lower() in ("1", "true", "yes", "on")

# 全局单例
doc_processor = DocumentProcessor()
vector_store = SimpleVectorStore(INDEX_DIR)
vector_store.load()
rag_pipeline = RAGPipeline(vector_store)
redis_cache = RedisCache()
llm_client = LLMClient()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    print("=" * 50)
    print("AI技术文档助手系统启动中...")
    print("=" * 50)

    # 初始化数据库
    try:
        with db_session() as db:
            db.execute(text("SELECT 1"))
        init_db()
        print("[OK] 数据库初始化完成")
    except Exception as e:
        print(f"[WARN] 数据库初始化警告: {e}")

    # 测试Redis连接
    if redis_cache.ping():
        print("[OK] Redis连接正常")
    else:
        print("[ERROR] Redis连接失败，部分功能受限")

    # 访问控制状态：漏配 ACCESS_CODE 是公网部署最常见也最贵的事故，
    # 启动时必须显式打出来，避免"以为设了、其实没生效"。
    if ACCESS_CODE:
        print("[OK] 访问口令已启用")
    else:
        print("[WARN] 未设置 ACCESS_CODE，任何访客都可直接调用接口（公网部署务必设置）")
    print(f"[INFO] 只读模式: {'开启（禁止上传/删除）' if READ_ONLY else '关闭'}")

    yield

    print("\n" + "=" * 50)
    print("系统已关闭")
    print("=" * 50)


app = FastAPI(
    title="AI技术文档助手",
    description="基于RAG的智能技术文档问答系统，支持Function Calling工具调用",
    version="1.0.0",
    lifespan=lifespan
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===== 访问口令 + 只读模式 =====

# 无需口令即可访问的路径。
# 页面本身与健康检查必须放行，否则浏览器连页面都拉不到，用户就没机会输入口令。
_PUBLIC_PATHS = {
    "/", "/index.html", "/favicon.ico",
    "/health", "/api",
    "/docs", "/redoc", "/openapi.json",
}

# 无需口令即可访问的路径前缀。
#
# /vendor/ 必须放行，这是个"开了口令反而白屏"的致命坑，已实测踩中：
#   index.html 用 <script src="vendor/vue.global.js">、<link href="vendor/mdi/...">
#   加载前端依赖，而浏览器**没有任何办法**给这类资源请求附加自定义请求头，
#   所以它们必然带不上 X-Access-Code，会被守卫拦成 401。
#   后果是 Vue 根本加载不出来 —— 连口令输入框自己都渲染不出来，
#   用户永远没机会输入口令，页面彻底死锁。
#   而这些文件全是 Vue / marked / DOMPurify / MDI 等公开的第三方静态资源，
#   不含任何业务数据，放行没有安全影响。
#
# 注意这里是"白名单"式的设计（默认拦截），而不是"列出 API 前缀去拦"：
# 前者漏配的后果是「某个静态资源 404/401，肉眼可见」，
# 后者漏配的后果是「新增的接口默认不设防，静默泄露」——
# 安全控制必须选失败时更显眼的那种。
_PUBLIC_PREFIXES = ("/vendor/",)


@app.middleware("http")
async def access_guard(request: Request, call_next):
    """统一访问守卫：先验口令，再判只读。

    用中间件而不是逐个路由挂 Depends，是因为静态页面挂载在 "/" 上、
    其余路由分散在多处，漏掉任何一个都等于留了个后门。
    """
    path = request.url.path

    # CORS 预检不携带自定义头，必须放行，否则浏览器直接判定跨域失败
    if (
        request.method == "OPTIONS"
        or path in _PUBLIC_PATHS
        or path.startswith(_PUBLIC_PREFIXES)
    ):
        return await call_next(request)

    # 用 compare_digest 而不是 == 做定长比较，避免通过响应耗时逐字符猜口令。
    # 先 encode 成 bytes：compare_digest 对 str 只支持 ASCII，
    # 口令里含中文时会直接抛 TypeError。
    if ACCESS_CODE and not secrets.compare_digest(
        request.headers.get("X-Access-Code", "").encode("utf-8"),
        ACCESS_CODE.encode("utf-8"),
    ):
        return JSONResponse(status_code=401, content={"detail": "访问口令无效，请重新输入"})

    # 只读模式：拒绝一切写操作，但 /chat 例外 ——
    # 提问对知识库而言是纯读取，也正是访客唯一该被允许做的事。
    if READ_ONLY and request.method in ("POST", "PUT", "PATCH", "DELETE") and path != "/chat":
        return JSONResponse(
            status_code=403,
            content={"detail": "演示环境为只读模式，不支持上传或删除文档"},
        )

    return await call_next(request)


# ===== 数据模型 =====

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    question: str
    stream: bool = False


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: List[str]
    tools_used: List[str]
    has_tool_call: bool


class DocumentInfo(BaseModel):
    id: int
    filename: str
    original_name: str
    chunk_count: int
    created_at: str


# ===== API端点 =====

@app.get("/api")
async def root():
    """API 元信息（根路径 / 留给前端页面，详见文件末尾的静态托管）"""
    return {"message": "AI技术文档助手API", "version": "1.0.0", "status": "running"}


@app.get("/health")
async def health_check():
    """健康检查：真实探测 Redis 与数据库，避免误报健康"""
    redis_ok = redis_cache.ping()
    try:
        with db_session() as db:
            db.execute(text("SELECT 1"))
        db_ok = True
    except Exception as e:
        print(f"[WARN] 数据库健康检查失败: {e}")
        db_ok = False
    return {
        "status": "healthy" if (redis_ok and db_ok) else "degraded",
        "redis": redis_ok,
        "db": db_ok,
    }


@app.get("/auth/verify")
async def verify_access_code():
    """校验访问口令，并告知前端当前是否处于只读模式。

    本接口同样受守卫保护：口令不对根本走不到这里（先被拦成 401），
    所以前端只要拿到 200，就说明口令有效。
    """
    return {
        "ok": True,
        "auth_required": bool(ACCESS_CODE),
        "read_only": READ_ONLY,
    }


def _write_file(path: str, content: bytes) -> None:
    """同步写文件（供线程池调用）"""
    with open(path, "wb") as f:
        f.write(content)


def _save_document_record(stored_name: str, original_name: str, chunks: List) -> int:
    """同步写入文档记录（供线程池调用）"""
    with db_session() as db:
        preview_text = ""
        if chunks:
            preview_text = doc_processor.extract_text_preview(chunks[0].page_content)
        doc = Document(
            filename=stored_name,
            original_name=original_name,
            content_preview=preview_text,
            chunk_count=len(chunks),
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)
        return doc.id


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """上传技术文档PDF

    优化点：
    - 内容哈希去重：同一份文档重复上传直接秒返回，不再重复解析
    - PyMuPDF 解析：实测比原 PyPDFLoader 快约 80 倍
    - 解析/切片/落盘均放入线程池，避免阻塞事件循环
    """
    if not (file.filename or "").lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="只支持PDF格式的文件")

    started = time.time()
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="文件内容为空")

        file_hash = hashlib.md5(content).hexdigest()
        original_name = file.filename

        # 1) 文件级去重：同一内容已索引过则跳过解析
        if vector_store.has_file(file_hash):
            elapsed = round(time.time() - started, 2)
            print(f"[UPLOAD] 命中文件去重，跳过解析: {original_name} ({elapsed}s)")
            return {
                "message": "该文档已存在，已跳过重复处理",
                "filename": original_name,
                "chunks": 0,
                "new_chunks": 0,
                "document_id": 0,
                "duplicated": True,
                "elapsed": elapsed,
            }

        # 2) 以内容哈希命名，天然避免同名文件副本堆积
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        stored_name = f"{file_hash[:8]}_{original_name}"
        file_path = os.path.join(UPLOAD_DIR, stored_name)
        await run_in_threadpool(_write_file, file_path, content)
        print(f"[UPLOAD] 处理文档: {original_name} ({len(content)} bytes)")

        # 3) 解析 + 切片（线程池执行）
        chunks = await run_in_threadpool(
            doc_processor.get_chunks_from_pdf, file_path, file_hash, original_name
        )
        print(f"[UPLOAD] 切片完成: {len(chunks)} chunks")

        # 4) 写入向量库
        added = await run_in_threadpool(vector_store.add_documents, chunks)

        # 5) 写入数据库
        document_id = 0
        try:
            document_id = await run_in_threadpool(
                _save_document_record, stored_name, original_name, chunks
            )
            print(f"[UPLOAD] 数据库记录已保存 (ID: {document_id})")
        except Exception as db_error:
            print(f"[WARN] 数据库保存失败: {db_error}")

        # 6) 新增了检索内容，让问答缓存失效
        #    此前缓存的答案很可能是在"检索不到相关内容"时生成的，
        #    例如先问了问题、拿到"无法回答"，之后才上传相关文档。
        #    若不清掉，用户上传后再问同一问题仍会命中旧缓存（TTL 1 小时），
        #    新文档等于白传。added == 0 表示内容未变，缓存依然有效，无需清理。
        invalidated = 0
        if added > 0:
            invalidated = await run_in_threadpool(_clear_qa_cache)
            if invalidated:
                print(f"[UPLOAD] 新增内容，已失效问答缓存 {invalidated} 条")

        elapsed = round(time.time() - started, 2)
        print(f"[UPLOAD] 完成，耗时 {elapsed}s")
        return {
            "message": "文档上传成功",
            "filename": original_name,
            "chunks": len(chunks),
            "new_chunks": added,
            "document_id": document_id,
            "duplicated": False,
            "invalidated_cache": invalidated,
            "elapsed": elapsed,
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] 上传处理失败: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"处理文档失败: {str(e)}")


@app.get("/documents")
async def list_documents():
    """获取已上传的文档列表"""
    try:
        with db_session() as db:
            docs = db.query(Document).order_by(Document.created_at.desc()).all()
            result = []
            for d in docs:
                result.append({
                    "id": d.id,
                    "filename": d.filename,
                    "original_name": d.original_name,
                    "chunk_count": d.chunk_count,
                    "created_at": str(d.created_at)
                })
            return result
    except Exception as e:
        print(f"[ERROR] 获取文档列表失败: {e}")
        return []


class DocumentDeleteResult(BaseModel):
    message: str
    removed_chunks: int = 0
    invalidated_cache: int = 0


def _clear_qa_cache() -> int:
    """清空问答缓存（同步，供线程池调用）

    删除文档后，旧答案很可能引用了已删除的内容，必须让缓存失效，
    否则用户再次提问会直接命中缓存、拿到"基于已删文档"的答案。
    """
    return redis_cache.clear_qa_cache()


@app.delete("/documents/{doc_id}", response_model=DocumentDeleteResult)
async def delete_document(doc_id: int):
    """删除文档：同步清理向量库切片、磁盘文件与失效的问答缓存"""
    try:
        # 1) 先取出记录信息，随即释放数据库连接（避免跨 await 长期占用连接池）
        with db_session() as db:
            doc = db.query(Document).filter(Document.id == doc_id).first()
            if not doc:
                raise HTTPException(status_code=404, detail="文档不存在")
            stored_name = doc.filename
            original_name = doc.original_name

        file_path = os.path.join(UPLOAD_DIR, stored_name)
        removed = 0

        # 2) 清理向量库：优先按内容哈希（最准确）
        if os.path.exists(file_path):
            file_hash = await run_in_threadpool(compute_file_hash, file_path)
            removed = await run_in_threadpool(
                vector_store.remove_by_file_hash, file_hash
            )

        # 3) 兜底：按来源文件名清理（兼容缺少 file_hash 元数据的旧索引）
        if removed == 0 and original_name:
            removed = await run_in_threadpool(
                vector_store.remove_by_source, original_name
            )

        # 4) 清理磁盘文件
        if os.path.exists(file_path):
            await run_in_threadpool(os.remove, file_path)

        # 5) 删除数据库记录
        with db_session() as db:
            db.query(Document).filter(Document.id == doc_id).delete()
            db.commit()

        # 6) 让引用了该文档的问答缓存失效
        invalidated = await run_in_threadpool(_clear_qa_cache)

        print(
            f"[DELETE] 文档 {doc_id} ({original_name}) 已删除，"
            f"清理切片 {removed}，失效缓存 {invalidated}"
        )
        return DocumentDeleteResult(
            message="文档已删除",
            removed_chunks=removed,
            invalidated_cache=invalidated,
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] 删除文档失败: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"删除文档失败: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """智能问答接口"""
    # 生成或验证session_id
    session_id = request.session_id or str(uuid.uuid4())
    
    # 缓存查询（命中时连同真实来源一起返回）
    try:
        cached_entry = redis_cache.get_cached_entry(request.question)
    except Exception:
        cached_entry = None
    if cached_entry:
        print(f"[CACHE] 命中缓存: {request.question[:30]}...")
        return ChatResponse(
            session_id=session_id,
            answer=cached_entry.get('answer', ''),
            sources=cached_entry.get('source_docs', []),
            tools_used=cached_entry.get('tools_used', []),
            has_tool_call=cached_entry.get('has_tool_call', False)
        )

    try:
        # 创建Agent实例
        agent = AIAgent(rag_pipeline, redis_cache, llm=llm_client)

        # 取最近若干轮对话作为上下文，实现多轮问答
        try:
            history = [
                {'role': m.get('role'), 'content': m.get('content')}
                for m in redis_cache.get_session_history(session_id, limit=6)
            ]
        except Exception:
            history = []

        # 处理问题（检索 + 大模型生成为同步阻塞操作，放入线程池避免阻塞事件循环）
        print(f"[CHAT] 处理问题: {request.question[:50]}...")
        result = await run_in_threadpool(
            agent.process_with_tools, request.question, history
        )
        print(f"[CHAT] 回答完成，来源={result.get('source')}，使用工具: {result.get('tools_used', [])}")
        
        # 保存对话到数据库
        try:
            with db_session() as db:
                conversation = Conversation(
                    session_id=session_id,
                    role="user",
                    content=request.question,
                    related_chunks=",".join(result.get('sources', [])),
                )
                db.add(conversation)
            
                assistant_msg = Conversation(
                    session_id=session_id,
                    role="assistant",
                    content=result['response'],
                    related_chunks=",".join(result.get('sources', [])),
                )
                db.add(assistant_msg)
                db.commit()
        except Exception as db_error:
            print(f"[WARN] 对话保存失败: {db_error}")
        
        # 更新会话历史到Redis
        # 注意：此时回答已经生成完毕，写历史失败绝不能让它变成 500 ——
        # 否则用户白等一次大模型耗时，最后只拿到"问答失败"。
        # Redis 在这里只承担多轮上下文记忆，降级为丢失上下文即可，不影响本次回答。
        for role, content in (("user", request.question), ("assistant", result['response'])):
            try:
                redis_cache.add_message(session_id, role, content)
            except Exception as cache_error:
                print(f"[WARN] 会话历史写入失败({role}): {cache_error}")

        # 问答缓存已由 Agent 内部写入（含来源与工具信息），此处不再重复写入

        return ChatResponse(
            session_id=session_id,
            answer=result['response'],
            sources=result.get('sources', []),
            tools_used=result.get('tools_used', []),
            has_tool_call=result.get('has_tool_call', False)
        )
    except Exception as e:
        print(f"[ERROR] 问答处理失败: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"问答失败: {str(e)}")


@app.get("/chat/history/{session_id}")
async def get_chat_history(session_id: str, limit: int = Query(20, ge=1, le=50)):
    """获取会话历史"""
    history = redis_cache.get_session_history(session_id, limit)
    
    try:
        with db_session() as db:
            db_messages = db.query(Conversation)\
                .filter(Conversation.session_id == session_id)\
                .order_by(Conversation.created_at.desc())\
                .limit(limit)\
                .all()
            db_history = [{"role": m.role, "content": m.content} for m in db_messages]
            return {"history": db_history, "redis_count": len(history)}
    except:
        return {"history": history, "redis_count": len(history)}


@app.get("/tools")
async def list_tools():
    """列出可用的工具函数"""
    return {
        "tools": FUNCTION_DESCRIPTORS,
        "count": len(FUNCTION_DESCRIPTORS)
    }


@app.get("/stats")
async def get_statistics():
    """获取系统统计信息"""
    try:
        with db_session() as db:
            doc_count = db.query(Document).count()
            conv_count = db.query(Conversation).count()
            return {
                "documents": doc_count,
                "conversations": conv_count,
                "redis_connected": redis_cache.ping()
            }
    except:
        return {"redis_connected": redis_cache.ping()}


# ===== 前端静态托管 =====
# 必须放在所有 API 路由之后注册：挂载到 "/" 会兜住一切未匹配路径，
# 若先注册，/upload、/chat 等接口就永远轮不到了。
# 同源部署的好处：访客只访问一个网址，不涉及跨域，也不会出现
# 「页面是 https、接口是 http」被浏览器拦截的混用问题。
if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
    print(f"[OK] 前端已托管: {FRONTEND_DIR}")
else:
    print(f"[WARN] 未找到前端目录，仅提供 API: {FRONTEND_DIR}")


if __name__ == "__main__":
    import uvicorn
    print("Starting server on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
