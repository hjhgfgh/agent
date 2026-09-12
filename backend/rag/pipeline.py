"""
RAG模块 - 文档加载、切片与检索

优化点：
1. PDF 解析改用 PyMuPDF（实测比 PyPDFLoader 快约 80 倍），并保留 PyPDFLoader 降级方案
2. 文件级内容去重：同一份文档重复上传时直接跳过解析
3. 中英文混合分词检索（bigram），修复中文整句无法命中的问题
4. 索引原子化落盘 + 分词结果缓存，避免文件损坏与重复计算
5. 支持按文件移除索引，删除文档时同步清理向量库
"""
import os
import re
import json
import hashlib
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Set

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LCDocument
from dotenv import load_dotenv

load_dotenv()

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# 英文/数字 token
_WORD_RE = re.compile(r"[a-z0-9_]+")
# 连续中文片段
_CN_RE = re.compile(r"[\u4e00-\u9fff]+")


def tokenize(text: str) -> Set[str]:
    """中英文混合分词。

    英文/数字按词切分，中文按二元组（bigram）切分。
    中文没有空格，若沿用 query.split() 会导致整句成为一个 token，永远匹配不到。
    """
    text = (text or "").lower()
    tokens: Set[str] = set(_WORD_RE.findall(text))
    for seg in _CN_RE.findall(text):
        if len(seg) == 1:
            tokens.add(seg)
        else:
            for i in range(len(seg) - 1):
                tokens.add(seg[i:i + 2])
    return tokens


def compute_chunk_hash(text: str) -> str:
    """切片内容哈希，用于切片级去重"""
    return hashlib.md5(text.strip().encode('utf-8')).hexdigest()[:16]


def compute_file_hash(path: str) -> str:
    """文件内容哈希（分块读取，支持大文件）"""
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class DocumentProcessor:
    def __init__(self):
        self.chunk_size = CHUNK_SIZE
        self.chunk_overlap = CHUNK_OVERLAP
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " ", ""],
        )

    # ===== PDF 解析 =====

    def load_pdf(self, pdf_path: str) -> List[LCDocument]:
        """加载 PDF：优先使用 PyMuPDF（快），失败时降级到 PyPDFLoader（兼容）"""
        try:
            docs = self._load_pdf_pymupdf(pdf_path)
            if docs:
                print(f"[PDF] {len(docs)} pages (PyMuPDF)")
                return docs
            print("[PDF] PyMuPDF 未提取到文本，尝试 PyPDFLoader")
        except Exception as e:
            print(f"[PDF] PyMuPDF 解析失败，降级 PyPDFLoader: {e}")
        return self._load_pdf_langchain(pdf_path)

    @staticmethod
    def _load_pdf_pymupdf(pdf_path: str) -> List[LCDocument]:
        import pymupdf  # PyMuPDF
        source = os.path.basename(pdf_path)
        docs: List[LCDocument] = []
        with pymupdf.open(pdf_path) as doc:
            for i, page in enumerate(doc):
                text = page.get_text("text")
                if text and text.strip():
                    docs.append(LCDocument(
                        page_content=text,
                        metadata={"source": source, "page": i},
                    ))
        return docs

    @staticmethod
    def _load_pdf_langchain(pdf_path: str) -> List[LCDocument]:
        from langchain_community.document_loaders import PyPDFLoader
        docs = PyPDFLoader(pdf_path).load()
        print(f"[PDF] {len(docs)} pages (PyPDFLoader)")
        return docs

    def split_documents(self, documents: List[LCDocument]) -> List[LCDocument]:
        chunks = self.text_splitter.split_documents(documents)
        print(f"[PDF] {len(chunks)} chunks")
        return chunks

    def get_chunks_from_pdf(self, pdf_path: str, file_hash: Optional[str] = None,
                            source_name: Optional[str] = None) -> List[LCDocument]:
        """解析 + 切片，并写入 file_hash / source 元数据（用于去重与检索来源展示）"""
        docs = self.load_pdf(pdf_path)
        if source_name:
            for d in docs:
                d.metadata["source"] = source_name
        chunks = self.split_documents(docs)
        if file_hash:
            for c in chunks:
                c.metadata["file_hash"] = file_hash
        return chunks

    def extract_text_preview(self, text: str, max_len: int = 100) -> str:
        text = text.replace('\n', ' ').strip()
        return text[:max_len] + "..." if len(text) > max_len else text


class SimpleVectorStore:
    """轻量关键词向量存储 - 不依赖外部 API / Embedding 模型"""

    # 检索为空时兜底使用的切片数（详见 fallback_context 的说明）
    FALLBACK_CHUNKS = 12

    def __init__(self, persist_dir: str = "./faiss_index"):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.chunks: List[LCDocument] = []
        self.index: Dict[str, int] = {}
        self._file_hashes: Set[str] = set()
        self._token_cache: Dict[int, Set[str]] = {}

    # ===== 去重辅助 =====

    def has_file(self, file_hash: str) -> bool:
        """判断该文件内容是否已经入库"""
        return file_hash in self._file_hashes

    # ===== 写入 / 移除 =====

    def add_documents(self, chunks: List[LCDocument], save: bool = True) -> int:
        """批量加入切片，返回真正新增的数量（自动按内容去重）"""
        new_count = 0
        for chunk in chunks:
            h = compute_chunk_hash(chunk.page_content)
            if h in self.index:
                continue
            self.index[h] = len(self.chunks)
            self.chunks.append(chunk)
            fh = (chunk.metadata or {}).get("file_hash")
            if fh:
                self._file_hashes.add(fh)
            new_count += 1
        print(f"[INDEX] 新增 {new_count} chunks，总计 {len(self.chunks)}")
        if save and new_count:
            self.save()
        return new_count

    def remove_by_file_hash(self, file_hash: str) -> int:
        """按文件移除其全部切片（用于删除文档）"""
        before = len(self.chunks)
        self.chunks = [
            c for c in self.chunks
            if (c.metadata or {}).get("file_hash") != file_hash
        ]
        self._file_hashes.discard(file_hash)
        self._reindex()
        removed = before - len(self.chunks)
        if removed:
            self.save()
        print(f"[INDEX] 移除 {removed} chunks (file_hash={file_hash[:8]})")
        return removed

    def remove_by_source(self, source_name: str) -> int:
        """按来源文件名移除切片（兼容没有 file_hash 元数据的旧数据）"""
        before = len(self.chunks)
        removed_hashes = {
            (c.metadata or {}).get("file_hash")
            for c in self.chunks
            if (c.metadata or {}).get("source") == source_name
        }
        self.chunks = [
            c for c in self.chunks
            if (c.metadata or {}).get("source") != source_name
        ]
        self._file_hashes -= {h for h in removed_hashes if h}
        self._reindex()
        removed = before - len(self.chunks)
        if removed:
            self.save()
        print(f"[INDEX] 移除 {removed} chunks (source={source_name})")
        return removed

    def _reindex(self) -> None:
        self.index = {compute_chunk_hash(c.page_content): i for i, c in enumerate(self.chunks)}
        self._token_cache.clear()

    # ===== 持久化 =====

    def save(self) -> None:
        """原子化写入，避免中途失败导致索引文件损坏"""
        save_path = self.persist_dir / "chunks.json"
        data = [{"content": c.page_content, "metadata": c.metadata} for c in self.chunks]
        fd, tmp_path = tempfile.mkstemp(dir=str(self.persist_dir), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            os.replace(tmp_path, save_path)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
        print(f"[INDEX] 已保存 {len(self.chunks)} chunks -> {save_path}")

    def load(self) -> bool:
        save_path = self.persist_dir / "chunks.json"
        if not save_path.exists():
            return False
        with open(save_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.chunks = [
            LCDocument(page_content=d["content"], metadata=d.get("metadata", {}))
            for d in data
        ]
        self._reindex()
        self._file_hashes = {
            (c.metadata or {}).get("file_hash") for c in self.chunks
        }
        self._file_hashes.discard(None)
        print(f"[INDEX] 已加载 {len(self.chunks)} chunks，覆盖 {len(self._file_hashes)} 个文件")
        return True

    # ===== 检索 =====

    def _chunk_tokens(self, idx: int) -> Set[str]:
        cached = self._token_cache.get(idx)
        if cached is None:
            chunk = self.chunks[idx]
            # 正文之外，把来源文件名一并纳入检索。
            # 用户经常按文件名相关的词提问（"简历""论文""接口文档"），
            # 而这些词未必出现在正文里 —— 只索引正文会白白漏掉这些命中。
            cached = tokenize(chunk.page_content) | tokenize(
                str((chunk.metadata or {}).get("source", ""))
            )
            self._token_cache[idx] = cached
        return cached

    def search(self, query: str, k: int = 5) -> List[LCDocument]:
        """中英文混合关键词检索：bigram 分词 + 重叠度打分"""
        q_tokens = tokenize(query)
        if not q_tokens:
            return []

        scored = []
        for idx in range(len(self.chunks)):
            overlap = len(q_tokens & self._chunk_tokens(idx))
            if overlap > 0:
                scored.append((overlap, idx))

        # 命中词越多越靠前；相同命中时优先更短的切片（信息更集中）
        scored.sort(key=lambda x: (-x[0], len(self.chunks[x[1]].page_content)))
        return [self.chunks[idx] for _, idx in scored[:k]]

    def fallback_context(self, limit: int = FALLBACK_CHUNKS) -> List[LCDocument]:
        """关键词检索命中 0 条时的兜底上下文：退回最近入库的若干切片。

        为什么必须有这层兜底：
        问句与正文的用词经常完全不重合，最典型的是总结/概括类提问 ——
        "总结一下简历内容"里的"总结""内容"在简历正文中根本不存在，
        关键词检索必然 0 命中。但文档确实在库里、用户在界面上也看得见，
        此时若把 context 置空，模型只能回答"未检索到相关文档内容"，
        用户看到的就是"明明上传了文档，它却说没有内容"。
        宁可多给一些上下文让模型自己判断，也不要谎报"没有内容"。
        """
        if limit <= 0:
            return []
        return self.chunks[-limit:]


class RAGPipeline:
    def __init__(self, vector_store: SimpleVectorStore):
        self.vector_store = vector_store
        self.prompt_template = """你是一个专业的技术文档助手。请根据以下参考资料回答用户的问题。
如果参考资料中没有相关信息，请如实告知用户"根据现有文档无法回答该问题"。
保持回答准确、简洁、专业。

【参考资料】
{context}

【用户问题】
{question}

【回答】"""

    @staticmethod
    def _source_label(meta: dict) -> str:
        """生成来源标注，如 'vue.pdf 第12页'"""
        src = meta.get("source", "文档")
        page = meta.get("page")
        return src if page is None else f"{src} 第{int(page) + 1}页"

    @classmethod
    def build_labelled_context(cls, docs: List[LCDocument]) -> str:
        """给每个片段加上来源标注，便于大模型在回答中准确引用出处"""
        blocks = []
        for i, doc in enumerate(docs, 1):
            label = cls._source_label(doc.metadata or {})
            blocks.append(f"[片段{i} | 来源：{label}]\n{doc.page_content}")
        return "\n\n".join(blocks)

    def get_context(self, question: str, top_k: int = 5) -> str:
        docs = self.vector_store.search(question, k=top_k)
        return self.build_labelled_context(docs)

    def format_prompt(self, question: str, context: str) -> str:
        return self.prompt_template.format(context=context, question=question)

    # 总结/概括类提问的触发词。这类问法的词（总结、概括、主要内容）通常
    # 不出现在正文里，关键词检索要么 0 命中，要么只命中零散片段 ——
    # 例如"总结一下简历内容"只会命中文件名带"简历"的那一份，漏掉其余文档。
    _SUMMARY_HINTS = (
        "总结", "概括", "概述", "归纳", "梳理", "主要内容",
        "讲了什么", "说了什么", "介绍了什么", "全文", "整篇", "整份",
    )

    @classmethod
    def is_summary_query(cls, question: str) -> bool:
        """判断是否为总结/概括类提问"""
        q = question or ""
        return any(hint in q for hint in cls._SUMMARY_HINTS)

    @staticmethod
    def _merge_unique(primary: List[LCDocument],
                      extra: List[LCDocument]) -> List[LCDocument]:
        """按正文内容去重合并，保持 primary 在前的顺序"""
        seen = set()
        merged: List[LCDocument] = []
        for doc in list(primary) + list(extra):
            if doc.page_content in seen:
                continue
            seen.add(doc.page_content)
            merged.append(doc)
        return merged

    def retrieve_and_format(self, question: str, top_k: int = 5) -> tuple:
        """一次检索，返回 (prompt, 带来源标注的context, sources)，避免重复检索"""
        docs = self.vector_store.search(question, k=top_k)

        # 库里有内容却检索为空时不能直接放弃（原因见 fallback_context）
        if not docs:
            docs = self.vector_store.fallback_context()
        elif self.is_summary_query(question):
            # 总结类提问需要看到更完整的文档，而不是只靠几个零散命中
            docs = self._merge_unique(docs, self.vector_store.fallback_context())

        context = self.build_labelled_context(docs)
        prompt = self.format_prompt(question, context)

        # 去重且保持顺序
        seen = set()
        unique_sources = []
        for doc in docs:
            label = self._source_label(doc.metadata or {})
            if label not in seen:
                seen.add(label)
                unique_sources.append(label)
        return prompt, context, unique_sources
