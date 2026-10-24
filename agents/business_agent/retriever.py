"""Local Chroma vector retrieval for the business knowledge base."""

from __future__ import annotations

# hashlib：根据文本等信息计算 SHA-256 哈希，用来生成知识片段的 ID
import hashlib
# os：读取环境变量
import os
import re
# Iterable：表示可以逐个遍历的数据，例如列表
from collections.abc import Iterable
# 简化数据类的定义
from dataclasses import dataclass
from pathlib import Path
# Protocol	定义一个对象需要提供哪些方法
from typing import Any, Protocol

import chromadb
from chromadb.errors import NotFoundError
from langchain_ollama import OllamaEmbeddings

# retriever.py：负责知识的准备、向量入库和相似度检索。

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KNOWLEDGE_DIR = PROJECT_ROOT / "data" / "career_knowledge"
DEFAULT_CHROMA_DIRECTORY = PROJECT_ROOT / "data" / "chroma"
# 默认集合名称。可以把集合暂时理解成数据库中的一张专门存放向量记录的表。
DEFAULT_COLLECTION_NAME = "career_knowledge_v1"
DEFAULT_EMBEDDING_MODEL = "bge-m3:latest"
DEFAULT_BATCH_SIZE = 64
MAX_RESULTS = 10


# 将文本按照这个格式来描述
@dataclass(frozen=True)
class KnowledgeChunk:
    """A retrievable passage with metadata suitable for source citations."""

    title: str # 文档标题
    source: str # 文档来源路径
    content: str # 知识片段正文
    score: float = 0.0 #入库时通常没有检索分数 检索阶段会根据ChromaDB返回的距离计算分数 下放入新的KnowledgeChunk


# Protocol 用来描述一个对象应该具备什么样的方法。
# KnowledgeRetriever 规定检索器需要什么能力
class KnowledgeRetriever(Protocol):
    """Minimal retriever contract consumed by the business Agent."""
    # 它要求对象提供：retrieve(query, limit=3)
    # 接收查询文本，返回 KnowledgeChunk 列表。
    def retrieve(self, query: str, *, limit: int = 3) -> list[KnowledgeChunk]:
        ...


class KnowledgeIndexEmptyError(RuntimeError):
    """Raised when a query targets an index that has not been populated."""


# Embedder 规定嵌入器需要什么能力 它要求提供两个方法：
# embed_documents(texts)：将多条文档转换为向量。
# embed_query(text)：将单条查询转换为向量。
class Embedder(Protocol):
    """Embedding methods used by the Chroma retriever."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


# 提取标题 这个函数接受两个参数
# content：Markdown 文件的全部文本。
# fallback：如果没有找到一级标题，就使用这个备用标题。
def _markdown_title(content: str, fallback: str) -> str:
    match = re.search(r"(?m)^#\s+(.+?)\s*$", content)
    return match.group(1).strip() if match else fallback


# 把长文档切成多个片段 _split_markdown() 将文档切分成知识片段。
def _split_markdown(
    content: str,
    *,
    title: str,
    source: str,
    max_chars: int = 700,
) -> list[KnowledgeChunk]:
    """Split Markdown paragraphs into bounded passages."""
    # 移除第一个一级标题
    body = re.sub(r"(?m)^#\s+.+?$", "", content, count=1).strip()
    # 按照空行拆分段落
    paragraphs = [
        paragraph.strip() # 按照空行切分文本。
        for paragraph in re.split(r"\n\s*\n", body) # 去除每个段落首尾的空白字符。
        if paragraph.strip() # 过滤空段落
    ]

    chunks: list[KnowledgeChunk] = []
    current: list[str] = [] # 暂时保存正在组合的段落
    current_length = 0 # 记录当前片段的估算字符长度
    for paragraph in paragraphs:
        if current and current_length + len(paragraph) + 2 > max_chars:
            chunks.append(KnowledgeChunk(title, source, "\n\n".join(current)))
            current = []
            current_length = 0
        current.append(paragraph)
        current_length += len(paragraph) + 2

    if current:
        chunks.append(KnowledgeChunk(title, source, "\n\n".join(current)))
    return chunks


# 确定知识目录。
def _resolve_project_path(value: str | Path | None, default: Path) -> Path:
    path = Path(value) if value is not None else default
    return path if path.is_absolute() else PROJECT_ROOT / path


# 批量读取知识文件
def load_knowledge_chunks(
    directory: str | Path | None = None,
) -> list[KnowledgeChunk]:
    """Load and chunk all Markdown documents under a knowledge directory."""
    # 确定知识目录。
    knowledge_directory = _resolve_project_path(directory, DEFAULT_KNOWLEDGE_DIR)
    # 检查目录是否存在 不存在就返回空列表
    if not knowledge_directory.is_dir():
        return []

    chunks: list[KnowledgeChunk] = []
    # rglob("*.md") 递归查找目录及子目录中的 Markdown 文件。
    for path in sorted(knowledge_directory.rglob("*.md")):
        # read_text() 读取文件内容。
        content = path.read_text(encoding="utf-8").strip()
        # 空文件就跳过
        if not content:
            continue
        # relative_to(...).as_posix() 生成相对来源路径。
        source = path.relative_to(knowledge_directory).as_posix()
        # _markdown_title() 提取标题。
        title = _markdown_title(content, path.stem.replace("_", " ").title())
        chunks.extend(
            # _split_markdown() 将文档切分成知识片段。
            _split_markdown(
                content,
                title=title,
                source=source,
            )
        )
    return chunks

# _stable_chunk_id()：为什么要生成稳定 ID？
def _stable_chunk_id(source: str, chunk_index: int, content: str) -> str:
    """Build an idempotent Chroma id from the source and passage content."""

    identity = f"{source}\0{chunk_index}\0{content}".encode("utf-8")
    # 返回十六进制形式的哈希字符串。
    return hashlib.sha256(identity).hexdigest()


def _positive_int(value: str | None, default: int) -> int:
    try:
        parsed = int(value) if value is not None else default
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


class ChromaKnowledgeRetriever:
    """Persist career documents in Chroma and retrieve them by vector similarity."""

    def __init__(
        self,
        directory: str | Path | None = None,
        *,
        client: Any | None = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        base_url: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        embedder: Embedder | None = None,
    ):
        # 数据库文件保存的位置。
        self.directory = _resolve_project_path(directory, DEFAULT_CHROMA_DIRECTORY)
        self.collection_name = collection_name # 使用哪个 ChromaDB 集合。
        self.embedding_model = embedding_model # 使用哪个嵌入模型。
        self.base_url = base_url # Ollama 服务地址
        self.batch_size = max(1, int(batch_size)) # 每批处理多少条文档。
        self._client = client
        self._embedder = embedder
        self._collection: Any | None = None

    # 获取 ChromaDB 客户端
    def _get_client(self) -> Any:
        # 检查有没有ChromaDB客户端 没有就创建一个持久化客户端 并将数据库目录传进去
        # PersistentClient 是持久化客户端，目的是将数据保存到指定目录，而不是只保存在临时内存中。
        if self._client is None:
            self._client = chromadb.PersistentClient(path=str(self.directory))
        return self._client

    # 获取 Ollama 嵌入模型
    def _get_embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = OllamaEmbeddings(
                model=self.embedding_model,
                base_url=self.base_url,
            )
        return self._embedder

    # 获取或创建知识集合
    def _get_collection(self) -> Any:
        if self._collection is None:
            # 先通过 _get_client() 获取数据库客户端。
            # 使用客户端获取指定名称的集合。
            # 如果集合不存在，就创建它。
            # 将集合对象保存在 self._collection 中。
            # 返回集合对象。
            self._collection = self._get_client().get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}, # 这是在指定 ChromaDB 使用余弦距离作为向量比较方式。
                embedding_function=None,
            )
        return self._collection

    #
    def _drop_collection(self) -> None:
        try: # 获取 ChromaDB 客户端。删除指定名称的集合。
            self._get_client().delete_collection(self.collection_name)
        except NotFoundError:
            pass
        self._collection = None

    def count(self) -> int:
        """Return the number of indexed passages."""

        return int(self._get_collection().count())

    # 将知识片段向量化并存入数据库
    def index_documents(
        self,
        # 已经加载、切分好的知识片段。
        chunks: Iterable[KnowledgeChunk], # Iterable：表示可以逐个遍历的数据，例如列表
        *,
        rebuild: bool = False,
    ) -> int:
        """Upsert passages and return the resulting collection size."""
        # 把输入转换成列表
        items = list(chunks)
        if not items:
            raise KnowledgeIndexEmptyError(
                "No knowledge chunks were found. Add Markdown documents under "
                f"{DEFAULT_KNOWLEDGE_DIR} and run scripts/ingest_knowledge.py."
            )
        # 判断是否需要重建集合
        if rebuild:
            # _drop_collection()：删除原有集合。
            self._drop_collection()

        collection = self._get_collection()

        source_positions: dict[str, int] = {} # 每份来源文档已经处理到第几个片段
        ids: list[str] = [] # 每个知识片段的唯一标识
        documents: list[str] = [] # 每个知识片段的正文
        metadatas: list[dict[str, str | int]] = [] # 每个片段的标题、来源和序号

        # 为每个片段生成 ID 和元数据
        for chunk in items:
            # source_positions 是一个字典 如果字典存在这个chunk_source 就返回对应的值 没有就 0
            chunk_index = source_positions.get(chunk.source, 0)
            source_positions[chunk.source] = chunk_index + 1
            # _stable_chunk_id()：生成稳定 ID？
            ids.append(_stable_chunk_id(chunk.source, chunk_index, chunk.content))
            documents.append(chunk.content)
            metadatas.append(
                {
                    "title": chunk.title,
                    "source": chunk.source,
                    "chunk_index": chunk_index,
                }
            )

        # 获取 Ollama 嵌入模型对象。默认模型是 bge-m3:latest。
        embedder = self._get_embedder()
        for start in range(0, len(items), self.batch_size):
            end = start + self.batch_size
            batch_documents = documents[start:end]
            # 将当前批次的文本转换为向量。
            embeddings = embedder.embed_documents(batch_documents)
            if len(embeddings) != len(batch_documents):
                raise ValueError("Embedding provider returned an unexpected batch size.")
            # 通过 upsert() 写入 ChromaDB
            collection.upsert(
                ids=ids[start:end],
                documents=batch_documents,
                metadatas=metadatas[start:end],
                embeddings=embeddings,
            )

        return int(collection.count())

    # 检索 retrieve()：根据用户问题，从数据库中找出相关知识。
    def retrieve(self, query: str, *, limit: int = 3) -> list[KnowledgeChunk]:
        """Embed a query and return the closest indexed passages."""

        normalized_query = query.strip()
        if not normalized_query or limit <= 0:
            return []

        collection = self._get_collection()
        indexed_count = int(collection.count())
        if indexed_count == 0:
            raise KnowledgeIndexEmptyError(
                "The Chroma knowledge index is empty. Run "
                "python scripts/ingest_knowledge.py before querying."
            )

        result_limit = min(limit, MAX_RESULTS, indexed_count)
        # 把用户问题转换成向量
        query_embedding = self._get_embedder().embed_query(normalized_query)
        # collection.query() 查询 ChromaDB:获得相关文本、元数据、距离
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=result_limit,
            # include：结果中需要哪些信息？
            include=["documents", "metadatas", "distances"],
        )

        # 取第一个元素
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        chunks: list[KnowledgeChunk] = []
        for document, metadata, distance in zip(documents, metadatas, distances):
            if document is None:
                continue
            values = metadata or {}
            source = str(values.get("source", ""))
            title = str(values.get("title", Path(source).stem))
            score = 1.0 - float(distance)
            chunks.append(
                KnowledgeChunk(
                    title=title,
                    source=source,
                    content=str(document),
                    score=score,
                )
            )

        chunks.sort(key=lambda chunk: chunk.score, reverse=True)
        return chunks[:result_limit]


# 读取项目的环境配置，创建并返回一个 ChromaKnowledgeRetriever 对象。
def create_default_business_retriever() -> ChromaKnowledgeRetriever:
    """Create the configured local Chroma retriever without rebuilding it."""

    return ChromaKnowledgeRetriever(
        directory=os.getenv("CHROMA_PERSIST_DIRECTORY", str(DEFAULT_CHROMA_DIRECTORY)),
        collection_name=os.getenv(
            "CHROMA_COLLECTION_NAME",
            DEFAULT_COLLECTION_NAME,
        ),
        embedding_model=os.getenv(
            "OLLAMA_EMBEDDING_MODEL",
            DEFAULT_EMBEDDING_MODEL,
        ),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        batch_size=_positive_int(
            os.getenv("CHROMA_BATCH_SIZE"),
            DEFAULT_BATCH_SIZE,
        ),
    )


__all__ = [
    "ChromaKnowledgeRetriever",
    "DEFAULT_CHROMA_DIRECTORY",
    "DEFAULT_COLLECTION_NAME",
    "KnowledgeChunk",
    "KnowledgeIndexEmptyError",
    "KnowledgeRetriever",
    "create_default_business_retriever",
    "load_knowledge_chunks",
]
