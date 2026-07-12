#!/usr/bin/env python3
"""
report_search.py
================
不依赖 iwencai 的本地研报语义搜索脚本。

基于东财 reportapi 下载的 PDF 研报（可用 industry_report_downloader.py 批量下载），
抽取文本、切分、向量化，构建本地可检索的向量索引，支持语义问答。

支持三种 embedding 后端：
  1. openai          OpenAI 兼容接口（text-embedding-3-small 等），需 LLM_API_KEY 或 OPENAI_API_KEY
  2. sentence-transformers  本地模型，零 key，首次下载模型
  3. bm25            纯本地关键词检索，无需任何模型和 key

依赖
----
必选：
  pip install requests numpy

根据后端任选其一：
  pip install pdfplumber                  # 推荐，PDF 文本抽取
  pip install pypdf                       # 备选
  pip install pdfminer.six                # 备选
  pip install sentence-transformers       # 若用 sentence-transformers

用法
----
# 1. 先索引（首次或 PDF 更新后运行）
python report_search.py --index ./reports/半导体

# 2. 语义搜索
python report_search.py --input ./reports/半导体 \
  --query "人形机器人产业链中丝杠和减速器的技术壁垒是什么" --top-k 5

# 3. 语义搜索 + 用 LLM 综合回答
python report_search.py --input ./reports/半导体 \
  --query "人形机器人产业链中丝杠和减速器的技术壁垒是什么" \
  --answer --top-k 5

# 4. 使用本地 sentence-transformers（零 key）
python report_search.py --index ./reports/半导体 --embedding sentence-transformers
python report_search.py --input ./reports/半导体 --embedding sentence-transformers \
  --query "丝杠和减速器技术壁垒"

# 5. 完全无 key：BM25 关键词检索
python report_search.py --index ./reports/半导体 --embedding bm25
python report_search.py --input ./reports/半导体 --embedding bm25 --query "丝杠 减速器"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import requests

warnings.filterwarnings("ignore")

# ─────────────────────────── 配置 ───────────────────────────

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL") or os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")

EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-3-small")
SENTENCE_MODEL = os.environ.get("SENTENCE_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

CACHE_DIR = Path(".report_search_cache")
CACHE_DIR.mkdir(exist_ok=True)

CHUNK_SIZE = int(os.environ.get("REPORT_SEARCH_CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.environ.get("REPORT_SEARCH_CHUNK_OVERLAP", "128"))

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"


# ─────────────────────────── PDF 文本抽取 ───────────────────────────

def _extract_with_pdfplumber(path: Path, max_pages: int | None = None) -> str:
    import pdfplumber
    chunks = []
    with pdfplumber.open(str(path)) as pdf:
        for i, page in enumerate(pdf.pages):
            if max_pages is not None and i >= max_pages:
                break
            chunks.append(page.extract_text() or "")
    return "\n".join(chunks)


def _extract_with_pypdf(path: Path, max_pages: int | None = None) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    chunks = []
    for i, page in enumerate(reader.pages):
        if max_pages is not None and i >= max_pages:
            break
        chunks.append(page.extract_text() or "")
    return "\n".join(chunks)


def _extract_with_pdfminer(path: Path, max_pages: int | None = None) -> str:
    from pdfminer.high_level import extract_text
    return extract_text(str(path), maxpages=max_pages) or ""


def extract_pdf_text(path: Path, max_pages: int | None = None) -> str:
    """优雅降级抽取 PDF 文本：pdfplumber → pypdf → pdfminer"""
    for fn in (_extract_with_pdfplumber, _extract_with_pypdf, _extract_with_pdfminer):
        try:
            return fn(path, max_pages)
        except Exception:
            continue
    return ""


def _has_pdf_extractor() -> bool:
    for mod in ("pdfplumber", "pypdf", "pdfminer"):
        try:
            __import__(mod)
            return True
        except Exception:
            continue
    return False


# ─────────────────────────── 文本切分 ───────────────────────────

def _split_to_sentences(text: str) -> list[str]:
    """按中文/英文句号/换行切分句子，保留句尾标点。"""
    text = re.sub(r"\n+", "\n", text)
    parts = re.split(r"([。！？!?.?]\s+)", text)
    sents = []
    buf = ""
    for p in parts:
        buf += p
        if re.match(r"[。！？!?]+\s*$", p):
            sents.append(buf.strip())
            buf = ""
    if buf.strip():
        sents.append(buf.strip())
    return [s for s in sents if s]


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    语义友好的滑动窗口切分：
    优先按句子边界切分，句子合并成约 chunk_size 字的块，块间重叠 overlap 字。
    """
    text = re.sub(r"\s+", "", text)
    if len(text) <= chunk_size:
        return [text] if text else []

    sentences = _split_to_sentences(text)
    if not sentences:
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size - overlap) if text[i:i+chunk_size]]

    chunks = []
    current = ""
    for s in sentences:
        if len(current) + len(s) <= chunk_size:
            current += s
        else:
            if current:
                chunks.append(current)
            current = s
            # 单句超过 chunk_size 直接截断
            while len(current) > chunk_size:
                chunks.append(current[:chunk_size])
                current = current[chunk_size - overlap:]
    if current:
        chunks.append(current)

    # 做重叠滑动窗口
    if overlap and len(chunks) > 1:
        final = []
        for i in range(len(chunks)):
            start = max(0, i - 1)
            merged = ""
            for j in range(start, i + 1):
                if len(merged) + len(chunks[j]) <= chunk_size:
                    merged += chunks[j]
                else:
                    break
            if merged:
                final.append(merged)
        chunks = final

    # 去重并保留顺序
    seen = set()
    out = []
    for c in chunks:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


# ─────────────────────────── Embedding 后端 ───────────────────────────

def _embed_openai(texts: list[str], model: str = EMBED_MODEL) -> np.ndarray:
    """OpenAI 兼容 /embeddings 接口。"""
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "使用 openai embedding 后端需要 OPENAI_API_KEY 或 LLM_API_KEY 环境变量。"
            "或者用 --embedding sentence-transformers / bm25 无需此 key。"
        )
    url = f"{OPENAI_BASE_URL}/embeddings"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    # OpenAI 限制每批 tokens / 数量，保守分批
    batch = 100
    all_vecs = []
    for i in range(0, len(texts), batch):
        batch_texts = texts[i:i+batch]
        payload = {"input": batch_texts, "model": model, "encoding_format": "float"}
        r = requests.post(url, json=payload, headers=headers, timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f"embedding API HTTP {r.status_code}: {r.text[:500]}")
        data = r.json()
        if "data" not in data:
            raise RuntimeError(f"embedding API 返回异常: {data}")
        vecs = sorted(data["data"], key=lambda x: x["index"])
        all_vecs.extend([v["embedding"] for v in vecs])
    return np.array(all_vecs, dtype=np.float32)


def _embed_sentence_transformers(texts: list[str]) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError(
            "使用 sentence-transformers 后端需安装：pip install sentence-transformers。"
            "错误：" + str(e)
        )
    model = SentenceTransformer(SENTENCE_MODEL)
    return np.array(model.encode(texts, normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)


def _tokenize(text: str) -> list[str]:
    """BM25 简单分词：中文按字、英文/数字按词。"""
    text = text.lower()
    # 中文按字
    cn = re.findall(r"[\u4e00-\u9fff]", text)
    # 英文/数字词
    words = re.findall(r"[a-z0-9]+", text)
    return cn + words


def _bm25_score(query: str, docs: list[str], k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    """纯 numpy 实现 BM25 评分。"""
    tokenized_docs = [_tokenize(d) for d in docs]
    tokenized_query = _tokenize(query)
    if not tokenized_query:
        return np.zeros(len(docs))

    # 构建词表
    vocab = set()
    for tokens in tokenized_docs:
        vocab.update(tokens)
    vocab = sorted(vocab)
    word_to_idx = {w: i for i, w in enumerate(vocab)}

    # 文档 term frequency
    doc_lens = np.array([len(t) for t in tokenized_docs], dtype=np.float32)
    avgdl = doc_lens.mean() if doc_lens.mean() > 0 else 1.0
    n_docs = len(docs)

    tf = np.zeros((n_docs, len(vocab)), dtype=np.float32)
    for i, tokens in enumerate(tokenized_docs):
        for t in tokens:
            if t in word_to_idx:
                tf[i, word_to_idx[t]] += 1

    # IDF
    df = np.count_nonzero(tf > 0, axis=0)
    idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)

    # BM25
    denom = tf + k1 * (1 - b + b * doc_lens[:, None] / avgdl)
    bm25 = idf * (tf * (k1 + 1)) / denom

    # query vector
    q_vec = np.zeros(len(vocab), dtype=np.float32)
    for t in tokenized_query:
        if t in word_to_idx:
            q_vec[word_to_idx[t]] += 1

    scores = bm25 @ q_vec
    return scores


def embed(texts: list[str], provider: str) -> tuple[np.ndarray, str]:
    """
    对 texts 做 embedding。返回 (embeddings, provider_name_or_metric)。
    openai / sentence-transformers 返回归一化向量；bm25 返回 BM25 分数矩阵（按 query 列）。
    """
    provider = provider.lower()
    if provider == "openai":
        vecs = _embed_openai(texts)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        vecs = vecs / norms
        return vecs, "openai"
    if provider == "sentence-transformers":
        vecs = _embed_sentence_transformers(texts)
        return vecs, "sentence-transformers"
    if provider == "bm25":
        # 文本不传，实际在 search 时计算；这里返回占位
        return np.array([]), "bm25"
    raise ValueError(f"未知 embedding 后端: {provider}")


def _cosine_similarity(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    """query_vec 与 doc_vecs 的 cosine 相似度。"""
    return (doc_vecs @ query_vec).flatten()


# ─────────────────────────── 索引 ───────────────────────────

class LocalIndex:
    """本地可持久化的向量/关键词索引。"""

    def __init__(self, root: Path, provider: str, chunks: list[dict] | None = None,
                 embeddings: np.ndarray | None = None, meta: dict | None = None):
        self.root = Path(root)
        self.provider = provider
        self.chunks = chunks or []
        self.embeddings = embeddings
        self.meta = meta or {}

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump({
                "root": str(self.root),
                "provider": self.provider,
                "chunks": self.chunks,
                "embeddings": self.embeddings,
                "meta": self.meta,
            }, f)

    @staticmethod
    def load(path: Path) -> "LocalIndex":
        with open(path, "rb") as f:
            data = pickle.load(f)
        return LocalIndex(
            root=Path(data["root"]),
            provider=data["provider"],
            chunks=data["chunks"],
            embeddings=data["embeddings"],
            meta=data["meta"],
        )

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """返回得分最高的 top_k 个 chunk。"""
        if self.provider == "bm25":
            corpus = [c["text"] for c in self.chunks]
            scores = _bm25_score(query, corpus)
        else:
            if self.embeddings is None or len(self.embeddings) == 0:
                raise RuntimeError("索引没有 embedding，请先重新索引。")
            if self.provider == "openai":
                q_vec = _embed_openai([query])
            elif self.provider == "sentence-transformers":
                q_vec = _embed_sentence_transformers([query])
            else:
                raise ValueError(f"未知 provider: {self.provider}")
            q_norm = np.linalg.norm(q_vec)
            if q_norm == 0:
                q_vec = np.zeros_like(q_vec)
            else:
                q_vec = q_vec / q_norm
            scores = _cosine_similarity(q_vec[0], self.embeddings)

        idx = np.argsort(scores)[::-1][:top_k]
        results = []
        for i in idx:
            if scores[i] <= 0:
                continue
            rec = self.chunks[i].copy()
            rec["score"] = float(scores[i])
            results.append(rec)
        return results


def _cache_path(root: Path, provider: str) -> Path:
    """索引缓存路径，与输入目录和 provider 绑定。"""
    key = str(root.resolve()) + "::" + provider
    name = hashlib.md5(key.encode()).hexdigest()[:12] + f"_{provider}.pkl"
    return CACHE_DIR / name


def build_index(root: Path, provider: str, max_pages: int | None = None,
                max_files: int | None = None) -> LocalIndex:
    """从 PDF 文件夹构建索引。"""
    if not _has_pdf_extractor():
        raise RuntimeError("未找到 PDF 文本提取库，请先安装：pip install pdfplumber（或 pypdf / pdfminer.six）")

    pdfs = sorted(root.rglob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"目录内未找到 PDF: {root}")
    if max_files:
        pdfs = pdfs[:max_files]

    print(f"发现 {len(pdfs)} 个 PDF，开始抽取文本...")
    chunks = []
    for i, p in enumerate(pdfs, 1):
        text = extract_pdf_text(p, max_pages)
        status = f"{len(text)} 字符" if text else "空(可能为扫描件/图片)"
        print(f"  [{i:3d}/{len(pdfs)}] {p.name[:50]:<50} → {status}")
        if not text:
            continue
        for c in chunk_text(text):
            chunks.append({
                "file": p.name,
                "text": c,
                "len": len(c),
            })

    if not chunks:
        raise RuntimeError("没有从 PDF 中抽取出有效文本，无法建立索引。")

    print(f"\n共生成 {len(chunks)} 个文本块，使用 {provider} 后端做 embedding...")
    embeddings = None
    if provider != "bm25":
        embeddings, _ = embed([c["text"] for c in chunks], provider)
        print(f"embedding 矩阵形状: {embeddings.shape}")

    idx = LocalIndex(root=root, provider=provider, chunks=chunks, embeddings=embeddings,
                     meta={"n_pdfs": len(pdfs), "n_chunks": len(chunks), "build_time": datetime.now().isoformat()})

    cache_path = _cache_path(root, provider)
    idx.save(cache_path)
    print(f"索引已保存到: {cache_path}")
    return idx


def load_or_build_index(root: Path, provider: str, rebuild: bool = False,
                        max_pages: int | None = None, max_files: int | None = None) -> LocalIndex:
    cache_path = _cache_path(root, provider)
    if not rebuild and cache_path.exists():
        print(f"加载已有索引: {cache_path}")
        idx = LocalIndex.load(cache_path)
        if idx.provider == provider:
            return idx
        print("索引 provider 不一致，重新构建。")
    return build_index(root, provider, max_pages, max_files)


# ─────────────────────────── LLM 回答综合 ───────────────────────────

def _llm_chat(messages: list[dict], model: str = LLM_MODEL, temperature: float = 0.3,
              max_tokens: int = 1200) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("未设置 OPENAI_API_KEY / LLM_API_KEY，无法调用 LLM 综合回答。")
    url = f"{OPENAI_BASE_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    for attempt in range(3):
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=120)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
            time.sleep(2 * (attempt + 1))
        except Exception as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"LLM 调用失败: {last_err}")


def answer_with_llm(query: str, results: list[dict], model: str = LLM_MODEL) -> str:
    """根据检索结果让 LLM 生成综合回答。"""
    context = "\n\n---\n\n".join(
        f"[来源: {r['file']}]\n{r['text']}" for r in results
    )
    prompt = (
        "你是一名专业的 A 股产业链研究分析师。请根据下方从研报 PDF 中提取的原始片段，"
        "回答用户的问题。回答需引用具体来源文件，必要时保留关键数据。"
        "若片段中没有足够信息，请明确说明。\n\n"
        f"用户问题：{query}\n\n"
        f"参考片段：\n\n{context}"
    )
    return _llm_chat([{"role": "user", "content": prompt}], model=model)


# ─────────────────────────── CLI ───────────────────────────

def main() -> None:
    global EMBED_MODEL, SENTENCE_MODEL
    parser = argparse.ArgumentParser(
        description="本地研报语义搜索（不依赖 iwencai）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例：
  # 索引
  python report_search.py --index ./reports/半导体

  # 搜索
  python report_search.py --input ./reports/半导体 --query "丝杠减速器技术壁垒" --top-k 5

  # 搜索 + LLM 综合
  python report_search.py --input ./reports/半导体 --query "丝杠减速器技术壁垒" --answer --top-k 5

  # 使用本地 sentence-transformers（零 key）
  python report_search.py --index ./reports/半导体 --embedding sentence-transformers
"""
    )
    parser.add_argument("--index", type=Path, help="索引 PDF 目录（首次或更新后运行）")
    parser.add_argument("--input", type=Path, help="搜索时使用的 PDF 目录")
    parser.add_argument("--query", "-q", type=str, help="搜索查询")
    parser.add_argument("--embedding", "-e", default="openai",
                        choices=["openai", "sentence-transformers", "bm25"],
                        help="embedding 后端（默认 openai）")
    parser.add_argument("--top-k", "-k", type=int, default=5, help="返回 top-k 个片段")
    parser.add_argument("--answer", action="store_true", help="使用 LLM 综合回答")
    parser.add_argument("--rebuild", action="store_true", help="强制重建索引")
    parser.add_argument("--max-pages", type=int, help="每份 PDF 最多读取前 N 页")
    parser.add_argument("--max-files", type=int, help="最多索引前 N 个 PDF")
    parser.add_argument("--embed-model", default=EMBED_MODEL, help="OpenAI embedding 模型")
    parser.add_argument("--sentence-model", default=SENTENCE_MODEL, help="sentence-transformers 模型名")
    args = parser.parse_args()

    EMBED_MODEL = args.embed_model
    SENTENCE_MODEL = args.sentence_model

    if args.index:
        idx = build_index(args.index, args.embedding, max_pages=args.max_pages, max_files=args.max_files)
        print("索引完成。")
        return

    if args.input:
        if not args.query:
            parser.error("搜索时需要 --query")
        idx = load_or_build_index(args.input, args.embedding, rebuild=args.rebuild,
                                  max_pages=args.max_pages, max_files=args.max_files)
        print(f"\n查询: {args.query}")
        print(f"后端: {args.embedding}")
        results = idx.search(args.query, top_k=args.top_k)

        if not results:
            print("未找到相关片段。")
            return

        print(f"\n找到 {len(results)} 个相关片段：\n")
        for i, r in enumerate(results, 1):
            print(f"--- {i}. {r['file']} (score: {r['score']:.4f}) ---")
            print(r["text"][:600] + ("..." if len(r["text"]) > 600 else ""))
            print()

        if args.answer:
            print("=" * 60)
            print("LLM 综合回答：")
            print("=" * 60)
            print(answer_with_llm(args.query, results))
        return

    parser.print_help()


if __name__ == "__main__":
    main()
