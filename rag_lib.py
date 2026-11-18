"""RAG 核心逻辑：切块、嵌入、检索、精排、生成。"""
import json
import os
import re

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
LLM_MODEL = "glm-4-flash"
EMBED_MODEL = "embedding-3"  # v1~v3 使用的 API 嵌入模型
LOCAL_EMBED_MODEL = "BAAI/bge-small-zh-v1.5"  # v4+ 使用的本地嵌入模型


def get_client():
    """创建 OpenAI 兼容客户端，API key 从 .env 读取。"""
    key = os.getenv("ZHIPU_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "还没配置 API key：请把 .env.example 复制一份改名为 .env，\n"
            "然后去 https://open.bigmodel.cn 注册登录 -> 右上角「API 密钥」-> 新建密钥，\n"
            "把 key 粘贴到 .env 文件里。"
        )
    return OpenAI(api_key=key, base_url=BASE_URL)


def split_text(text, chunk_size=500, overlap=50):
    """切块：每段最多 chunk_size 字，段间重叠 overlap 字，避免语义被切断在边界上。"""
    text = text.strip()
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def embed_texts(client, texts):
    """调用 API 把文字转为向量（v1~v3 使用，已由 embed_texts_local 取代）。"""
    vectors = []
    for text in texts:
        resp = client.embeddings.create(model=EMBED_MODEL, input=text)
        vectors.append(resp.data[0].embedding)
    return vectors


_local_embedder = None


def get_local_embedder():
    """懒加载本地嵌入模型：加载一次后复用，避免每次请求重复加载。"""
    global _local_embedder
    if _local_embedder is None:
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        os.environ.setdefault("FASTEMBED_CACHE_PATH", "./.model_cache")
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        from fastembed import TextEmbedding

        _local_embedder = TextEmbedding(LOCAL_EMBED_MODEL)
    return _local_embedder


def embed_texts_local(texts):
    """本地模型嵌入（bge-small-zh-v1.5，512 维）：零 API 成本、离线、毫秒级。"""
    model = get_local_embedder()
    return [vec.tolist() for vec in model.embed(texts)]


def cosine_similarity(a, b):
    """余弦相似度：a·b / (|a|·|b|)，取值 -1~1，越接近 1 越相似。"""
    a = np.asarray(a)
    b = np.asarray(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def retrieve(index, query_vec, k=3):
    """线性扫描找出与问题最相似的 k 段（v1 实现，v2 起由 chromadb 取代）。"""
    scored = []
    for i, vec in enumerate(index["vectors"]):
        score = cosine_similarity(query_vec, vec)
        scored.append((score, i))
    scored.sort(reverse=True)
    return [(index["chunks"][i], score) for score, i in scored[:k]]


def build_qa_prompt(question, contexts):
    """拼接问答提示词；「只根据资料回答」是减少幻觉的关键约束。"""
    context_text = "\n\n".join(f"[资料{i + 1}] {text}" for i, (text, _) in enumerate(contexts))
    return (
        "你是一个文档问答助手。请只根据下面的资料回答用户的问题；"
        "如果资料中没有答案，就直说「资料里没有提到」，不要编造。\n\n"
        f"{context_text}\n\n问题：{question}"
    )


def ask_llm(client, question, contexts):
    """基于检索资料生成回答（一次性返回）。"""
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "你是一个严谨的文档问答助手，只根据给定资料回答。"},
            {"role": "user", "content": build_qa_prompt(question, contexts)},
        ],
        temperature=0.3,
    )
    return resp.choices[0].message.content


def ask_llm_stream(client, question, contexts):
    """同 ask_llm，但以流式方式逐片段产出（配合 SSE 实现打字机效果）。"""
    stream = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "你是一个严谨的文档问答助手，只根据给定资料回答。"},
            {"role": "user", "content": build_qa_prompt(question, contexts)},
        ],
        temperature=0.3,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def rerank_llm(client, question, candidates, top_k=3):
    """LLM 精排（RankGPT 思路）：按「能否回答问题」对候选段落打分排序。

    embedding 相似度只能衡量整体语义相似性，无法判断段落能否真正回答问题，
    因此粗筛之后需要精排。candidates 为 [(文本, 相似度), ...]，返回重排后的前 top_k 个。
    """
    candidates_text = "\n\n".join(
        f"[{i}] {text}" for i, (text, _) in enumerate(candidates)
    )
    prompt = (
        "下面有若干段资料（编号从 0 开始）和一个用户问题。\n"
        "你的任务：判断每段资料与问题的相关程度，并按相关度从高到低排序。\n"
        "要求：\n"
        "1. 先逐段给出 0-10 的相关度分数和一句话理由；\n"
        "2. 最后一行只输出排序结果：按相关度从高到低排列的编号列表，"
        "用方括号包裹，例如 [2, 0, 5]，除此之外不要输出任何内容。\n\n"
        f"{candidates_text}\n\n问题：{question}"
    )
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "你是一个检索排序助手，按用户要求的格式输出。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    answer = resp.choices[0].message.content
    # 优先解析方括号编号列表（如 [2, 0, 5]），解析失败时退回全文字符提取
    brackets = re.findall(r"\[[\d,\s]+\]", answer)
    if brackets:
        answer = brackets[-1]
    picked = []
    seen = set()
    for num in re.findall(r"\d+", answer):
        i = int(num)
        if i not in seen and 0 <= i < len(candidates):
            seen.add(i)
            picked.append(i)
    if not picked:
        # 模型输出异常时保持原始顺序兜底
        picked = list(range(min(top_k, len(candidates))))
    return [candidates[i] for i in picked[:top_k]]


def save_index(index, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False)


def load_index(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
