# rag_lib.py —— RAG 项目的核心逻辑
# 一个 RAG 问答系统分四步：
#   1) 切块：把长文档切成小段（chunk）
#   2) 嵌入：把每段文字变成向量（embedding）
#   3) 检索：把你的问题也变成向量，找出最相似的几段（retrieve）
#   4) 生成：把找出的段落 + 你的问题一起交给 LLM，生成回答（generate）
import json
import os
import re

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # 读取 .env 文件里的 API key 到环境变量

# 智谱的接口是 OpenAI 兼容格式，所以直接用 openai 库，把地址改一下就行
BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
LLM_MODEL = "glm-4-flash"      # 生成回答用的模型（免费）
EMBED_MODEL = "embedding-3"    # v1~v3 用的 API 嵌入模型
LOCAL_EMBED_MODEL = "BAAI/bge-small-zh-v1.5"  # v4 用的本地嵌入模型（免费、离线、512 维）


def get_client():
    """创建 API 客户端。key 从 .env 里读，没配就提示怎么配。"""
    key = os.getenv("ZHIPU_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "还没配置 API key：请把 .env.example 复制一份改名为 .env，\n"
            "然后去 https://open.bigmodel.cn 注册登录 -> 右上角「API 密钥」-> 新建密钥，\n"
            "把 key 粘贴到 .env 文件里。"
        )
    return OpenAI(api_key=key, base_url=BASE_URL)


def split_text(text, chunk_size=500, overlap=50):
    """把长文本切成小段，每段最多 chunk_size 个字，段与段重叠 overlap 个字。
    重叠的作用：避免一句话正好被截断在段的边界上，导致检索时丢上下文。"""
    text = text.strip()
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def embed_texts(client, texts):
    """把一组文字变成向量。
    向量就是一串浮点数（比如 1024 个），语义相近的文字，向量在空间里的方向也相近。"""
    vectors = []
    for text in texts:  # 一条一条调，简单直观；以后文档多了可以改成批量
        resp = client.embeddings.create(model=EMBED_MODEL, input=text)
        vectors.append(resp.data[0].embedding)
    return vectors


_local_embedder = None  # 模型缓存：只加载一次，之后复用（加载要几秒，别每次提问都重载）


def get_local_embedder():
    """拿到本地嵌入模型。第一次调用时才真正加载（这叫"懒加载"），之后直接复用。"""
    global _local_embedder
    if _local_embedder is None:
        # 国内下载模型文件走 hf-mirror 镜像；模型缓存放进项目里的 .model_cache 文件夹
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        os.environ.setdefault("FASTEMBED_CACHE_PATH", "./.model_cache")
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # 关掉 Windows 上的无害警告
        from fastembed import TextEmbedding  # 用到了才 import，程序启动更快
        _local_embedder = TextEmbedding(LOCAL_EMBED_MODEL)
    return _local_embedder


def embed_texts_local(texts):
    """用本地模型把文字转成向量（v4 新增）。
    和 embed_texts（调智谱 API）对比：
      - 免费：不花 API 钱
      - 快：本机计算，不用等网络
      - 离线：断网也能建库和检索
    代价是效果比大模型 embedding 稍差一点点，但对 RAG 检索来说完全够用。
    """
    model = get_local_embedder()
    return [vec.tolist() for vec in model.embed(texts)]


def cosine_similarity(a, b):
    """余弦相似度：衡量两个向量方向有多接近，取值 -1 ~ 1，越接近 1 越相似。
    公式：a·b / (|a|·|b|)"""
    a = np.asarray(a)
    b = np.asarray(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def retrieve(index, query_vec, k=3):
    """在索引里找出和问题最相似的 k 个段落，按相似度从高到低返回。"""
    scored = []
    for i, vec in enumerate(index["vectors"]):
        score = cosine_similarity(query_vec, vec)
        scored.append((score, i))
    scored.sort(reverse=True)
    return [(index["chunks"][i], score) for score, i in scored[:k]]


def build_qa_prompt(question, contexts):
    """把检索到的资料和问题拼成给 LLM 的提示词。
    提示词里明确要求「只根据资料回答」，这是 RAG 减少幻觉（编造）的关键。"""
    context_text = "\n\n".join(f"[资料{i + 1}] {text}" for i, (text, _) in enumerate(contexts))
    return (
        "你是一个文档问答助手。请只根据下面的资料回答用户的问题；"
        "如果资料中没有答案，就直说「资料里没有提到」，不要编造。\n\n"
        f"{context_text}\n\n问题：{question}"
    )


def ask_llm(client, question, contexts):
    """把检索到的资料和问题一起交给 LLM 生成回答（一次性返回全部文字）。"""
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "你是一个严谨的文档问答助手，只根据给定资料回答。"},
            {"role": "user", "content": build_qa_prompt(question, contexts)},
        ],
        temperature=0.3,  # 低一点，回答更稳定、少发挥
    )
    return resp.choices[0].message.content


def ask_llm_stream(client, question, contexts):
    """和 ask_llm 一样，但用"流式"输出：一小段一小段地产生回答（打字机效果）。
    生成式 AI 的一大体验优化：用户不用干等十几秒才看到完整结果，
    而是看到文字一个个蹦出来——网页版（webapp.py）就用它。"""
    stream = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "你是一个严谨的文档问答助手，只根据给定资料回答。"},
            {"role": "user", "content": build_qa_prompt(question, contexts)},
        ],
        temperature=0.3,
        stream=True,  # 关键参数：开启流式，返回值变成一个"片段生成器"
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def rerank_llm(client, question, candidates, top_k=3):
    """重排序（rerank）：让 LLM 从候选段落里挑出真正能回答问题的。
    向量检索只是"粗筛"——它只能判断"整体像不像"，所以你见过社保块排在年假块前面。
    这一步是"精排"——把候选段落全部编号交给 LLM，让它按"能不能回答问题"重新排序。
    参数 candidates 是 [(段落文字, 相似度), ...] 的列表，返回重排后的前 top_k 个。
    """
    # 给每段编上号，让 LLM 只回复编号，好解析
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
        temperature=0,  # 排序任务要稳定，不随机发挥
    )
    answer = resp.choices[0].message.content
    # 先找最后一行方括号里的编号列表（如 [2, 0, 5]）；找不到就从整段回复里提取数字
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
        # 模型输出异常时退回原始顺序，保证系统不会崩
        picked = list(range(min(top_k, len(candidates))))
    return [candidates[i] for i in picked[:top_k]]


def save_index(index, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False)


def load_index(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
