"""两段式检索问答：向量粗筛 top-6 → LLM 精排 top-3 → 生成。"""
import sys

import chromadb

import rag_lib


def main():
    if len(sys.argv) < 2:
        print('用法：python ask_rerank.py "你的问题"')
        return
    question = sys.argv[1]

    db = chromadb.PersistentClient(path="./chroma_db")
    collection = db.get_collection(name="docs")
    print(f"已连接向量数据库，共 {collection.count()} 段")

    try:
        client = rag_lib.get_client()
    except RuntimeError as e:
        print(e)
        return

    # ① 粗筛：向量检索召回 top-6（宁可多拿不能漏）
    query_vec = rag_lib.embed_texts_local([question])[0]
    result = collection.query(query_embeddings=[query_vec], n_results=6)
    candidates = list(zip(result["documents"][0], result["distances"][0]))
    print("\n—— ① 粗筛：向量检索的 6 段候选 ——")
    for i, (text, dist) in enumerate(candidates):
        print(f"[{i}] 相似度 {1 - dist:.3f}  {text[:60]}...")

    # ② 精排：LLM 按「能否回答问题」重排，取 top-3
    print("\n—— ② 精排：LLM 重排序后的前 3 ——")
    hits = rag_lib.rerank_llm(client, question, [(t, 1 - d) for t, d in candidates], top_k=3)
    for text, score in hits:
        print(f"✓ {text[:60]}...")

    print("\n—— AI 回答 ——")
    answer = rag_lib.ask_llm(client, question, hits)
    print(answer)


if __name__ == "__main__":
    main()
