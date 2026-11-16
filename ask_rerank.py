# ask_rerank.py —— v4 版提问：向量粗筛 + LLM 精排（rerank）+ 本地嵌入
# 用法：python ask_rerank.py "你的问题"
#
# v2 的问题：向量相似度只能判断"整体像不像"，经常把无关段落排前面（你亲眼见过）。
# v3 的解法是业界标准套路"两段式检索"：
#   ① 粗筛：向量检索先拿回候选 6 段——宁可多拿，不能漏
#   ② 精排：把 6 段编号交给 LLM，按"能不能回答问题"重新排序，取前 3
# v4：问题转向量也换成本地免费模型（和建库用的是同一个模型）
import sys

import chromadb

import rag_lib


def main():
    if len(sys.argv) < 2:
        print('用法：python ask_rerank.py "你的问题"')
        return
    question = sys.argv[1]

    # 打开向量数据库
    db = chromadb.PersistentClient(path="./chroma_db")
    collection = db.get_collection(name="docs")
    print(f"已连接向量数据库，共 {collection.count()} 段")

    try:
        client = rag_lib.get_client()
    except RuntimeError as e:
        print(e)
        return

    # 1. 问题转成向量（v4：本地模型，和建库时用的同一个模型——必须一致才能比较）
    query_vec = rag_lib.embed_texts_local([question])[0]

    # 2. 粗筛：向量检索拿回 6 段候选（比 v2 的 3 段更多）
    result = collection.query(query_embeddings=[query_vec], n_results=6)
    candidates = list(zip(result["documents"][0], result["distances"][0]))
    print("\n—— ① 粗筛：向量检索的 6 段候选（注意看顺序对不对）——")
    for i, (text, dist) in enumerate(candidates):
        print(f"[{i}] 相似度 {1 - dist:.3f}  {text[:60]}...")

    # 3. 精排：交给 LLM 重新排序
    print("\n—— ② 精排：LLM 重排序后的前 3 ——")
    hits = rag_lib.rerank_llm(client, question, [(t, 1 - d) for t, d in candidates], top_k=3)
    for text, score in hits:
        print(f"✓ {text[:60]}...")

    # 4. 生成回答（和 v1/v2 一样）
    print("\n—— AI 回答 ——")
    answer = rag_lib.ask_llm(client, question, hits)
    print(answer)


if __name__ == "__main__":
    main()
