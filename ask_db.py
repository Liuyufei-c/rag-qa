# ask_db.py —— v2 版提问：向量数据库检索
# 用法：python ask_db.py "你的问题"
import sys

import chromadb

import rag_lib


def main():
    if len(sys.argv) < 2:
        print('用法：python ask_db.py "你的问题"')
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

    query_vec = rag_lib.embed_texts_local([question])[0]

    result = collection.query(query_embeddings=[query_vec], n_results=3)
    hits = list(zip(result["documents"][0], result["distances"][0]))

    print("\n—— 检索到的最相关段落（相似度从高到低）——")
    for text, dist in hits:
        print(f"[相似度 {1 - dist:.3f}] {text[:80]}...")

    print("\n—— AI 回答 ——")
    answer = rag_lib.ask_llm(client, question, [(t, 1 - d) for t, d in hits])
    print(answer)


if __name__ == "__main__":
    main()
