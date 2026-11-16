# ask.py —— 提问：问题 → 向量 → 检索最相关段落 → LLM 生成回答
# 用法：python ask.py "你的问题"
import sys

import rag_lib


def main():
    if len(sys.argv) < 2:
        print('用法：python ask.py "你的问题"')
        return
    question = sys.argv[1]

    try:
        index = rag_lib.load_index("index.json")
    except FileNotFoundError:
        print("还没建索引：请先运行  python build_index.py data/样本文档.md")
        return
    print(f"已加载索引，共 {len(index['chunks'])} 段")

    try:
        client = rag_lib.get_client()
    except RuntimeError as e:
        print(e)
        return

    # 1. 把问题也变成向量
    query_vec = rag_lib.embed_texts(client, [question])[0]

    # 2. 检索最相似的 3 段
    hits = rag_lib.retrieve(index, query_vec, k=3)
    print("\n—— 检索到的最相关段落（相似度从高到低）——")
    for text, score in hits:
        print(f"[相似度 {score:.3f}] {text[:80]}...")

    # 3. 交给 LLM 生成回答
    print("\n—— AI 回答 ——")
    answer = rag_lib.ask_llm(client, question, hits)
    print(answer)


if __name__ == "__main__":
    main()
