# build_index.py —— v1 版建索引：文档 → 切块 → API 嵌入 → 保存为 JSON
# 用法：python build_index.py data/样本文档.md
import sys

import rag_lib


def main():
    if len(sys.argv) < 2:
        print('用法：python build_index.py 文档路径')
        return
    doc_path = sys.argv[1]

    with open(doc_path, encoding="utf-8") as f:
        text = f.read()

    try:
        client = rag_lib.get_client()
    except RuntimeError as e:
        print(e)
        return

    chunks = rag_lib.split_text(text)
    print(f"文档切成了 {len(chunks)} 段（每段最多 500 字，重叠 50 字）")

    print("正在把每段转成向量（调用智谱 embedding 接口，几秒钟）...")
    vectors = rag_lib.embed_texts(client, chunks)

    index = {"doc": doc_path, "chunks": chunks, "vectors": vectors}
    rag_lib.save_index(index, "index.json")
    print("完成 ✓ 索引已保存到 index.json")


if __name__ == "__main__":
    main()
