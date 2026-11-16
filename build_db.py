# build_db.py —— 把文档存进向量数据库（v4 版建库，支持多个文档，嵌入完全本地化）
# 用法：python build_db.py data/文档1.md data/文档2.md ...
#
# 版本演化：
#   v1：向量存进 index.json（普通文件），检索时一条一条比（线性扫描）
#   v2：向量存进 chromadb（向量数据库），内部自动建 HNSW 索引，毫秒级检索
#   v4：嵌入从"调智谱 API"换成"本地免费模型"——建库不再需要 API key、不花钱、离线可用
import os
import sys

import chromadb

import rag_lib


def main():
    if len(sys.argv) < 2:
        print('用法：python build_db.py data/文档1.md data/文档2.md ...')
        return

    # 打开数据库。如果里面已有旧数据，先清空重来（避免旧文档的块残留混进来）
    db = chromadb.PersistentClient(path="./chroma_db")  # 数据会存进 chroma_db 文件夹
    existing = [c.name for c in db.list_collections()]
    if "docs" in existing:
        db.delete_collection("docs")
        print("检测到旧数据库，已清空重建")
    collection = db.get_or_create_collection(
        name="docs",
        metadata={"hnsw:space": "cosine"},  # 用余弦距离来比较向量像不像
    )

    total = 0
    for doc_path in sys.argv[1:]:
        # 读文档
        with open(doc_path, encoding="utf-8") as f:
            text = f.read()
        doc_name = os.path.basename(doc_path)

        # 1. 切块（和 v1 完全一样）
        chunks = rag_lib.split_text(text)
        print(f"[{doc_name}] 切成了 {len(chunks)} 段")

        # 2. 每段转成向量（v4：用本地模型，免费且快；第一次运行会先下载模型）
        vectors = rag_lib.embed_texts_local(chunks)

        # 3. 存进向量数据库。编号带上文档名，这样以后能知道每段来自哪个文件
        collection.add(
            ids=[f"{doc_name}-{i}" for i in range(len(chunks))],  # 每段一个编号
            embeddings=vectors,   # 向量
            documents=chunks,     # 原文
        )
        total += len(chunks)

    print(f"完成 ✓ 共 {total} 段已存入向量数据库 chroma_db/")
    print('提示：提问用  python ask_rerank.py "你的问题"')


if __name__ == "__main__":
    main()
