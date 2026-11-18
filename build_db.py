"""建库：文档 → 切块 → 本地嵌入 → 存入 chromadb。支持多文档与全量重建。"""
import os
import sys

import chromadb

import rag_lib


def main():
    if len(sys.argv) < 2:
        print('用法：python build_db.py data/文档1.md data/文档2.md ...')
        return

    # 全量重建：先清空旧集合，避免旧文档的块残留
    db = chromadb.PersistentClient(path="./chroma_db")
    if "docs" in [c.name for c in db.list_collections()]:
        db.delete_collection("docs")
        print("检测到旧数据库，已清空重建")
    collection = db.get_or_create_collection(
        name="docs",
        metadata={"hnsw:space": "cosine"},
    )

    total = 0
    for doc_path in sys.argv[1:]:
        with open(doc_path, encoding="utf-8") as f:
            text = f.read()
        doc_name = os.path.basename(doc_path)

        chunks = rag_lib.split_text(text)
        print(f"[{doc_name}] 切成了 {len(chunks)} 段")

        vectors = rag_lib.embed_texts_local(chunks)

        collection.add(
            ids=[f"{doc_name}-{i}" for i in range(len(chunks))],
            embeddings=vectors,
            documents=chunks,
        )
        total += len(chunks)

    print(f"完成 ✓ 共 {total} 段已存入向量数据库 chroma_db/")
    print('提示：提问用  python ask_rerank.py "你的问题"')


if __name__ == "__main__":
    main()
