"""RAG 网页版：FastAPI + SSE 流式问答服务。"""
import json
import time

import chromadb
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse

import rag_lib

app = FastAPI(title="RAG 智能问答")


def sse(obj):
    """把一条消息包装成 SSE 格式（data: {...} 后接空行）。"""
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


@app.get("/")
def index():
    """首页。"""
    return FileResponse("static/index.html")


@app.post("/ask")
def ask(payload: dict):
    """问答接口：按「边做边发」顺序推送状态、粗筛候选、精排结果与流式回答。

    重排序是耗时较长的 LLM 调用，若等全部完成再返回，前端将长时间无响应；
    因此本地可完成的步骤（状态提示、向量粗筛）先推，慢步骤边跑边推。
    """
    question = (payload.get("question") or "").strip()
    if not question:
        return {"error": "问题不能为空"}

    def event_stream():
        try:
            t0 = time.time()
            yield sse({"type": "thinking", "text": "正在向量检索…"})

            db = chromadb.PersistentClient(path="./chroma_db")
            collection = db.get_collection(name="docs")
            query_vec = rag_lib.embed_texts_local([question])[0]
            result = collection.query(query_embeddings=[query_vec], n_results=6)
            candidates = list(zip(result["documents"][0], result["distances"][0]))
            t1 = time.time()
            yield sse({"type": "thinking",
                       "text": f"✓ 向量检索完成，召回 {len(candidates)} 段候选（{t1 - t0:.2f} 秒）"})
            yield sse({"type": "candidates", "candidates": [
                {"text": text, "score": round(1 - dist, 3)} for text, dist in candidates
            ]})

            yield sse({"type": "thinking", "text": "正在用 LLM 重排序…"})
            client = rag_lib.get_client()
            hits = rag_lib.rerank_llm(
                client, question, [(t, 1 - d) for t, d in candidates], top_k=3
            )
            t2 = time.time()
            yield sse({"type": "thinking",
                       "text": f"✓ 重排序完成，选出最相关的 {len(hits)} 段（{t2 - t1:.2f} 秒）"})
            yield sse({"type": "chunks", "chunks": [
                {"text": text, "score": round(score, 3)} for text, score in hits
            ]})

            yield sse({"type": "thinking", "text": "正在生成回答…"})
            for delta in rag_lib.ask_llm_stream(client, question, hits):
                yield sse({"type": "token", "text": delta})
            t3 = time.time()
            yield sse({"type": "done", "elapsed": round(t3 - t0, 1),
                       "gen_elapsed": round(t3 - t2, 1)})
        except Exception as e:
            yield sse({"type": "error", "text": str(e)})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    print("RAG 网页版已启动：请在浏览器打开  http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
