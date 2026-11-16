# webapp.py —— 把 RAG 问答从命令行变成网页版
# 用法：python webapp.py   然后浏览器打开 http://127.0.0.1:8000
#
# 新增的两个"面试关键词"：
#   FastAPI：Python 写 Web 接口的主流框架，启动后访问 /docs 能看到自动生成的接口文档
#   SSE（Server-Sent Events）：服务器向浏览器"流式推送"消息，
#     回答一个字一个字蹦出来，就是 ChatGPT 那种打字机效果。
#     和 WebSocket 的区别：SSE 只能服务器→浏览器单向推，但实现简单，生成式回答够用
import json
import time

import chromadb
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse

import rag_lib

app = FastAPI(title="RAG 智能问答")


def sse(obj):
    """把一条消息包装成 SSE 格式。SSE 的规定：每条消息一行 data: 开头，空行结尾。"""
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


@app.get("/")
def index():
    """首页：把 static/index.html 发给浏览器。"""
    return FileResponse("static/index.html")


@app.post("/ask")
def ask(payload: dict):
    """问答接口。浏览器把问题 POST 过来，我们流式把结果推回去。

    注意这里的顺序设计——为什么"边做边发"而不是"做完再发"？
    因为重排序是十几秒的 LLM 调用，如果做完再发，用户只能盯着空白页干等。
    所以：快的步骤（向量检索）做完立刻把结果推出去，慢的步骤（重排、生成）边跑边推。
    """
    question = (payload.get("question") or "").strip()
    if not question:
        return {"error": "问题不能为空"}

    def event_stream():
        try:
            # ① 向量检索：本地计算，毫秒级完成
            t0 = time.time()
            yield sse({"type": "thinking", "text": "正在向量检索…"})

            db = chromadb.PersistentClient(path="./chroma_db")
            collection = db.get_collection(name="docs")
            query_vec = rag_lib.embed_texts_local([question])[0]
            result = collection.query(query_embeddings=[query_vec], n_results=6)
            candidates = list(zip(result["documents"][0], result["distances"][0]))
            t1 = time.time()
            yield sse({"type": "thinking", "text": f"✓ 向量检索完成，召回 {len(candidates)} 段候选（{t1 - t0:.2f} 秒）"})
            # 候选段落附在消息里，前端放进"展开查看"折叠区，不占版面
            yield sse({"type": "candidates", "candidates": [
                {"text": text, "score": round(1 - dist, 3)} for text, dist in candidates
            ]})

            # ② 重排序：最慢的一步（LLM 调用），边等边告诉用户
            yield sse({"type": "thinking", "text": "正在用 LLM 重排序…"})
            client = rag_lib.get_client()
            hits = rag_lib.rerank_llm(
                client, question, [(t, 1 - d) for t, d in candidates], top_k=3
            )
            t2 = time.time()
            yield sse({"type": "thinking", "text": f"✓ 重排序完成，选出最相关的 {len(hits)} 段（{t2 - t1:.2f} 秒）"})
            yield sse({"type": "chunks", "chunks": [
                {"text": text, "score": round(score, 3)} for text, score in hits
            ]})

            # ③ 流式生成：回答一个字一个字推给页面（打字机效果）
            yield sse({"type": "thinking", "text": "正在生成回答…"})
            for delta in rag_lib.ask_llm_stream(client, question, hits):
                yield sse({"type": "token", "text": delta})
            t3 = time.time()
            yield sse({"type": "done", "elapsed": round(t3 - t0, 1),
                       "gen_elapsed": round(t3 - t2, 1)})
        except Exception as e:
            # 流已经开始发了，出错只能通过消息告诉前端，没法返回普通错误响应
            yield sse({"type": "error", "text": str(e)})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    print("RAG 网页版已启动：请在浏览器打开  http://127.0.0.1:8000")
    print("（按 Ctrl+C 停止服务）")
    uvicorn.run(app, host="127.0.0.1", port=8000)
