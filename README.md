# RAG 文档问答（寒假实习项目）

## 这是什么

RAG = Retrieval-Augmented Generation（检索增强生成）。解决的核心问题：让 LLM 回答"你私有文档里的问题"——因为大模型没训练过你的文档，直接把文档全文塞给它又太长太贵。RAG 的思路是：先**检索**出文档里和问题最相关的几段，再让 LLM **基于这几段生成**回答。

四步流程，对应代码：

| 步骤 | 干什么 | 代码位置 |
|---|---|---|
| 1 切块 | 把长文档切成小段 | `split_text()` in rag_lib.py |
| 2 嵌入 | 每段文字转成向量（语义相近 → 向量相近） | `embed_texts()` |
| 3 检索 | 问题也转成向量，找最相似的 k 段 | `retrieve()` + `cosine_similarity()` |
| 4 生成 | 检索结果 + 问题一起交给 LLM | `ask_llm()` |

## 怎么跑

1. 依赖已装好（.venv 虚拟环境里：openai、numpy、python-dotenv）
2. 拿智谱 API key：open.bigmodel.cn 注册登录（需实名）→ 「API 密钥」→ 新建
3. 把 `.env.example` 复制一份改名为 `.env`，key 填进去
4. 建索引：`.venv/Scripts/python build_index.py data/样本文档.md`
5. 提问：`.venv/Scripts/python ask.py "试用期最长是多久？"`

## 文件结构

```
rag-qa/
  .venv/           虚拟环境（装好的依赖在这里面）
  data/            放你的文档
  rag_lib.py       核心逻辑：切块/嵌入/检索/生成
  build_index.py   建索引（离线跑一次）
  ask.py           提问（每次问答都跑这个）
  .env             你的 API key（不要发给别人）
  index.json       建好的索引（程序自动生成）
```

## 常见报错

- "还没配置 API key" → 看第 3 步，.env 建好了没、key 填了没
- "还没建索引" → 先运行 build_index.py
- 换自己的文档：把文件丢进 data/ 文件夹，然后 `build_index.py data/你的文件.md`（先支持纯文本 .md/.txt）

## 升级计划（后面几周逐个加，每项都是面试谈资）

1. 本地 embedding 模型（bge-small-zh）+ chromadb 向量库——不依赖 API、理解向量库原理
2. PDF / Word 文档解析
3. 混合检索（BM25 关键词 + 向量）+ rerank 重排
4. SSE 流式输出 + FastAPI 做成 Web 接口
5. RAGAS 评估检索和回答质量
