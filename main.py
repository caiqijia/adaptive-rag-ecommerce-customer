from contextlib import asynccontextmanager
import os
import threading

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from tortoise.contrib.fastapi import register_tortoise

from config import TORTOISE_ORM, collection, RERANK_MODEL_NAME
from routers.doc_router import doc_router
from routers.agent_router import agent_router
from routers.debug_router import debug_router
from services.bm25_service import bm25_service
from common.logger import logger


def _warmup_rerank() -> None:
    """
    后台预热精排模型（不阻塞服务启动）。

    【为什么需要预热】
    rerank 是懒加载的：第一个请求到达时才加载模型。本地 2.2GB 权重在 CPU 上
    ~10-20s，这段时间用户看到的是"思考中"卡住——功能正常，但体验像坏了。
    启动时后台加载，首个请求直接命中已就绪的单例，延迟归零。

    【为什么不阻塞 lifespan】
    加载耗时不可控（磁盘慢/杀毒软件扫描会更久），阻塞会让服务迟迟不可访问。
    另起守护线程：加载期间服务照常响应，rerank 未就绪时走降级路径。

    【为什么只在"本地目录存在"时预热】
    若配置的是 HF 仓库名（模型未下载），预热会触发一次 2.2GB 下载——
    这不该在用户不知情时发生在启动阶段。那种情况下保持懒加载：
    首个请求触发下载，失败则熔断降级（见 rerank_service）。
    """
    if not os.path.isdir(RERANK_MODEL_NAME):
        logger.info(
            f"rerank 模型非本地目录（{RERANK_MODEL_NAME}），跳过预热，"
            f"首次使用时会按需加载/下载"
        )
        return

    def _load() -> None:
        # 延迟 import：避免模块加载期就拉起 FlagEmbedding（拖慢启动）
        from services.rerank_service import get_reranker
        logger.info("后台预热 rerank 模型（不阻塞服务启动）...")
        if get_reranker() is not None:
            logger.info("rerank 模型预热完成，首请求无需等待加载")
        # 失败时 get_reranker 内部已记录错误并熔断，这里无需再处理

    threading.Thread(target=_load, name="rerank-warmup", daemon=True).start()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    服务启动/关闭钩子（FastAPI 0.100+ 推荐用 lifespan 替代已废弃的 on_event）。

    内存索引的"持久化恢复"是落地系统与 Demo 的分水岭之一：
    Demo 假设"服务不重启、数据不变化"；落地必须回答"重启后状态从哪来"。
    """
    logger.info("服务启动：开始从 Chroma 重建 BM25 内存索引...")
    try:
        # collection.get 不带 where = 拿全量 chunk（含 documents + metadatas）。
        # Chroma 的 id 就是 f"{doc_id}_{i}"，直接当 chunk_id 回填，
        # 与上传时 add_chunk 写入的 meta 完全一致 → 重建后两路主键天然对齐
        snapshot = collection.get(include=["documents", "metadatas"])
        count = 0
        for cid, text, meta in zip(
            snapshot["ids"], snapshot["documents"], snapshot["metadatas"]
        ):
            meta = meta or {}   # 无 metadata 的 chunk 兜底空 dict
            bm25_service.add_chunk(text, {
                "doc_id": meta.get("doc_id"),
                "chunk_id": cid,
            })
            count += 1
        logger.info(f"BM25索引重建完成，共加载 {count} 个chunk")
    except Exception as e:
        # 重建失败不阻塞启动：BM25 退化为空（双路→向量单路），
        # 之后上传新文档会走 add_chunk 自动补回。
        # 可用 GET /debug/status 核对两路数量是否一致（in_sync 字段）
        logger.error(f"BM25索引重建失败（服务继续启动，检索将降级为向量单路）: {repr(e)}")

    # 精排模型后台预热（失败不影响启动，降级路径已由 rerank_service 兜住）
    _warmup_rerank()

    yield   # ← yield 之前=启动阶段，之后=关闭阶段

    logger.info("服务关闭")


app = FastAPI(title="Adaptive-RAG电商客服系统", lifespan=lifespan)

# 允许独立部署的静态页面（如项目展示站）跨域调用问答接口。
# allow_credentials=False：鉴权信息不随请求发送，配合 "*" 通配符即可。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    # Chrome 私有网络访问（PNA）：公网 https 页面请求本机 127.0.0.1 时的预检放行
    allow_private_network=True,
)

# 注册Tortoise ORM
register_tortoise(
    app,
    config=TORTOISE_ORM,
    generate_schemas=True,
    add_exception_handlers=True
)

# 挂载路由
app.include_router(doc_router)
app.include_router(agent_router)
app.include_router(debug_router)

@app.get("/")
async def index_page():
    """
    前端演示页入口：http://127.0.0.1:8000/
    """
    return FileResponse("index.html")

@app.get("/health")
async def health_check():
    logger.info("健康检查")
    return {"status":"ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run('main:app', host="127.0.0.1", port=8000, reload=True)
