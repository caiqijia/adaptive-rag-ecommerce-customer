r"""
debug_router.py — 调试与评测接口（召回详情可视化 + FAQ 管理 + 索引健康度）

【这个文件在项目里的角色】
    线上接口（agent_router）追求"快和稳"，调试接口追求"看得见"。
    没有 debug 层的 RAG 系统，出问题只能靠猜——这是 Demo 和落地系统的
    又一个分水岭。三个区块对应三种日常：

    1. GET  /debug/retrieve   召回详情：给定 query，把"双路召回 → RRF → rerank"
                               全过程的中间结果原样返回，不生成回答。
                               → 评测 Step 0/6 诊断"为什么这个问题检索不到"的第一现场
    2. FAQ 管理 CRUD           B 类知识库维护（增/查/删/重载缓存）
    3. GET  /debug/status      索引健康度：Chroma 与 BM25 两路数量核对
                               → 抓"两路数据漂移"这种不报错但协同失效的隐蔽故障

【求职考点】为什么调试接口必须"不走生成"：
    RAG 的故障分两层——检索层（召回不对）和生成层（证据够但说错话）。
    混在一起查，定位成本是整链路；拆开后，检索问题用 /debug/retrieve
    单独复现，一眼能看到是哪一路召回丢的、RRF 融合名次对不对。
    分层定位，是工程能力最直接的体现。
"""

import asyncio
import time
from collections import Counter
from typing import Dict, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config import RRF_K, RERANK_TOP_K, collection
from common.logger import logger
from common.response import success_resp
from models.faq import FAQ, FAQResp

# 复用 agent_service 已暴露的检索组件——调试接口与线上链路走同一份代码，
# 保证"调试时看到的"就是"线上发生的"（两套实现 = 调试通过上线照错）
from services.agent_service import (
    vector_recall,
    bm25_recall,
    faq_matcher,
)
from services.bm25_service import bm25_service
from services.rrf_service import to_ranked_list, rrf_fuse
from services.rerank_service import rerank, reranker_status, reset_reranker

debug_router = APIRouter(tags=["调试与评测"], prefix="/debug")


# ============================================================
# 0. 内部工具：FAQ 缓存重载
# ============================================================

async def _reload_faq_cache() -> int:
    """
    FAQ 写操作后立即重载内存缓存（强一致策略）。

    FAQMatcher 把 FAQ 问题向量常驻内存（preload），数据库增删后缓存不刷新
    会导致"库里改了、线上不生效"。这里重置其加载标志后重新预热。

    Returns:
        int: 重载后的缓存条数

    【求职考点】缓存失效策略的选择：
    - 写后立即重载（本接口）：强一致，FAQ 是低频写场景，成本可接受
    - TTL 过期：最终一致，适合高频写/读多写少的大规模缓存
    - 生产完善方向：给 FAQMatcher 加公开 reload() 方法 + 版本号，
      避免调试接口直接碰私有属性（_loaded/_cache）
    """
    faq_matcher._loaded = False
    faq_matcher._cache = []
    await faq_matcher.preload()
    return len(faq_matcher._cache)


# ============================================================
# 1. GET /debug/status — 索引健康度
# ============================================================

@debug_router.get("/status")
async def debug_status():
    """
    两路索引一致性体检。

    Returns:
        data: {
            "chroma_total": int,   # 向量库 chunk 总数
            "bm25_total": int,     # 关键词库 chunk 总数
            "in_sync": bool,       # 两路是否完全一致
            "diff_by_doc": {       # 不一致时按 doc_id 列出差异（chunk数对比）
                "doc_id": [chroma_count, bm25_count], ...
            },
            "faq_cache": {"loaded": bool, "count": int}
        }

    【求职考点】为什么需要这个接口：
    混合检索的两路（Chroma + BM25）是两份独立存储，靠 chunk_id 对齐。
    上传漏同步 BM25、删除只删一边、服务重启没重建——都会造成"漂移"：
    不报错、不 500，只是混合检索静默退化成单路。diff_by_doc 直接把
    漂移定位到具体文档，不用翻日志猜。
    """
    def _collect():
        """同步采集（Chroma 本地读，放线程池避免阻塞事件循环）"""
        snapshot = collection.get(include=["metadatas"])
        ids = snapshot.get("ids") or []
        metas = snapshot.get("metadatas") or []
        chroma_doc: Counter = Counter()
        for cid, meta in zip(ids, metas):
            meta = meta or {}
            # 优先用 metadata 里的 doc_id；老数据没有时从 chunk_id 规则反推
            doc_id = meta.get("doc_id", str(cid).rsplit("_", 1)[0])
            chroma_doc[str(doc_id)] += 1
        bm25_doc: Counter = Counter(
            str(m.get("doc_id")) for m in bm25_service.meta
        )
        return chroma_doc, bm25_doc

    chroma_doc, bm25_doc = await asyncio.to_thread(_collect)

    # 按 doc_id 粒度 diff：不只报"总量不等"，还告诉你是哪份文档缺了几个 chunk
    diff = {
        k: [chroma_doc.get(k, 0), bm25_doc.get(k, 0)]
        for k in set(chroma_doc) | set(bm25_doc)
        if chroma_doc.get(k, 0) != bm25_doc.get(k, 0)
    }
    data = {
        "chroma_total": sum(chroma_doc.values()),
        "bm25_total": sum(bm25_doc.values()),
        "in_sync": not diff,
        "diff_by_doc": diff,
        "faq_cache": {"loaded": faq_matcher._loaded, "count": len(faq_matcher._cache)},
        # 精排模型健康状况：circuit_open=True 说明加载失败已熔断，
        # 此时 rerank 静默降级为 RRF 顺序——**低置信自省失效，拒答第一道防线没了**。
        # 这类"降级但不报错"的状态必须能在体检里看到，否则精度悄悄掉了都没人知道。
        "rerank": reranker_status(),
    }
    if diff:
        # 漂移是隐患不是故障：记录 WARNING 级日志，服务照常运行
        logger.warning(f"检测到两路索引漂移: {diff}")
    return success_resp(data=data)


# ============================================================
# 2. GET /debug/retrieve — 召回全链路详情（评测 Step 0 的核心工具）
# ============================================================

@debug_router.get("/retrieve")
async def debug_retrieve(query: str, top_k: int = RERANK_TOP_K):
    """
    给定 query，原样返回"双路召回 → RRF 融合 → rerank 精排"的全部中间结果。

    Args:
        query:  检索用 query（此处不做 Query Rewrite——调试就是查原始 query 的表现）
        top_k:  精排保留条数（默认 config.RERANK_TOP_K=3）

    Returns:
        data: {
            "query": str,
            "timing": {"total_ms", "vector_ms", "bm25_ms", "rerank_ms"},  # 分段耗时
            "vector_recall": [...],   # 向量通路原始召回（含 similarity）
            "bm25_recall": [...],     # 关键词通路原始召回（含 bm25_score）
            "fused": [...],           # RRF 融合结果（含 hit_detail 两路名次）
            "rerank": {...}           # 精排结果 + passed/top1_score 自省结论
        }

    【求职考点】三个设计点：
    1. 与线上同一份代码：vector_recall/bm25_recall/rrf_fuse/rerank 全部复用
       agent_service 的组件——"调试所见即线上所为"，不会出现调试通过上线照错
    2. 分段耗时埋点：优化必须有度量。哪一路慢、rerank 占多少，
       这个接口的数字就是依据（没有度量就没有优化）
    3. 同步阻塞调用一律 asyncio.to_thread：
       embedding 是网络 IO、rerank 是本地模型推理（CPU/GPU 密集），
       直接在 async 函数里 await 调用会卡死事件循环——其他用户的请求
       全部排队，这是 FastAPI 最常见的性能事故
    """
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="query 不能为空")
    query = query.strip()
    t0 = time.perf_counter()

    # ---- 向量通路（网络 IO → 线程池）----
    vec = await asyncio.to_thread(vector_recall, query)
    t1 = time.perf_counter()

    # ---- 关键词通路（纯内存计算，但 jieba 分词也有开销，同样丢线程池）----
    bm = await asyncio.to_thread(bm25_recall, query)
    t2 = time.perf_counter()

    # ---- RRF 融合（纯 Python 字典运算，微秒级，直接算）----
    fused = rrf_fuse(
        to_ranked_list(vec, score_key="similarity"),
        to_ranked_list(bm, score_key="bm25_score"),
        k=RRF_K,
    )

    # ---- rerank 精排（模型推理 → 线程池）----
    rr = await asyncio.to_thread(rerank, query, fused, top_k=top_k)
    t3 = time.perf_counter()

    logger.info(f"debug召回: query={query} 向量{len(vec)}条 BM25{len(bm)}条 "
                f"融合{len(fused)}条 精排top1={rr.get('top1_score')} "
                f"总耗时{(t3 - t0) * 1000:.0f}ms")
    return success_resp(data={
        "query": query,
        "timing": {
            "total_ms": round((t3 - t0) * 1000, 1),
            "vector_ms": round((t1 - t0) * 1000, 1),
            "bm25_ms": round((t2 - t1) * 1000, 1),
            "rerank_ms": round((t3 - t2) * 1000, 1),
        },
        "vector_recall": vec,
        "bm25_recall": bm,
        "fused": fused,
        "rerank": rr,
    })


# ============================================================
# 3. FAQ 管理（B 类知识库维护）
# ============================================================

class FAQCreate(BaseModel):
    """FAQ 新增请求体"""
    question: str
    answer: str


@debug_router.post("/faq")
async def create_faq(req: FAQCreate):
    """
    新增 FAQ 标准问答（B 类路由的知识来源）。

    查重：完全相同的问题不允许重复入库——否则 FAQMatcher 匹配到哪条全看
    余弦计算的浮点运气，标准答案不唯一就失去"标准"的意义。

    写后立即重载缓存（见 _reload_faq_cache）。
    """
    question = req.question.strip()
    answer = req.answer.strip()
    if not question or not answer:
        raise HTTPException(status_code=400, detail="question 和 answer 均不能为空")
    if await FAQ.get_or_none(question=question):
        raise HTTPException(status_code=400, detail="相同问题已存在，请先删除旧条目")
    faq = await FAQ.create(question=question, answer=answer)
    count = await _reload_faq_cache()
    logger.info(f"FAQ新增 id={faq.id}，缓存重载后共{count}条")
    return success_resp(data=FAQResp.model_validate(faq))


@debug_router.get("/faq/list")
async def faq_list():
    """FAQ 全量列表（调试/审核用）"""
    faqs = await FAQ.all()
    return success_resp(data=[FAQResp.model_validate(f) for f in faqs])


@debug_router.delete("/faq/{faq_id}")
async def delete_faq(faq_id: int):
    """删除 FAQ（删后立即重载缓存，保证线上不再命中已删条目）"""
    faq = await FAQ.get_or_none(id=faq_id)
    if not faq:
        raise HTTPException(status_code=404, detail="FAQ不存在")
    await faq.delete()
    count = await _reload_faq_cache()
    logger.info(f"FAQ删除 id={faq_id}，缓存重载后共{count}条")
    return success_resp(msg="FAQ已删除")


@debug_router.post("/faq/reload")
async def reload_faq():
    """
    手动重载 FAQ 缓存（应急用）。

    正常路径下增删接口已自动重载；此接口用于"缓存疑似不同步"时
    的人工修复——重载一次，让内存与数据库强制对齐。
    """
    count = await _reload_faq_cache()
    return success_resp(data={"count": count}, msg=f"FAQ缓存已重载，当前{count}条")


# ============================================================
# 4. POST /debug/rerank/reset — 重置精排模型熔断器
# ============================================================

@debug_router.post("/rerank/reset")
async def reset_rerank_breaker():
    """
    重置 rerank 模型熔断器，让下次检索重新尝试加载模型。

    【使用场景】
    模型首次下载失败（断网 / 镜像不通 / 磁盘满）后，rerank_service 会熔断——
    后续请求不再尝试加载，直接降级为 RRF 顺序，保证服务可用。
    修好网络或下载完模型后，调本接口即可恢复精排能力，**不必重启服务**。

    Returns:
        data: {"rerank": {...}}  重置后的模型状态

    【求职考点】为什么要给熔断器配一个复位入口：
    熔断器的价值是"快速失败保护"，副作用是"修复后无法自愈"。
    生产上的标准做法是提供手动复位（本接口）+ 定时半开探测（未来的改进方向：
    每隔 N 分钟允许一次试探请求，成功则自动闭合熔断）。
    只有熔断没有复位，等于把一个可恢复故障变成必须重启的运维事故。
    """
    reset_reranker()
    return success_resp(
        data={"rerank": reranker_status()},
        msg="rerank 熔断器已重置，下次检索将重新尝试加载模型",
    )
