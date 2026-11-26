"""
agent_router.py — 多智能体问答 SSE 流式接口（前端 + 评测脚本的入口）

【事件协议】（前端按 type 分发处理）
    start         请求受理（含 query）
    route         Router 分类结果（A/B/C/D/E）
    faq           仅 B 类命中时：FAQ 原文 + 相似度分
    decompose     仅 D 类：子问题列表
    round         每轮"检索+精排"的结果（轮次/query/top1 精排分/是否通过/候选数）
    answer_start  开始生成回答
    chunk         答案文本分片（打字机效果，前端拼接）
    citations     引用列表（回答流完后发，供前端点击溯源）
    refused       拒答（reason + 话术）
    done          结束汇总（refused/rounds/route）
    error         异常（SSE 里不抛异常裸断连接，而是发 error 事件后优雅关闭）

【两个接口的分工】
    POST /agent/chat       SSE 流式 —— 给人用（前端），体验优先
    POST /agent/chat_sync  普通 JSON  —— 给评测脚本用（eval/），简单优先

【当前实现的诚实边界 + 升级路径】
    agent_run() 是原子调用（内部同步编排全部跑完才返回），所以 route/round 等
    阶段事件是" pipeline 结束后从 trace 反推"发出的，answer 是分片流式。
    要做到"每个阶段实时到达前端"，见文件末尾【进阶】注释里的 on_event 改造方案。
"""
import asyncio
import json
from typing import Dict, List

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from common.logger import logger
from services.agent_service import agent_run, LLMError

agent_router = APIRouter(tags=["多智能体问答"], prefix="/agent")


# ============================================================
# 1. 请求体定义（Pydantic 自动做类型校验，脏数据进不了编排层）
# ============================================================

class ChatRequest(BaseModel):
    """
    问答请求体。

    query:   用户原始问题（必填，空串会被 400 拒绝）
    history: 对话历史 [{"role": "user"/"assistant", "content": ...}, ...]，
             多轮指代消解的关键输入（Query Rewrite 消费它）
    """
    query: str
    history: List[Dict] = []


# ============================================================
# 2. SSE 工具函数
# ============================================================

def sse_pack(event: str, data: Dict) -> str:
    """
    把一条事件打包成 SSE 协议文本。
        event: 事件名（对应协议里的 type，如 "chunk"）
        data:  事件数据（dict，会被序列化成 JSON）

    returns -> str: 形如 "event: chunk\\ndata: {...}\\n\\n" 的完整事件文本

    三个细节：
    1. data 用 json.dumps 且 ensure_ascii=False——中文直接以 UTF-8 传输，
       否则前端收到的是 \\uXXXX 转义串，还得自己解一层
    2. 结尾必须 "\\n\\n"（一个换行结束 data 行，一个空行结束整个事件）
    3. data 内部不能有裸换行——JSON 序列化已保证（\\n 会被转义成 "\\\\n"）
    """
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def stream_text_chunks(text: str, chunk_size: int = 8, delay: float = 0.02):
    """
    把完整文本切成小片，逐片 yield（打字机效果）。

        text:       完整文本（回答或拒答话术）
        chunk_size: 每片字符数。8 是体验甜点：再小前端渲染抖动，再大不像流式
        delay:      片间间隔（秒）。两个作用：打字机节奏 + await 让出事件循环

    yields ->str: 文本分片
    """
    for i in range(0, len(text), chunk_size):
        yield text[i:i + chunk_size]
        await asyncio.sleep(delay)


# ============================================================
# 3. SSE 主接口（前端消费）
# ============================================================

@agent_router.post("/chat")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    """
    多智能体问答 —— SSE 流式接口。

    流程：受理 → 跑完整 pipeline（agent_run）→ 按协议依次推事件：
          start → route → [faq/decompose] → round × N → answer_start
          → chunk × N → citations → done
          拒答时：start → route → round × N → refused → done
        req: ChatRequest（query + history）

    returns -> StreamingResponse: text/event-stream 流式响应

    raises -> HTTPException 400: query 为空
    （流式响应已经开始就不能再抛 4xx/5xx 了，所以参数校验必须放在建流之前）
    """
    # ---- 参数校验：必须在进入流式之前 ----
    # 一旦返回 StreamingResponse，HTTP 状态码就固定为 200 了，
    # 之后任何错误只能通过 error 事件告知前端（无法再返回 400/500）
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="query 不能为空")

    async def event_generator():
        """事件生成器：FastAPI 逐段消费，yield 一段就立刻写给前端"""
        # ---- 1. 受理事件 ----
        yield sse_pack("start", {"query": req.query})

        try:
            # ---- 2. 执行完整 pipeline ----
            # agent_run 内部已把同步编排丢进线程池（asyncio.to_thread），
            # 这里直接 await 不会阻塞事件循环，其他用户的请求照常处理
            result = await agent_run(req.query, req.history)
            trace = result.get("trace", {})

            # ---- 3. 路由事件：让前端立刻看到"这个问题被怎么处理" ----
            # （Adaptive 系统的可观察性：用户能感知到系统"想清楚了"再答）
            yield sse_pack("route", {"route": result["route"]})

            # ---- 3b. B 类 FAQ 命中事件（仅命中时发）----
            faq_hit = trace.get("faq_hit")
            if faq_hit:
                yield sse_pack("faq", {"question": faq_hit["question"],
                                       "score": faq_hit["score"]})

            # ---- 3c. D 类拆解事件（仅 D 类发）----
            if trace.get("sub_questions"):
                yield sse_pack("decompose", {"sub_questions": trace["sub_questions"]})

            # ---- 4. 每轮检索事件（从 trace 反推，见文件头【诚实边界】）----
            # 前端可渲染"第1轮检索中… 精排分0.82 ✓ 通过"的进度条——
            # 重试过程对用户可见，是可信度的重要组成部分（用户知道系统在努力）
            for i, rnd in enumerate(trace.get("rounds", [])):
                yield sse_pack("round", {
                    "round": i + 1,
                    "query": rnd.get("query"),
                    "top1_score": rnd.get("top1_score"),
                    "passed": rnd.get("passed"),
                    "candidate_count": rnd.get("candidate_count"),
                    "degraded": rnd.get("degraded", False),
                })

            # ---- 5. 拒答 or 回答 ----
            if result["refused"]:
                # 拒答：话术短，整体发一个 refused 事件（无需分片）
                # reason 同时进 trace，供 debug_router/评测统计拒答率
                yield sse_pack("refused", {
                    "reason": result["refuse_reason"],
                    "answer": result["answer"],
                })
            else:
                # 回答：先发 answer_start（前端可切"思考中"→"输出中"的 UI 状态），
                # 再分片推答案，最后发 citations（引用溯源）
                yield sse_pack("answer_start", {})
                async for chunk in stream_text_chunks(result["answer"]):
                    yield sse_pack("chunk", {"text": chunk})
                yield sse_pack("citations", {
                    "citations": result["citations"],
                    "used_citations": result["used_citations"],
                })

            # ---- 6. 结束汇总 ----
            yield sse_pack("done", {
                "refused": result["refused"],
                "rounds": result["rounds"],
                "route": result["route"],
            })

        except Exception as e:
            # ---- 7. 异常兜底 ----
            # SSE 连接里直接 raise = 前端只看到"连接莫名断开"，没有任何上下文
            # 正确做法：发一个 error 事件把错误原因告诉前端，再正常结束生成器
            # （HTTPException 之外的所有异常都兜住——流式接口的健壮性底线）
            logger.error(f"SSE问答流程异常: {repr(e)}", exc_info=True)

            # ---- 错误分类提示（技术细节只进日志，用户文案说人话）----
            # 限流类：DashScope 账户配额/并发超限。这不是代码 bug，
            # 提示里给出可执行动作（稍后重试/检查额度），而不是甩错误码
            # 其他类：一句话带过 + 建议重试。repr(e) 里的 code/message/链接
            # 对用户是噪音，全部留在日志里供排查
            if isinstance(e, LLMError) and e.retryable:
                msg = ("AI 服务当前繁忙（账户调用量已达上限），请稍等片刻再试。"
                       "如持续出现，请检查 DashScope 账户额度或明日自动恢复。")
            else:
                msg = "服务暂时开小差，请稍后重试；若持续出现请联系管理员。"
            yield sse_pack("error", {"message": msg})

    # ---- 流式响应头 ----
    # X-Accel-Buffering: no → 告诉 Nginx"别缓冲"，否则所有事件被攒成一次性发送
    # Cache-Control: no-cache → 防止浏览器/代理缓存事件流
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================
# 4. 同步接口（评测脚本消费，不走 SSE）
# ============================================================

@agent_router.post("/chat_sync")
async def chat_sync(req: ChatRequest) -> Dict:
    """
    多智能体问答 —— 同步 JSON 接口（给 eval/ 评测脚本用）。

    req: ChatRequest（query + history）
    returns -> Dict: agent_run() 的完整返回（含 trace，评测指标的数据源）

    raises -> HTTPException 400: query 为空
    """
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="query 不能为空")
    try:
        result = await agent_run(req.query, req.history)
        return result
    except LLMError as e:
        # 限流/LLM 故障：500 + 分类文案（评测脚本会记录 detail，便于归因）
        logger.error(f"同步问答接口LLM异常: {repr(e)}", exc_info=True)
        detail = ("AI 服务当前繁忙（账户调用量已达上限），请稍后重试"
                  if e.retryable else f"问答失败: {e}")
        raise HTTPException(status_code=500, detail=detail)
    except Exception as e:
        logger.error(f"同步问答接口异常: {repr(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"问答失败: {repr(e)}")
