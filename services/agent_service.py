"""
agent_service.py — Adaptive-RAG 多智能体编排层

【完整链路】
    用户原始query
        ↓
    agent_run() 入口（async：预加载FAQ缓存 + 丢线程池跑同步编排）
        ↓
    Router Agent 五分类：A寒暄 / B FAQ / C单跳 / D复合多跳 / E越界
        ↓
    C/D 进入重试循环：
        第一轮：原始query
        ├─ 向量召回（embedding_service + Chroma）
        ├─ BM25关键词召回（bm25_service）
        └─ rrf_service.rrf_fuse() 两路召回融合
        ↓
        rerank_service.rerank() 精排 + 置信自省
        ↓
        if passed=True  → 生成Agent产出带引用的客服回答 → Critic溯源校验
        elif passed=False 且 未到最大重试次数 → QueryRewriteAgent改写query，进入下一轮检索
        else → Critic判定证据不足，拒答，不编造答案，引导人工客服

【文件结构】（区块编号即阅读顺序）
    1. LLM 封装      call_llm / extract_json —— 所有 Agent 的唯一 LLM 出口
    2. Router Agent  route_query —— 五分类，决定算力分配
    3. Rewrite Agent rewrite_query —— 重试轮的查询改写
    4. Decompose     decompose_query —— D 类复合问题拆解
    5. FAQ Matcher   FAQMatcher —— B 类语义直出
    6. 检索编排       vector_recall / bm25_recall / retrieve_round
    7. 生成 Agent    generate_answer —— 带引用回答
    8. Critic Agent  verify_answer —— 溯源校验守门
    9. 主编排        run_pipeline / agent_run
    10. Mock 自测    python services/agent_service.py 直接运行

【如何调用】
    from services.agent_service import agent_run
    result = await agent_run(query, history)
    返回结构（字段契约）见 _pack() / _refuse() 的 docstring。

【设计哲学】
    "多智能体"不是多个 LLM，而是职责单一、可独立评测、可替换、通过结构化 JSON
    通信的协作单元。本文件每个 Agent 都可以单独抽出来跑评测
"""
import asyncio
import json
import re
import sys
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional

from config import (
    LLM_MODEL,
    LLM_BASE_URL,
    LLM_ENABLE_THINKING,
    DASHCOPE_API_KEY,
    VECTOR_TOP_K,
    BM25_TOP_K,
    RRF_K,
    RERANK_TOP_K,
    DISTANCE_THRESHOLD,
    AGENT_MAX_RETRY_TIMES,
    AGENT_QUERY_REWRITE_NUM,
    collection,
)
from common.logger import logger
from services.bm25_service import bm25_service
from services.embedding_service import get_embedding
from services.rrf_service import to_ranked_list, rrf_fuse
from services.rerank_service import rerank
# 提示词模板集中管理在 services/prompt_templates.py（评测锁版本 / 运营改话术 / A/B 替换）
from services.prompt_templates import (
    ROUTER_PROMPT, REWRITE_PROMPT, DECOMPOSE_PROMPT,
    ANSWER_PROMPT, SYNTHESIS_PROMPT, VERIFY_PROMPT, CHITCHAT_PROMPT,
    REFUSE_REPLIES, DEFAULT_REFUSE,
)

#   FAQ 语义命中的余弦阈值。
#   阈值不是拍脑袋，要用评测集校准：
#   调高 → 命中更准但更多问题掉到 C 类走检索（时延变差）
#   调低 → B 类直出变多但可能答错问题（准确率变差）
#   校准方法：在评测集上扫一遍 0.80~0.90，看"FAQ 直出准确率"和"C 类召回率"的拐点
FAQ_MATCH_THRESHOLD = 0.85

# D 类最多拆几个子问题：防止模型把简单问题拆散、或无限递归拆解
# （限一层 + 限条数，是客服场景 <3s 时延硬约束下的经验值）
MAX_SUB_QUESTIONS = 4

# 上下文最长字符数：防止 rerank 后的 top_k 文本总量撑爆 prompt
# （text-embedding-v2 单次上限 25 条不涉及这里；这是给 LLM prompt 控长）
MAX_CONTEXT_CHARS = 6000

# A 类寒暄的本地模板库：高频模式化场景，规则优先于 LLM。
# 【为什么】寒暄是客服场景最高频的输入（"在吗""你好"占开场白的大头），
# 但答案高度模式化——为一句"您好我在的"付一次 LLM 调用（~1s 时延 + 计费
# token），ROI 极低，且演示/压测时高频点击很容易把账户配额打满（限流）。
# 这是"按需求做技术选型"的直接体现：LLM 留给真正需要理解力的任务
# （分类/改写/生成/审核），模式化回复用本地规则，零成本零时延。
# 【兜底链】本地规则 → 未命中走 LLM → LLM 故障也有静态兜底，三层不会全崩。
# 编译选项 re.IGNORECASE：用户会打 "Hello"/"HELLO"/"Thank you" 各种大小写，
# 中文不受影响，英文寒暄必须靠它兜住——漏一个大小写形态，规则命中率就不好看
# 【边界：只有"纯寒暄"才短路】句尾允许语气词与标点（"你好呀""在吗亲""谢谢啦~"），
# 但不允许再接任何业务内容。这个边界踩过坑：早期正则写成 `^(...|大家好?)` 没有右锚定，
# 等价于匹配前缀「大家」——所有「大家电…」开头的真实业务问题（保修/三包/预约配送）
# 都被判成寒暄，直接回一句"欢迎光临"，业务问题整个丢失。280 条评测里实测 8 条中招，
# 占路由错误的一半以上。同类问题还有"你好，退货运费谁出"——用户先打招呼再说事是客服
# 场景最高频的输入形态，前缀匹配会把这类复合句全部吃掉。
# 修法：右锚定 + 句尾只容忍语气词，语义上等价于"整句就是一个寒暄"。
_GREET_TAIL = r"[呀啊哈呢啦哦嘛~～。，,亲\s]*"

CHITCHAT_RULES = [
    (re.compile(rf"^(在吗|在么|在不在|在否|在线的?吗|有人吗|有人在不){_GREET_TAIL}$", re.IGNORECASE),
     "您好，我在的～请问有什么可以帮您？"),
    (re.compile(rf"^(你好|您好|哈喽|哈罗|嗨|hi|hello|hey|大家(好)?){_GREET_TAIL}$", re.IGNORECASE),
     "您好呀，欢迎光临～请问有什么可以帮您的？"),
    (re.compile(rf"^(你是谁|你是什么|你是机器人吗|是人工吗|人工吗|机器人吗|你是真人吗){_GREET_TAIL}$", re.IGNORECASE),
     "我是电商智能客服，可以为您解答商品、订单、配送、售后相关问题；"
     "如果问题较复杂，我会为您转接人工客服。"),
    (re.compile(rf"^(谢谢|感谢|辛苦了|多谢|thanks?|thank\s*you){_GREET_TAIL}$", re.IGNORECASE),
     "不客气～能帮到您就好！还有别的问题随时问我。"),
    (re.compile(rf"^(再见|拜拜|先这样|没事了|晚安|bye){_GREET_TAIL}$", re.IGNORECASE),
     "好的，随时欢迎再来～祝您购物愉快！"),
]
# 本地规则和 LLM 都没兜住时的最终静态兜底（保证 A 类永远有回复，不会白屏）
CHITCHAT_FALLBACK = "您好，我是电商智能客服，有什么可以帮您的吗？"


def _local_chitchat(query: str) -> Optional[str]:
    """
    A 类寒暄的本地规则匹配（不调用 LLM）。

    Args:
        query: 用户原始输入

    Returns:
        Optional[str]: 命中规则返回模板回复；未命中返回 None（调用方再走 LLM）

    实现细节：去掉首尾标点再匹配——用户会打"在吗？""在吗。""在吗~~~"，
    正则必须容忍这些噪声，否则"规则命中率"在真实输入下会很难看。

    【为什么是 fullmatch 而不是 search】短路的前提是"这一句只说了寒暄"。
    用 search 会让"你好，退货运费谁出"这种"打招呼 + 提问"的复合句命中寒暄，
    用户真正想问的业务问题被一句"欢迎光临"吃掉——这是比拒答更糟的失败：
    用户会觉得系统根本没在听。宁可多花一次 LLM 调用，也不能吞掉业务意图。
    """
    q = query.strip().rstrip("？?！!。.~～ \n\t")
    for pattern, reply in CHITCHAT_RULES:
        if pattern.fullmatch(q):
            return reply
    return None


# ============================================================
# 1. LLM 封装 + JSON 提取
# ============================================================

class LLMError(RuntimeError):
    """
    LLM 调用失败的自定义异常。

    【为什么要自定义而不直接 RuntimeError】
    上层（agent_router 的 SSE 接口）需要区分两种失败来做不同的用户提示：
        retryable=True  → 限流类（账户配额/瞬时并发超限）：提示"服务繁忙，稍后再试"
        retryable=False → 其他错误（参数/网络/服务端故障）：提示"服务暂时不可用"
    直接 RuntimeError 携带一大串原始错误码，前端只能把 repr(e) 怼给用户看，
    用户看到的是一堆 "Throttling.RateQuota https://help.aliyun.com/..." 这种
    运维细节——既看不懂，也暴露内部实现。分类提示是产品体验的基本功。

    【求职考点】异常设计即接口设计：异常携带的语义信息决定了上层能做什么。
    """
    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable   # True=限流类瞬时错误，值得退避重试


def _is_rate_limit_error(http_code: int, err_body: str) -> bool:
    """
    判断一次 HTTP 错误是否为"限流"类。

    限流的两个特征码（阿里云文档）：
        - HTTP 429
        - 错误体含 "Throttling"（如 Throttling.RateQuota 账户配额 /
          Throttling.TPS 每秒并发）或 "rate limit"
    限流≠故障：账号没事，是"单位时间调用太密"或"免费额度用尽"。
    退避几秒往往能恢复（瞬时并发型）；持续型（配额耗尽）则快速失败，
    前端给出明确人工指引——重试 10 次让用户干等 30 秒反而是更差的体验。
    """
    return http_code == 429 or "Throttling" in err_body or "rate limit" in err_body.lower()


def call_llm(messages: List[Dict], temperature: float = 0.3, max_tokens: int = 1024,
             max_retries: int = 2) -> str:
    """
    统一的 LLM 调用出口（所有 Agent 的唯一出口，便于 mock 和加横切逻辑）

    实现方式：DashScope **compatible-mode** 端点（OpenAI 兼容协议）+ 标准库
    urllib。为什么不继续用 dashscope SDK 的 Generation.call：
    1. 已安装 SDK 1.27.8 对新模型（qwen3.x 系列）的原生端点路由报 url error，
       这是 1.x SDK 的已知限制；compatible-mode 端点新旧模型通吃
    2. OpenAI 兼容协议 = 行业标准接口，将来换供应商（OpenAI / DeepSeek 官方 /
       本地 vLLM）只改 config 里的 LLM_BASE_URL + LLM_MODEL，本函数零改动
    3. 只依赖标准库，少一个第三方依赖就少一分供应链风险

    Args:
        messages:    OpenAI 风格消息列表，如 [{"role": "system", "content": ...}]
        temperature: 采样温度。确定性任务（分类/审核）用 0.1~0.3；
                     要多样性的任务（改写）用 0.7
        max_tokens:  最大生成 token 数。按任务估：分类 128 / 改写 256 / 生成 1024
        max_retries: 限流类错误的最大重试次数（默认 2，即最多 3 次尝试）。
                     只对限流重试：超时/参数错误重试没有意义，快速失败让上层降级。

    Returns:
        str: LLM 输出的文本内容

    Raises:
        LLMError: 所有失败路径（retryable 标记是否为限流类，供上层分类提示）

    【为什么限流要退避重试】
     DashScope 免费额度有 QPS/日配额限制。演示时连续点几个问题就可能撞上
     瞬时并发上限——这不是代码 bug，是配额设计。指数退避（1s→2s）是业界
     标准做法：给服务端缓冲时间，大部分瞬时限流 1~2 次重试内就恢复了。
    """
    url = f"{LLM_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {DASHCOPE_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        # qwen3 混合推理模型必须显式关闭思考：否则分类/审核类任务会先输出
        # 一大段推理过程，max_tokens 被撑爆且 JSON 解析必失败（见 config 注释）
        "enable_thinking": LLM_ENABLE_THINKING,
    }

    delay = 1.0  # 首次重试等待 1 秒，之后翻倍（1s → 2s）
    for attempt in range(max_retries + 1):
        try:
            req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                         headers=headers, method="POST")
            # timeout=60：LLM 生成最长等 60s。注意它是"每次尝试"的超时，
            # 不是总预算——总预算 = 60 × 尝试次数 + 退避时间
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
            content = data.get("choices", [{}])[0].get("message", {}).get("content")
            if content is None:
                raise LLMError(f"LLM返回结构异常: {json.dumps(data)[:200]}", retryable=False)
            return content

        except urllib.error.HTTPError as e:
            # HTTP 层错误（4xx/5xx）：错误体是 JSON，含 code/message
            err_body = ""
            try:
                err_body = e.read().decode()[:300]
            except Exception:
                pass
            retryable = _is_rate_limit_error(e.code, err_body)
            if retryable and attempt < max_retries:
                logger.warning(
                    f"LLM限流，{delay:.0f}s 后重试（第{attempt + 2}次尝试）: "
                    f"HTTP {e.code} {err_body[:150]}"
                )
                time.sleep(delay)
                delay *= 2
                continue
            # 不可重试 / 重试次数用尽 → 抛给上层（agent_router 会转成用户友好的提示）
            raise LLMError(f"LLM调用失败: HTTP {e.code} {err_body}", retryable=retryable)

        except LLMError:
            raise  # 上面主动抛的结构异常，原样上抛
        except Exception as e:
            # 网络层错误（连接超时/DNS/代理）：不重试，快速失败走上层降级
            raise LLMError(f"LLM调用失败: {repr(e)}", retryable=False)

    raise LLMError("LLM调用失败: 重试次数已用尽", retryable=True)  # 理论上不可达，防御性


def extract_json(raw: str) -> Optional[Dict]:
    """
    从 LLM 输出中提取 JSON（LLM 爱裹 ```json 代码块、爱加前后缀，必须清洗）。

    Args:
        raw: LLM 的原始输出文本

    Returns:
        Optional[Dict]: 解析成功的 dict；失败返回 None（由调用方决定降级策略）

    """
    if not raw:
        return None
    text = raw.strip()
    # 剥掉 markdown 代码块围栏（```json ... ```）
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    # 截取第一个 { 到最后一个 }，容忍 LLM 在 JSON 前后说的废话
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        # 不抛异常：解析失败是"预期内的脏数据"，返回 None 让上层降级
        logger.warning(f"JSON解析失败，原始输出: {raw[:200]}")
        return None


def _fmt_history(history: Optional[List[Dict]]) -> str:
    """
    把对话历史格式化成 prompt 文本。

    Args:
        history: [{"role": "user"/"assistant", "content": ...}, ...] 或 None

    Returns:
        str: 多行文本，无历史时返回占位符（占位符比空字符串更利于 LLM 理解语境）
    """
    if not history:
        return "（无历史）"
    lines = []
    for h in history[-6:]:  # 只带最近6轮：控prompt长度，也防旧信息干扰指代消解
        role = "用户" if h.get("role") == "user" else "客服"
        lines.append(f"{role}: {h.get('content', '')}")
    return "\n".join(lines)

# ============================================================
# 2. Router Agent
# ============================================================

def route_query(question: str, history: List[Dict]) -> str:
    """
    Router Agent：决定这个问题"配用什么级别的处理"。

    分级路由（L0 本地规则 → L1 LLM 分类）：
        L0：命中寒暄白名单 → 直接判 A，0 次 LLM 调用
        L1：其余全部走 LLM 五分类
    【为什么加 L0】Router 在每个请求的关键路径上，是固定开销；而寒暄是客服
    场景最高频的输入（开场白的大头）。为"在吗"这种确定性极高的输入付一次
    LLM 调用（~1s + 计费 + 限流风险），ROI 极低。这不是绕过 Adaptive——
    恰恰是 Adaptive "按需分配算力"思想的延伸：算力预算留给真正需要
    理解力的 query。账户限流时，这条路径保证系统不全军覆没。
    【为什么只有 A 类做短路】E 类（越界）的判定涉及合规，本地正则误判
    一个正常问题为越界 = 误拒，代价远高于省一次 LLM 调用，必须 LLM 精确判断。

    Args:
        question: 用户当前问题（原始 query，未改写）
        history:  对话历史（用于结合上下文分类，如"那它呢"依赖上文的订单语境）

    Returns:
        str: "A" / "B" / "C" / "D" / "E" 之一
    """
    # ---- L0：寒暄白名单短路，零成本零时延 ----
    if _local_chitchat(question):
        logger.info(f"Router L0规则短路: {question} → A（0次LLM调用）")
        return "A"

    # ---- L1：LLM 五分类 ----
    raw = call_llm(
        [{"role": "system", "content": ROUTER_PROMPT.format(
            history=_fmt_history(history), question=question)}],
        temperature=0.1,   # 分类是确定性任务，低温保稳定（同一个问题多次分类结果一致）
        max_tokens=128,    # 分类输出很短，128 足够；给大了浪费计费 token
    )
    data = extract_json(raw)
    cat = str((data or {}).get("category", "C")).strip().upper()
    if cat not in ("A", "B", "C", "D", "E"):
        # 分类失败/脏数据 → 降级 C 类走完整检索链路：宁可多检索，不可不检索
        # （降级方向的选择逻辑：漏检索=答不上，误拒=把能答的拒了，前者代价更低）
        logger.warning(f"Router输出异常，降级为C类: {raw[:100]}")
        return "C"
    logger.info(f"Router分类: {cat} | {data.get('reason', '')}")
    return cat


# ============================================================
# 3. Query Rewrite Agent（重试轮的改写）
# ============================================================

def rewrite_query(question: str, history: List[Dict], n: int = AGENT_QUERY_REWRITE_NUM) -> List[str]:
    """
    Query Rewrite Agent：结合对话历史消解指代、补全省略、规范口语。

    Args:
        question: 检索失败前的原始 query
        history:  对话历史（指代消解的关键输入）
        n:        生成几条改写（config.AGENT_QUERY_REWRITE_NUM=3）

    Returns:
        List[str]: 去重后的改写列表；可能为空（LLM 故障时），调用方需处理

    例："那它呢？" → "已发货订单可以申请退货吗"
    """
    raw = call_llm(
        [{"role": "system", "content": REWRITE_PROMPT.format(
            history=_fmt_history(history), question=question, n=n)}],
            temperature=0.7,   # 改写要多样性，高温（同低温会让 3 条改写几乎一样，白费）
            max_tokens=256,
    )
    data = extract_json(raw)
    queries = [q.strip() for q in (data or {}).get("queries", []) if q and q.strip()]
    # 去重 + 去掉与原始 query 完全相同的（相同的重搜没有意义，纯粹浪费一轮）
    seen, uniq = set(), []
    for q in queries:
        if q != question and q not in seen:
            seen.add(q)
            uniq.append(q)
    logger.info(f"QueryRewrite生成{len(uniq)}条: {uniq}")
    return uniq


# ============================================================
# 4. Decompose Agent（D 类多跳拆解）
# ============================================================

def decompose_query(question: str, history: List[Dict]) -> List[str]:
    """
    Decompose Agent：把复合问题拆成可独立检索的子问题（限一层）。

    Args:
        question: 用户的复合问题
        history:  对话历史

    Returns:
        List[str]: 子问题列表（可能为空）；上限 MAX_SUB_QUESTIONS 条

    例："订单还没到，退货的话运费谁出"
        → ["订单未送达时可以退货吗", "退货时运费由谁承担"]

    """
    raw = call_llm(
        [{"role": "system", "content": DECOMPOSE_PROMPT.format(
            history=_fmt_history(history), question=question, max_sub=MAX_SUB_QUESTIONS)}],
        temperature=0.3,
        max_tokens=256,
    )
    data = extract_json(raw)
    subs = [s.strip() for s in (data or {}).get("sub_questions", []) if s and s.strip()]
    subs = subs[:MAX_SUB_QUESTIONS]   # 双保险：prompt 里限了，代码里再截一次
    logger.info(f"Decompose拆解: {subs}")
    return subs


# ============================================================
# 5. FAQ Matcher（B 类语义直出）
# ============================================================

class FAQMatcher:
    """
    B 类 FAQ 语义匹配器：把 FAQ 问题向量化后常驻内存，用户 query 直接算余弦。
    命中 → 直出标准答案，不检索不生成（<0.5s，体验最好）。

    """

    def __init__(self):
        # FAQ 缓存：[{"question","answer","embedding"}]，启动时一次性批量构建
        self._cache: List[Dict] = []
        self._loaded = False

    async def preload(self):
        """
        服务启动/调用前预热：批量向量化一次，之后查询只 embed 用户 query。
        幂等：已加载则直接返回（每个进程只做一次批量 embed，省 API 调用）。
        """
        if self._loaded:
            return
        from models.faq import FAQ  # 延迟导入：避免未初始化 Tortoise 时 import 炸掉
        faqs = await FAQ.all()
        if not faqs:
            self._loaded = True
            return
        texts = [f.question for f in faqs]
        # 批量向量化放线程池：get_batch_embedding 是同步阻塞 IO，
        # 直接 await 会卡死 FastAPI 的事件循环
        embeddings = await asyncio.to_thread(_get_batch_embedding_safe, texts)
        self._cache = [
            {"question": f.question, "answer": f.answer, "embedding": emb}
            for f, emb in zip(faqs, embeddings)
        ]
        self._loaded = True
        logger.info(f"FAQ缓存加载完成，共{len(self._cache)}条")

    def match(self, question: str) -> Optional[Dict]:
        """
        余弦相似度 ≥ FAQ_MATCH_THRESHOLD → 返回命中，否则 None。

        Args:
            question: 用户问题

        Returns:
            Optional[Dict]: {"question","answer","score"} 或 None
        """
        if not self._cache:
            return None
        q_vec = get_embedding(question)
        best, best_score = None, -1.0
        for item in self._cache:
            score = _cosine(q_vec, item["embedding"])
            if score > best_score:
                best, best_score = item, score
        if best_score >= FAQ_MATCH_THRESHOLD:
            logger.info(f"FAQ语义命中: score={best_score:.3f} q={best['question']}")
            return {"question": best["question"], "answer": best["answer"], "score": round(best_score, 4)}
        return None


def _cosine(a: List[float], b: List[float]) -> float:
    """
    余弦相似度
    """
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _get_batch_embedding_safe(texts: List[str]) -> List[List[float]]:
    """批量向量化同步封装（embedding_service 的同步版），供 FAQMatcher.preload 用"""
    from services.embedding_service import get_batch_embedding
    return get_batch_embedding(texts)


# 模块级单例：整个进程共用一份 FAQ 缓存（重复加载 = 重复烧 embedding API）
faq_matcher = FAQMatcher()


# ============================================================
# 6. 检索编排：向量召回 / BM25 召回 / RRF 融合
# ============================================================

def vector_recall(query: str, top_k: int = VECTOR_TOP_K) -> List[Dict]:
    """
    向量召回（语义通路）：query 向量化 → Chroma 查询 → 欧氏距离转相似度。

    Args:
        query:  本轮检索用 query（可能是原始 query，也可能是改写后的）
        top_k:  向量侧召回条数（config.VECTOR_TOP_K=6）

    Returns:
        List[Dict]: [{"chunk_id","doc_id","text","similarity"}]，similarity 越大越相关
    """
    q_vec = get_embedding(query)
    res = collection.query(
        query_embeddings=[q_vec],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    out = []
    # Chroma 返回的四个列表严格同序同长，用 zip 平行遍历（位置即对应关系）
    for cid, doc, meta, dist in zip(
        res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        dist = float(dist)
        if dist > DISTANCE_THRESHOLD:
            continue  # 超距候选直接丢弃：连"可能相关"都算不上，不进融合
        out.append({
            # Chroma 的 id 就是 f"{doc_id}_{i}"，天然唯一 → 直接当 chunk_id 用
            # （这也是 RRF 用 chunk_id 做主键的原因：同文档不同 chunk 必须独立计分）
            "chunk_id": str(cid),
            "doc_id": str((meta or {}).get("doc_id", cid)),
            "text": doc,
            # 欧氏距离越小越相关 → 1/(1+d) 映射成 0~1 的"越大越相关"，
            # 统一方向后才能进 to_ranked_list 按降序排名
            "similarity": round(1.0 / (1.0 + dist), 6),
        })
    return out


def bm25_recall(query: str, top_k: int = BM25_TOP_K) -> List[Dict]:
    """
    BM25 关键词召回（精确通路）：jieba 分词后按词项匹配度打分。

    Args:
        query:  本轮检索用 query
        top_k:  BM25 侧召回条数（config.BM25_TOP_K=6）

    Returns:
        List[Dict]: [{"chunk_id","doc_id","text","bm25_score"}]，分数越大越相关

    """
    hits = bm25_service.search(query, top_k=top_k)
    out = []
    for h in hits:
        meta = h.get("meta") or {}
        # chunk_id 必须与 Chroma id 同规则（f"{doc_id}_{i}"），否则 RRF 主键对不齐、
        # 两路无法融合——这是混合检索最典型的静默 bug：不报错，但协同效应归零
        out.append({
            "chunk_id": str(meta.get("chunk_id", "")),
            "doc_id": str(meta.get("doc_id", "")),
            "text": h["text"],
            "bm25_score": h["score"],
        })
    # 丢掉没有 chunk_id 的结果：宁可少一路候选，不可让两路主键错位
    return [x for x in out if x["chunk_id"]]


def retrieve_round(query: str) -> List[Dict]:
    """
    一轮完整召回：向量 + BM25 → RRF 融合 → 全量候选（不截断，交给 rerank）。

    Args:
        query: 本轮检索用 query

    Returns:
        List[Dict]: rrf_fuse() 的输出（按 rrf_score 降序），两路都无结果时返回 []

    """
    vec = vector_recall(query)
    bm = bm25_recall(query)
    if not vec and not bm:
        logger.info(f"两路召回均无结果: query={query}")
        return []
    # to_ranked_list 把两路不同量纲的分数（similarity / bm25_score）统一转成"名次"，
    # 这是 RRF 能跨系统比较的前提——不比分数，只比排名
    fused = rrf_fuse(
        to_ranked_list(vec, score_key="similarity"),
        to_ranked_list(bm, score_key="bm25_score"),
        k=RRF_K,
    )
    return fused


# ============================================================
# 7. 生成 Agent（带引用）
# ============================================================

def _build_context(results: List[Dict]) -> str:
    """
    把精排结果拼成带编号的上下文，编号即引用号 [1][2]...

    Args:
        results: rerank 精排后的 top_k 结果

    Returns:
        str: 形如 "[1] 文本A\\n[2] 文本B" 的上下文字符串
    """
    parts, total = [], 0
    for i, r in enumerate(results):
        piece = f"[{i + 1}] {r['text']}"
        if total + len(piece) > MAX_CONTEXT_CHARS:
            break  # 超长截断：省 token 也防 prompt 超限（截断顺序即精排序，损失最小）
        parts.append(piece)
        total += len(piece)
    # 显式声明可用编号范围。实测模型会凭空写出 [4]（上下文只有 3 条）：
    # 前端拿这个编号跳不到任何引用，Critic 也可能因此判失败。
    # prompt 里直接给死边界，比事后过滤更有效（过滤是兜底，不是主策略）。
    bound = (f"\n\n（以上为全部 {len(parts)} 条资料，引用编号只能使用 [1] 至 [{len(parts)}]，"
             f"不得使用此范围外的编号）")
    return "\n".join(parts) + bound


def generate_answer(query: str, results: List[Dict], history: List[Dict], sub_mode: bool = False):
    """
    生成 Agent：基于精排证据生成带引用的回答。

    Args:
        query:    本轮检索用 query（rerank 用的哪个，这里就用哪个，保证一致性）
        results:  rerank 精排后的证据列表（含 chunk_id/doc_id/text）
        history:  对话历史
        sub_mode: True 用 D 类综合模板（SYNTHESIS_PROMPT），False 用 C 类模板

    Returns:
        Tuple[str, List[Dict]]: (答案文本, 引用列表)
        引用列表元素：{"index","chunk_id","doc_id","rerank_score","text"(截断)}

    """
    context = _build_context(results)
    template = SYNTHESIS_PROMPT if sub_mode else ANSWER_PROMPT
    content = call_llm(
        [{"role": "system", "content": template.format(
            context=context, history=_fmt_history(history), question=query)}],
        temperature=0.3,   # 生成任务允许适度多样性，但不能高——客服话术要稳
        max_tokens=1024,
    )
    citations = [
        {
            "index": i + 1,               # 与上下文中 [i+1] 编号严格对应
            "chunk_id": r["chunk_id"],    # 溯源到具体 chunk（debug_router 可回查原文）
            "doc_id": r["doc_id"],        # 溯源到具体文档
            "rerank_score": r.get("rerank_score"),  # 精排分：低分引用前端可置灰警示
            "text": r["text"][:80],       # 截断：citation 只做溯源展示，不必全文
            "full_text": r["text"],       # 评测/调试用：关键词命中判定不能只看 80 字摘要
        }
        for i, r in enumerate(results)
    ]
    return content, citations


def extract_used_citations(answer: str, max_index: Optional[int] = None) -> List[int]:
    """
    从回答文本里提取实际出现的引用编号（正则抓 [n]）。

    Args:
        answer: 生成 Agent 产出的回答文本
        max_index: 证据条数上限。传入后会丢弃 [0] / [max_index+1] 这类越界编号
                   （模型偶发编造不存在的编号，见下）

    Returns:
        List[int]: 升序去重的编号列表，如 [1, 2]

    用途：前端只高亮"回答真正引用了的"证据；评测"引用忠实度"指标。

    【为什么要 max_index 兜底】
    实测模型会写出 "[4]"，而上下文只有 3 条证据。这类编号前端跳不到任何引用、
    会让"引用溯源"这个卖点变成断链。prompt 里已经声明了合法范围（_build_context），
    但 LLM 的指令遵循不是 100%——**凡是"模型可能不听话"的地方，代码都要有兜底**。
    这里选择"丢弃 + 告警"而不是"改写答案文本"：答案内容是对的，只是标注多余，
    替换文本反而有改坏内容的风险。
    """
    nums = {int(m) for m in re.findall(r"\[(\d+)\]", answer)}
    if max_index is not None:
        invalid = sorted(n for n in nums if n < 1 or n > max_index)
        if invalid:
            logger.warning(f"回答出现越界引用编号 {invalid}（有效范围 1-{max_index}），已忽略")
        nums = {n for n in nums if 1 <= n <= max_index}
    return sorted(nums)


# ============================================================
# 8. Critic Agent（溯源校验守门）
# ============================================================

def verify_answer(answer: str, results: List[Dict]) -> Dict:
    """
    Critic Agent：判断答案是否被检索证据支撑。
    这是拒答的第二道防线（第一道在 rerank 的低置信自省）。

    Args:
        answer:  生成 Agent 产出的回答
        results: 生成时使用的证据列表（审核必须针对同一批证据）

    Returns:
        Dict: {"passed": bool, "degraded": bool, "reason": str}
        - passed=True  答案通过校验
        - degraded=True 表示这次判定是"审核器故障时的放行"，监控要能看到
    """
    context = _build_context(results)
    try:
        raw = call_llm(
            [{"role": "system", "content": VERIFY_PROMPT.format(context=context, answer=answer)}],
            temperature=0.1,   # 审核要确定性：同一个答案多次审核结论必须一致
            max_tokens=256,
        )
        data = extract_json(raw)
        if data is None:
            # 审核器输出无法解析 → 放行但打标（fail-open + degraded）
            return {"passed": True, "degraded": True, "reason": "critic_output_unparsable"}
        return {"passed": bool(data.get("passed")), "degraded": False,
                "reason": str(data.get("reason", ""))}
    except Exception as e:
        # LLM 调用本身异常（网络/限流）→ 同样 fail-open + degraded
        # 注意：异常必须被兜住，否则一次审核器故障会让整个问答 500
        logger.error(f"Critic审核异常，fail-open放行: {repr(e)}")
        return {"passed": True, "degraded": True, "reason": f"critic_error: {repr(e)}"}


# ============================================================
# 9. 主编排
# ============================================================

def _pack(answer: str, route: str, citations: List[Dict], used: List[int],
          rounds: int, trace: Dict) -> Dict:
    """
    成功回答的统一返回结构（SSE 接口和评测脚本消费同一份契约）。

    Returns:
        Dict: {
            "answer": str,            # 回答文本（含 [1][2] 引用标注）
            "refused": False,         # 固定 False（拒答走 _refuse）
            "refuse_reason": None,
            "route": str,             # 最终生效的路由（B 未命中会是 "C(fallback_from_B)"）
            "citations": List[Dict],  # 全部引用（index/chunk_id/doc_id/rerank_score/text）
            "used_citations": List[int],  # 回答中实际出现的引用编号
            "rounds": int,            # 实际检索了几轮（0=未检索，如 A/B 类）
            "trace": Dict,            # 全过程现场数据，debug_router 可视化的数据源
        }
    """
    return {
        "answer": answer,
        "refused": False,
        "refuse_reason": None,
        "route": route,
        "citations": citations,
        "used_citations": used,
        "rounds": rounds,
        "trace": trace,
    }


def _refuse(reason: str, route: str, rounds: int, trace: Dict) -> Dict:
    """
    拒答统一出口：不编造 + 引导人工客服

    Args:
        reason: 拒答原因，取 REFUSE_REPLIES 的 key：
                "low_confidence"（检索层低置信）/ "verify_failed"（生成层校验失败）/
                "out_of_scope"（越界敏感）/ "rewrite_failed"（改写也失败）
        route:  当前路由分类
        rounds: 已执行的检索轮数
        trace:  全过程现场数据（拒答时保留现场，供复盘"为什么拒"）

    Returns:
        Dict: 与 _pack 同构，但 refused=True、answer 为人工引导话术

    """
    return {
        "answer": REFUSE_REPLIES.get(reason, DEFAULT_REFUSE),
        "refused": True,
        "refuse_reason": reason,
        "route": route,
        "citations": [],
        "used_citations": [],
        "rounds": rounds,
        "trace": trace,
    }


def _search_and_rerank(query: str, trace: Dict) -> Dict:
    """
    一轮"检索 + 精排"的公共步骤，结果记入 trace。

    Args:
        query: 本轮检索用 query
        trace: 全局 trace 字典（原地追加 rounds 记录）

    Returns:
        Dict: rerank() 的原始返回（results/top1_score/passed/degraded）

    trace 里每轮记录：query、候选数、top1 精排分、是否通过、是否降级、
    前 3 篇的 hit_detail（两路各自名次）——这就是 debug_router 要展示的全部信息。
    """
    candidates = retrieve_round(query)
    rr = rerank(query, candidates, top_k=RERANK_TOP_K)
    trace["rounds"].append({
        "query": query,
        "candidate_count": len(candidates),
        "top1_score": rr.get("top1_score"),
        "passed": rr.get("passed"),
        "degraded": rr.get("degraded", False),
        "hits": [r.get("hit_detail") for r in rr.get("results", [])[:3]],
    })
    return rr


def run_pipeline(query: str, history: List[Dict], matcher: FAQMatcher) -> Dict:
    """
    同步编排主函数（内部全是同步调用，由 agent_run 丢线程池执行）。
    结构严格对齐文件头部的链路图：Router → 分类分治 → 重试循环 → 生成/拒答。

    Args:
        query:   用户原始 query（第一轮检索用它，之后轮用改写）
        history: 对话历史
        matcher: FAQMatcher 实例（注入而非直接用全局，方便测试替换）

    Returns:
        Dict: _pack() 或 _refuse() 的结构，见两者 docstring
    """
    # trace：全过程现场数据。评测时用它算"平均检索轮数""拒答率"，
    # 调试时用它回答"这个回答是怎么来的/为什么被拒"
    trace: Dict = {"route": None, "rounds": [], "sub_questions": [], "verify": None}

    # ---- Router：先分类，再决定花多少算力（Adaptive 的入口） ----
    route = route_query(query, history)
    trace["route"] = route

    # ---- A 寒暄：直接答，不检索 ----
    # （检索是时延和成本的浪费；且寒暄检索出来的东西对回答没有帮助）
    # 三层兜底：本地规则 → LLM → 静态模板。第一层就挡掉最高频的输入，
    # 账户限流时（如演示连续点击）这条路径依然可用——见 CHITCHAT_RULES 注释
    if route == "A":
        content = _local_chitchat(query)
        if content:
            logger.info(f"A类寒暄本地规则命中，0次LLM调用: {query}")
            return _pack(content, route, [], [], 0, trace)
        try:
            content = call_llm(
                [{"role": "system", "content": CHITCHAT_PROMPT.format(
                    history=_fmt_history(history), question=query)}],
                temperature=0.5, max_tokens=128,
            )
        except LLMError as e:
            # LLM 也不可用时用静态兜底：寒暄场景宁可朴素也不能没有响应
            logger.warning(f"A类寒暄LLM失败，使用静态兜底: {e}")
            content = CHITCHAT_FALLBACK
        return _pack(content, route, [], [], 0, trace)

    # ---- E 越界/敏感：合规兜底，直接拒答转人工 ----
    # （这是电商场景特有的设计：知识库无权处理的操作必须挡住）
    if route == "E":
        return _refuse("out_of_scope", route, 0, trace)

    # ---- FAQ 直出：B 类命中直出；C 类先做一次"零号探测"，不中再走检索 ----
    # 【为什么 C 类也要探测】
    #   Router 给 B 的判据是"知识库有标准答案"，可 LLM 看不到知识库——这个判据对它
    #   不可观测。实测 280 条评测里 B 一次都没输出过（连 ROUTER_PROMPT 自带示例
    #   "支持七天无理由吗"都被判成 C），FAQ 表灌了 62 条也永远查不到，
    #   直出通道等于不存在。
    #   "该不该直出"本质是一次检索判断，交给向量相似度比交给 LLM 猜可靠：
    #   C 类进检索链路前先算一次 FAQ 余弦，命中（≥ FAQ_MATCH_THRESHOLD）就直出，
    #   不中再走原来的 QueryRewrite 重试循环——额外开销只是一次 embedding。
    #   D 类不探测：多跳问题需要两条以上证据，单条 FAQ 只答一半反而更糟。
    if route in ("B", "C"):
        route_in = route
        hit = matcher.match(query)
        if hit:
            # 命中：直出人工审定的标准答案。citation 记 FAQ 原文（chunk_id=None 表示非 chunk 来源）
            return {
                "answer": hit["answer"],
                "refused": False,
                "refuse_reason": None,
                "route": "B",
                "citations": [{"index": 1, "chunk_id": None, "doc_id": None,
                               "rerank_score": hit["score"], "text": hit["question"]}],
                "used_citations": [],
                "rounds": 0,
                "trace": {**trace, "faq_hit": hit},
            }
        # 未命中：交给检索链路。trace 里保留原始判定，便于区分两种情况：
        #   "B→C(fallback)" = Router 以为有标准答案，库里其实没有（分类器错，系统兜住）
        #   "C"             = 本来就是单跳知识题，走检索是正常路径
        logger.info(f"FAQ未命中({route_in})，走C类检索链路")
        route = "C"
        trace["route"] = "C(fallback_from_B)" if route_in == "B" else "C"

    # ---- D 复合多跳：先分解，子问题各走一轮检索，合并证据 ----
    if route == "D":
        subs = decompose_query(query, history)
        trace["sub_questions"] = subs
        if subs:
            # 每个子问题独立走一轮完整检索（子问题层面不重试，重试预算留给顶层）
            merged: Dict[str, Dict] = {}
            for sub in subs:
                for cand in retrieve_round(sub):
                    cid = cand["chunk_id"]
                    # 按 chunk_id 去重，保留 rrf_score 更高的那份（不同子问题可能召回同一 chunk）
                    if cid not in merged or cand["rrf_score"] > merged[cid]["rrf_score"]:
                        merged[cid] = cand
            # 用原始复合 query 统一精排：对齐用户的真实意图，而非子问题的局部意图
            # （用子问题精排会把"运费谁出"的证据排到"能不能退"前面，打乱最终答案的重点）
            rr = rerank(query, list(merged.values()), top_k=RERANK_TOP_K)
            trace["rounds"].append({
                "query": query, "candidate_count": len(merged),
                "top1_score": rr.get("top1_score"), "passed": rr.get("passed"),
                "degraded": rr.get("degraded", False),
                "sub_mode": True,
            })
            if rr.get("passed"):
                answer, citations = generate_answer(query, rr["results"], history, sub_mode=True)
                used = extract_used_citations(answer, max_index=len(rr["results"]))
                verdict = verify_answer(answer, rr["results"])
                trace["verify"] = verdict
                if verdict["passed"]:
                    return _pack(answer, "D", citations, used, 1, trace)
                return _refuse("verify_failed", "D", 1, trace)
        # D 类没拆出来 / 精排没通过 → 落入下面的统一重试循环（用原始 query 再试）
        logger.info("D类分解检索未通过，落入统一重试循环")

    # ---- C 类（及 D 类兜底）统一重试循环：完全对应链路图 ----
    # 第 1 轮 = 原始 query；之后每轮用 QueryRewrite 的新 query
    max_rounds = AGENT_MAX_RETRY_TIMES + 1   # config=2 → 最多 3 轮（原始 + 2 次改写）
    rewrite_pool: List[str] = []
    for rnd in range(max_rounds):
        if rnd == 0:
            round_query = query
        else:
            # 按需生成改写池：第一次未过时才调 LLM，不预加载浪费
            if not rewrite_pool:
                rewrite_pool = rewrite_query(query, history, AGENT_QUERY_REWRITE_NUM)
                if not rewrite_pool:
                    # 改写器也故障 → 没有新 query 可试，直接拒答（附原因便于监控）
                    return _refuse("rewrite_failed", route, rnd, trace)
            round_query = rewrite_pool.pop(0)

        logger.info(f"第{rnd + 1}/{max_rounds}轮检索: {round_query}")
        rr = _search_and_rerank(round_query, trace)

        if rr.get("passed"):
            answer, citations = generate_answer(round_query, rr["results"], history)
            used = extract_used_citations(answer, max_index=len(rr["results"]))
            verdict = verify_answer(answer, rr["results"])
            trace["verify"] = verdict
            if verdict["passed"]:
                return _pack(answer, route, citations, used, rnd + 1, trace)
            # 生成层守门失败：改写重试救不了"证据够但说错话"，直接拒答
            # （重试的预算只花在"证据不足"上，不花在"说错话"上——后者该换 prompt 或模型）
            return _refuse("verify_failed", route, rnd + 1, trace)

        # passed=False 且还有轮次 → 循环继续，进入改写重试
        logger.info(f"第{rnd + 1}轮精排未过(top1={rr.get('top1_score')})，准备改写重试")

    # 轮次用完仍不过 → 证据不足，拒答转人工（链路图的 else 分支）
    return _refuse("low_confidence", route, max_rounds, trace)


async def agent_run(query: str, history: Optional[List[Dict]] = None) -> Dict:
    """
    对外唯一入口（async）。agent_router 的 SSE 接口消费这个函数。

    Args:
        query:   用户原始 query
        history: 对话历史 [{"role","content"}, ...]，无历史传 None

    Returns:
        Dict: run_pipeline 的返回（字段契约见 _pack / _refuse）

    """
    await faq_matcher.preload()
    return await asyncio.to_thread(run_pipeline, query, history or [], faq_matcher)


# ============================================================
# 10. Mock 自测（python services/agent_service.py 直接运行）
# ============================================================

if __name__ == "__main__":
    """
    编排逻辑回归测试——不需要真实 API / 模型，所有外部调用点用 mock 替换。

    【测试设计原则】只 mock 最底层出口（call_llm / retrieve_round / rerank），
    Router / Rewrite / Decompose / 生成 / Critic 这些 Agent 的真实代码路径
    照常执行。这样"这条路径调了几次 LLM"的断言才可信——例如场景 2 验证
    "A 类寒暄本地命中时 LLM 零调用"，这是账户限流时的生命线特性。

    覆盖 8 个场景：
        1. C 类正常流（检索过→生成→Critic 过）
        2. A 类寒暄·本地规则命中（0 次 LLM，限流时仍可用）
        3. A 类寒暄·规则未命中 → 走 LLM（兜底链第二层）
        4. 低置信拒答（原始 + 2 次改写共 3 轮全不过）
        5. Critic 拒答（证据够但答案不被支撑→不重试）
        6. E 类越界（零检索，合规兜底）
        7. B 类 FAQ 直出（不检索不生成）
        8. D 类多跳（拆解→分头检索→合并→统一精排）
    """
    # 以脚本方式运行时 __name__ == "__main__"，当前模块即被测对象
    svc = sys.modules[__name__]

    _PASS: List[str] = []
    _FAIL: List[str] = []

    def check(name: str, cond: bool, detail: str = ""):
        (_PASS if cond else _FAIL).append(name)
        print(f"  {'✅' if cond else '❌'} {name}" + (f"  [{detail}]" if detail and not cond else ""))

    # ---------- mock 基础设施 ----------
    class _MockMatcher:
        """FAQMatcher 替身：未命中（走检索链路）"""
        def match(self, q):
            return None

    class _HitMatcher:
        """FAQMatcher 替身：永远命中（B 类直出场景）"""
        def match(self, q):
            return {"question": "支持七天无理由退货吗",
                    "answer": "支持七天无理由退货。", "score": 0.91}

    llm_counter = {"n": 0}                          # LLM 调用计数（场景断言核心）
    router_reply = {"category": "C", "reason": "mock"}   # Router 可变回复

    def fake_call_llm(messages: List[Dict], **kwargs) -> str:
        """按 prompt 内容分发假回复——各 Agent 的解析逻辑照常真实执行"""
        llm_counter["n"] += 1
        content = messages[0]["content"]
        if "意图分类器" in content:      # ROUTER_PROMPT
            return json.dumps(router_reply, ensure_ascii=False)
        if "查询改写器" in content:      # REWRITE_PROMPT
            return json.dumps({"queries": ["改写query甲", "改写query乙"]}, ensure_ascii=False)
        if "问题分解器" in content:      # DECOMPOSE_PROMPT
            return json.dumps({"sub_questions": ["订单现在到哪里了", "退货的话运费谁承担"]},
                              ensure_ascii=False)
        if "答案审核员" in content:      # VERIFY_PROMPT
            return json.dumps({"passed": True, "reason": "mock校验通过"}, ensure_ascii=False)
        if "电商客服问答助手" in content:  # ANSWER / SYNTHESIS prompt
            return "支持七天无理由退货[1]，具体以商品页说明为准。"
        if "电商客服助手" in content:    # CHITCHAT prompt
            return "您好呀，这是 mock 的寒暄回复。"
        return "mock reply"

    _FAKE_CAND = [{"chunk_id": "12_0", "doc_id": 12, "text": "支持七天无理由退货",
                   "rrf_score": 0.0325}]

    def _patch_all():
        svc.call_llm = fake_call_llm
        svc.retrieve_round = lambda q: [dict(c) for c in _FAKE_CAND]
        svc.rerank = lambda q, cands, top_k=5: {
            "results": [{"chunk_id": "12_0", "doc_id": 12, "text": "支持七天无理由退货",
                         "rerank_score": 0.82, "hit_detail": {"vector": 1, "bm25": 2}}],
            "top1_score": 0.82, "passed": True, "degraded": False}

    def _restore_all():
        for name, fn in _ORIG.items():
            setattr(svc, name, fn)

    _ORIG = {n: getattr(svc, n) for n in
             ("call_llm", "retrieve_round", "rerank", "_search_and_rerank", "verify_answer")}

    try:
        _patch_all()

        # ---------- 场景 1：C 类正常流 ----------
        print("\n[场景1] C 类正常流（检索过 → 生成 → Critic 过）")
        llm_counter["n"] = 0
        svc._search_and_rerank = lambda q, trace: {
            "results": [{"chunk_id": "12_0", "doc_id": 12, "text": "支持七天无理由退货",
                         "rerank_score": 0.82}],
            "top1_score": 0.82, "passed": True, "degraded": False}
        r = svc.run_pipeline("支持七天无理由退货吗", [], _MockMatcher())
        check("route=C", r["route"] == "C")
        check("未拒答", r["refused"] is False)
        check("引用含chunk_id", bool(r["citations"]) and r["citations"][0]["chunk_id"] == "12_0")
        check("LLM=3次（Router+生成+Critic）", llm_counter["n"] == 3, f"实际{llm_counter['n']}")

        # ---------- 场景 2：A 类·本地规则命中（限流生命线） ----------
        print("\n[场景2] A 类寒暄·本地规则命中（L0短路+本地回复，全程 0 次 LLM）")
        router_reply["category"] = "A"
        llm_counter["n"] = 0
        r = svc.run_pipeline("在吗", [], _MockMatcher())
        check("route=A", r["route"] == "A")
        check("未拒答", r["refused"] is False)
        check("本地模板回复", "我在的" in r["answer"], r["answer"])
        check("LLM=0次（Router被L0短路，寒暄本地模板）", llm_counter["n"] == 0, f"实际{llm_counter['n']}")
        check("0轮检索", r["rounds"] == 0)

        # ---------- 场景 3：A 类·规则未命中走 LLM ----------
        print("\n[场景3] A 类寒暄·规则未命中 → 走 LLM（兜底链第二层）")
        llm_counter["n"] = 0
        r = svc.run_pipeline("今天天气真不错啊", [], _MockMatcher())
        check("LLM=2次（Router+寒暄生成）", llm_counter["n"] == 2, f"实际{llm_counter['n']}")
        check("未拒答", r["refused"] is False)
        router_reply["category"] = "C"

        # ---------- 场景 4：低置信拒答 ----------
        print("\n[场景4] 低置信拒答（原始 + 2 次改写共 3 轮全不过）")

        def _fake_low_conf(q: str, trace: Dict) -> Dict:
            # 注意：必须自己 append trace——真实 _search_and_rerank 的职责之一就是记现场，
            # mock 绕不过它（这正是"只 mock 出口、不 mock 内部职责"的代价，换来断言可信）
            trace["rounds"].append({"query": q, "candidate_count": 0,
                                    "top1_score": 0.05, "passed": False, "degraded": False})
            return {"results": [], "top1_score": 0.05, "passed": False, "degraded": False}

        svc._search_and_rerank = _fake_low_conf
        r = svc.run_pipeline("这个问题知识库里肯定没有", [], _MockMatcher())
        check("拒答", r["refused"] is True)
        check("reason=low_confidence", r["refuse_reason"] == "low_confidence")
        check("共3轮检索", len(r["trace"]["rounds"]) == 3, f"实际{len(r['trace']['rounds'])}")

        # ---------- 场景 5：Critic 拒答 ----------
        print("\n[场景5] Critic 拒答（证据够但答案不被支撑 → 不重试）")
        svc._search_and_rerank = lambda q, trace: {
            "results": [{"chunk_id": "12_0", "doc_id": 12, "text": "支持七天无理由退货",
                         "rerank_score": 0.9}],
            "top1_score": 0.9, "passed": True, "degraded": False}
        svc.verify_answer = lambda ans, results: {"passed": False,
                                                  "reason": "mock校验失败", "degraded": False}
        r = svc.run_pipeline("退货时效多久", [], _MockMatcher())
        check("reason=verify_failed", r["refused"] and r["refuse_reason"] == "verify_failed")
        check("只1轮（说错话不靠重试救）", r["rounds"] == 1, f"实际{r['rounds']}")
        # 场景专属 patch 用完立即恢复——否则污染后续场景（场景间必须隔离）
        svc.verify_answer = _ORIG["verify_answer"]

        # ---------- 场景 6：E 类越界 ----------
        print("\n[场景6] E 类越界（零检索，合规兜底）")
        router_reply["category"] = "E"
        llm_counter["n"] = 0
        r = svc.run_pipeline("帮我把订单价格改成1元", [], _MockMatcher())
        check("route=E", r["route"] == "E")
        check("reason=out_of_scope", r["refused"] and r["refuse_reason"] == "out_of_scope")
        check("LLM仅1次（只有Router）", llm_counter["n"] == 1, f"实际{llm_counter['n']}")
        check("0轮检索", r["rounds"] == 0)
        router_reply["category"] = "C"

        # ---------- 场景 7：B 类 FAQ 直出 ----------
        print("\n[场景7] B 类 FAQ 直出（不检索不生成）")
        router_reply["category"] = "B"
        r = svc.run_pipeline("支持七天无理由退货吗", [], _HitMatcher())
        check("route=B", r["route"] == "B")
        check("未拒答", r["refused"] is False)
        check("0轮检索", r["rounds"] == 0)
        check("FAQ命中进trace", r["trace"].get("faq_hit") is not None)
        router_reply["category"] = "C"

        # ---------- 场景 8：D 类多跳 ----------
        print("\n[场景8] D 类多跳（拆解 → 分头检索 → 合并 → 统一精排）")
        router_reply["category"] = "D"
        r = svc.run_pipeline("订单还没到，退货的话运费谁出", [], _MockMatcher())
        check("route=D", r["route"] == "D")
        check("拆解出2个子问题", len(r["trace"]["sub_questions"]) == 2)
        check("未拒答", r["refused"] is False)
        check("1轮合并检索(sub_mode)", r["trace"]["rounds"][0].get("sub_mode") is True)
        router_reply["category"] = "C"

    finally:
        _restore_all()

    print(f"\n{'=' * 48}\n结果: {len(_PASS)} 通过 / {len(_FAIL)} 失败")
    if _FAIL:
        print("失败项:", _FAIL)
        raise SystemExit(1)
    print("全部通过 ✅")

