"""
rerank_service.py —— Rerank 精排服务（BGE-reranker-v2-m3）

【在检索链路中的位置】
    rrf_fuse() 输出全量融合候选
        │
        ▼
    rerank() ──► top_k 精排结果 ──────────────► 生成 Agent（带引用 [1][2]）
        │
        └─► passed=False（低置信）──► 上层触发：query 重写重检 / 拒答转人工

【模型选型：BAAI/bge-reranker-v2-m3】
    - 中文场景效果好（智源 BGE 系列，专为中文优化）
    - 输出经 sigmoid 归一化到 0~1 的概率 → 可以直接设绝对阈值
    - 本地部署、免费、无 API 依赖、数据不出境

【依赖】pip install FlagEmbedding
      首次运行自动下载约 2.2GB 模型。下载源与超时由 config.py 顶部的
      HF_ENDPOINT（默认 hf-mirror.com 镜像）控制。

【运维要点：模型加载的三种故障场景】
    1. 首次下载慢/断网  → 熔断后走降级（RRF 顺序），服务不瘫
    2. 加载失败反复重试  → 熔断标记，避免每个请求都卡一次下载超时
    3. 多请求并发加载    → 加锁，防止几份 2.2GB 模型同时进内存把机器打爆
    这三条是"本地大模型服务"上线必踩的坑，比模型精度更影响可用性。
"""
import math
import threading
from typing import Dict, List, Optional

from config import (
    RERANK_MODEL_NAME,
    RERANK_DEFAULT_TOP_K,
    RERANK_SCORE_THRESHOLD,
    RERANK_MAX_CANDIDATE,
)
from common.logger import logger

# ============================================================
# 模型单例 + 熔断器（进程级状态）
# ============================================================
# _reranker：加载成功的模型实例。None 表示"还没加载"或"加载失败"。
_reranker = None

# _load_failed：熔断开关。
# 【为什么需要熔断】模型加载失败（断网/磁盘满/显存不足）后，如果每次都重试，
# 每个用户请求都要先等一次下载超时（默认 10s×5 次重试），体验是"永远转圈"。
# 熔断后直接返回 None → 走降级 → 用户 1 秒内拿到 RRF 顺序的答案。
# 【怎么恢复】修好网络/模型后调用 reset_reranker() 或重启服务。
_load_failed = False

# _load_lock：加载锁。
# 【为什么必须加锁】run_pipeline 跑在线程池里，多个并发请求会在不同线程
# 同时执行 get_reranker()。无锁时 3 个请求 = 3 份 2.2GB 模型进内存 →
# 直接 OOM 或长时间 GC 停顿。加锁后只有一个线程真正加载，其余等待复用。
_load_lock = threading.Lock()


def get_reranker() -> Optional[object]:
    """
    获取 Reranker 模型单例（懒加载 + 熔断 + 并发保护）。

    Returns:
        FlagReranker 实例；若已熔断（之前加载失败过）则返回 None，
        调用方据此降级到 RRF 顺序。
    """
    global _reranker, _load_failed

    # 快路径：已加载，直接返回（无锁，避免每次检索都抢锁）
    if _reranker is not None:
        return _reranker

    # 熔断路径：之前失败过，不再重试（快速失败，保护用户体验）
    if _load_failed:
        return None

    with _load_lock:
        # 双检：拿到锁后可能已有其他线程加载完成，避免重复加载
        if _reranker is not None:
            return _reranker
        if _load_failed:
            return None

        # 延迟 import：没装 FlagEmbedding 时不影响项目其他模块运行
        # （这条 import 也是暴露"缺依赖"和"缺模型"两类不同故障的分界点）
        try:
            from FlagEmbedding import FlagReranker
        except ImportError as e:
            logger.error(f"未安装 FlagEmbedding，rerank 将永久降级: {e}")
            _load_failed = True
            return None

        logger.info(f"开始加载 rerank 模型: {RERANK_MODEL_NAME}（首次需下载约2.2GB）")
        try:
            _reranker = FlagReranker(RERANK_MODEL_NAME, use_fp16=False)
            logger.info("rerank 模型加载完成")
            return _reranker
        except Exception as e:
            # 【熔断】此处必须吞异常并置标记，不能让异常冒泡——
            # 否则异常会穿透到 rerank() 的 except 分支，虽然也能降级，
            # 但 _load_failed 没置位，下一个请求又会重新尝试下载。
            _load_failed = True
            logger.error(
                f"rerank 模型加载失败，已熔断（后续请求直接降级）：{e}\n"
                f"  排查指引：\n"
                f"    1) 检查能否访问 HF_ENDPOINT={__import__('os').environ.get('HF_ENDPOINT')}\n"
                f"    2) 或改用 ModelScope 离线下载后设置 RERANK_MODEL_PATH 环境变量\n"
                f"       modelscope download --model BAAI/bge-reranker-v2-m3 --local_dir ./models/bge-reranker-v2-m3",
                exc_info=True,
            )
            return None


def reset_reranker() -> None:
    """
    重置熔断器（不卸载已加载的模型）。

    用途：修好网络 / 下载完模型后，无需重启服务即可重新尝试加载。
    供 debug 接口或运维脚本调用。
    """
    global _load_failed
    _load_failed = False
    logger.info("rerank 熔断器已重置，下次调用将重新尝试加载模型")


def reranker_status() -> Dict:
    """
    查询模型状态（给 debug_router 的 /debug/status 展示用）。

    Returns:
        {"loaded": bool, "circuit_open": bool, "model": str}
        circuit_open=True 表示已熔断，rerank 当前处于降级状态。
    """
    return {
        "loaded": _reranker is not None,
        "circuit_open": _load_failed,
        "model": RERANK_MODEL_NAME,
    }


# ============================================================
# 精排主逻辑
# ============================================================

def _finalize(scored_items: List[Dict], top_k: int, score_threshold: float) -> Dict:
    """
    纯逻辑部分：排序 + top_k 截断 + 低置信判断。不依赖模型，可单测。
    """
    # 按精排分降序。注意：这里覆盖了 RRF 的顺序——
    # 这正是精排的意义：cross-encoder 认为真正相关的文档顶到前面
    ranked = sorted(scored_items, key=lambda x: x["rerank_score"], reverse=True)

    # top1 分数是"自省"的依据：连最高的都没过阈值，说明整体证据不足
    top1_score = ranked[0]["rerank_score"] if ranked else 0.0

    return {
        "results": ranked[:top_k],
        "top1_score": top1_score,
        "passed": top1_score >= score_threshold,
        "degraded": False,
    }


def rerank(query: str, candidates: List[Dict], top_k: int = RERANK_DEFAULT_TOP_K,
           score_threshold: float = RERANK_SCORE_THRESHOLD) -> Dict:
    """
    对 RRF 融合后的候选做 cross-encoder 精排。

    Args:
        query:           本轮检索用的 query（须与召回时用同一个，保证一致性）
        candidates:      rrf_fuse() 的输出列表，元素含
                         chunk_id / doc_id / text / rrf_score / hit_detail
        top_k:           精排后保留几篇交给生成 Agent
        score_threshold: 低置信阈值，top1 低于它判定证据不足

    Returns:
        {
          "results":    [...],   # 按 rerank_score 降序的 top_k 篇
          "top1_score": float | None,  # 最高精排分（降级时为 None）
          "passed":     bool,    # 是否通过置信度检查
          "degraded":   bool,    # 是否走了降级（模型不可用时回退 RRF 顺序）
        }
        results 每篇字段：
          chunk_id / doc_id / text / rerank_score
          rrf_score / hit_detail（原样透传，供 debug_router 展示两阶段全过程）
    """
    # 空候选直接返回，不打模型（省一次前向，也避免空列表送进模型报错）
    if not candidates:
        logger.info("rerank 输入候选为空，直接返回")
        return {
            "results": [],
            "top1_score": 0.0,
            "passed": False,
            "degraded": False,
        }

    # 保护：截断候选数量，防止送入过多 chunk，推理超时、显存溢出
    if len(candidates) > RERANK_MAX_CANDIDATE:
        logger.warning(f"rerank 候选数量 {len(candidates)} 超过上限 {RERANK_MAX_CANDIDATE}，做截断")
        candidates = candidates[:RERANK_MAX_CANDIDATE]

    # 基础清洗：过滤空文本、空白文本，避免 rerank 推理报错
    valid_candidates = []
    for item in candidates:
        text = item.get("text", "").strip()
        if text:
            valid_candidates.append(item)
        else:
            logger.warning(f"chunk_id={item.get('chunk_id')} 文本为空，跳过精排")
    candidates = valid_candidates

    if not candidates:
        logger.info("过滤后无有效候选，直接返回")
        return {
            "results": [],
            "top1_score": 0.0,
            "passed": False,
            "degraded": False,
        }

    try:
        reranker = get_reranker()

        # ---- 模型不可用（已熔断）→ 降级：直接用 RRF 顺序 ----
        # 【设计取舍】降级时 passed=True：
        #   拿不到精排分时不擅自拒答——不能因为辅助组件坏了就误伤正常用户。
        #   误拒（把能答的问题拒掉）比偶尔的不精确更伤体验，且降级状态会
        #   通过 degraded=True 透传到 trace 和日志，问题不会被掩盖。
        if reranker is None:
            logger.warning(f"rerank 模型不可用（熔断中），降级为 RRF 顺序；query={query}")
            return {
                "results": candidates[:top_k],
                "top1_score": None,
                "passed": True,
                "degraded": True,
            }

        logger.info(f"开始精排，query={query}, 有效候选数={len(candidates)}")
        # cross-encoder 的输入是 (query, text) pair 列表，一次前向拿全部分数。
        # 注意这里的 pair 语义：query 和 text 拼接过模型
        pairs = [[query, item["text"]] for item in candidates]
        raw_scores = reranker.compute_score(pairs)

        # 【sigmoid 归一化】bge-reranker 输出的是未归一化的 logit（可能 -10~+10），
        # 经 sigmoid 映射到 0~1 才能当"相关概率"设绝对阈值。
        # 这一步不能省：阈值 0.3 是在归一化尺度上校准出来的，直接用 logit 会让
        # "passed" 的判断完全失真（logit=0.3 对应概率其实是 0.57）。
        norm_scores = [1.0 / (1.0 + math.exp(-float(s))) for s in raw_scores]

        # 组装结果：用 {**item, ...} 生成新 dict，不修改调用方传入的对象（无副作用）
        scored_items = [
            {**item, "rerank_score": round(float(s), 6)}
            for item, s in zip(candidates, norm_scores)
        ]

        result = _finalize(scored_items, top_k, score_threshold)
        logger.info(f"精排完成，top1_score={result['top1_score']:.4f}, passed={result['passed']}")
        return result

    except Exception as e:
        # 推理期异常（模型 OOM / 输入超长 / 依赖冲突）→ 同样降级
        logger.error(f"rerank 推理异常，触发降级！query={query}, err={str(e)}", exc_info=True)
        return {
            "results": candidates[:top_k],
            "top1_score": None,
            "passed": True,
            "degraded": True,
        }
