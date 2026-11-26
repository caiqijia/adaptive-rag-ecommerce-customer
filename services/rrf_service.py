# 把向量检索和 BM25 两路召回的结果合并成一个排序列表
from typing import List,Dict
# RRF 平滑系数，论文实证 60 最稳健：
#   调大 → 头部更平滑，单路第 1 名优势被削弱，融合更保守
#   调小 → 单路第 1 名更主导，接近"取各路第一名"
DEFAULT_K = 60

def to_ranked_list(results: List[Dict], score_key: str = "score") -> List[Dict]:
    """
    把任意一路"检索结果"转换成 RRF 输入格式（带 rank 的列表）。
    统一成 {"doc_id", "text", "rank"} 三元组
        results:   一路检索的结果列表，元素需含 doc_id / text / score_key 字段
        score_key: 表示"相关度分数"的字段名，越大代表越相关。
    Returns:[{"doc_id": str, "text": str, "rank": int}, ...]，rank 从 1 开始
    """
    # 防御性重排：正常上游应已按分数降序排好，这里再排一次，
    # 保证即使传入乱序结果，名次也真实反映相关度
    ordered = sorted(results, key=lambda x: x[score_key], reverse=True)

    return [
        {
            # chunk_id 作为唯一标识，字符串化防止类型不一致导致匹配失败
            "chunk_id": str(item["chunk_id"]),
            # 保留doc_id，用于溯源属于哪一份文档
            "doc_id": str(item["doc_id"]),
            "text": item["text"],
            # enumerate 从 0 开始，手动 +1：RRF 论文定义 rank 从 1 起，
            # 第 1 名得分 1/(k+1)。若用 0 起，排序不变但绝对分整体偏移，
            # 会影响日后基于 rrf_score 绝对值的阈值自省
            "rank": idx + 1,
        }
        for idx, item in enumerate(ordered)
    ]


def rrf_fuse(vector_list: List[Dict],bm25_list: List[Dict],k: int = DEFAULT_K,) -> List[Dict]:
    """
    RRF 排名融合：把向量检索与 BM25 检索的两路候选按名次融合排序。
        使用 chunk_id 作为唯一key,同一个doc下多个chunk不会互相覆盖合并
        vector_list: 向量检索的带 rank 列表 [{"doc_id","text","rank"}]
        bm25_list:   BM25 检索的带 rank 列表，格式同上（建议都用 to_ranked_list() 生成，保证 rank/doc_id 规范）
        k:           RRF 平滑系数，默认 60
    Returns:
        按 rrf_score 降序的融合列表：
        [{"chunk_id": str, "doc_id": str, "text": str, "rrf_score": float, "hit_detail": dict}]
        - rrf_score:  两路 1/(k+rank) 的累加总分
        - hit_detail: 该chunk在各路的名次，如 {"vector": 2, "bm25": 1},供 debug_router 可视化召回来源，下游 Rerank 可忽略
    """
    # 累加器：chunk_id → {"doc_id":文档id, "text": 文档文本, "score": RRF 总分, "hits": 各路名次}
    # 核心：chunk_id作为唯一key，同文档不同chunk独立计分，不再互相覆盖
    score_map: Dict[str, Dict] = {}

    # 向量、BM25 两个循环结构完全同构，合并成一次遍历（DRY）。
    for system_name, ranked_list in (("vector", vector_list), ("bm25", bm25_list)):
        for item in ranked_list:
            # 再次兜底转 str：即使上游没用 to_ranked_list，也保证两路 key 可比
            chunk_id = str(item["chunk_id"])
            doc_id = str(item["doc_id"])
            r = item["rank"]
            contribution = 1.0 / (k + r)

            if chunk_id not in score_map:
                # 首次见到该chunk：登记信息，初始化累加器
                score_map[chunk_id] = {
                    "doc_id": doc_id,
                    "text": item["text"],
                    "score": 0.0,
                    "hits": {}
                }

            # FastAPI 的 JSON 序列化会直接报错，这里转成原生 float
            score_map[chunk_id]["score"] += float(contribution)
            score_map[chunk_id]["hits"][system_name] = r

    # 把累加器摊平回 list of dict。
    # 字段与两路检索输出保持同构（doc_id / text / 分数），
    # 下游 Rerank 无需关心分数是哪一路产生的
    fused_result = [
        {
            "chunk_id": chunk_id,
            "doc_id": val["doc_id"],
            "text": val["text"],
            # round 到 6 位：RRF 分数量级约 0.016，6 位小数足够区分
            # 不同名次组合，同时避免浮点尾差干扰日志比对和评测脚本
            "rrf_score": round(float(val["score"]), 6),
            "hit_detail": val["hits"],
        }
        for chunk_id, val in score_map.items()
    ]

    # 按 RRF 总分降序
    fused_result.sort(key=lambda x: x["rrf_score"], reverse=True)
    return fused_result
