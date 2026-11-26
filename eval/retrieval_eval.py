"""
检索质量评测脚本（Step 0 baseline 的起点）

【为什么单独写这个脚本，而不是直接问前端】
    前端只能看单条 query 的过程，回答"这套检索到底好不好"需要**批量、可复现、有指标**。
    "感觉还行"不是工程结论；Recall@k / MRR / 平均耗时才是能写进简历、能和优化后对比的数字。

【它测什么】
    只测检索链路（双路召回 → RRF → rerank），不生成答案。
    生成质量受 LLM 随机性影响，检索质量是确定的——把变量拆开，优化才有方向。

【怎么判定命中】
    每条用例带 expected_keywords（多个用 | 分隔，命中任意一个即算命中）。
    这是"弱标注"：不要求 chunk 完全等于标准答案，只要求它包含能回答问题的那句话。
    对 500 字粗粒度 chunk 来说，这是唯一现实的判法。

【用法】
    python eval/retrieval_eval.py                       # 内置 12 条冒烟用例（含 rerank）
    python eval/retrieval_eval.py --no-rerank           # 只测 RRF 召回（快 10 倍，适合调参）
    python eval/retrieval_eval.py --csv eval/test_dataset.csv
    python eval/retrieval_eval.py --top-k 5

【CSV 格式】（列名固定，顺序无关）
    question,category,expected_keywords,expected_doc
    "衣服买大了能退吗",C,"同款不同尺码|七天无理由",退货与退款政策.pdf
    expected_keywords 用 | 分隔；expected_doc 可留空（留空则只算 chunk 级命中）

【求职考点】
    Recall@k 与 Precision@1 要分开看：
      - Recall@3 低 = 检索没找到，生成再强也没用（召回瓶颈）
      - Recall@3 高但 Precision@1 低 = 找到了但排不上去，靠 rerank 解决（排序瓶颈）
    这两个指标指向完全不同的优化动作，混在一起看会误判。
"""
import argparse
import asyncio
import csv
import logging
import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
# config 里 Tortoise/Chroma 是相对路径，必须在项目根执行
os.chdir(BASE_DIR)

# 内置冒烟用例：每条覆盖一个真实咨询场景，期望关键词取自知识库原文的独特片段
# （选"同款不同尺码"这种只在一处出现的短语，避免用"退货"这类到处都有的词导致假命中）
BUILTIN_CASES = [
    ("衣服买大了能退吗", "C", "同款不同尺码", "退货与退款政策.pdf"),
    ("多少钱包邮", "C", "满 99 元包邮", "配送与物流服务说明.pdf"),
    ("生鲜坏了怎么赔", "C", "坏果比例退款", "常见问题FAQ手册.pdf"),
    ("空调安装要另外收费吗", "C", "80 元/米", "配送与物流服务说明.pdf"),
    ("退款一般多久到账", "C", "1 至 3 个工作日", "退货与退款政策.pdf"),
    ("内衣拆封了还能退吗", "C", "不支持七天无理由退货", "常见问题FAQ手册.pdf"),
    ("积分怎么抵扣", "C", "100 积分抵 1 元", "常见问题FAQ手册.pdf"),
    ("定制的窗帘尺寸不对能退吗", "C", "定制商品", "退货与退款政策.pdf"),
    ("赠品需要退回去吗", "C", "赠品标价", "退货与退款政策.pdf"),
    ("换货运费谁出", "C", "首次换货免运费", "常见问题FAQ手册.pdf"),
    ("新疆能发货吗", "C", "199 元包邮", "配送与物流服务说明.pdf"),
    ("快递丢件怎么赔", "C", "补发同款", "配送与物流服务说明.pdf"),
]


def load_cases(csv_path: str):
    """从 CSV 读用例；期望关键词按 | 拆成列表

    A/E 类没有 expected_keywords，检索评测会跳过；C/D 是主要评测对象。
    """
    cases = []
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            q = (row.get("question") or "").strip()
            kw = (row.get("expected_keywords") or "").strip()
            if not q or not kw:
                continue
            cases.append({
                "id": (row.get("id") or "").strip(),
                "question": q,
                "category": (row.get("category") or "").strip() or "-",
                "keywords": [k.strip() for k in kw.split("|") if k.strip()],
                "expected_doc": (row.get("expected_doc") or "").strip(),
            })
    return cases


def rebuild_bm25() -> int:
    """
    从 Chroma 重建 BM25 内存索引。

    服务里这步由 main.py 的 lifespan 做；脚本是独立进程，必须自己来——
    否则 BM25 是空索引，双路召回会静默退化成向量单路，测出来的数字全是错的。
    """
    from config import collection
    from services.bm25_service import bm25_service

    snapshot = collection.get(include=["documents", "metadatas"])
    for cid, text, meta in zip(snapshot["ids"], snapshot["documents"], snapshot["metadatas"]):
        meta = meta or {}
        bm25_service.add_chunk(text, {"doc_id": meta.get("doc_id"), "chunk_id": cid})
    return len(snapshot["ids"])


def _summarize(details):
    n = len(details)
    if not n:
        return {"n": 0, "recall": 0.0, "precision1": 0.0, "mrr": 0.0, "avg_cost": 0.0}
    return {
        "n": n,
        "recall": sum(1 for d in details if d["rank"]) / n,
        "precision1": sum(1 for d in details if d["rank"] == 1) / n,
        # MRR：正确结果排第 1 得 1 分、第 2 得 1/2……没命中得 0，取平均。
        "mrr": sum(1.0 / d["rank"] for d in details if d["rank"]) / n,
        "avg_cost": sum(d["cost"] for d in details) / n,
    }


def evaluate(cases, top_k: int, use_rerank: bool):
    """跑全部用例，返回整体指标、分品类指标与逐条明细"""
    from services.agent_service import retrieve_round, _search_and_rerank

    details = []
    for c in cases:
        question = c["question"]
        t0 = time.time()
        if use_rerank:
            trace = {"rounds": []}
            rr = _search_and_rerank(question, trace)
            results = rr.get("results", [])[:top_k]
            top1_score = rr.get("top1_score")
            passed = rr.get("passed")
            candidates = trace["rounds"][0]["candidate_count"] if trace["rounds"] else 0
        else:
            results = retrieve_round(question)[:top_k]
            top1_score, passed, candidates = None, None, len(results)

        rank = None
        for i, r in enumerate(results, 1):
            if any(k in r["text"] for k in c["keywords"]):
                rank = i
                break

        details.append({
            "id": c["id"],
            "question": question,
            "category": c["category"],
            "rank": rank,
            "top1_score": top1_score,
            "passed": passed,
            "candidates": candidates,
            "cost": time.time() - t0,
        })

    by_cat = {}
    for cat in sorted({d["category"] for d in details}):
        by_cat[cat] = _summarize([d for d in details if d["category"] == cat])
    return {
        "overall": _summarize(details),
        "by_category": by_cat,
        "details": details,
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description="检索质量评测（Recall@k / Precision@1 / MRR）")
    parser.add_argument("--csv", help="评测集 CSV 路径；不传则用内置 12 条冒烟用例")
    parser.add_argument("--top-k", type=int, default=3, help="看前 k 条（默认 3，与 RERANK_TOP_K 一致）")
    parser.add_argument("--no-rerank", action="store_true", help="只测 RRF 召回，跳过精排（快很多）")
    parser.add_argument("--categories", default="", help="只看指定品类，逗号分隔，如 C,D")
    parser.add_argument("-v", "--verbose", action="store_true", help="打印每条命中的原文片段")
    args = parser.parse_args()

    if args.csv:
        if not os.path.exists(args.csv):
            print(f"评测集不存在：{args.csv}")
            return
        cases = load_cases(args.csv)
    else:
        cases = [{"id": "", "question": q, "category": c, "keywords": [k],
                  "expected_doc": doc}
                 for q, c, k, doc in BUILTIN_CASES]

    if args.categories:
        wanted = {x.strip() for x in args.categories.split(",") if x.strip()}
        cases = [c for c in cases if c["category"] in wanted]

    if not cases:
        print("没有可用用例（CSV 为空、缺少 question / expected_keywords 列，或品类过滤为空）")
        return

    logging.disable(logging.INFO)   # BM25 每个 chunk 打一行，会淹掉结果

    n_chunk = rebuild_bm25()
    print(f"知识库 {n_chunk} 个 chunk | 用例 {len(cases)} 条 | "
          f"{'含 rerank' if not args.no_rerank else '纯召回（无 rerank）'}\n")

    metrics = evaluate(cases, args.top_k, not args.no_rerank)

    for d in metrics["details"]:
        mark = "OK  " if d["rank"] else "MISS"
        score = f"{d['top1_score']:.2f}" if d["top1_score"] is not None else " -- "
        pos = f"第{d['rank']}位" if d["rank"] else "未命中"
        print(f"{mark} [{d['category']}] top1={score} 候选={d['candidates']:<3} "
              f"{d['cost']:>4.1f}s  {d['question']}  → {pos}")

    print(f"\n{'='*64}")
    print(f"{'品类':<4} {'N':>4} {'Recall@'+str(args.top_k):>12} {'Precision@1':>12} {'MRR':>7} {'耗时':>8}")
    for cat, m in metrics["by_category"].items():
        hit = sum(1 for d in metrics["details"] if d["category"] == cat and d["rank"])
        print(f"{cat:<4} {m['n']:>4} {m['recall']:>11.0%} {m['precision1']:>12.0%} "
              f"{m['mrr']:>7.3f} {m['avg_cost']:>7.2f}s  ({hit}/{m['n']})")
    m = metrics["overall"]
    hit = sum(1 for d in metrics["details"] if d["rank"])
    print(f"{'ALL':<4} {m['n']:>4} {m['recall']:>11.0%} {m['precision1']:>12.0%} "
          f"{m['mrr']:>7.3f} {m['avg_cost']:>7.2f}s  ({hit}/{m['n']})")

    miss = [d for d in metrics["details"] if not d["rank"]]
    if miss:
        print(f"\n未命中 {len(miss)} 条：")
        for d in miss:
            print(f"  - [{d['category']}] {d['question']}")


if __name__ == "__main__":
    asyncio.run(main())
