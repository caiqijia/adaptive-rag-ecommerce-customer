"""
FAQ 语义阈值校准（以评测集为标尺扫 FAQ_MATCH_THRESHOLD）

【为什么需要校准】
    FAQMatcher 的命中判定是一条硬阈值：余弦 ≥ 0.85 就直出标准答案，否则降级走检索。
    这个 0.85 是拍脑袋定的，而它的两侧代价完全不对称：
      - 定太高：B 类问法口语一点就撞不上，本该 0.3s 直出的问题退化成一整条检索链路，
                检索不到还会回"资料未提及"——用户体感是"答不上来"
      - 定太低：语义沾边的非 FAQ 问题被硬塞一个标准答案，答非所问比不答更糟
    所以在评测集上把阈值扫一遍，看两个错误率随阈值怎么交换，比凭感觉调有依据。

【口径说明】
    - B 类命中率：50 条里有几条能直出（越高越好）
    - 非 B 误命中率：A/C/D/E 里被 FAQ 表抢走的有几条（越低越好）。
      这部分问题实际应该走检索或寒暄，被直出就是答错。
      注意只有 Router 判成 B 才会进 FAQMatcher，这里按最坏情况统计
      （假设全部都会进），得到的是"上限风险"，不是线上真实误命中率。

【用法】
    python eval/faq_threshold_scan.py             # 全量扫描并打印对比表
    python eval/faq_threshold_scan.py --detail    # 额外列出各阈值下的翻转明细
"""
import argparse
import asyncio
import csv
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
os.chdir(BASE_DIR)

from tortoise import Tortoise

from config import TORTOISE_ORM
from models.faq import FAQ

EVAL_CSV = os.path.join(BASE_DIR, "eval", "test_dataset.csv")
# 扫描区间：低于 0.75 会出现大量"只是同属售后话题"的伪命中，高于 0.90 B 类几乎全灭
THRESHOLDS = [0.75, 0.78, 0.80, 0.82, 0.84, 0.85, 0.86, 0.88, 0.90]


def load_eval() -> list:
    with open(EVAL_CSV, encoding="utf-8-sig") as f:
        return [
            {"id": r["id"], "question": r["question"], "category": r["category"]}
            for r in csv.DictReader(f)
        ]


def best_match(qv, faq_vecs, _cosine) -> tuple:
    """返回 (最高相似度, 对应 faq 下标)"""
    best_i, best_s = 0, -1.0
    for i, fv in enumerate(faq_vecs):
        s = _cosine(qv, fv)
        if s > best_s:
            best_i, best_s = i, s
    return best_s, best_i


async def main() -> None:
    parser = argparse.ArgumentParser(description="扫描 FAQ 语义匹配阈值")
    parser.add_argument("--detail", action="store_true", help="列出阈值区间的翻转明细")
    args = parser.parse_args()

    from services.agent_service import _cosine
    from services.embedding_service import get_batch_embedding

    await Tortoise.init(config=TORTOISE_ORM)
    try:
        faqs = await FAQ.all()
    finally:
        await Tortoise.close_connections()
    if not faqs:
        print("faq 表为空，先跑 scripts/seed_faq.py")
        return

    rows = load_eval()
    print(f"faq 表 {len(faqs)} 条，评测集 {len(rows)} 条，开始向量化...")
    faq_vecs = get_batch_embedding([f.question for f in faqs])
    q_vecs = get_batch_embedding([r["question"] for r in rows])

    # 每条 query 只算一次 top1，之后换阈值只是重新比大小
    scored = []
    for r, qv in zip(rows, q_vecs):
        s, i = best_match(qv, faq_vecs, _cosine)
        scored.append({**r, "score": s, "faq_q": faqs[i].question})

    b = [x for x in scored if x["category"] == "B"]
    non_b = [x for x in scored if x["category"] != "B"]

    print(f"\n{'阈值':<7}{'B 命中率':<12}{'非B 误命中':<12}{'误命中明细(A/C/D/E)'}")
    print("-" * 72)
    for t in THRESHOLDS:
        b_hit = sum(1 for x in b if x["score"] >= t)
        mis = [x for x in non_b if x["score"] >= t]
        per_cat = {}
        for x in mis:
            per_cat[x["category"]] = per_cat.get(x["category"], 0) + 1
        cat_str = " ".join(f"{k}:{v}" for k, v in sorted(per_cat.items())) or "-"
        print(f"{t:<7}{b_hit:>3}/{len(b)} = {b_hit / len(b) * 100:>4.0f}%   "
              f"{len(mis):>3}/{len(non_b)} = {len(mis) / len(non_b) * 100:>4.1f}%   {cat_str}")

    # B 类分布：一眼看出阈值卡掉了哪些"其实问法正确"的条目
    print(f"\nB 类相似度分布（{len(b)} 条，200 分位）：")
    scores = sorted(x["score"] for x in b)
    for p in (0, 10, 25, 50, 75, 90, 100):
        idx = min(int(len(scores) * p / 100), len(scores) - 1)
        print(f"  P{p:<3} {scores[idx]:.3f}")

    if args.detail:
        print("\n非 B 被高分抢走的问题（相似度 ≥0.80，按分数降序）：")
        for x in sorted([x for x in non_b if x["score"] >= 0.80],
                        key=lambda y: -y["score"]):
            print(f"  {x['score']:.3f} [{x['category']}] {x['question']} → {x['faq_q']}")
        print("\nB 类未达 0.85 的（候选条目可人工核对是否本来就该命中）：")
        for x in sorted([x for x in b if x["score"] < 0.85], key=lambda y: -y["score"]):
            print(f"  {x['score']:.3f} {x['question']} → {x['faq_q']}")


if __name__ == "__main__":
    asyncio.run(main())
