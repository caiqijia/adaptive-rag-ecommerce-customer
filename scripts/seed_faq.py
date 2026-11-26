"""
FAQ 标准问答导入脚本（离线批量写入 faq 表）

【为什么需要这个脚本】
    `faq` 表是 B 类问题的"高速通道"：FAQMatcher 把每个 question 向量化后常驻内存，
    用户问题算余弦相似度 ≥ FAQ_MATCH_THRESHOLD 就直接返回标准答案，不检索、不生成（<0.5s）。
    表里 0 条时这条通道等于不存在——B 类问法全部降级走 C 类检索链路，
    检索命中不了就回"资料未提及"，这在评测里表现为 B 类准确率被系统性低估。

    数据源用 `data/seed_docs/_src/03_faq_handbook.html` 而不是 PDF：
    PDF 解析能把文字拿回来，但"哪段是问题、哪段是答案"的边界是丢了，
    靠正则从纯文本里猜 Q/A 配对，比直接读结构化 HTML 脆得多。

【为什么 faq 表和知识库 chunk 是两套数据】
    FAQ 手册的 PDF 已经切块进了 Chroma/BM25（C 类检索还能命中它，作为兜底）。
    这里再灌一份进 faq 表，不是重复存储，而是二级加速：
      - B 类：faq 表直出，命中即返回，零检索零生成
      - C 类：检索链路兜底，检索到 FAQ 手册的 chunk 也能答
    两套数据的失效互不影响：删了 faq 表，系统退化成"只有检索"，仍然能跑。

【幂等性】
    按 question 判重：库里已有的问题跳过，可反复执行。
    --reset 先清空 faq 表再全量重建（手册改版后用）。

【用法】
    python scripts/seed_faq.py --dry-run     # 只解析、打印统计，不写库
    python scripts/seed_faq.py               # 导入（跳过已存在）
    python scripts/seed_faq.py --reset       # 清空后重建
    python scripts/seed_faq.py --check       # 导入并跑 B 类命中自检

【灌完之后必须重载缓存】
    正在跑的服务进程里，FAQMatcher 的 _cache 是启动时构建的快照，
    直接改库不会生效。两种办法：
      1. POST /debug/faq/reload  （不重启服务，推荐）
      2. 重启服务（agent_run 首次调用会 preload）
"""
import argparse
import asyncio
import csv
import html
import os
import re
import sys
from collections import Counter

# 让脚本能以 `python scripts/seed_faq.py` 的方式直接运行
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
# config.py 里 Tortoise 连接串是相对路径 "sqlite://ecom_rag.db"，
# 不在项目根目录执行会静默创建一个空库（和 seed_knowledge.py 同一个坑）
os.chdir(BASE_DIR)

from tortoise import Tortoise

from config import TORTOISE_ORM, data_path
from models.faq import FAQ

FAQ_HTML = os.path.join(data_path, "seed_docs", "_src", "03_faq_handbook.html")
EVAL_CSV = os.path.join(BASE_DIR, "eval", "test_dataset.csv")

# 一行 qa 块的结构：<div class="qa"> <div class="q">问题</div> <div class="a">答案</div> </div>
# 用正则而不是 BeautifulSoup，是因为这份手册是自建模板生成的：div 不嵌套、无 <br>、
# 答案里只有 <span class="nb"> 这类行内高亮，结构完全可控。多一个依赖不值。
QA_RE = re.compile(
    r'<div class="qa">\s*<div class="q">(.*?)</div>\s*<div class="a">(.*?)</div>\s*</div>',
    re.S,
)
H2_RE = re.compile(r"<h2>(.*?)</h2>", re.S)


def _clean(fragment: str) -> str:
    """剥掉行内标签、还原 HTML 实体、压掉源码换行造成的多余空白"""
    text = re.sub(r"<[^>]+>", "", fragment)
    text = html.unescape(text)          # &nbsp; → 空格，&amp; → &
    return re.sub(r"\s+", " ", text).strip()


def parse_handbook(path: str) -> list:
    """
    解析 FAQ 手册 HTML，按章节归属返回 [{section, question, answer}]。

    【为什么要记 section】
        faq 表本身只存 question/answer（B 类直出用不到分类），
        但导入时打印分布、以及排查"哪一类问题的答案质量差"时，
        章节是最省事的分组维度，所以解析阶段保留、打印完即丢。
    """
    with open(path, encoding="utf-8") as f:
        src = f.read()

    # 先按 <h2> 把正文切段，就能给每条 qa 打上章节标签
    pieces = H2_RE.split(src)
    items = []
    section = "未分类"
    # split 带捕获组时，结果是 [前言, 标题1, 段1, 标题2, 段2, ...]
    for i in range(1, len(pieces), 2):
        section = _clean(pieces[i])
        for m in QA_RE.finditer(pieces[i + 1]):
            question = _clean(m.group(1))
            answer = _clean(m.group(2))
            items.append({"section": section, "question": question, "answer": answer})
    return items


def print_stats(items: list) -> None:
    """导入前先看一眼：条数、章节分布、字段长度是否触到模型上限"""
    per_section = {}
    for it in items:
        per_section[it["section"]] = per_section.get(it["section"], 0) + 1

    print(f"解析到 {len(items)} 条问答：")
    for sec, cnt in per_section.items():
        print(f"  {sec}: {cnt} 条")

    q_max = max(len(it["question"]) for it in items)
    a_max = max(len(it["answer"]) for it in items)
    print(f"  问题最长 {q_max} 字（模型 max_length=1024），答案最长 {a_max} 字（TextField 无上限）")

    # 问题重复会让 FAQMatcher 的 argmax 变成任取其一（两条都 1.0 分），答案可能互相矛盾
    counter = Counter(it["question"] for it in items)
    dup = [q for q, n in counter.items() if n > 1]
    print(f"  重复问题：{'无' if not dup else dup}")


async def import_items(items: list, reset: bool = False) -> tuple:
    """写入 faq 表，按 question 判重。返回 (新增数, 跳过数)"""
    if reset:
        deleted = await FAQ.all().delete()
        print(f"  [reset] 已清空 faq 表（删除 {deleted} 条）")

    existing = {f.question for f in await FAQ.all()}
    new_items = [it for it in items if it["question"] not in existing]
    skipped = len(items) - len(new_items)

    if new_items:
        # 一次 bulk_create 拼成单条多值 INSERT，比 62 次 create() 少 61 次事务开销
        await FAQ.bulk_create([FAQ(question=it["question"], answer=it["answer"]) for it in new_items])

    print(f"  [ok] 新增 {len(new_items)} 条，跳过已存在 {skipped} 条")
    return len(new_items), skipped


async def check_b_class() -> None:
    """
    B 类命中自检：拿评测集里 50 条真实用户问法去撞 faq 表。

    【为什么重要】
        faq 表灌进去不等于生效。阈值（0.85）是之前按"表里有数据"的假设定的，
        如果手册的问题表述和用户口语差太远，余弦上不去，B 类照样降级。
        这个自检直接给"B 类能被直出的比例"，是这次导入唯一有说服力的验收指标。

    【为什么不调 faq_matcher.match()】
        它内部逐条 embed，50 条 = 50 次 API 往返。这里批量 embed（25 条/批）后
        本地算余弦，2 次请求跑完，且不依赖服务进程里那个可能过期的缓存。
    """
    from services.agent_service import FAQ_MATCH_THRESHOLD, _cosine
    from services.embedding_service import get_batch_embedding

    faqs = await FAQ.all()
    if not faqs:
        print("faq 表为空，跳过自检")
        return
    if not os.path.exists(EVAL_CSV):
        print(f"评测集不存在（{EVAL_CSV}），跳过自检")
        return

    with open(EVAL_CSV, encoding="utf-8-sig") as f:
        queries = [r["question"] for r in csv.DictReader(f) if r["category"] == "B"]

    faq_vecs = get_batch_embedding([f.question for f in faqs])
    q_vecs = get_batch_embedding(queries)

    hits, low = 0, []
    for q, qv in zip(queries, q_vecs):
        scores = [_cosine(qv, fv) for fv in faq_vecs]
        best_i = max(range(len(scores)), key=scores.__getitem__)
        if scores[best_i] >= FAQ_MATCH_THRESHOLD:
            hits += 1
        else:
            low.append((q, faqs[best_i].question, scores[best_i]))

    print(f"\nB 类自检（阈值 {FAQ_MATCH_THRESHOLD}）：{hits}/{len(queries)} 命中 "
          f"= {hits / len(queries) * 100:.1f}%")
    if low:
        print(f"未命中的 {len(low)} 条（其中相似度最高的候选如下，可判断是阈值问题还是手册缺条目）：")
        for q, cand, score in low[:8]:
            print(f"  {score:.3f}  「{q}」 → 最近条目「{cand}」")


async def cmd_list() -> None:
    faqs = await FAQ.all().order_by("id")
    if not faqs:
        print("faq 表为空")
        return
    for f in faqs:
        print(f"{f.id:>3}  {f.question}")
    print(f"\n合计 {len(faqs)} 条")


async def main() -> None:
    parser = argparse.ArgumentParser(description="导入 FAQ 手册问答到 faq 表")
    parser.add_argument("--src", default=FAQ_HTML, help="手册 HTML 路径")
    parser.add_argument("--reset", action="store_true", help="清空 faq 表后全量重建")
    parser.add_argument("--dry-run", action="store_true", help="只解析打印，不写库")
    parser.add_argument("--check", action="store_true", help="导入后跑 B 类命中自检（耗 embedding 额度）")
    parser.add_argument("--list", action="store_true", help="列出表内现有条目后退出")
    args = parser.parse_args()

    if not os.path.exists(args.src):
        print(f"手册不存在：{args.src}")
        return

    items = parse_handbook(args.src)
    print_stats(items)
    print()

    if args.dry_run:
        print("dry-run：未写库。样例 3 条：")
        for it in items[:3]:
            print(f"  [{it['section']}] Q: {it['question']}")
            print(f"      A: {it['answer'][:60]}...")
        return

    await Tortoise.init(config=TORTOISE_ORM)
    await Tortoise.generate_schemas()
    try:
        if args.list:
            await cmd_list()
            return

        await import_items(items, reset=args.reset)
        print(f"  [db]   faq 表当前共 {await FAQ.all().count()} 条")

        if args.check:
            await check_b_class()
        else:
            print("\n提示：正在运行的服务需 POST /debug/faq/reload 才能用上新数据")
    finally:
        await Tortoise.close_connections()


if __name__ == "__main__":
    asyncio.run(main())
