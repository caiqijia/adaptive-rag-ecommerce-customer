"""
Adaptive 端到端基线评测

和 retrieval_eval.py 的区别：
  - retrieval_eval 只测“召回+精排”，不调 LLM；
  - 本脚本调用 agent_service.agent_run()，走完整的 Router → 分治 → 检索 → 生成 → Critic，
    因此能产出路由准确率、拒答双指标、答案关键词命中、平均轮数、端到端时延。

指标口径：
  - route_acc      : 实际路由 == 期望路由（B 未命中 FAQ 会记 C，按严格口径算错）
  - refusal_acc    : 是否拒答 == 期望是否拒答
  - false_refusal  : 本应回答却被拒的比例（误拒率）
  - missed_refusal : 本应拒答却回答的比例（漏拒率）
  - answer_acc     : 答案文本命中 expected_keywords 的比例（D 需同时命中 min_hits 个）
  - evidence_acc   : 引用 chunk 命中 expected_keywords 的比例（区分“没检索到”与“生成了但没说”）
  - verify_pass    : Critic 校验通过率（None 表示该路径没有走 Critic，不计入分母）
  - avg_rounds     : 平均检索轮数
  - latency        : 端到端耗时（含 LLM、精排）

用法：
    python eval/adaptive_eval.py --csv eval/test_dataset.csv
    python eval/adaptive_eval.py --csv eval/test_dataset.csv --no-rerank
    python eval/adaptive_eval.py --csv eval/test_dataset.csv --per-category 20 --out eval/results/adaptive_100.jsonl
    python eval/adaptive_eval.py --csv eval/test_dataset.csv --per-category 20 --resume

注意：
  - 全量 280 条 + CPU 精排（约 18s/条）会非常久；建议先用 --per-category 分层抽样跑基线。
  - --no-rerank 是消融基线：跳过精排、passed 恒为 True，因此没有“低置信拒答”，只剩 Critic 拒答。
"""
import argparse
import asyncio
import csv
import json
import logging
import os
import random
import sys
import time
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
os.chdir(BASE_DIR)


def load_cases(path):
    cases = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            q = (row.get("question") or "").strip()
            if not q:
                continue
            kw = (row.get("expected_keywords") or "").strip()
            cases.append({
                "id": (row.get("id") or "").strip(),
                "question": q,
                "category": (row.get("category") or "").strip() or "-",
                "keywords": [k.strip() for k in kw.split("|") if k.strip()],
                "expected_doc": (row.get("expected_doc") or "").strip(),
                "expected_route": (row.get("expected_route") or row.get("category") or "").strip(),
                "expected_refusal": int((row.get("expected_refusal") or "0").strip() or 0),
                "min_hits": int((row.get("min_hits") or "1").strip() or 1),
            })
    return cases


def sample_cases(cases, per_category, seed):
    if not per_category or per_category <= 0:
        return cases
    rng = random.Random(seed)
    by_cat = defaultdict(list)
    for c in cases:
        by_cat[c["category"]].append(c)
    out = []
    for cat in sorted(by_cat):
        items = list(by_cat[cat])
        rng.shuffle(items)
        out.extend(items[:per_category])
    return out


def rebuild_bm25():
    from config import collection
    from services.bm25_service import bm25_service
    snapshot = collection.get(include=["documents", "metadatas"])
    for cid, text, meta in zip(snapshot["ids"], snapshot["documents"], snapshot["metadatas"]):
        bm25_service.add_chunk(text, {"doc_id": (meta or {}).get("doc_id"), "chunk_id": cid})
    return len(snapshot["ids"])


def install_no_rerank():
    """消融：跳过 cross-encoder，直接按 RRF 顺序取 top_k；passed 恒为 True。"""
    import services.agent_service as A

    def fake_rerank(query, candidates, top_k=3):
        results = sorted(candidates, key=lambda x: x.get("rrf_score", 0), reverse=True)[:top_k]
        return {"results": results, "top1_score": None, "passed": True, "degraded": True}

    A.rerank = fake_rerank


def keywords_in(text, keywords):
    return [k for k in keywords if k in (text or "")]


async def run_case(case):
    from services.agent_service import agent_run

    t0 = time.time()
    try:
        result = await agent_run(case["question"])
        latency = time.time() - t0
    except Exception as e:  # 单条失败不终止整批评测
        return {
            **{k: case[k] for k in ("id", "question", "category", "keywords",
                                    "expected_route", "expected_refusal", "min_hits")},
            "error": repr(e), "latency": time.time() - t0,
            "route": None, "trace_route": None, "route_ok": False,
            "refused": None, "refuse_reason": None, "refusal_ok": False,
            "out_of_scope_ok": False,
            "answer": "", "rounds": 0, "citations": [], "trace": {},
            "answer_hits": [], "evidence_hits": [],
            "answer_ok": None if not case["keywords"] else False,
            "evidence_ok": None if not case["keywords"] else False,
            "faq_hit": False, "verify_passed": None, "trace_rounds": [],
            "last_top1": None, "sub_mode": False, "used_citations": [],
        }

    answer = result.get("answer") or ""
    citations = result.get("citations") or []
    # citation.text 是 80 字摘要；评测命中必须看 full_text，否则长 chunk 会被误判为未命中
    cite_text = " ".join((c.get("full_text") or c.get("text") or "") for c in citations)
    keywords = case["keywords"]
    min_hits = case["min_hits"] or 1
    answer_hits = keywords_in(answer, keywords)
    evidence_hits = keywords_in(cite_text, keywords)
    trace = result.get("trace") or {}
    trace_rounds = trace.get("rounds") or []
    verify = trace.get("verify")
    faq_hit = bool(trace.get("faq_hit"))

    return {
        "id": case["id"],
        "question": case["question"],
        "category": case["category"],
        "expected_route": case["expected_route"],
        "route": result.get("route"),
        "trace_route": trace.get("route"),
        "route_ok": result.get("route") == case["expected_route"],
        "expected_refusal": case["expected_refusal"],
        "refused": bool(result.get("refused")),
        "refuse_reason": result.get("refuse_reason"),
        "refusal_ok": bool(result.get("refused")) == bool(case["expected_refusal"]),
        "out_of_scope_ok": (not case["expected_refusal"]) or
                           (result.get("refuse_reason") == "out_of_scope"),
        "keywords": keywords,
        "answer_hits": answer_hits,
        "evidence_hits": evidence_hits,
        "answer_ok": (len(answer_hits) >= min_hits) if keywords else None,
        # FAQ 直出没有检索 chunk，标准答案本身就是证据，直接记命中
        "evidence_ok": True if (faq_hit and keywords) else
                       ((len(evidence_hits) >= min_hits) if keywords else None),
        "faq_hit": faq_hit,
        "verify_passed": None if verify is None else bool(verify.get("passed")),
        "rounds": result.get("rounds", 0),
        "trace_rounds": trace_rounds,
        "last_top1": trace_rounds[-1].get("top1_score") if trace_rounds else None,
        "sub_mode": any(r.get("sub_mode") for r in trace_rounds),
        "citations": citations,
        "used_citations": result.get("used_citations") or [],
        "latency": latency,
        "error": None,
    }


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else 0.0


def _rate(vals):
    """分母为空时返回 None，避免 A/E 类把“不适用”显示成 0%。"""
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def _pct(vals):
    vals = [v for v in vals if v is not None]
    return f"{_mean(vals):.0%}" if vals else "-"


def summarize(rows):
    n = len(rows)
    if not n:
        return {}
    answer_rows = [r for r in rows if r.get("answer_ok") is not None]
    evidence_rows = [r for r in rows if r.get("evidence_ok") is not None]
    verify_rows = [r for r in rows if r.get("verify_passed") is not None]
    expected_no_refuse = [r for r in rows if not r["expected_refusal"]]
    expected_refuse = [r for r in rows if r["expected_refusal"]]
    lat = sorted(r["latency"] for r in rows)
    return {
        "n": n,
        "errors": sum(1 for r in rows if r.get("error")),
        "route_acc": _rate([1.0 if r["route_ok"] else 0.0 for r in rows]),
        "refusal_acc": _rate([1.0 if r["refusal_ok"] else 0.0 for r in rows]),
        "false_refusal": _rate([1.0 if r["refused"] else 0.0 for r in expected_no_refuse]),
        "missed_refusal": _rate([0.0 if r["refused"] else 1.0 for r in expected_refuse]),
        "out_of_scope_acc": _rate([1.0 if r["out_of_scope_ok"] else 0.0
                                   for r in expected_refuse]),
        "answer_acc": _rate([1.0 if r["answer_ok"] else 0.0 for r in answer_rows]),
        "evidence_acc": _rate([1.0 if r["evidence_ok"] else 0.0 for r in evidence_rows]),
        "faq_hit": _rate([1.0 if r["faq_hit"] else 0.0
                          for r in rows if r["category"] == "B"]),
        "verify_pass": _rate([1.0 if r["verify_passed"] else 0.0 for r in verify_rows]),
        "avg_rounds": _mean([r["rounds"] for r in rows]),
        "avg_latency": _mean([r["latency"] for r in rows]),
        "p50_latency": lat[len(lat) // 2],
        "p90_latency": lat[min(len(lat) - 1, int(len(lat) * 0.9))],
        "_answer_n": len(answer_rows),
        "_evidence_n": len(evidence_rows),
        "_verify_n": len(verify_rows),
        "_no_refuse_n": len(expected_no_refuse),
        "_refuse_n": len(expected_refuse),
    }


def print_summary(rows):
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)

    print(f"\n{'='*118}")
    print(f"{'品类':<5}{'N':>4}{'错误':>5}{'路由':>7}{'拒答':>7}{'误拒':>7}"
          f"{'答案':>7}{'证据':>7}{'FAQ':>7}{'Critic':>8}{'轮数':>6}{'P50':>8}{'P90':>8}")
    order = ["A", "B", "C", "D", "E"]
    for cat in order + [c for c in sorted(by_cat) if c not in order]:
        if cat not in by_cat:
            continue
        m = summarize(by_cat[cat])
        print(f"{cat:<5}{m['n']:>4}{m['errors']:>5}{_pct([m['route_acc']]):>7}"
              f"{_pct([m['refusal_acc']]):>7}{_pct([m['false_refusal']]):>7}"
              f"{_pct([m['answer_acc']]):>7}{_pct([m['evidence_acc']]):>7}"
              f"{_pct([m['faq_hit']]) if m['faq_hit'] is not None else '-':>7}"
              f"{_pct([m['verify_pass']]) if m['_verify_n'] else '-':>8}"
              f"{m['avg_rounds']:>6.2f}{m['p50_latency']:>7.1f}s{m['p90_latency']:>7.1f}s")
    m = summarize(rows)
    print(f"{'ALL':<5}{m['n']:>4}{m['errors']:>5}{_pct([m['route_acc']]):>7}"
          f"{_pct([m['refusal_acc']]):>7}{_pct([m['false_refusal']]):>7}"
          f"{_pct([m['answer_acc']]):>7}{_pct([m['evidence_acc']]):>7}"
          f"{_pct([m['faq_hit']]) if m['faq_hit'] is not None else '-':>7}"
          f"{_pct([m['verify_pass']]) if m['_verify_n'] else '-':>8}"
          f"{m['avg_rounds']:>6.2f}{m['p50_latency']:>7.1f}s{m['p90_latency']:>7.1f}s")
    print(f"答案分母={m['_answer_n']} 证据分母={m['_evidence_n']} "
          f"Critic分母={m['_verify_n']} 应答={m['_no_refuse_n']} 应拒={m['_refuse_n']}")

    # 失败样本按“路有错 > 拒答错 > 答案错 > 证据错”排序，只打前 30 条
    def bad(r):
        return (not r["route_ok"], not r["refusal_ok"], not r["out_of_scope_ok"],
                r["answer_ok"] is False, r["evidence_ok"] is False, bool(r.get("error")))

    failures = [r for r in rows if any(bad(r))]
    failures.sort(key=bad, reverse=True)
    if failures:
        print(f"\n失败样本（前 30 / 共 {len(failures)}）：")
        for r in failures[:30]:
            flags = []
            if not r["route_ok"]:
                flags.append(f"route {r['route']}≠{r['expected_route']}")
            if not r["refusal_ok"]:
                flags.append(f"refused={r['refused']} expected={r['expected_refusal']}")
            if r["answer_ok"] is False:
                flags.append("answer_miss")
            if r["evidence_ok"] is False:
                flags.append("evidence_miss")
            if r.get("error"):
                flags.append(f"error={r['error'][:60]}")
            print(f"  - [{r['category']}] {r['question']}  | " + ", ".join(flags))


async def main():
    parser = argparse.ArgumentParser(description="Adaptive 端到端基线评测")
    parser.add_argument("--csv", default=os.path.join(BASE_DIR, "eval", "test_dataset.csv"))
    parser.add_argument("--categories", default="", help="逗号分隔，如 A,B,C,D,E")
    parser.add_argument("--per-category", type=int, default=0, help="每类抽 N 条；0=全量")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-rerank", action="store_true", help="消融：跳过精排")
    parser.add_argument("--out", default=os.path.join(BASE_DIR, "eval", "results", "adaptive_baseline.jsonl"))
    parser.add_argument("--resume", action="store_true", help="跳过 out 中已有 id")
    parser.add_argument("--sleep", type=float, default=0.2, help="每条之间休眠，避开限流")
    args = parser.parse_args()

    cases = load_cases(args.csv)
    if args.categories:
        wanted = {x.strip() for x in args.categories.split(",") if x.strip()}
        cases = [c for c in cases if c["category"] in wanted]
    cases = sample_cases(cases, args.per_category, args.seed)
    if not cases:
        print("没有用例")
        return

    logging.disable(logging.INFO)

    # Tortoise：FAQMatcher.preload() 需要 FAQ 表
    from tortoise import Tortoise
    from config import TORTOISE_ORM
    await Tortoise.init(config=TORTOISE_ORM)

    n_chunk = rebuild_bm25()
    if args.no_rerank:
        install_no_rerank()
        mode = "no-rerank（消融）"
    else:
        from services.rerank_service import get_reranker
        t0 = time.time()
        get_reranker()   # 预热，避免把 20s 加载算进第一条
        mode = f"rerank（预热 {time.time() - t0:.1f}s）"

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    done_ids = set()
    if args.resume and os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["id"])
                except Exception:
                    pass

    print(f"知识库 {n_chunk} chunk | 用例 {len(cases)} 条 | {mode} | 结果 {args.out}")
    if done_ids:
        print(f"resume：已有 {len(done_ids)} 条，跳过")

    rows = []
    with open(args.out, "a" if args.resume else "w", encoding="utf-8") as f:
        for i, case in enumerate(cases, 1):
            if case["id"] in done_ids:
                continue
            row = await run_case(case)
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            flag = "ERR " if row.get("error") else ("OK  " if row["route_ok"] else "ROUTE")
            print(f"[{i}/{len(cases)}] {flag} [{row['category']}] "
                  f"route={row['route']} refused={row['refused']} "
                  f"rounds={row['rounds']} {row['latency']:.1f}s {row['question']}")
            if args.sleep:
                await asyncio.sleep(args.sleep)

    # resume 时把旧结果也读回来一起汇总
    if args.resume and os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]

    print_summary(rows)

    summary = {
        "mode": mode,
        "csv": args.csv,
        "per_category": args.per_category,
        "n": len(rows),
        "overall": summarize(rows),
        "by_category": {cat: summarize([r for r in rows if r["category"] == cat])
                        for cat in sorted({r["category"] for r in rows})},
    }
    summary_path = args.out + ".summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n汇总已写入 {summary_path}")

    await Tortoise.close_connections()


if __name__ == "__main__":
    asyncio.run(main())
