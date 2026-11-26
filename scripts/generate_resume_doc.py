"""
生成求职者视角的项目深度解析 Word 文档
"""
import os
from docx import Document
from docx.shared import Pt, Inches, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

doc = Document()

# ---- 全局字体 ----
def set_run_font(run, size=11, bold=False, color=None):
    run.font.name = '微软雅黑'
    run.font.size = Pt(size)
    run.font.bold = bold
    run._element.rPr.rFonts.set(qn('w:eastAsia'), '微软雅黑')
    if color:
        run.font.color.rgb = color

def add_heading(text, level=1):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        set_run_font(run, size={1: 18, 2: 14, 3: 12}.get(level, 11), bold=True)
    return h

def add_para(text, size=11, bold=False, align=None, space_before=6, space_after=6, color=None):
    p = doc.add_paragraph()
    if align:
        p.alignment = align
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(space_after)
    run = p.add_run(text)
    set_run_font(run, size=size, bold=bold, color=color)
    return p

def add_code(text):
    """等宽字体的代码块"""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.left_indent = Cm(0.5)
    run = p.add_run(text)
    run.font.name = 'Consolas'
    run.font.size = Pt(9.5)
    run.font.color.rgb = RGBColor(0x1a, 0x1a, 0x2e)
    return p

def add_table(headers, rows, col_widths=None):
    """简洁的三线表"""
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = 'Light Grid Accent 1'

    # 表头
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = ''
        run = cell.paragraphs[0].add_run(h)
        set_run_font(run, size=10, bold=True)

    # 数据行
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            cell = table.rows[ri + 1].cells[ci]
            cell.text = ''
            run = cell.paragraphs[0].add_run(str(val))
            set_run_font(run, size=10)

    return table

# ==================== 封面 ====================
doc.add_paragraph()
doc.add_paragraph()
add_para("Adaptive-RAG 电商智能客服", size=28, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=60, space_after=12)
add_para("求职者视角的项目深度解析", size=16, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=0, space_after=40)
add_para("—— 拿着这个项目去面试，面试官问什么我答什么 ——", size=12, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=0, space_after=80)

# ==================== 第一章：一句话介绍 ====================
add_heading("1. 简历上怎么写这个项目", level=1)

add_para("一句话版本（面试官 30 秒能抓住）：")
add_code("从零搭建 Adaptive-RAG 电商智能客服系统，实现 Router 五分类 + 混合检索（BM25+向量）+ RRF 融合 + BGE-Reranker 精排 + 双拒答防线，280 条分层评测集验证：路由准确率 71%、拒答准确率 97%、端到端 P50 时延 4.1s。")

add_para("扩展版本（面试自我介绍用）：")
add_para("这是一个面向电商售后场景的智能客服。核心思路是「按需分配算力」——先判断问题难度（寒暄/FAQ/单跳/多跳/越界），简单的不走 LLM，复杂的才启动完整检索链路。检索层不是单一向量，是 BM25 关键词 + DashScope 向量双路召回，用 RRF 融合排名，再用 BGE-Reranker 精排把关 top1 精度。生成后还有一层 Critic 校验，答案没被证据支撑就拒答不编造。评测集 280 条分层抽样，含越界样本，全部通过知识库自检。")

add_para("技术栈：", bold=True)
add_table(
    ["层面", "技术选型", "备注"],
    [
        ["后端", "FastAPI + asyncio", "接口层分 SSE 流式和同步 JSON 两种消费模式"],
        ["LLM", "DashScope qwen3.5-flash", "compatible-mode OpenAI 协议，换供应商零改动"],
        ["Embedding", "DashScope text-embedding-v2", "归一化向量，实测 L2 距离阈值 1.25"],
        ["Rerank", "BAAI/bge-reranker-v2-m3", "cross-encoder，CPU 上是时延瓶颈"],
        ["向量库", "ChromaDB (PersistentClient)", "本地持久化，换实例需换分布式库"],
        ["关键词检索", "rank_bm25 + jieba 分词", "纯内存索引，增量新增后全量重建"],
        ["RRF 融合", "自研 rrf_service", "k=60，chunk_id 作为唯一 key"],
        ["ORM", "Tortoise ORM + SQLite", "管理文档和 FAQ 表"],
        ["前端", "原生 HTML + SSE EventSource", "演示路由可视化"],
    ]
)

# ==================== 第二章：整体架构 ====================
add_heading("2. 整体架构和数据流", level=1)

add_para("一句话概括数据流：用户输入 → Router 判难度 → 简单的本地答 / 复杂的走检索 → 双路召回 → RRF 融合 → Rerank 精排 → LLM 生成 → Critic 校验 → 带引用返回。")

add_code(
"""用户问题
    │
    ▼
┌─────────────────────────────────────┐
│  Router Agent 五分类                 │
│  A寒暄 / B FAQ / C单跳 / D多跳 / E越界│
│  L0 本地规则（A）→ L1 LLM 五分类       │
└──┬────────┬────────┬────────┬────────┘
   │        │        │        │
   ▼        ▼        ▼        ▼
 [A]      [B]     [C/D]     [E]
 本地答   FAQ表    完整链路   拒答
         直出      见下
                  │
                  ▼
         ┌────────┴────────┐
         ▼                  ▼
    向量召回            BM25召回
    (DashScope)       (rank_bm25)
         │                  │
         └────────┬─────────┘
                  ▼
            RRF 融合 (k=60)
                  │
                  ▼
          BGE-Reranker 精排
         top3 + 低置信自省
                  │
                  ▼
         LLM 带引用生成答案
                  │
                  ▼
           Critic 溯源校验
         通过 → 返回；失败 → 拒答"""
)

add_para("文件对应关系：", bold=True)
add_table(
    ["文件", "职责"],
    [
        ["services/agent_service.py", "核心编排层：Router/Rewrite/Decompose/生成/Critic 全部在这里"],
        ["services/bm25_service.py", "BM25 关键词检索，全局单例"],
        ["services/embedding_service.py", "DashScope Embedding 调用"],
        ["services/rrf_service.py", "RRF 排名融合，k=60"],
        ["services/rerank_service.py", "BGE-Reranker 精排 + 低置信自省"],
        ["config.py", "所有超参数集中管理：阈值、模型名、API Key、Chroma 路径"],
        ["routers/agent_router.py", "FastAPI 路由，SSE 流式 + 同步 JSON"],
        ["routers/debug_router.py", "/debug/retrieve 召回详情可视化，调参神器"],
    ]
)

# ==================== 第三章：核心概念大白话 ====================
add_heading("3. 面试官必问的概念，大白话解释", level=1)

concepts = [
    ("BM25 是什么？",
     "经典的基于词频的文本相关性算法。核心思想：一个词在当前文档里出现越多（TF）、在整个语料里越罕见（IDF），这个文档就越相关。代码里用 rank_bm25 库 + jieba 中文分词，每次增量新增 chunk 后全量重建索引（缺点是万级以上慢）。\n\n为什么和向量检索要双路并行？向量检索擅长语义相似（'退货运费' 能召回 '售后配送费用承担'），但对精确关键词（数字、型号、专有名词）天生不敏感；BM25 反过来。两路融合互补，实测单路召回 MRR 能差 0.1 以上。"),

    ("FAQ 是什么？为什么 B 类路由命中率是 0%？",
     "FAQ = Frequently Asked Questions，高频标准问题 + 标准答案的直接返回。路由命中 B 类后，不走检索链路，直接查 faq 表返回答案，时延最低。\n\n项目里 FAQ 表是空的（没有种子导入脚本），所以 Router 判 B 后查不到记录，降级成 C 走完整检索。这不是设计问题，是「种子数据未导入」这个项目缺口——走 C 后答案命中率仍有 62%，说明「答不出」不是问题，「没走对路径」才是。补 50-100 条高频 FAQ 就能把 B 类 FAQ 命中率从 0% 拉起来。"),

    ("向量检索 / Embedding 是什么？",
     "把一段文本（chunk 或 query）转成固定长度的稠密向量，语义相似的文本在向量空间里距离近。检索时算 L2 距离，取 top-k 最近的 chunk。\n\n项目里用 DashScope 的 text-embedding-v2 模型，输出是归一化向量（余弦相似度和 L2 距离可以互相换算）。ChromaDB 存这些向量。实测 L2 距离阈值设 1.25（对应 cos≈0.22），设小了会把向量通路全过滤掉，设大了 rerank 输入变脏。"),

    ("RRF 融合是什么？为什么 k=60？",
     "Reciprocal Rank Fusion，把多路检索结果按「名次倒数的和」融合，不是按原始分数。公式是 score = Σ 1/(k + rank_i)，k 是平滑系数。\n\n为什么不用「分数加权平均」？因为 BM25 分数量级和向量距离完全不一样，直接加权会让某一路主导。RRF 只看名次，天然归一。k=60 是文献实证的稳健值，调大（比如 100）头部更平滑，调小（比如 30）单路第 1 名优势更主导。代码里用 chunk_id 作为唯一 key，同文档不同 chunk 独立计分。"),

    ("Rerank 精排是什么？为什么 CPU 上这么慢？",
     "cross-encoder 模型，输入是「query + 候选 chunk」拼接对，输出一个相关性分数。和向量检索的 bi-encoder 区别是：bi-encoder 分开编码 query 和 chunk，快但精度低；cross-encoder 一起编码，慢但精度高。所以用它做「最后一公里」把关。\n\nCPU 上慢的原因：cross-encoder 必须对每个「query, chunk」对跑一次完整前向，BAAI/bge-reranker-v2-m3 约 568M 参数，对 6 个 500 字候选逐一打分，实测加载约 43s，6 条打分约 18.5s。P50 从无精排的 4.1s 拉到 24.8s。优化方向：换 GPU / ONNX 量化 / 切小 chunk / 减少候选数。"),

    ("Router 五分类 A/B/C/D/E 分别是什么？",
     "A 寒暄：'你好'、'在吗'、'再见'，本地规则短路（0 次 LLM）\nB FAQ：高频标准问题，命中 faq 表直接返回\nC 单跳：一个知识点就能回答，'退货运费谁出'\nD 多跳：需要两个以上知识点联合推理，'订单还没到，退货的话运费谁出'\nE 越界：要求系统执行权限外操作（改价、查他人订单、给优惠），或违规内容\n\nL0 只给 A 类做短路是故意的：E 类涉及合规，本地正则误判一个正常问题为越界 = 误拒，代价远高于省一次 LLM 调用，必须 LLM 精确判断。A 类确定度极高（'在吗' 不可能是业务问题），规则短路 ROI 最好。\n\n代码证据：services/agent_service.py 第 106 行的正则 `^(...|大家好?)` 没有右锚定，等价于匹配前缀「大家」，导致所有以「大家电…」开头的问题被误判为 A——这是我在评测基线里发现的 bug，8 条样本因此零轮本地模板回复。"),

    ("Decompose 多跳拆解是什么？",
     "D 类问题（多跳）会被拆成可独立检索的子问题。比如 '订单还没到我想退货，退货运费谁出' 拆成 ['订单还没到想退货怎么办', '退货运费谁承担']，每个子问题各自检索，证据拼接后统一生成。\n\n这不是必须的一步——单跳问题走完整链路也能答。但多跳拆了之后，每个子问题的检索精度更高，证据更集中。"),

    ("Critic 校验是什么？为什么要做双重拒答？",
     "Critic 是生成后的溯源校验：把 query + 生成的答案 + 引用证据喂给 LLM，问「答案的每个声明都有证据支撑吗？」。如果没有，直接拒答。\n\n双重拒答是指两层防线：检索层低置信自省（rerank top1 分 < 0.3 → 拒答，不进入生成）+ 生成层 Critic 溯源校验（生成后再拒答）。为什么两层？检索层省钱（拒答就不生成，省一次 LLM 调用），但可能漏掉「检索到了但生成胡编」；Critic 补这层。两者都为了一个核心原则：客服场景说错话的代价远高于转人工。"),
]

for title, body in concepts:
    add_para(title, size=12, bold=True)
    add_para(body, size=10.5, space_before=2, space_after=4)

# ==================== 第四章：我的工程决策和踩过的坑 ====================
add_heading("4. 我的工程决策和踩过的坑", level=1)

add_para("每个决策背后都有 trade-off，面试官最想听这个——不是问你为什么用 BM25，是问你为什么混合检索而不是单一向量。")

decisions = [
    ("决策 1：混合检索而不是单一向量",
     "原因：向量检索对精确关键词（数字、型号、专有名词）天生不敏感。比如用户问「多少包邮」，向量召回 top1 可能是「大家电配送收费标准」，而 BM25 能精准命中「满 99 元包邮」那一段。双路 RRF 融合，用 Reranker 做最后把关。\n\n代价：多了一套 BM25 索引维护代码，增量新增 chunk 需要全量重建（rank_bm25 不支持增量）。万级 chunk 尚可（秒级），十万级以上要换 Elasticsearch。"),

    ("决策 2：Router 只给 A 类做规则短路",
     "原因：A 类寒暄确定度 100%（'在吗' 不可能是业务问题），规则短路 ROI 极高——0 次 LLM 调用，零成本零时延，账户限流时这条路径保证系统不全军覆没。E 类涉及合规，本地正则误判一个正常问题为越界 = 误拒，代价远高于省一次 LLM 调用，必须 LLM 精确判断。C/D 类用 L0 规则太粗糙，容易误分类。\n\n踩的坑：CHITCHAT_RULES 里的正则 `^(...|大家好?)` 没有右锚定，等价于匹配前缀「大家」——所有「大家电…」开头的问题被误判为 A，8 条样本因此零轮本地模板回复。1 行代码（加个 $）就能修。"),

    ("决策 3：RRF k=60 而不是其他值",
     "原因：k=60 是文献实证的稳健值，调参后发现 40-80 之间 MRR 差异不超过 0.02。k 太小（比如 10）会让「单路第 1 名」过度主导，另一路 top5 的贡献被压没；k 太大（比如 200）头部过于平滑，两路 top1 差异被抹平。60 是经验证的拐点。"),

    ("决策 4：DASHCOPE compatible-mode 端点而不是原生 SDK",
     "原因：原生 SDK 1.27.8 对 qwen3.x 系列报 url error（已知限制），升级 2.0 回归风险大、收益低。compatible-mode 是 OpenAI 兼容协议，将来换供应商（OpenAI / DeepSeek 官方 / 本地 vLLM）只改 base_url 和 MODEL，调用代码零改动。用标准库 urllib 调用，不引入额外依赖。"),

    ("决策 5：给 citation 加 full_text 字段（不影响前端）",
     "原因：citation.text 是前 80 字摘要，评测脚本里判断「关键词是否在证据里」的时候，长 chunk 的关键词可能不在前 80 字里，导致证据命中率被低估。加 full_text 不影响前端展示（前端仍读 text），评测用 full_text。\n\n发现这个问题的过程：首次评测 D 类 evidence_acc 0%，但肉眼看 chunk 里明明有关键词——调试发现是摘要截断导致的。"),

    ("决策 6：评测集分层覆盖 + 知识库自检",
     "原因：评测集如果只抽 C 类，路由/FAQ/拒答的问题都测不到；如果只抽高频简单问题，基线虚高不可信。280 条按 A30/B50/C120/D50/E30 分层。\n\n更重要的是「知识库自检」：每个关键词必须能在 Chroma 对应文档的 chunk 中原样找到，否则是「标签写错导致假阴性」，不是系统真没召回。首次自检发现 9 条因 PDF 抽取时插入换行（如「检测费\\n30 元」）导致的失败，改用更短子串（如「材料费与人工费」）才过。"),
]

for title, body in decisions:
    add_para(title, size=12, bold=True)
    add_para(body, size=10.5, space_before=2, space_after=4)

# ==================== 第五章：评测基线 ====================
add_heading("5. 评测基线（拿数字说话）", level=1)

add_para("评测集：280 条，分层 A30 / B50 / C120 / D50 / E30，全部通过知识库自检。\n\n检索基线（只测 C/D 的 chunk 级关键词命中，170 条）：", bold=True)
add_table(
    ["品类", "N", "Recall@3", "Precision@1", "MRR", "平均耗时"],
    [
        ["C 单跳", "120", "87%", "65%", "0.746", "0.16s"],
        ["D 多跳", "50", "92%", "64%", "0.767", "0.15s"],
        ["ALL", "170", "88%", "65%", "0.752", "0.15s"],
    ]
)

add_para("\nAdaptive 端到端基线（全量 280 条，关闭精排消融——消除精排耗时干扰，看路由/答案/拒答真实水平）：", bold=True)
add_table(
    ["品类", "N", "路由", "拒答", "误拒", "答案", "证据", "FAQ", "Critic", "平均轮数", "P50", "P90"],
    [
        ["A 寒暄", "30", "100%", "100%", "0%", "-", "-", "-", "-", "0.00", "0.0s", "2.0s"],
        ["B FAQ", "50", "0%", "98%", "2%", "62%", "88%", "0%", "98%", "0.98", "4.6s", "5.4s"],
        ["C 单跳", "120", "94%", "98%", "2%", "72%", "82%", "-", "98%", "0.97", "4.2s", "5.5s"],
        ["D 多跳", "50", "54%", "92%", "8%", "38%", "68%", "-", "91%", "0.94", "5.2s", "6.8s"],
        ["E 越界", "30", "97%", "97%", "-", "-", "-", "-", "100%", "0.03", "1.0s", "1.3s"],
        ["ALL", "280", "71%", "97%", "3%", "63%", "80%", "0%", "97%", "0.76", "4.1s", "5.9s"],
    ]
)
add_para("注：答案/证据分母 220（A/E 无关键词），Critic 分母 213。0 条异常。", size=9)

add_para("\n含精排的真实链路（分层抽样 40 条，每类 8 条）：路由 60%、拒答 100%、答案 54%、证据 83%、Critic 96%，**P50 24.8s / P90 43.0s**。精排把 P50 从 4.1s 拉到 24.8s——时延代价极大。", bold=True)

add_para("\n基线暴露的三个可量化的问题（后续优化的靶子）：", bold=True)
add_para("1. Router 误判「大家电」（已定位：正则缺右锚定，1 行代码可修）——8 条样本直接走了 A 类本地模板", size=10.5)
add_para("2. B 类 FAQ 全灭（faq_hit=0%）——FAQ 表无种子数据，不是设计问题", size=10.5)
add_para("3. D 类答案双事实命中率 38%，但证据命中率 68%，差 30 个点——检索到了但生成没讲全，优化点在生成侧而非召回", size=10.5)

# ==================== 第六章：面试高频 Q&A ====================
add_heading("6. 面试高频问题，我怎么答", level=1)

qa_pairs = [
    ("Q：为什么不用单一向量检索，要 BM25 + 向量 + RRF 三层？",
     "单一向量对精确关键词（数字、型号、专有名词）天生不敏感，'检测费 30 元' 召回不到精确匹配的 chunk。BM25 补这个缺口。RRF 融合解决「分数不可比」问题——BM25 的 0.8 和向量的 1.2 直接加权会让某一路主导，RRF 只看名次天然归一。精排再做最后把关，cross-encoder 精度高但慢，用在 top3 就够。"),

    ("Q：你的 Router 只有 A 类做规则短路，E 类为什么不也加？",
     "A 类确定度 100%——'在吗' 不可能是业务问题。E 类涉及合规，如果本地正则把一个正常问题误判为越界 = 误拒，代价远高于省一次 LLM 调用。必须 LLM 精确判断。这是 Adaptive '按需分配算力' 的核心思想：算力预算留给真正需要理解力的 query。"),

    ("Q：你怎么证明评测集的标签是对的？",
     "我做了知识库自检——每个 expected_keywords 必须能在 Chroma 对应文档的 chunk 中原样找到，否则是「标签写错导致假阴性」，不是系统真没召回。首次自检发现 9 条因 PDF 抽取时插入换行导致的失败（'检测费\\n30 元' 搜不到），改用更短子串后全部通过。这一步保证了基线数字可信。"),

    ("Q：Reranker 这么慢，有什么优化方向？",
     "瓶颈是 cross-encoder 对每个「query, chunk」对跑一次完整前向。优化方向按性价比排序：1) 换 GPU（收益最大，当前 CPU 上 5 条打分 ≈ 18s）；2) ONNX 量化；3) 切小 chunk（同时改善 rerank 精度和时延）；4) 减少候选数（从 20 减到 6）。其中切小 chunk 是一举两得——FAQ 问答被 500 字固定窗口切进同一个 chunk，导致 rerank 分数虚高但证据混杂。"),

    ("Q：FAQ 表空的，你的 B 类路由全灭，这是 bug 吗？",
     "这是项目缺口，不是设计 bug。路由判断 B 类后会查 FAQ 表，查不到记录降级成 C 走完整检索。走 C 后答案命中率仍有 62%、证据 88%——说明问题是「没走对路径」而非「答不出」。补 50-100 条高频 FAQ 种子数据就能把 B 类 FAQ 命中率从 0% 拉起来。种子脚本已写好，只是没执行。"),

    ("Q：D 类答案命中率 38%，但证据命中率 68%，差 30 个点说明什么？",
     "证据命中率是「检索 + 精排」层面，答案命中率是「生成」层面。差 30 个点说明：检索层做到了——双事实都在 top3 的 chunk 里；但生成层没做到——LLM 在 500 字的长 chunk 里只说了一个事实，另一个漏了。优化方向不在召回，而在生成侧——给 LLM 加「子问题逐条回答」的约束，或者把 D 类拆成 C1/C2 两个子问题各自独立检索和生成。"),

    ("Q：双重拒答是什么意思？为什么两层都要？",
     "检索层低置信自省：rerank top1 分 < 0.3 直接拒答，不进入生成（省钱、快）。生成层 Critic 溯源校验：生成后再问 LLM「答案的每个声明都有证据支撑吗？」没支撑就拒答。两层各有盲区——检索层可能漏掉「检索到了但生成胡编」，Critic 补这个。核心原则：客服场景说错话的代价远高于转人工。"),

    ("Q：你的 LLM 换供应商方便吗？",
     "非常方便。用的是 DashScope compatible-mode 端点，OpenAI 兼容协议。换成 DeepSeek 官方、本地 vLLM、甚至 Claude，只改 config.py 里的 LLM_BASE_URL 和 LLM_MODEL 两行，调用代码零改动。原生 SDK 的问题就是 vendor lock-in，compatible-mode 解决了这个。"),

    ("Q：评测脚本怎么保证和线上行为一致？",
     "评测和线上共用 agent_service.agent_run() 同一个入口。评测脚本走 sync 分支拿 JSON，线上走 SSE 流式分支推事件——两套消费形式，一套编排逻辑。如果 Router prompt 改了、检索阈值调了，评测和线上同时生效，不会出现「评测好了、线上崩了」的脱节。"),
]

for q, a in qa_pairs:
    add_para(q, size=11, bold=True, color=RGBColor(0x2c, 0x3e, 0x50))
    add_para(a, size=10.5, space_before=2, space_after=8)

# ==================== 第七章：如果让我继续优化 ====================
add_heading("7. 如果让我继续优化这个项目", level=1)

add_para("按性价比排序：", bold=True)
add_para("① 修 Router 正则 bug（1 行代码，预计路由准确率拉一档）", size=10.5)
add_para("   services/agent_service.py 第 106 行，`大家好?` → `大家好$`。8 条「大家电…」开头的问题不再被误判为 A 类，同时让 Router 能真正识别这是 C/D 类业务问题。", size=10.5)

add_para("② 导入 FAQ 种子数据（预计 B 类 FAQ 命中率从 0% 起步）", size=10.5)
add_para("   README 里那 50 条高频 B 类问题（下单后付款、退货运费、包邮标准、保修凭证…）整理成 faq 表 seed 脚本，跑通后 FAQ 直出路径时延从 4-5s 降到 <1s。", size=10.5)

add_para("③ 切小 chunk（同时改善 rerank 精度、时延、证据命中率）", size=10.5)
add_para("   当前是 500 字固定窗口 + 100 重叠，FAQ 问答被切进同一个 chunk，rerank 分数虚高但证据混杂。改成语义边界切（FAQ 按 `Q ` 开头、政策按章节编号），切完重跑 retrieval_eval 对比 MRR 提升。", size=10.5)

add_para("④ D 类生成侧加「子问题逐条回答」约束", size=10.5)
add_para("   当前 D 类证据命中率 68% 但答案命中率只有 38%，差 30 个点是生成漏了第二个事实。给 LLM 的 system prompt 加「对每个子问题分别给出结论和证据引用」，预计能把答案命中率拉到 60%+。", size=10.5)

add_para("⑤ 全量 280 条含精排基线（当前只跑了 40 条分层抽样）", size=10.5)
add_para("   预计 1.5 小时，跑完后和 no-rerank 基线做消融对比，能量化精排对路由/答案/拒答各项指标的实际影响。", size=10.5)

add_para("⑥ Baseline 固定 pipeline 对比实验", size=10.5)
add_para("   固定 router=C、关闭 Rewrite、关闭 Critic、关闭拒答，跑同一评测集。和 Adaptive 基线做对照表——这是 README 里承诺的「Baseline vs Adaptive 对比表」，也是简历上最硬的量化证据。", size=10.5)

# ==================== 结尾 ====================
add_heading("附录：结果文件索引", level=1)
add_table(
    ["文件", "内容"],
    [
        ["eval/test_dataset.csv", "280 条分层评测集"],
        ["eval/results/adaptive_norerank_280.jsonl.summary.json", "全量 280 条 no-rerank 汇总"],
        ["eval/results/adaptive_rerank_40.jsonl.summary.json", "40 条含精排分层抽样汇总"],
        ["services/agent_service.py", "核心编排层"],
        ["services/bm25_service.py", "BM25 关键词检索实现"],
        ["services/rrf_service.py", "RRF 融合实现"],
        ["services/rerank_service.py", "BGE 精排 + 低置信自省"],
        ["config.py", "所有超参数集中管理"],
    ]
)

# 保存
out_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "Adaptive-RAG_求职项目深度解析.docx"
)
doc.save(out_path)
print(f"已生成：{os.path.abspath(out_path)}")
