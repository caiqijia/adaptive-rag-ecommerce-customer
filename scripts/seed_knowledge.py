"""
知识库种子数据导入脚本（离线批量导入，不走 HTTP）

【为什么需要这个脚本】
    doc_router 的 /doc/upload_pdf 是给前端用的，一次只能传一个文件、必须先把服务跑起来。
    搭演示环境、做评测、给别人复现项目时，这些前提都很麻烦。
    这个脚本直接调用 service 层，一条命令把 data/seed_docs/ 下的 PDF 全部灌进知识库：
        DB 元数据（Tortoise） + 向量库（Chroma） + 内存索引（BM25）
    三处一次写齐，避免"向量库有、BM25 没有"的链路断层（那会让 RRF 融合直接失效）。

【幂等性】
    默认按 file_name 判重：已导入过的文件跳过，可反复执行。
    --reset 先删同名文档再重新导入（改了 PDF 内容后用）。

【用法】
    python scripts/seed_knowledge.py            # 导入 seed_docs 下全部 PDF（跳过已存在）
    python scripts/seed_knowledge.py --reset    # 删除同名文档后重新导入
    python scripts/seed_knowledge.py --list     # 只看当前库里有哪些文档
    python scripts/seed_knowledge.py --delete 1 2   # 按 doc_id 删除（清理测试残留）

【求职考点】
    数据导入的"事务性"是面试常被追问的点。这里 Chroma 和 SQLite 是两个独立存储，
    没有跨库事务，所以顺序是「先写 DB 拿 id → 再写 Chroma → 最后写 BM25」，
    任何一步失败都能靠 id 定位并清理，不会留下无法追踪的孤儿数据。
"""
import argparse
import asyncio
import os
import sys
import uuid

# 让脚本能以 `python scripts/seed_knowledge.py` 的方式直接运行
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
# 【关键】config.py 里 Tortoise 连接串是相对路径 "sqlite://ecom_rag.db"，
# 不在项目根目录执行会静默创建一个空库，然后一切看起来都"成功"但查不到数据。
os.chdir(BASE_DIR)

from tortoise import Tortoise

from config import TORTOISE_ORM, collection, data_path
from models.document import Document
from services.pdf_service import parse_pdf, split_text
from services.embedding_service import get_batch_embedding
from services.bm25_service import bm25_service

SEED_DIR = os.path.join(data_path, "seed_docs")


async def purge_doc(doc: Document) -> None:
    """删除一个文档在三处存储中的全部痕迹（DB 记录 / Chroma 向量 / BM25 索引 / 磁盘文件）"""
    collection.delete(where={"doc_id": doc.id})       # 向量库：按 doc_id 批量删该文档所有 chunk
    bm25_service.delete_by_doc_id(doc.id)             # 内存索引：同步摘掉，避免检索命中已删内容
    await doc.delete()                                # 元数据最后删，前面失败还能靠记录重试
    # 磁盘文件：只在确认落在 data/ 目录内时才删，避免异常记录路径误删系统文件
    if doc.save_path and os.path.exists(doc.save_path):
        if os.path.abspath(doc.save_path).startswith(os.path.abspath(data_path)):
            os.remove(doc.save_path)
    print(f"  [purge] 已删除 doc_id={doc.id} ({doc.file_name})")


async def import_pdf(pdf_path: str, reset: bool = False) -> None:
    file_name = os.path.basename(pdf_path)

    existing = await Document.filter(file_name=file_name).first()
    if existing:
        if not reset:
            print(f"  [skip] {file_name} 已存在 (doc_id={existing.id})，如需重导加 --reset")
            return
        print(f"  [reset] 发现同名文档，先清理 doc_id={existing.id}")
        await purge_doc(existing)

    pdf_text = parse_pdf(pdf_path)
    chunks = split_text(pdf_text)
    if not chunks:
        print(f"  [warn] {file_name} 解析不到文本，跳过（可能是扫描件／加密 PDF）")
        return

    # 复制一份到 data/ 下用 uuid 命名——和上传接口保持一致的落盘策略，
    # 这样 seed 文档和用户上传的文档在磁盘上没有区别，删除逻辑不需要分叉
    file_uuid = str(uuid.uuid4())
    save_path = os.path.join(data_path, f"{file_uuid}.pdf")
    with open(pdf_path, "rb") as src, open(save_path, "wb") as dst:
        dst.write(src.read())

    doc = await Document.create(
        file_uuid=file_uuid,
        file_name=file_name,
        save_path=save_path,
        file_size=os.path.getsize(save_path),
    )

    embeddings = get_batch_embedding(chunks)
    # chunk_id 与 Chroma 主键同规则：BM25 侧靠它和向量侧对齐，RRF 融合才有意义。
    # 服务重启时 main.py 的 lifespan 也按这个规则从 Chroma 重建 BM25。
    ids = [f"{doc.id}_{i}" for i in range(len(chunks))]
    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=[{"doc_id": doc.id, "chunk_id": cid} for cid in ids],
    )
    for cid, chunk_text in zip(ids, chunks):
        bm25_service.add_chunk(chunk_text, {"doc_id": doc.id, "chunk_id": cid})

    print(f"  [ok]   {file_name} → doc_id={doc.id}，{len(chunks)} 个 chunk 已入向量库与 BM25")


async def cmd_delete(doc_ids: list) -> None:
    """按 doc_id 删除文档（清理测试残留／过期资料用），三处存储与磁盘文件一并清理"""
    for did in doc_ids:
        doc = await Document.filter(id=did).first()
        if not doc:
            print(f"  [skip] doc_id={did} 不存在")
            continue
        await purge_doc(doc)


async def cmd_list() -> None:
    docs = await Document.all().order_by("id")
    if not docs:
        print("知识库为空")
        return
    snapshot = collection.get(include=["metadatas"])
    per_doc = {}
    for meta in snapshot["metadatas"]:
        if meta:
            per_doc[meta.get("doc_id")] = per_doc.get(meta.get("doc_id"), 0) + 1
    print(f"{'doc_id':<8}{'chunk':<8}{'文件名'}")
    for d in docs:
        print(f"{d.id:<8}{per_doc.get(d.id, 0):<8}{d.file_name}")
    print(f"\n合计 {len(docs)} 个文档，{len(snapshot['ids'])} 个 chunk")


async def main() -> None:
    parser = argparse.ArgumentParser(description="导入 data/seed_docs 下的种子 PDF 到知识库")
    parser.add_argument("--reset", action="store_true", help="删除同名文档后重新导入")
    parser.add_argument("--list", action="store_true", help="仅列出当前库内文档")
    parser.add_argument("--delete", nargs="+", type=int, metavar="ID",
                        help="按 doc_id 删除文档，例如 --delete 1 2")
    args = parser.parse_args()

    await Tortoise.init(config=TORTOISE_ORM)
    await Tortoise.generate_schemas()

    try:
        if args.list:
            await cmd_list()
            return

        if args.delete:
            print(f"删除 {len(args.delete)} 个文档 ...")
            await cmd_delete(args.delete)
            print()
            await cmd_list()
            return

        if not os.path.isdir(SEED_DIR):
            print(f"种子目录不存在：{SEED_DIR}")
            return

        pdfs = sorted(
            os.path.join(SEED_DIR, f)
            for f in os.listdir(SEED_DIR)
            if f.lower().endswith(".pdf")
        )
        if not pdfs:
            print(f"{SEED_DIR} 下没有 PDF")
            return

        print(f"开始导入 {len(pdfs)} 个 PDF ...")
        for pdf in pdfs:
            await import_pdf(pdf, reset=args.reset)

        print()
        await cmd_list()
    finally:
        await Tortoise.close_connections()


if __name__ == "__main__":
    asyncio.run(main())
