from fastapi import APIRouter, File, UploadFile, HTTPException
import uuid
import aiofiles
import os
from models.document import Document, DocumentResp
from services.pdf_service import parse_pdf, split_text
from services.bm25_service import bm25_service
from services.embedding_service import get_batch_embedding
from config import data_path, collection
from common.logger import logger
from common.response import success_resp
import chromadb


doc_router = APIRouter(tags=['文档管理'],prefix='/doc')

os.makedirs(data_path,exist_ok=True)

# 上传PDF
@doc_router.post("/upload_pdf")
async def upload_pdf(file:UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400,detail="只允许上传PDF文件！")
    file_uuid_str = str(uuid.uuid4())
    new_file_name = f"{file_uuid_str}.pdf"
    # 把上传的 PDF 二进制存到本地磁盘 data_path，用 uuid 命名避免重名覆盖。
    full_save_path = os.path.join(data_path,new_file_name)
    logger.info(f"开始上传文件：{file.filename},uuid={file_uuid_str}")
    async with aiofiles.open(full_save_path,"wb") as f:
        content = await file.read()
        await f.write(content)
    file_size = len(content)
    # 写入数据库记录,在关系型数据库（Tortoise）里记一条元数据,方便以后查文件名、大小、路径、处理状态
    doc_record = await Document.create(
        file_uuid = file_uuid_str,
        file_name = file.filename,
        save_path = full_save_path,
        file_size = file_size
    )
    logger.info(f"文件上传入库成功,doc_id={doc_record.id}, file_uuid={file_uuid_str}")
    # return DocumentResp.model_validate(doc_record)

    # 解析PDF，切chunk
    pdf_text = parse_pdf(full_save_path)
    chunks = split_text(pdf_text)
    if not chunks:
        raise HTTPException(400, "PDF内无有效文本")

    # 批量向量化
    embeddings = get_batch_embedding(chunks)
    ids = [f"{doc_record.id}_{i}" for i in range(len(chunks))]
    # metadata 里的 chunk_id 与 Chroma id 同规则（f"{doc_id}_{i}"）：
    # BM25 侧靠它和向量侧主键对齐，RRF 才能融合；服务重启重建 BM25 也靠它回填
    metadatas = [{"doc_id":doc_record.id,"chunk_id":cid} for cid in ids]

    # 存入chroma向量库
    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas
    )
    logger.info(f"向量入库完成，chunk数量:{len(chunks)}")
    # ========= 增量维护 BM25 索引 =========
    # add_chunk 内部会重建 BM25Okapi 实例（rank_bm25 不支持真正的增量更新，
    # 只能整体重建——这是它的已知局限；语料到十万级以上应换 ES/OpenSearch）
    # meta 必须带 chunk_id：这是与向量侧对齐的融合主键，缺了 RRF 协同效应归零
    for i, chunk_text in enumerate(chunks):
        bm25_service.add_chunk(chunk_text, {
            "doc_id": doc_record.id,
            "chunk_id": ids[i],
        })
    logger.info("BM25索引维护完成")
    return success_resp(data=DocumentResp.model_validate(doc_record))

# 查询文档列表
@doc_router.post('/list')
async def get_doc_list():
    docs = await Document.all()
    return success_resp(data=[DocumentResp.model_validate(d) for d in docs])

# 删除文档接口
@doc_router.delete("/{doc_id}")
async def delete_doc(doc_id:int):
    doc = await Document.get_or_none(id = doc_id)
    if not doc:
        raise HTTPException(404, "文档不存在")
    # 删除向量库
    collection.delete(where={"doc_id":doc_id})
    # 删除本地pdf
    if os.path.exists(doc.save_path):
        os.remove(doc.save_path)
    await doc.delete()
    # ========= 同步移除 BM25 内该文档片段 =========
    # 向量库（Chroma）与关键词库（BM25）必须同步删，否则残留 chunk 会被
    # 检索出来但点不开原文——混合检索两路的数据一致性是运维基本要求
    bm25_service.delete_by_doc_id(doc_id)
    logger.info(f"文档删除成功 id={doc_id},已同步清理BM25索引")
    return success_resp(msg="文档已删除")

