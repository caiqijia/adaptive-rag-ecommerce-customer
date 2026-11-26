# 向量召回
import dashscope 
from dashscope import TextEmbedding
from fastapi import HTTPException
from config import DASHCOPE_API_KEY,EMBEDDING_MODEL
import asyncio

dashscope.api_key = DASHCOPE_API_KEY

EMBED_BATCH_SIZE = 25  # text_embedding_v2 单次最多 25 条，超出整个请求会被拒

def embed_batch(batch:list[str])->list[list[float]]:
    """
    内部辅助函数：负责真正调用 DashScope 接口获取一批文本的向量。
    参数 batch 是一个字符串列表，返回值是一个二维列表（每个文本对应一个由浮点数组成的向量）。
    """
    resp = TextEmbedding.call(
        model = EMBEDDING_MODEL,input=batch
    )
    if resp.status_code !=200:
        raise RuntimeWarning(f"embedding接口失败:{resp.code}{resp.message}")
    else:
        # 如果调用成功，使用列表推导式从返回结果中提取出所有文本对应的 embedding 向量并返回
        return [i["embedding"] for i in resp.output["embeddings"]]

# 单文本向量化
def get_embedding(text:str)->list[float]:
    try:
        return embed_batch([text])[0]
    except Exception as e:
        raise HTTPException(status_code=500,detail=f"embedding调用失败:{repr(e)}")

# 批量文本向量化
def get_batch_embedding(chunks:list[str])->list[list[float]]:
    try:
        result = []
        for i in range(0,len(chunks),EMBED_BATCH_SIZE):
            result.extend(embed_batch(chunks[i:i+EMBED_BATCH_SIZE]))
        return result

    except Exception as e:
        raise HTTPException(status_code=500,detail=f"批量embedding调用失败:{repr(e)}")

async def get_batch_embedding_async(chunks:list[str])->list[list[float]]:
    """
    获取大批量文本的向量表示（异步版本）。
    在 FastAPI 等异步框架中，为了防止耗时的网络请求卡住整个服务器
    """
    # asyncio.to_thread 的作用是把一个普通的同步函数（get_batch_embedding）放到一个单独的线程中去执行
    return await asyncio.to_thread(get_batch_embedding,chunks)
