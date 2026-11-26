# 基于 BM25 算法的中文关键词检索服务
"""
把切好的文本块chunk存进内存里的 BM25 索引，用户查询时按关键词匹配度返回最相关的片段。
"""
from rank_bm25 import BM25Okapi
import jieba
from typing import List, Dict,Optional
from common.logger import logger  # 注意：导入 logger 对象本身；写 from common import logger 拿到的是模块，调 .info 会 AttributeError


class BM25Service:
    def __init__(self):
        # 存储原始chunk文本，索引下标和列表下标一一对应
        self.corpus: List[str] = []
        # 存储每个chunk的元信息：doc_id, chunk_id等
        self.meta: List[Dict] = []
        # BM25实例，初始为空，添加文档后才构建
        self.bm25: Optional[BM25Okapi] = None

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        中文分词工具
        用jieba做分词，去掉空格，作为BM25输入
        """
        words = jieba.lcut(text)
        # 简单过滤空白字符
        return [w.strip() for w in words if w.strip()]

    def add_chunk(self, chunk_text: str, meta_info: Dict):
        """
        给 BM25 服务加一个增量新增单个文本块
        chunk_text: 切分好的文本片段
        meta_info: 元数据，例如 {"doc_id":1, "chunk_id":101}
        """
        self.corpus.append(chunk_text)
        self.meta.append(meta_info)
        # 每次新增后重建索引
        tokenized_corpus = [self.tokenize(doc) for doc in self.corpus]
        # 用新的分词结果重新构造 BM25Okapi 实例
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info(f"BM25新增chunk，当前总片段数量：{len(self.corpus)}")

    def delete_by_doc_id(self, doc_id: int):
        """
        根据doc_id删除该文档下所有chunk，重建索引
        把不想删的全部挑出来，组成新列表，替换掉旧列表
        """
        new_corpus = []
        new_meta = []
        for text, m in zip(self.corpus, self.meta):
            if m.get("doc_id") != doc_id:
                new_corpus.append(text)
                new_meta.append(m)
        self.corpus = new_corpus
        self.meta = new_meta
        # 重新构建索引
        if len(self.corpus) > 0:
            tokenized_corpus = [self.tokenize(doc) for doc in self.corpus]
            self.bm25 = BM25Okapi(tokenized_corpus)
        else:
            self.bm25 = None
        logger.info(f"BM25删除doc_id={doc_id}，剩余chunk数量：{len(self.corpus)}")

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        BM25检索入口
        query: 用户原始问题
        top_k: 返回前k条结果
        return: [{"text":xxx, "meta":{}, "score":bm25分数}]
        """
        if self.bm25 is None or len(self.corpus) == 0:
            logger.warning("BM25索引为空，检索无结果")
            return []

        # 问题分词
        tokenized_query = self.tokenize(query)
        # 获取每个文档的bm25得分
        scores = self.bm25.get_scores(tokenized_query)

        # 打包（分数，文本，元数据）
        scored_items = list(zip(scores, self.corpus, self.meta))
        # 按分数从高到低排序
        scored_items.sort(key=lambda x: x[0], reverse=True)

        # 截取top_k
        res = []
        for score, text, meta in scored_items[:top_k]:
            res.append({
                "text": text,
                "meta": meta,
                "score": float(score)
            })
        logger.info(f"BM25检索完成，query={query}, 命中{len(res)}条")
        return res


# 全局单例，整个项目共用一套BM25索引
bm25_service = BM25Service()    