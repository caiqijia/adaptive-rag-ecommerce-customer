import os

# ========= .env 加载（可选依赖，不装也能跑）=========
# 【为什么需要】
#   原来只读系统环境变量：在 IDE 里跑没问题（IDE 注入了环境变量），
#   但换一台机器 `uvicorn main:app` 就会因为 DASHCOPE_API_KEY=None 而全线 401。
#   面试官 clone 下来最可能栽在这一步——"跑不起来"比"效果一般"致命得多。
# 【为什么用 try/except 而不是写进 requirements 强制依赖】
#   .env 只是开发期便利：生产环境用容器/密钥管理注入环境变量，
#   此时装不装 python-dotenv 都无所谓。可选依赖不该让主流程 import 失败。
# 【为什么放在文件最前面】
#   .env 的值必须在下面 os.getenv() 读取之前写入 os.environ 才生效。
try:
    from dotenv import load_dotenv
    # 显式指定路径：uvicorn 的 cwd 不一定是项目根，用相对路径会读不到
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass

# ========= HuggingFace 下载配置（必须在 huggingface_hub 被导入之前设置）=========
# 【背景】rerank 模型 BAAI/bge-reranker-v2-m3（约 2.2GB）首次使用要从 HuggingFace 下载。
# 实测 huggingface.co 在境内不可达（连接超时 WinError 10060，8s 失败），
# 而 hf-mirror.com 是官方镜像（实测 0.4s 可达）。
#
# 【为什么用 setdefault 而不是直接赋值】
#   保留用户显式指定的优先级：如果部署环境已设置 HF_ENDPOINT
#   （指向企业内网代理或离线目录），不应该被这里的默认值覆盖。
#   这是"有默认值但不劫持配置"的工程惯例。
#
# 【为什么必须放在 config.py 顶部】
#   huggingface_hub 在 import 时就会读取 HF_ENDPOINT 并据此初始化常量，
#   之后再改 os.environ 不生效。config.py 是 rerank_service 的前置依赖，
#   而 FlagEmbedding 是在 rerank_service 内部延迟 import 的——
#   所以只要在 config 模块加载时设置好，就能保证 huggingface_hub
#   第一次被导入时读到的就是镜像地址。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# 缩短超时：huggingface_hub 默认单次连接超时 10s + 内部重试 5 次，
# 最坏情况一个请求要卡 1 分钟以上（日志里的 "Retrying in 1s [Retry 1/5]" 就是这个）。
# 缩到 10s 连接 / 30s 单文件下载，失败能快速暴露而不是无限拖住用户请求。
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "10")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")

import chromadb
# ========= Dashscope 大模型&Embedding&Rerank =========
DASHCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
EMBEDDING_MODEL = "text-embedding-v2"
RERANK_MODEL = "bge-reranker-v2-m3"
# LLM 模型选型（2026-10-27 从 qwen-turbo 迁移）：
#   qwen-turbo 免费额度已耗尽（Throttling.RateQuota），且已从免费额度列表下架。
#   qwen3.5-flash 是控制台"免费额度"页显示名 "qwen3-flash" 的真实 API ID
#   （阿里云页面显示名会去掉小版本号：qwen3.5-flash → qwen3-flash），
#   持有 1M token 免费额度；flash 定位 = qwen-turbo 的对应替代
#   （轻量 / 快速 / 低成本），覆盖 Router / Rewrite / Decompose / 生成 / Critic 全链路。
#   备选（同额度池）：qwen3.5-27b（页显 qwen3-27b）、kimi-k3、deepseek-v4-flash-0731。
LLM_MODEL = "qwen3.5-flash"
# DashScope compatible-mode 端点（OpenAI 兼容协议）。
# 为什么不继续用 dashscope SDK 的 Generation.call（原生端点）：
#   1. 已安装 SDK 1.27.8 对新模型（qwen3.x 系列）的原生端点路由报 url error，
#      这是 1.x SDK 的已知限制；升级 2.0 亦可但回归风险大、收益低
#   2. compatible-mode 新旧模型通吃，且是 OpenAI 兼容协议——将来换供应商
#      （OpenAI / DeepSeek 官方 / 本地 vLLM）只改这个 URL 和 MODEL，调用代码零改动
#   3. 调用只用标准库 urllib，不引入额外依赖
LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# qwen3 系列是"混合推理"模型：不关思考的话，连 Router 分类都会先输出一大段
# 推理过程，max_tokens=128 直接被思考内容撑爆、JSON 解析必失败。
# 客服场景（分类/改写/生成/审核）全部不需要深度推理，统一关闭。
LLM_ENABLE_THINKING = False

# CHROMA_PRESISR_PATH = "./chroma_db"
# COLLECTION_NAME = "ecommerce_knowledge"

# ========= Chroma向量库配置 =========
# # 获取当前项目根目录的绝对路径
base_dir = os.path.dirname(os.path.abspath(__file__))# 得到 adaptive_rag_ecommerce_customer/
# 本地chroma_db文件夹路径
chroma_db_path = os.path.join(base_dir,"chroma_db")

# 初始化 ChromaDB 客户端，指定数据持久化保存到 "./chroma_db" 文件夹
chromadb_client = chromadb.PersistentClient(path=chroma_db_path)

# 表名
collection_name = "ecommerce_knowledge"
# 获取或创建一个名为 "rag_docs" 的集合对象
collection = chromadb_client.get_or_create_collection(name=collection_name)

# ========= 文件存储 =========
data_path = os.path.join(base_dir,"data")

# ========= Tortoise ORM 数据库配置 =========
TORTOISE_ORM = {
    "connections": {
        "default": "sqlite://ecom_rag.db"
    },
    "apps": {
        "models": {
            "models": ["models.document", "models.faq"],
            "default_connection": "default"
        }
    }
}

# ========= 检索超参数 =========
VECTOR_TOP_K = 6
BM25_TOP_K = 6
RRF_K = 60
RERANK_TOP_K = 3
# 距离阈值：Chroma 默认返回 L2 欧氏距离（越小越相关）
# 【为什么是 1.25 而不是 0.8】
#   text-embedding-v2 输出的是归一化向量，两点间 L2 距离 d 与余弦相似度满足
#   cos = 1 - d²/2，即 d=0.8 时要求 cos > 0.68——中文短 query 与 500 字长 chunk
#   之间几乎不可能达到（实测 40 个 chunk 的真实 top6 距离全部落在 1.00~1.39，
#   对应余弦 0.03~0.46）。阈值卡 0.8 会让向量通路**全部被过滤**，
#   双路召回退化成 BM25 单路，RRF 的协同效应直接归零，且日志里毫无异常。
#   1.25 对应 cos ≈ 0.22：只砍掉"明显不相关"的尾巴，精度交给 rerank 把关。
# 【为什么不设更大】
#   向量召回负责"别漏"，rerank 负责"别错"。放宽召回是一回事，
#   把明显无关的也算候选会让 rerank 的输入变脏、top1 分数虚高。1.25 是实测拐点。
DISTANCE_THRESHOLD = 1.25

# ========== Rerank 精排配置 ==========
# 模型来源优先级（三级兜底，越靠前越优先）：
#   1. 环境变量 RERANK_MODEL_PATH —— 部署时显式指定（离线包 / 共享盘）
#   2. 项目内 local_models/bge-reranker-v2-m3 —— 自动探测，存在就直接用
#   3. HuggingFace 仓库名 —— 首次运行经 HF_ENDPOINT 镜像下载约 2.2GB
#
# 【为什么优先项目内目录而不是 HF 缓存】
#   HF 缓存目录（~/.cache/huggingface）会被各类清理工具当垃圾清掉，
#   且缓存的 snapshots 是软链接结构，拷到别的机器会断链。
#   放项目内目录 + 加 .gitignore，换机器直接拷目录就能跑，可移植性最好。
#
# 【首次下载命令（二选一）】
#   A. 走 HF 镜像：
#      huggingface-cli download BAAI/bge-reranker-v2-m3 \
#          --local-dir ./local_models/bge-reranker-v2-m3
#   B. 走 ModelScope（国内更稳）：
#      modelscope download --model BAAI/bge-reranker-v2-m3 \
#          --local_dir ./local_models/bge-reranker-v2-m3
_RERANK_LOCAL_DIR = os.path.join(base_dir, "local_models", "bge-reranker-v2-m3")
RERANK_MODEL_NAME = (
    os.getenv("RERANK_MODEL_PATH")
    or (_RERANK_LOCAL_DIR if os.path.isdir(_RERANK_LOCAL_DIR) else "BAAI/bge-reranker-v2-m3")
)
RERANK_DEFAULT_TOP_K = 5
RERANK_SCORE_THRESHOLD = 0.3
RERANK_MAX_CANDIDATE = 20

# ========== Adaptive Agent 配置 ==========
AGENT_MAX_RETRY_TIMES = 2        # 最多重试几次：原始query算第一轮，最多再改写2次
AGENT_QUERY_REWRITE_NUM = 3      # 每次改写生成几条不同query
