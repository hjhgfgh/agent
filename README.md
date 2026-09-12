# AI 技术文档助手

上传 PDF 技术文档后，可以就文档内容提问，由大模型结合检索到的片段生成回答。
后端 FastAPI，前端 Vue 3，MySQL + Redis 存储，Docker Compose 一键启动。

---

## 一、技术栈

| 层次 | 实际使用 | 说明 |
|------|---------|------|
| 后端框架 | FastAPI + Uvicorn 0.34 | |
| 大模型 | agnes-3.0-flash（OpenAI 兼容接口） | 换服务只需改 `LLM_BASE_URL` / `LLM_MODEL` |
| 工具调用 | LangChain（Function Calling） | Agent 自主决定是否调用工具 |
| 检索 | **自研关键词检索**（`SimpleVectorStore`） | 中文 bigram 切分 + 重叠度打分，**不使用 Embedding 模型** |
| 会话 / 缓存 | Redis 7 | 多轮对话历史 + 问答结果缓存 |
| 数据库 | MySQL 8 + SQLAlchemy 2 | 文档元数据、会话、缓存记录 |
| 前端 | Vue 3（依赖已本地化，不走 CDN） | 单页应用，由后端同源托管 |
| 部署 | Docker Compose | 后端 + MySQL + Redis 一键拉起 |

### 关于检索方式（重要）

`backend/rag/pipeline.py` 里的 `SimpleVectorStore` 是自研的**关键词检索**：
中文按 bigram（双字）切分，用重叠度打分排序，**不依赖任何 Embedding 模型或向量库**。

这是有意的工程取舍：

- **好处**：零外部 API 依赖、零额外成本、冷启动快、镜像不需要预置模型文件
- **代价**：对同义改写召回较弱（问"响应式原理"可能匹配不到只写了"数据双向绑定"的段落）

若要升级为语义检索，把 `SimpleVectorStore` 换成 FAISS + Embedding 即可。
`requirements.txt` 里的 `faiss-cpu` 是为这一步预留的，**当前代码并未 import 它**。

---

## 二、Docker 一键启动（推荐）

### 前置条件

Docker Desktop（Windows / macOS）或 Docker Engine + Compose（Linux）。

### 1. 配置

根目录 `backend/.env` 是**唯一的应用配置来源**（本地与 Docker 共用同一份）：

```env
# --- 大模型 ---
LLM_API_KEY=你的密钥
LLM_BASE_URL=https://apihub.agnes-ai.cn/v1
LLM_MODEL=agnes-3.0-flash
LLM_TIMEOUT=60
LLM_MAX_TOKENS=2000

# --- 数据库 / 缓存（本地开发填 localhost）---
DATABASE_URL=mysql+pymysql://root:123456@localhost:3306/tech_doc_assistant
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# --- 访问控制（公网部署必看第四节）---
# ACCESS_CODE=你的口令
# READ_ONLY=true
```

> `DATABASE_URL` 和 `REDIS_HOST` 里的 `localhost` **在 Docker 下不用改**。
> `docker-compose.yml` 会覆盖成容器服务名（`mysql` / `redis`）——
> 因为容器里的 `localhost` 指向容器自己，会 100% 连不上。

### 2. 启动

```bash
docker compose up -d --build
```

### 3. 访问

打开 <http://localhost:8000>

### 常用命令

```bash
docker compose ps                    # 查看状态（三个服务都应是 healthy）
docker compose logs -f backend       # 跟踪后端日志
docker compose up -d --build backend # 只重建后端（改 Python 代码后）
docker compose down                  # 停止，数据保留
docker compose down -v               # 停止并删除数据（索引、上传的 PDF 都会丢）
```

---

## 三、本地开发（不用 Docker）

需要本机已有 MySQL 和 Redis。

```bash
pip install -r backend/requirements.txt

cd backend
python -c "from database.models import init_db; init_db()"
uvicorn main:app --host 0.0.0.0 --port 8000
```

前端访问 <http://localhost:8000> 即可（后端会一并托管）。

> ⚠️ 不要加 `--reload`：Windows 下会导致端口占用和新 worker 启动失败，
> 详见 `run_backend.py` 顶部注释。也可以直接 `python run_backend.py`。

---

## 四、访问控制（公网部署必读）

在 `backend/.env` 中设置：

```env
ACCESS_CODE=your-secret-code   # 不设 = 任何人拿到地址都能调 /chat 烧你的额度
READ_ONLY=true                 # 演示环境建议开，只允许提问
```

改完重启：`docker compose up -d --build backend`

### 行为

| 场景 | 结果 |
|------|------|
| 未设 `ACCESS_CODE` | 不做鉴权，启动日志打印 `[WARN] 未设置 ACCESS_CODE` |
| 设了 `ACCESS_CODE` | 除下方白名单外，所有接口都要求请求头 `X-Access-Code` |
| 口令错误 / 未带 | `401`，前端自动弹回口令输入框 |
| `READ_ONLY=true` | 上传、删除返回 `403`，前端直接隐藏这两个入口 |

**免口令白名单**（`backend/main.py` 的 `_PUBLIC_PATHS` / `_PUBLIC_PREFIXES`）：

- `/`、`/index.html`、`/favicon.ico` —— 页面本身，否则用户没机会输入口令
- `/health` —— 必须公开，否则 Docker HEALTHCHECK 一直失败，容器永远不 healthy
- `/docs`、`/redoc`、`/openapi.json`、`/api`
- **`/vendor/` 整个前缀** —— 页面用 `<script src>` / `<link href>` 加载这些前端依赖，
  浏览器**无法给这类请求附加自定义请求头**。若不放行，开启口令后 Vue 加载不出来，
  连口令输入框自己都渲染不出来，页面彻底死锁。这些都是公开的第三方静态资源，无业务数据。

> 新增前端静态目录时，记得同时加进 `_PUBLIC_PREFIXES`，否则开了口令后该目录会 401。

---

## 五、部署到服务器

1. 装 Docker：

```bash
curl -fsSL https://get.docker.com | sh
```

2. 拉代码（私有仓库先配好 SSH key 或访问令牌）：

```bash
git clone <你的仓库地址> ai-tech-doc-assistant
cd ai-tech-doc-assistant
```

3. 创建 `backend/.env`（内容见第二节），**务必设置 `ACCESS_CODE`**。

4. 想直接用 80 端口：

```bash
echo "BACKEND_PORT=80" > .env
docker compose up -d --build
```

5. 放行端口。云服务器要在**安全组**里放行，Ubuntu 还要过 ufw：

```bash
ufw allow 80
```

**安全提醒**：`docker-compose.yml` 刻意**没有**给 MySQL 和 Redis 做端口映射。
它们是裸奔状态（Redis 无密码），映射到公网等于任人读写，这是最常见的入侵入口。
后端是唯一对外暴露的服务。

---

## 六、API 接口

| 方法 | 路径 | 鉴权 | 说明 |
|------|------|:----:|------|
| GET | `/health` | 免 | 健康检查（真实探测 MySQL + Redis） |
| GET | `/auth/verify` | 需 | 校验口令，返回 `{ok, auth_required, read_only}` |
| POST | `/upload` | 需 | 上传 PDF，切片并建索引 |
| GET | `/documents` | 需 | 文档列表 |
| DELETE | `/documents/{doc_id}` | 需 | 删除文档 |
| POST | `/chat` | 需 | 问答（只读模式下唯一允许的写方法） |
| GET | `/chat/history/{session_id}` | 需 | 会话历史 |
| GET | `/tools` | 需 | 可用工具列表 |
| GET | `/stats` | 需 | 系统统计 |
| GET | `/api` | 免 | 服务信息 |

交互式文档：<http://localhost:8000/docs>

---

## 七、项目结构

```
ai-tech-doc-assistant/
├── Dockerfile                  # 后端镜像（含前端静态文件）
├── docker-compose.yml          # 后端 + MySQL + Redis 编排
├── .dockerignore / .gitignore / .gitattributes
├── backend/
│   ├── main.py                 # FastAPI 主应用 + 访问守卫中间件
│   ├── requirements.txt
│   ├── .env                    # 配置（已被 git 忽略，需自行创建）
│   ├── rag/pipeline.py         # RAG 核心：PDF 解析、切片、关键词检索
│   ├── agent/agent.py          # Function Calling 与 4 个工具
│   ├── cache/redis_cache.py    # 会话历史 + 问答缓存
│   ├── database/models.py      # SQLAlchemy 模型
│   ├── llm/client.py           # 大模型客户端（OpenAI 兼容）
│   ├── documents/              # 上传的 PDF（运行时数据，不入仓库）
│   └── faiss_index/            # 索引存储（运行时数据，不入仓库）
├── frontend/
│   ├── index.html              # Vue 3 单页应用
│   └── vendor/                 # 前端依赖本地化
├── database/init.sql           # 建表脚本（首次启动自动执行）
└── _verify_frontend.js         # 前端静态校验脚本（开发工具）
```

---

## 八、内置工具（Function Calling）

Agent 按需自动调用：

| 工具 | 作用 |
|------|------|
| `calculator` | 数学计算 |
| `get_current_time` | 当前时间 |
| `count_words` | 统计字数 |
| `extract_keywords` | 提取关键词 |

---

## 九、前端依赖本地化

`frontend/vendor/` 自带全部前端依赖，**不依赖任何 CDN**：

| 依赖 | 版本 | 大小 |
|------|------|------|
| vue | 3.5.42 | 594 KB |
| marked | 12.0.2 | 35 KB |
| dompurify | 3.4.15 | 29 KB |
| @mdi/font | 7.4.47 | 347 KB CSS + 394 KB woff2 |

解决了两个问题：

1. **访客浏览器必须能访问 CDN** —— 一旦被墙或超时，Vue 加载不出来页面就白屏，
   而这个错误发生在访客浏览器里，服务器日志上完全看不到，极难排查。
2. **浮动版本号** —— 原先是 `vue@3`、`marked@12` 这类写法，
   上游一发新版就可能把页面搞坏；现在锁定到具体版本。

MDI 字体只保留了 woff2 一种格式（原包含 eot/woff/ttf 共约 2.4 MB 冗余）——
本项目要 Vue 3 + ES6 + fetch，能跑起来的浏览器必然支持 woff2。

升级依赖：替换 `frontend/vendor/` 下的文件即可。

---

## 十、后续可做

- [ ] 把关键词检索升级为 FAISS 向量检索（`faiss-cpu` 已在依赖里，尚未接入）
- [ ] WebSocket 流式输出（当前是一次性返回完整回答）
- [ ] 清理 `requirements.txt` 中未使用的依赖（`faiss-cpu`、`huggingface-hub`）
- [ ] 索引增量更新，避免每次上传重建
