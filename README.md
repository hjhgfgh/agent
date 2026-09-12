# AI 技术文档助手

上传 PDF 技术文档后，可以就文档内容提问，由大模型结合检索到的片段生成回答。
后端 FastAPI，前端 Vue 3，MySQL + Redis 存储，Docker Compose 一键启动。

---

## 一、技术栈

| 层次 | 实际使用 | 说明 |
|------|---------|------|
| 后端框架 | FastAPI + Uvicorn 0.34 | |
| 大模型 | agnes-2.5-flash（OpenAI 兼容接口） | 换服务只需改 `LLM_BASE_URL` / `LLM_MODEL` |
| PDF 解析 / 切片 | PyMuPDF 为主，LangChain `PyPDFLoader` 降级兜底；切片用 `RecursiveCharacterTextSplitter` | 见 `rag/pipeline.py` |
| 工具调用 | **OpenAI 原生 Function Calling**（自研 `to_openai_tools` 转换） | 不经 LangChain Agent；Agent 自主决定是否调用工具 |
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

`backend/.env` 是**唯一的应用配置来源**（本地与 Docker 共用同一份）：

```env
# --- 大模型 ---
LLM_API_KEY=你的密钥
LLM_BASE_URL=https://apihub.agnes-ai.cn/v1
LLM_MODEL=agnes-2.5-flash
LLM_TIMEOUT=120
LLM_MAX_TOKENS=4000

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
READ_ONLY=true                 # 演示环境建议开：只放行 /chat 与 /session/cleanup
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

> 以下步骤已在 **阿里云 ECS** 上实测通过：Alibaba Cloud Linux 3.2104 LTS、2 核 2 GiB、
> 华南1（深圳）D、40 GiB ESSD、公网带宽按流量计费。

### 1. 装 Docker

系统是 RHEL 系（`dnf`），**不要**用 `get.docker.com` 一键脚本，按下面来：

```bash
# Alibaba Cloud Linux 3 的 $releasever 是 "3"，而 docker-ce 仓库只有 7/8/9，
# 不处理会直接 404。这里固定成 8（Anolis 8 / RHEL8 兼容）。
dnf install -y dnf-plugins-core
curl -fsSL -o /etc/yum.repos.d/docker-ce.repo \
  https://mirrors.aliyun.com/docker-ce/linux/centos/docker-ce.repo
sed -i 's|\$releasever|8|g; s|download.docker.com|mirrors.aliyun.com/docker-ce|g' \
  /etc/yum.repos.d/docker-ce.repo
dnf install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
systemctl enable --now docker
```

实测装到 Docker 26.1.3 + Compose v2.27.0。

### 2. 配镜像加速器（**必做**）

国内直连 Docker Hub 会超时 —— 首次没配加速器时，15 分钟只拉下来 29 MB，
连接还落在 Cloudflare 的 CDN 上，约 30 KB/s。

```bash
mkdir -p /etc/docker
cat > /etc/docker/daemon.json <<'EOF'
{
  "registry-mirrors": [
    "https://docker.m.daocloud.io",
    "https://docker.1panel.live"
  ],
  "log-driver": "json-file",
  "log-opts": { "max-size": "10m", "max-file": "3" },
  "live-restore": true
}
EOF
systemctl restart docker
```

加速器实测（拉 `alpine:latest` ≈3 MB）：

| 加速器 | 耗时 |
|--------|------|
| `docker.m.daocloud.io` | **8 s** |
| `docker.1panel.live` | **10 s** |
| `docker.xuanyuan.me` | 26 s |
| `docker.1ms.run` | 47 s |
| `dockerpull.org` / `docker.1panel.top` | 失败 |

换源后三个基础镜像（python 133 MB + redis 39 MB + mysql 799 MB）**2 分钟**拉完。

> `log-opts` 那两行不是可有可无的：小磁盘机器上容器日志无限增长会把
> 系统盘写满，进而整个 Docker 挂掉。

### 3. 小内存机器加 swap

2 GiB 内存构建时（pip 装依赖）有 OOM 风险，挂 2 GiB swap 兜底：

```bash
fallocate -l 2G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
echo '/swapfile none swap sw 0 0' >> /etc/fstab
echo 'vm.swappiness=10' >> /etc/sysctl.conf && sysctl -w vm.swappiness=10
```

实测运行期三容器合计约 570 MB，很宽裕；swap 在构建阶段做保险。

### 4. 放代码 + 建配置

```bash
# 私有仓库先配好 SSH key 或访问令牌
git clone <你的仓库地址> ai-tech-doc-assistant
cd ai-tech-doc-assistant
```

创建 `backend/.env`（内容见第二节），**务必设置 `ACCESS_CODE`**；再建根目录 `.env`：

```bash
echo "BACKEND_PORT=80" > .env
echo "MYSQL_ROOT_PASSWORD=$(openssl rand -hex 12)" >> .env
```

### 5. 构建并启动

ECS 上阿里云 pypi 走内网，实测 550 KB/s vs 清华 342 KB/s，所以构建时切阿里云源：

```bash
docker compose build \
  --build-arg PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
  --build-arg PIP_TRUSTED_HOST=mirrors.aliyun.com
docker compose up -d
```

> `docker compose up` 本身不支持 `--build-arg`，所以必须分成 `build` + `up` 两步。

### 6. 放行端口

云服务器要在**安全组**里放行 80（入方向）；系统防火墙如果开着也要放：

```bash
# Alibaba Cloud Linux 3 / CentOS
# firewall-cmd --permanent --add-port=80/tcp && firewall-cmd --reload
# Ubuntu
# ufw allow 80
```

**验证**：`curl http://<公网IP>/health` 应返回
`{"status":"healthy","redis":true,"db":true}`。

**安全提醒**：`docker-compose.yml` 刻意**没有**给 MySQL 和 Redis 做端口映射。
它们是裸奔状态（Redis 无密码），映射到公网等于任人读写，这是最常见的入侵入口。
后端是唯一对外暴露的服务。

---

## 六、API 接口

| 方法 | 路径 | 鉴权 | 说明 |
|------|------|:----:|------|
| GET | `/health` | 免 | 健康检查（真实探测 MySQL + Redis） |
| GET | `/auth/verify` | 需 | 校验口令，返回 `{ok, auth_required, read_only}` |
| POST | `/upload` | 需 | 上传 PDF，切片并建索引（只读模式返回 `403`） |
| GET | `/documents` | 需 | 文档列表（内置文档在前） |
| DELETE | `/documents/{doc_id}` | 需 | 删除文档（内置文档返回 `403`，只读模式全部 `403`） |
| POST | `/session/cleanup` | 需 | 清空访客上传的临时文档，内置文档不受影响 |
| POST | `/chat` | 需 | 问答（只读模式放行的两个写方法之一） |
| GET | `/chat/history/{session_id}` | 需 | 会话历史 |
| GET | `/tools` | 需 | 可用工具列表 |
| GET | `/stats` | 需 | 系统统计 |
| GET | `/api` | 免 | 服务信息 |

> **只读模式（`READ_ONLY=true`）放行的写方法只有两个**：`/chat` 与 `/session/cleanup`。
> 前者是访客唯一该被允许做的事；后者是"把环境恢复成初始状态"，只会让知识库更干净 ——
> 若一并拦掉，页面每次打开都会留一条 403，且残留的临时文档再也清不掉。
> `POST /upload`、`DELETE /documents/{id}` 一律返回 `403`。

交互式文档：<http://localhost:8000/docs>

---

## 七、项目结构

```
ai-tech-doc-assistant/
├── Dockerfile                  # 后端镜像（含前端静态文件）
├── docker-compose.yml          # 后端 + MySQL + Redis 编排
├── .dockerignore / .gitignore / .gitattributes
├── README.md
├── run_backend.py              # 本地启动脚本（绕开 Windows 下 --reload 的端口占用）
├── backend/
│   ├── main.py                 # FastAPI 主应用 + 访问守卫中间件
│   ├── requirements.txt
│   ├── .env                    # 配置（已被 git 忽略，需自行创建）
│   ├── builtin_docs/           # 内置文档：随镜像发布、常驻知识库（见下方说明）
│   │   └── vuejs-official-guide.pdf
│   ├── rag/pipeline.py         # RAG 核心：PDF 解析、切片、关键词检索
│   ├── agent/agent.py          # Function Calling 与 4 个工具
│   ├── cache/redis_cache.py    # 会话历史 + 问答缓存
│   ├── database/models.py      # SQLAlchemy 模型
│   ├── llm/client.py           # 大模型客户端（OpenAI 兼容）
│   ├── documents/              # 访客上传的 PDF（运行时数据，走 upload_data 卷）
│   └── faiss_index/            # 索引存储（运行时数据，走 faiss_data 卷）
├── frontend/
│   ├── index.html              # Vue 3 单页应用
│   └── vendor/                 # 前端依赖本地化（vue / marked / dompurify / mdi）
├── database/init.sql           # 建表脚本（首次启动自动执行）
└── documents/                  # 原始素材（不入仓库、不进镜像）
```

### 内置文档 vs 临时文档

`backend/builtin_docs/` 里的 PDF 是**随镜像发布**的常驻知识库，与访客上传的文档走两套生命周期：

| | 内置文档 | 临时文档 |
|---|---|---|
| 存放位置 | `backend/builtin_docs/`（**在镜像内**） | `backend/documents/`（走 `upload_data` 卷） |
| 数据库标记 | `is_builtin = 1` | `is_builtin = 0` |
| 能否删除 | 接口层拒绝，返回 `403` | 可删，或被 `/session/cleanup` 批量回收 |
| 生命周期 | 容器重建后仍在，换服务器重新部署也一定在 | 关掉网页后再打开页面时被自动清空 |

**为什么另放一个目录**：`upload_data` 卷在服务器首次部署时是空的，内置文档若放在
`UPLOAD_DIR` 里，"开箱即有内容可问"就无从谈起；镜像内的文件不随卷清空而消失。

**索引时机**：容器启动时由 `lifespan` 起一个**后台线程**处理，不阻塞启动 ——
首次要解析 2 MB 的 PDF、切出上千个切片再全量写盘，同步做会让健康检查
（`start-period 20s`）超时，容器会被判定不健康而反复重启。日志会打印：

```
[BUILTIN] 首次索引: VueJS官方文档.pdf
[BUILTIN] 完成: VueJS官方文档.pdf，新增 xxx 个切片
```

**刷新页面**即可看到内容，服务本身不为它等待。索引幂等（靠内容哈希判断），容器重启不会重复切分。

> 磁盘文件名刻意用纯英文（`vuejs-official-guide.pdf`）：容器 locale 未设置时 Python 的
> 文件系统编码可能回退成 ASCII，中文文件名在 `os.listdir` / `open` 时会直接抛错，
> 而这个错误只在 Linux 容器出现，本地 Windows 开发完全复现不了。
> 界面上的显示名走 `main.py` 的 `BUILTIN_DISPLAY_NAMES` 映射，不受此限制。

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
| vue | 3.5.42 | 580 KB |
| marked | 12.0.2 | 35 KB |
| dompurify | 3.4.15 | 29 KB |
| @mdi/font | 7.4.47 | 338 KB CSS + 394 KB woff2 |

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
- [ ] 清理 `requirements.txt` 中未使用的依赖（`faiss-cpu`、`huggingface-hub`、`langchain-openai`）
- [ ] 索引增量更新，避免每次上传重建
