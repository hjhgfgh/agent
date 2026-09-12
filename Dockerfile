# ============================================================
# AI技术文档助手 - 后端镜像
#
# 构建上下文必须是「项目根目录」（不是 backend/），原因：
#   main.py 用 BASE_DIR/../frontend 定位前端目录，
#   镜像里必须保留 backend/ 与 frontend/ 的兄弟关系，
#   否则启动时会打印"未找到前端目录，仅提供 API"，页面彻底打不开。
#
# 构建： docker build -t ai-tech-doc-assistant .
# ============================================================
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Asia/Shanghai \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# pip 走国内源。服务器上直连 pypi.org 经常几十 KB/s 甚至超时，
# 构建阶段就会卡死；用 ARG 是为了构建时还能覆盖。
#
# ⚠️ 默认源为什么不是阿里云：实测（家用宽带，同一文件 boto3 索引页）
#     腾讯云 1.46 MB/s ｜ 清华 1.34 MB/s ｜ pypi.org 188 KB/s ｜ 阿里云 161 KB/s
#   阿里云只比直连官方快一点点，装 300MB 依赖要几十分钟，
#   实测触发过 908 秒读超时导致整个构建白跑。清华比阿里云快约 8 倍。
#
# 换源示例（服务器上哪个快就用哪个）：
#   docker build --build-arg PIP_INDEX_URL=https://mirrors.cloud.tencent.com/pypi/simple \
#                --build-arg PIP_TRUSTED_HOST=mirrors.cloud.tencent.com -t ai-tech-doc-assistant .
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn
ENV PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_TRUSTED_HOST=${PIP_TRUSTED_HOST}

WORKDIR /app

# libgomp1 是 faiss-cpu 的硬依赖（OpenMP 运行时）。
# slim 镜像默认不带，缺失时 import faiss 会直接报
#   ImportError: libgomp.so.1: cannot open shared object file
# 这类错误在构建期不报、运行期才炸，必须在这里装掉。
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 tzdata \
 && rm -rf /var/lib/apt/lists/*

# 依赖单独占一层：以后只改 Python 代码时，这一层直接命中缓存，
# 不必每次重新装几分钟的依赖。
COPY backend/requirements.txt ./backend/requirements.txt
# --default-timeout / --retries 是踩坑后加的：
# pip 默认 15 秒读超时、5 次重试，网络一抖动就整体失败，
# 而失败前已经白跑十几分钟（实测触发过一次，908 秒后才报错退出）。
RUN pip install --default-timeout=60 --retries=8 -r backend/requirements.txt

# 代码
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# 运行期数据目录。镜像里只留空目录，
# 真实数据由 docker-compose 的卷挂载进来 —— 容器删了重建，索引和上传的 PDF 都还在。
RUN mkdir -p /app/backend/faiss_index /app/backend/documents

WORKDIR /app/backend

EXPOSE 8000

# 健康检查直接打 /health（它真的会探测 MySQL 与 Redis，不是无脑返回 200）
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status==200 else 1)"

# 刻意不加 --reload：Windows 下它会导致端口占用与新 worker 启动失败，
# 详见 run_backend.py 顶部注释。容器里同样不需要热重载。
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
