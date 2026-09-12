"""
启动脚本 - 用于快速启动后端服务

为什么刻意不加 --reload：
    Windows 下 uvicorn 的 --reload 在触发重启时，旧 worker 经常来不及释放
    8000 端口，新 worker 会以
        [Errno 10048] error while attempting to bind on address ('0.0.0.0', 8000)
    启动失败并退出；而旧 worker 可能已经卡死、却仍然占着监听套接字。
    结果是端口处于 LISTENING 状态但不再处理任何请求 —— 所有接口无限挂起
    （前端表现为"上传超时""回答不出来"），且该进程往往需要提权才能杀掉。

    需要改代码自动重启时，请手动执行并在出问题后参考 README 的排查步骤。
"""
import subprocess
import sys
import os

PORT = os.getenv("BACKEND_PORT", "8000")


def main():
    backend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend')

    print("正在启动 AI 技术文档助手后端服务...")
    print(f"工作目录: {backend_dir}")
    print(f"访问地址: http://localhost:{PORT}")
    print(f"API文档: http://localhost:{PORT}/docs")
    print("=" * 50)

    cmd = [
        sys.executable, "-m", "uvicorn",
        "main:app",
        "--host", "0.0.0.0",
        "--port", PORT,
    ]

    subprocess.run(cmd, cwd=backend_dir)


if __name__ == "__main__":
    main()
