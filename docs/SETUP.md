# 部署说明

本仓库将 [Video-3D-LLM](https://github.com/LaVi-Lab/Video-3D-LLM) 的官方 2.5D 推理链路与五任务评分器接入 FastAPI 界面，供逐样本观察和输入实验使用。它不是完整模型或数据发布包。

## 先准备上游模型和数据

按 Video-3D-LLM 上游说明建立环境、下载 checkpoint 和数据，保留其代码及数据布局。需要 posed RGB/depth/pose、EmbodiedScan metadata、五任务问题/GT、相应物体 proposals 和官方 scorer 依赖。ScanNet 数据由有使用权限者准备；模型、数据、参考全量预测和实验日志均不包含在本仓库。

实验室已用环境为 Python 3.10、Torch 2.1.2、Transformers 4.40.0.dev0。`dev0` 是原环境中的开发版本，不能理解为可任意替换的稳定 pip 版本；应先遵循上游的环境说明，再安装界面层依赖。Java 用于 METEOR。不要与原生 VTC 的 Transformers 环境混装。

## 配置启动

```bash
git clone https://github.com/YuPeng-bit/Video-3D-LLM-WebUI.git
cd Video-3D-LLM-WebUI
export V3D_REPO=/path/to/Video-3D-LLM
export V3D_CKPT=/path/to/Video3D-LLM-LLaVA-Qwen-Uniform-32
export V3D_PYTHON=/path/to/video3dllm-env/bin/python
"$V3D_PYTHON" -m pip install -r requirements-webui.txt
HOST=127.0.0.1 PORT=8099 sh start_webui.sh
```

`V3D_REPO` 应指向已准备数据的上游仓库。`CUDA_VISIBLE_DEVICES=1` 可限制 GPU；未指定时启动脚本选择空闲显存最多的一张可见卡，模型第一次推理才加载。数据文件清单和评测划分见 `app/config.py`，帧索引来自 EmbodiedScan metadata，而不是目录扫描。

远程访问：`ssh -N -L 8099:127.0.0.1:8099 USER@SSH_HOST`，然后打开 `http://127.0.0.1:8099`。可选 `tunnel.py` 使用运行时 SSH 参数、密钥/agent 或交互密码，不内置任何服务器凭据。

停止当前项目：`PORT=8099 sh stop_webui.sh`；重启当前项目：`PORT=8099 sh restart_webui.sh`。停止工具按工作目录、uvicorn 参数和端口匹配，不停止其他项目。它不会自动强制结束没有响应的进程。

## 使用与验收

默认同上游 uniform32、官方文本模板和解码参数；页面可选题目、帧、prompt 和思考提示词。五任务 GT 只用于展示与生成后的评分。Video3D checkpoint 不存在原生 `enable_thinking` 开关，此处的思考选项是提示词实验，与 VTC 原生思考不同。

恢复资产后，可按 README 运行 `tools/verify_metrics.py`、`tools/compare_official.py`；后者需要实验室官方 baseline 文件。新机器先验证页面预览和少量真实推理，不把 README 中历史实验数字当作新机器验收结果。

结果写入 `results/webui/runs.jsonl`，日志和缓存保留在本地；它们被 `.gitignore` 排除。前端位于 `web/`，模型/预处理在 `app/engine.py`，评测在 `app/metrics.py`，接口与串行队列在 `app/server.py`。

机器上的真实路径可保存到本机 `.env.local`，启动脚本会读取该文件；它被 `.gitignore` 排除。GitHub 中不包含任何实验室地址、绝对路径或私有状态记录。
