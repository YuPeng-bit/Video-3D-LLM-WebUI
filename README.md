# Video-3D-LLM WebUI

为 Video-3D-LLM 的官方 2.5D 推理和评测提供逐样本交互界面，支持 ScanQA、SQA3D、Scan2Cap、ScanRefer、Multi3DRefer。

- 从任务中选择问题，查看 GT 和场景 RGB/深度帧。
- 默认沿用官方 uniform32 采样、文本模板与解码参数。
- 可修改帧、问题、附加 prompt 和生成参数，观察实际送入模型的输入。
- 保存原始输出、预测与官方评分；历史结果和缓存存放在本地。
- 模型按需加载，单 GPU 串行推理；提供显存释放与空闲自动卸载。

## 部署

先读 [部署说明](docs/SETUP.md)，准备 [Video-3D-LLM](https://github.com/LaVi-Lab/Video-3D-LLM) 上游代码、模型和有权限使用的数据，再运行 `sh start_webui.sh`，默认端口 8099。

本仓库只包含 WebUI 源码，不包含模型、数据、原始实验日志或登录凭据。机器路径通过环境变量或不入库的 `.env.local` 提供。

## 结构

`app/config.py` 管理任务与配置；`app/store.py` 读取问题、GT 与官方帧索引；`app/engine.py` 复用上游预处理与模型；`app/metrics.py` 复用官方评分器；`app/server.py` 提供 API、队列与记录；`web/` 为原生 HTML/CSS/JS 前端。

`tools/` 包含小样本推理、评分比对和安全服务管理工具。停止/重启只匹配本项目目录及选定端口，不匹配其他 WebUI，不自动强制结束无关进程。`tunnel.py` 是可选的 SSH 转发工具，连接参数由运行时传入。

## 边界与来源

默认帧索引来自 EmbodiedScan metadata，不是目录扫描。GT 只用于展示和生成后的官方评分。逐样本指标不代表完整 benchmark 成绩；恢复数据后应在新机器自行验收。

