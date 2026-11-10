# CC Media Studio 运行方式

安装和使用步骤见 [README](../README.md)，MCP 配置见 [MCP 指南](mcp.md)，模型策略与实测见 [性能说明](performance.md)。

GUI、ComfyUI 和 MCP 连接同一个本机后台，共用任务队列、账号配置、Kaggle 连接和已生成图片。HTTP API 使用相同后台。Kaggle 独立 Notebook 内嵌相同云端运行时。

## 自动准备与复用

首次使用只需 Kaggle 用户名与 API Key。自动模式准备私有专用 Notebook、双 T4 会话及所选模型。优先读取该模型的私有 Dataset 缓存，没有缓存时从固定版本的公开 Hugging Face 权重下载，显示真实文件字节进度。

同一模型的后续请求复用已加载的实例。Qwen 在自动模式下另外缓存一个 Q4 副本，两卡同时处理 CFG 正负分支；编码器在提示词编码和采样之间迁移，权重不重复加载。重新连接同一内核会保留主模型、编码器与副本。切换模型释放当前实例，已下载文件继续保留。

## 加速配置

所有入口默认 `auto`，双卡优先。GUI 的“加速策略”、Kaggle Notebook 下拉框均可选择 `split`，使用原编码 / 生成分卡模式。CLI 为 `--acceleration auto|split`；没有保存策略时，可用 `CC_MEDIA_ACCELERATION=auto|split` 设置默认值。

Qwen 使用 `cfg_parallel`；Z-Image Turbo 和 SDXL-Lightning 使用 `encoder_split`。实际执行方式与分段耗时保存在生成结果的 `usage` 中。Qwen 加速异常会清理副本并恢复原调用，按相同请求和种子生成；该内核后续使用原路径。

## 会话与后台

空闲保活默认 15 分钟，可设为 0–1440 分钟。下载、准备、生图、排队期间不触发空闲释放。查询状态与 GPU 用量不延长倒计时。

关闭 GUI 或 MCP 客户端后，本机后台继续处理任务和空闲计时。Kaggle 会话仍存活时，重新连接继续复用；会话已释放时，下一次自动请求准备新会话并读取缓存。手动 Jupyter 模式和独立 Kaggle Notebook 的会话由用户管理。

账号配置位于当前用户目录 `.kaggle-qwen-client/settings.json`，任务和 PNG 位于同目录的 `studio/`。ComfyUI、GUI 与 MCP 使用同一系统用户时共享这些设置。

## 更新客户端

先完成队列，再按 [README 的更新步骤](../README.md#更新) 停止本机后台、更新代码和依赖，然后重新启动。Kaggle 会话仍存活时，客户端可重新连接并复用已加载模型。
