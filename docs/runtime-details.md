> 历史运行细节。当前 GUI、ComfyUI、MCP 已共用独立后台，安装及后台生命周期以根目录 README 和 MCP 指南为准。

> 本页中的文件路径和命令以仓库根目录为基准。

# CC Media Studio · Kaggle

三个入口共用同一套常驻模型：本机 GUI、ComfyUI 节点、Kaggle 独立 Notebook。同一个 Kaggle 内核还在运行时只加载一次；关闭编辑器网页或重启本机客户端后，可以接回原模型。

## 本机 GUI

需要 **Python 3.11 或更新版本**。Windows 双击 `start_windows.bat`；macOS 双击 `start_mac.command`。首次启动器自动安装客户端依赖，并打开 CC Media Studio 窗口。未安装原生窗口依赖时会打开本机浏览器页面：`http://127.0.0.1:8787`。

只填写自己的 **Kaggle 用户名、API Key / API Token**。账号需要完成 Kaggle 验证并有可用 GPU 额度。程序会自动创建私有专用 Notebook、申请 T4 ×2、取得 Jupyter 连接，并检查已有模型。无需自己填写 Jupyter 地址或预先准备 Dataset。专用私有 Notebook 的元数据会保留启动任务编号（不含 Key 或连接地址），新装客户端也能通过账号认证找回仍在运行的模型。

输入提示词后点击生成。固定输出一张 512×512 PNG，默认 20 步、seed 42。下载阶段显示真实字节、百分比和速度；模型加载、INT8 保存、上传等阶段显示实际状态，不伪造进度百分比。

## 从零准备与私有缓存

1. 优先检查自己的私有 `用户名/qwen-image-studio-assets`。
2. 没有缓存时，从公开 Hugging Face 仓库下载固定版本的 Qwen Image 2.1 组件与 Q4_K_M GGUF；约 **23.51 GB**。支持断点续传，LFS 大文件校验 SHA256。
3. 文本编码器首次在 GPU 1 加载为 INT8，主模型放在 GPU 0。自动保存 INT8 编码器及必要组件，并创建私有 Dataset。首次会多花保存和上传时间。
4. 新内核优先拉取这份私有缓存，直接加载已准备的 INT8/Q4 权重。同一内核的下一张图片继续复用内存模型。

缓存只创建新 Dataset，不覆盖已有同名、不兼容或无法验证的 Dataset。缓存读写失败时页面会提示，仍可使用公开权重生图。缓存保存成功会显示“私有模型缓存已保存”。私有 Dataset 拉取采用 4 路并行，支持断点续传并显示真实字节进度；上传显示文件/阶段状态。

公开来源及固定版本：

| 组件 | 仓库 | 版本 |
| --- | --- | --- |
| 编码器、VAE、配置、处理器 | `Qwen/Qwen-Image-2.1` | `d26bb61231c349cf6b7896fa83353113880e1ba3` |
| Q4_K_M 主模型 | `azukivc/Qwen-Image-2.1-GGUF` | `4e43fd824763d3d24233630c343462592f68ca49` |

## ComfyUI 独立使用

将整个 `comfy_kaggle_node` 文件夹放入 `ComfyUI/custom_nodes/`，在 ComfyUI 的 Python 环境安装 `requirements.txt`，重启 ComfyUI。

先在 GUI 保存一次账号配置，之后 GUI 和本机 HTTP 服务都可以关闭。或者在启动 ComfyUI 前设置 `KAGGLE_USERNAME` 与 `KAGGLE_API_TOKEN`（旧 Key 可用 `KAGGLE_KEY`）。

添加 **CC Media Image**，将 `IMAGE` 接到 **Save Image** 的 `images`。节点直接启动或连接 Kaggle 后端，不通过本机 HTTP 服务，凭据不写入工作流。

## Kaggle 直接运行

导入 `kaggle_standalone.ipynb`，选择 **GPU T4 ×2**、开启 Internet，运行所有单元格。输入提示词并点击按钮，第一次自动下载和加载，之后直接复用。此 Notebook 已嵌入全部必要运行代码，无需下载本机项目。

不填任何账号信息也可以生成，文件保存在当前会话临时磁盘。如果希望自动创建和读取跨会话私有缓存，展开 Notebook 中的可选设置，填写自己的 Kaggle 用户名与 API Key。不要保存带凭据的 Notebook 输出。

## 关闭、重连和固定 API

| 操作 | 后续行为 |
| --- | --- |
| 关闭 Notebook 编辑器网页 | Kaggle GPU 会话仍存活时继续生成。 |
| 再次请求或重启 GUI / ComfyUI | 重新连接同一个内核，模型仍在时直接复用。 |
| Kaggle 会话被回收或主动停止 | 自动模式下次请求申请新会话，首次加载一次；私有缓存存在时优先读取。 |
| 网络暂时中断 | 报错，不自动重发已提交的生图；用户可检查后再次请求。 |
| 远端已有不兼容模型或内核选择不明确 | 报错，不加载第二份模型。 |

关闭客户端只断开连接，**不会停止 Kaggle GPU 会话**。Kaggle 的额度、闲置回收与最长会话时限仍生效，用完请在 Kaggle 停止会话。

GUI 运行时仍提供 `POST http://127.0.0.1:8787/v1/images/generations`。使用高级设置中显示的本机 API 密钥作为 `Authorization: Bearer ...`，它与 Kaggle Key 不同。请求示例见 `test_api_example.py`。

```json
{"prompt":"An orange fox near a teacup, natural photography","size":"512x512","steps":20,"seed":42}
```

返回 `data[0].b64_json` 为 PNG；`usage` 中 `runtime_id` 标识模型实例，`model_load_count` 是加载次数，`model_reused` 是本次是否复用，`generation_number` 是已完成生图次数；另有 `asset_source`、`cache_status` 与耗时。

高级设置保留手动 Jupyter 会话模式及旧批处理模式。批处理每次新建任务；常驻模式不会自动切换为批处理。

## 验证与配置

```bash
python -m unittest discover -s tests -v
node --test tests/test_ui.cjs
```

本地检查通过 40 项 Python 测试与 6 项页面逻辑测试。

真实 GPU 运行记录见 `verification.json`，示例图见 `examples/`。已验证干净内核从私有 Dataset 加载并生成 512×512 图片，未使用 Hugging Face 下载。实测覆盖账号自动启动、公开权重下载、GUI API、Kaggle Notebook 及同一模型实例的连续请求。Windows/macOS 原生窗口、完整 ComfyUI 画布需要在对应环境验收。

配置默认位于本机用户目录 `.kaggle-qwen-client/settings.json`；会话连接缓存位于同目录 `sessions/`。包含敏感信息，按当前用户私有权限保存。不要分享这些文件或加入分发包。本包不包含真实凭据。
