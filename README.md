# CC Media Studio · Kaggle

在 Windows、Linux、macOS 上使用自己的 Kaggle GPU 生图。客户端只负责界面、请求和图片显示，模型下载、加载、推理都在 Kaggle 完成；本机无需显卡、CUDA、PyTorch 或模型权重。

## 快速开始：clone → 安装依赖 → 运行

准备 **Git、Python 3.11+** 和可联网的电脑。推荐 Python 3.11 或 3.12。私有仓库需要先用有访问权限的 GitHub 账号完成 Git 认证；Kaggle Key 不用于 GitHub 认证。

使用有访问权限的 GitHub 账号克隆本私有仓库：

```bash
git clone https://github.com/cc6pzupa/free-media-studio.git qwen-image-studio
cd qwen-image-studio
```

### Windows（PowerShell 或命令提示符）

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe desktop.py
```

### Linux / macOS

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python desktop.py
```

程序会打开 GUI。未安装原生窗口组件时，在系统浏览器打开同样的界面：**http://127.0.0.1:8787**。保持启动进程运行，按 `Ctrl+C` 可退出 GUI 服务。已启动的独立后台会继续处理任务；停止后台请在队列结束后运行 `python studio_worker.py --stop`。

第一次使用只填写 **Kaggle 用户名 + API Key / API Token**，然后输入提示词、点击生成。无需自己填写 Jupyter 地址，也无需先准备 Dataset。

### 可选：独立窗口或自动安装

默认依赖已足够运行浏览器 GUI。Windows/macOS 如需独立窗口，在同一个虚拟环境内额外安装：

```bash
python -m pip install -r requirements-desktop.txt
python desktop.py
```

上面的 `python` 指已激活虚拟环境的 Python；未激活时，使用前文对应的 `.venv` Python 完整路径。原生窗口不可用时会退回系统浏览器。也可以用 `python desktop.py --browser` 固定使用浏览器界面。

不想手动创建环境，可直接运行 `python launch.py`。它创建独立的 `.easy-venv`，安装依赖，并在依赖清单变化后自动更新；Windows/macOS 会尝试安装原生窗口组件。

| 系统 | 自动启动入口 |
| --- | --- |
| Windows | 双击 `start_windows.bat` |
| macOS | `bash start_mac.command` |
| Linux | `bash start_linux.sh` |

## 使用流程

1. 在 GUI 填 Kaggle 用户名和 API Key，按需勾选“记住配置”。账号需已完成 Kaggle 验证并有可用 GPU 额度。
2. 输入提示词，点击生成。程序自动创建或找回私有专用 Notebook，并申请双 T4 会话。
3. 模型已在当前内核中时直接生成。新内核优先读取自己的私有 `用户名/qwen-image-studio-assets`。
4. 完全首次使用时，从公开 Hugging Face 下载约 **23.51 GB** 权重，显示真实进度；准备 INT8 编码器和 Q4 主模型，自动保存约 **16 GB** 私有 Dataset 缓存。
5. 接收一张 **512×512 PNG**，可在 GUI 保存。后续请求继续复用已加载的模型。

同一个 Kaggle 内核存活时模型只加载一次。关闭 Notebook 编辑器网页、退出或重启客户端，不会主动停止 GPU 会话；新装客户端也能通过专用私有 Notebook 的启动任务编号恢复连接。Kaggle 会话结束后需要加载一次，已有私有缓存时无需重新从 HF 下载、量化。

缓存下载支持 **4 路并行、断点续传、真实字节进度**；上传和模型加载显示实际阶段。首次保存私有缓存需要额外等待。程序不会覆盖同名的不兼容 Dataset。

## ComfyUI：独立运行，无需保持 GUI 打开

在 `ComfyUI/custom_nodes` 中 clone 本仓库，然后用 **ComfyUI 所用的 Python** 安装依赖并重启 ComfyUI：

```bash
git clone https://github.com/cc6pzupa/free-media-studio.git ComfyUI/custom_nodes/qwen-image-studio
<ComfyUI的Python> -m pip install -r ComfyUI/custom_nodes/qwen-image-studio/requirements.txt
```

以上路径以 ComfyUI 的上级目录为基准；已经进入 `ComfyUI/custom_nodes` 时，直接 clone 到 `qwen-image-studio` 即可。

先在 GUI 保存一次账号配置，或在启动 ComfyUI 前设置 `KAGGLE_USERNAME`、`KAGGLE_API_TOKEN`（旧 Key 可用 `KAGGLE_KEY`）。添加 **CC Media Image**，将 `IMAGE` 接到 **Save Image** 的 `images`。节点通过独立后台连接 Kaggle，不经过 GUI 的 HTTP 服务，凭据不写入工作流。本机 Torch、NumPy、Pillow 由 ComfyUI 环境提供。

## Kaggle：直接运行 Notebook

上传或导入 [`kaggle_standalone.ipynb`](kaggle_standalone.ipynb)，选择 **GPU T4 ×2**、开启 Internet，运行所有单元格，然后在 Notebook 的界面中输入提示词并生成。

Notebook 已嵌入必要运行代码，无需本机客户端或已有 Dataset。不填额外凭据也可从 HF 下载后生图；如需自动保存和读取跨会话私有缓存，展开可选设置，填写自己的 Kaggle 用户名和 API Key。Key 使用后会清空输入框；不要保存带凭据的 Notebook 输出。

## 本地 MCP：第四个入口

安装 `requirements-mcp.txt`，在 AI 客户端注册 `mcp_server.py`，即可通过对话调用生图。提供 `studio_status`、`generate_image`、`get_job`、`get_image` 四个工具，支持异步队列、真实进度、PNG 返回、请求去重和任务恢复。

GUI、ComfyUI 和 MCP 共用独立后台；关掉 GUI 后 MCP 仍可运行。账号配置复用本机保存内容，Key 不进入工具参数。

完整安装步骤、Windows/macOS/Linux 配置示例和恢复说明见 **[本地 MCP 使用指南](docs/mcp.md)**。

## 本机 API 与启动参数

GUI 运行时提供：

```text
POST http://127.0.0.1:8787/v1/images/generations
Authorization: Bearer <GUI高级设置中显示的本机API密钥>
```

```json
{"prompt":"An orange fox beside a teacup, natural photography","size":"512x512","steps":20,"seed":42}
```

返回 `data[0].b64_json` 为 PNG；`usage` 包含耗时、`runtime_id`、`model_load_count`、`model_reused`、`generation_number`、`asset_source`、`cache_status`。这个本机 API 密钥与 Kaggle Key 不同。示例：[test_api_example.py](test_api_example.py)。

```bash
python desktop.py --browser
python desktop.py --port 8788
python desktop.py --no-open --port 8787
```

`--no-open` 仅启动本机服务，不打开窗口，适合无桌面环境和启动检查；服务始终只监听本机地址。`--port 0` 可供测试自动选择空闲端口。通过 `launch.py` 启动时也可传递这些参数。

## 更新与测试

先等队列完成，停止本机后台并关闭 GUI / MCP 客户端，再在仓库目录执行：

```bash
python studio_worker.py --stop
git pull
python -m pip install -r requirements.txt
python desktop.py
```

这里的 `python` 同样指本项目虚拟环境中的 Python。自动启动器会在下次启动时检查依赖清单是否变化。

MCP 使用者更新后还需执行 `python -m pip install -r requirements-mcp.txt`。

开发检查（Node.js 只用于页面测试，普通使用无需安装；这些测试不需要真实 Kaggle Key，也不会申请 GPU）：

```bash
python -m unittest discover -s tests -v
node --test tests/test_ui.cjs
```

Linux 安装 MCP 依赖后的完整检查通过 **49 项 Python 测试、6 项页面逻辑测试**。MCP 协议测试使用真实 SDK 客户端和模拟 GPU 后端，不消耗 Kaggle 额度；真实 GPU 后端的既有验证记录仍保留。已验证本地 Git clone 后的手动启动和自动环境启动。

[verification.json](verification.json) 保留 Kaggle 验证结论摘要（已移除私有资源标识），[examples](examples) 包含结果图。已验证 HF 首次下载、私有缓存加载、跨客户端恢复会话，以及同一模型实例的连续请求。Windows/macOS 原生窗口与完整 ComfyUI 画布尚未实机验收；仓库包含三系统自动测试配置，上传后由 GitHub Actions 执行。

## 配置与常见问题

- **本机没有显卡可以吗？** 可以。模型与推理在 Kaggle，客户端不安装 GPU 依赖。
- **Linux 创建虚拟环境失败？** 某些发行版需先安装对应 Python 的 `venv` 组件，然后重试。
- **浏览器没有自动打开？** 手动访问启动日志中的本机地址；也可加 `--browser`。
- **端口被占用？** 使用 `--port 8788`，不会把请求交给不属于本程序的已有服务。
- **关闭 GUI 就停止计费/额度消耗吗？** 不会自动停止 Kaggle GPU，会话用完请在 Kaggle 主动停止。
- **Key 保存在哪里？** 本机用户目录 `.kaggle-qwen-client/settings.json`，连接记录在同目录 `sessions/`；POSIX 系统使用私有文件权限，Windows 使用用户目录的系统访问控制。不要上传或分享这些文件。

仓库的 `.gitignore` 已排除本机配置、会话凭据、虚拟环境和下载权重；仓库不包含真实 Key。权重来源、固定版本、旧版手动连接和批处理说明见 [运行细节](docs/runtime-details.md)。
