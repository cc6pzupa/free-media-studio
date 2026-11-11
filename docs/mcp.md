# 本地 MCP：第四个入口

**简体中文** | [English](mcp.en.md)

让支持本地 stdio MCP 的 AI 客户端调用 Kaggle 生成图片和视频。GUI、ComfyUI 节点和 MCP 现在都连接同一个独立后台，后台拥有唯一任务队列、Kaggle 连接与图片记录；Kaggle 独立 Notebook 继续复用同一套云端运行时代码。

本机不需要显卡、CUDA、Torch。MCP 依赖独立安装，不会增加普通 GUI 使用者的安装要求。

## 安装

在仓库根目录，用本项目的虚拟环境 Python 执行：

```bash
python -m pip install -r requirements-mcp.txt
```

这里的 `python` 指 `.easy-venv` 内的 Python。没有激活环境时：

```powershell
# Windows
.\.easy-venv\Scripts\python.exe -m pip install -r requirements-mcp.txt
```

```bash
# macOS / Linux
.easy-venv/bin/python -m pip install -r requirements-mcp.txt
```

MCP 使用官方 Python SDK 2.x，支持自动协商新协议和旧版初始化连接。

## 配置账号

推荐先在 GUI 保存一次 Kaggle 用户名与 Key，然后关闭 GUI。MCP 自动读取同一用户目录中的配置，包括默认“双卡优先”的加速策略。`get_job` 的 `usage.parallel_strategy` 显示实际执行方式；Qwen 双卡 CFG 为 `cfg_parallel`，编码 / 生成分卡为 `encoder_split`。

也可在 MCP 客户端为这个服务设置环境变量：

- `KAGGLE_USERNAME`：Kaggle 用户名。
- `KAGGLE_API_TOKEN`：Kaggle API Token；旧 Key 可用 `KAGGLE_KEY`。
- `CC_MEDIA_ACCELERATION`：可选，`auto`（默认双卡优先）或 `split`（原分卡），用于未保存加速设置的配置。
- `EASY_CONFIG_PATH`：可选，指定另一份本机账号配置路径。

已保存的 GUI 配置优先于环境变量。一个配置目录共用一个后台和一个账号；不同账号应使用不同的 `EASY_CONFIG_PATH`。正在排队或生成时不能切换账号。首次启动后台时会读取配置，客户端不会通过工具参数询问或返回 Key。

未勾选“记住配置”时，账号只在后台内存中保留；关闭 GUI 不会清除运行中的后台配置，停止后台后需要重新配置。

## 接入 MCP 客户端

在支持 `mcpServers` 配置格式的客户端中添加以下内容，并把路径替换为 clone 后的真实绝对路径。其他客户端请在其“本地 MCP / stdio”设置中填写相同的可执行文件与参数。

### Windows

```json
{
  "mcpServers": {
    "cc-media-studio": {
      "command": "C:/absolute/path/free-media-studio/.easy-venv/Scripts/python.exe",
      "args": ["C:/absolute/path/free-media-studio/mcp_server.py"]
    }
  }
}
```

### macOS / Linux

```json
{
  "mcpServers": {
    "cc-media-studio": {
      "command": "/absolute/path/free-media-studio/.easy-venv/bin/python",
      "args": ["/absolute/path/free-media-studio/mcp_server.py"]
    }
  }
}
```

Windows 上，先在 GUI 保存账号并提交一次任务，让独立后台保持运行，再连接 MCP。也可以在单独的终端启动后台并保持该终端打开：

```powershell
.\.easy-venv\Scripts\python.exe studio_worker.py
```

macOS / Linux 可由 MCP 自动启动独立后台。

不要在终端手动运行 MCP 之后等待网页打开。MCP 由 AI 客户端启动，通过标准输入输出通信；它本身不打开窗口，也不是 GUI 的 HTTP 地址。

默认空闲保活 15 分钟，可在 GUI 修改；`studio_status` 同时返回 GPU 额度、保活配置及倒计时。查询额度不启动 GPU，也不重置保活计时。自动释放需要本机后台在线，只作用于自动账号模式的当前会话。

## 六个工具

| 工具 | 参数 | 返回 |
| --- | --- | --- |
| `studio_status` | 无 | 账号是否配置、排队数量、当前任务、最近一次模型使用情况 |
| `generate_image` | `prompt`、`request_id`；可选 `seed=42`、`model_id`、`steps`（省略时采用模型默认步数） | 立即返回任务 ID，不等待首次下载或生图 |
| `get_job` | `job_id` | `queued/running/complete/error/interrupted`、真实阶段、下载字节与百分比、复用信息 |
| `get_image` | 已完成的 `job_id` | MCP `image/png` 内容，以及图片尺寸和耗时等元数据 |
| `generate_video` | `prompt`、`request_id`；可选 `model_id`、`frames`、`seed=42`、`steps` | 立即返回视频任务 ID |
| `get_video` | 已完成的视频 `job_id` | 本机 MP4 路径、文件 URI、文件大小与耗时 |

每个图片任务输出一张 512×512 PNG；视频任务输出一个 MP4。`request_id` 是 AI 客户端为每次预期生成分配的唯一编号，例如 `fox-20261109-001`。同一提交重试时复用该编号，会取回原任务；相同编号搭配不同参数会报错，避免静默生成第二张。

`studio_status` 不申请 GPU。`last_usage` 是历史信息，不保证 Kaggle 当前仍存活；实际生成时后端会重新检查会话和模型。

## 一次完整调用

1. 用户说“帮我生成一只窗边的小狐狸”。
2. AI 调用 `studio_status`，确认本机已有配置。
3. AI 调用 `generate_image(prompt=..., request_id=...)`，获得 `job_id`。
4. 后台自动启动或找回 Kaggle 会话，复用热模型；必要时读取私有缓存或首次下载。
5. AI 每隔至少两秒调用 `get_job(job_id)` 查询进度。
6. 状态为 `complete` 后调用 `get_image(job_id)`。支持图片展示的客户端可直接显示图片；PNG 同时保留在本机。

下载和复制阶段显示真实字节；加载、上传等阶段只显示实际状态，不伪造完成百分比。

## 后台、持久化与恢复

后台按需启动，只绑定 `127.0.0.1`，使用本机私有连接记录认证。MCP 的 stdout 只用于协议；普通日志在 stderr，后台日志单独保存。

默认数据目录：`~/.kaggle-qwen-client/studio/`（Windows 同样位于当前用户目录）：

- `jobs.sqlite3`：任务记录、参数、进度和使用统计，不保存 Kaggle Key。
- `images/<job_id>.png`：已完成的图片。
- `images/<job_id>.mp4`：已完成的视频。
- `connection.json`：本机后台地址与访问凭据，不要分享。
- `worker.log`：后台诊断日志。

自定义 `EASY_CONFIG_PATH` 时，数据位于该配置文件旁边的 `studio/`。账号、连接记录、任务和图片只保存在本机；它们已被 `.gitignore` 排除。

关闭 GUI、断开 MCP 或重启 MCP 客户端不会停止后台，任务继续执行，原任务 ID 仍可查询。GUI 重新打开后也能取回 MCP 提交的任务；ComfyUI 无需 GUI 窗口或 GUI 的 HTTP 服务。

如果后台进程本身崩溃，已经执行中的任务会标为 `interrupted`，不会自动重发，因为远端可能仍在运行。已完成的图片仍可读取；未开始的队列任务可在同一配置下恢复。不要把“本机连接中断”当作远端任务已取消。

需要关闭后台时，先等队列结束，再执行：

```bash
python studio_worker.py --stop
```

这只停止本机后台，不会停止 Kaggle GPU 会话。Kaggle 用完仍需在 Kaggle 主动停止。

更新代码前也应先完成任务、停止本机后台，再 `git pull` 和更新依赖，之后重启 GUI / MCP 客户端，使后台使用新版本代码。

## 模型选择

`generate_image` 的 `model_id` 可选 `qwen-image-2.1`、`z-image-turbo` 或 `sdxl-lightning`。省略 `steps` 时分别使用 20、9、4 步；SDXL-Lightning 固定为 4 步。

```text
generate_image(prompt="An adult catgirl by a window", request_id="catgirl-turbo-001", model_id="z-image-turbo")
```

模型切换时释放旧模型；同一模型继续复用。新增模型使用 GUI 保存的自动账号配置。返回的 usage 包含 model_id、编码、采样及解码耗时。


## 通过 MCP 生成视频

视频入口位于本仓库的 `feature/video-three-tiers` 预览分支。先按上面的步骤保存账号、安装 MCP 依赖并连接客户端。

| 档位 | `model_id` | 默认步数 |
| --- | --- | --- |
| 效率 | `ltx-video-2b-distilled` | 固定 8 |
| 平衡 | `wan21-t2v-1.3b` | 30 |
| 质量 | `wan22-ti2v-5b` | 30 |

省略 `frames` 时生成 33 帧，可设为 9、17、25……81。省略 `steps` 使用该模型的默认值。

可以直接对 AI 说：“用 CC Media Studio 的效率档生成一段窗边读书的成年猫娘视频，33 帧。完成后给我 MP4 文件。”

对应调用顺序：

```text
studio_status()
generate_video(prompt="An adult anime catgirl reading by a sunlit window. Her hair moves gently in the breeze.", request_id="cc-video-001", model_id="ltx-video-2b-distilled", frames=33, seed=42)
get_job(job_id="<generate_video 返回的 job_id>")
get_video(job_id="<同一个 job_id>")
```

`generate_video` 只提交任务。保留返回的 `job_id`，每隔至少两秒查询 `get_job`，直到 `state` 为 `complete`，再调用 `get_video`。首次使用显示下载字节与百分比，生成时显示采样步数，之后依次解码和导出。

`get_video` 返回 `path`、`uri`、`mime_type=video/mp4`、`size_bytes` 与 `usage`。AI 客户端可以用本机路径打开视频或提供文件下载。视频不作为 PNG 图片返回。

图片与视频共用一个串行队列。连续请求同一视频模型会复用活跃进程；切换到图片或另一模型会释放旧模型。GUI 保存的加速策略和空闲保活时间同样适用于 MCP 视频任务。
