# 语言模型、多账号与本机 AI 工具（预览） / Language models, accounts and local AI tools (preview)

> 预览功能，已在 Kaggle 双 T4 上实测。需要 **自动账号模式**。图片和视频功能不受影响。

## 语言模型

聊天页 `/chat`（图片页左侧、视频页和画廊顶部都有入口）：

1. 输入模型名称，例如 `Qwen3-8B`、`Qwen3-32B`、`Qwen3-30B-A3B`，或 Hugging Face 仓库 `Qwen/Qwen3-8B-GGUF`。
2. 点 **查找并加载**。Kaggle 会话在 Hugging Face 上查找这个模型的 GGUF 量化版本，按文件大小和上下文长度估算显存与内存，自动选出能放进本会话、质量最高的一个，然后下载并加载。
3. 卡片显示所选版本、运行位置（全部在 GPU / GPU + 内存 / 仅 CPU）和预计速度；**其他版本** 可以改用更快或质量更高的版本。

选择规则：

- 只考虑原版模型及其直接转换（官方、unsloth、bartowski 等）；去除安全限制的改版、微调版和其他衍生模型不会被选中。
- 优先全部放进 GPU；放不下时，把部分层放到内存（速度主要取决于内存部分）；纯 CPU 时优先速度。
- 推理由 Ollama 完成：自动安装，自动把模型分到两块 T4 和内存。

### Kaggle 双 T4 实测（2026-11-18，生成 256–320 token）

| 模型 / 版本 | 位置 | 速度 |
| --- | --- | --- |
| Qwen3-8B Q4_K_M | GPU | 33–37 token/秒 |
| Qwen3-8B Q8_0（自动选择） | GPU | 27 token/秒 |
| Qwen3-8B Q4_K_M | GPU + 内存（一半层在 CPU） | 6.4 token/秒 |
| Qwen3-8B Q4_K_M | 仅 CPU（4 核） | 3.6 token/秒 |
| Qwen3-30B-A3B Q6_K（自动选择） | GPU | 9–10 token/秒（正在排查，见下） |

首次加载包括下载：Qwen3-8B Q8_0 约 4 分钟，Qwen3-30B-A3B Q6_K（25 GB）约 7–11 分钟；同一会话内再次使用直接复用。

同一个 Kaggle 会话里，语言模型和图片、视频模型互相替换（显存不够同时放下），切换时自动释放。要同时使用，请用两个账号。

### OpenAI 兼容接口

```http
POST http://127.0.0.1:8787/v1/chat/completions
Authorization: Bearer <本机 API Key>

{"model": "Qwen3-8B", "stream": true, "messages": [{"role": "user", "content": "你好"}]}
```

- `GET /v1/models` 列出已加载的模型。
- `model` 填已加载模型的名称，或 `default`；填另一个模型名会先自动加载该模型（首次需下载）。
- 支持 `stream`、`temperature`、`top_p`、`max_tokens`、`stop`、`seed`、`presence_penalty`、`frequency_penalty`。暂不支持工具调用和图片输入。
- 本机 API Key 在聊天页“在其他工具中调用”里复制。勾选“记住账号配置”后 Key 才会固定。

## 多个 Kaggle 账号

在“账号与连接”中保存多个账号（各自填写用户名和 Key，勾选记住）。对话框上方列出已保存的账号：

- **切换**：之后的新任务交给新账号。原账号的会话不会结束：正在运行的任务照常完成，排队的任务仍在原账号上执行，会话按它自己的空闲保活自动释放。切回时直接复用已加载的模型。
- **释放会话**：立即结束该账号的 Kaggle 会话，节省额度。
- **移除**：删除这个账号保存的 Key，不影响正在运行的会话。

账号需分别属于不同的人：Kaggle 规定一人一个账号，同一个人用多个账号叠加免费额度会有封号风险。

## 本机 AI 工具

聊天页的 **本机 AI 工具** 会查找这台电脑上能用 OpenAI 接口的工具，并一键接入当前模型：

| 工具 | 接入方式 |
| --- | --- |
| OpenCode | 在 `~/.config/opencode/opencode.json` 新增 “CC Media (Kaggle)” 提供方（带注释的 `.jsonc` 需手动添加） |
| Aider | 写入单独的 `~/.aider.cc-media.conf.yml`，用 `aider --config ~/.aider.cc-media.conf.yml` 启动，不改默认配置 |
| Continue（VS Code / JetBrains） | 在 `~/.continue/config.yaml` 的 models 中新增一项 |
| Open WebUI | 一键启动并连接（http://127.0.0.1:8080） |
| Cherry Studio、Chatbox | 显示需要填写的 Base URL、Key 和模型名 |
| Codex CLI | 暂不支持：新版只接受 OpenAI Responses API |

修改前会把原文件备份为 `<文件名>.bak-ccmedia`。未安装的 OpenCode、Aider、Open WebUI 可以一键安装（需要 npm，或 uv / pipx）。

## English

**Language models.** Open `/chat`, enter a model name (`Qwen3-8B`, `Qwen3-32B`, `Qwen3-30B-A3B`) or a GGUF repository, and select **Find and load**. The Kaggle session searches Hugging Face for GGUF builds of that model, estimates GPU and RAM needs from the file sizes and context length, and loads the highest-quality build that fits: all on the GPUs if possible, otherwise split with RAM, or CPU only. Only the original model and plain conversions are considered. Ollama runs the model and spreads it across both T4s and RAM. Measured on dual T4: Qwen3-8B 27 tok/s at Q8_0 and 33–37 tok/s at Q4_K_M on GPU, 6.4 tok/s half on CPU, 3.6 tok/s CPU only. Language, image and video models replace each other on one session.

**OpenAI-compatible API.** `POST /v1/chat/completions` (streaming or not) and `GET /v1/models` with the local API key. Naming another model in `model` loads it first. Tool calls and image input are not supported yet. The key stays the same only when the account is saved with Remember.

**Several accounts.** Save each account under Account & connection; the dialog lists them with Switch, Release session and Remove. Switching never ends the other account's session: its running and queued tasks finish there, and it is released by its own idle timer. Accounts must belong to different people (Kaggle allows one account per person).

**Local AI tools.** The chat page finds OpenCode, Aider, Continue, Open WebUI, Cherry Studio, Chatbox and Codex CLI on this computer and connects them to the current model: a provider in `opencode.json`, a separate Aider config file, one Continue model entry, Open WebUI started already connected, and the settings to copy for desktop apps. Codex CLI accepts only the Responses API and is not supported yet. Changed files are backed up as `<name>.bak-ccmedia`; missing tools can be installed with npm, uv or pipx.
