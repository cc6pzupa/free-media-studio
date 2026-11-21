## 新功能 / What's new

所有生成都在你自己的 Kaggle 会话上进行，以下功能已在 Kaggle 双 T4 上实测。All generation runs on your own Kaggle session; everything below was tested on Kaggle dual T4.

- **语言模型（预览）**：聊天页 `/chat` 与 OpenAI 兼容接口 `/v1/chat/completions`。输入模型名（如 Qwen3-8B、Qwen3-30B-A3B），自动从 Hugging Face 选出能放进双 T4 的最佳 GGUF 量化版本：Qwen3-8B 约 27–37 token/秒，Qwen3-30B-A3B 约 55–66 token/秒。支持工具调用和图片输入，也可以在不占 GPU 额度的 CPU 会话上运行（Qwen3-30B-A3B 约 5 token/秒）。
- **多个 Kaggle 账号**：保存并切换账号，切换不会结束其他账号的会话。账号需分别属于不同的人。
- **本机 AI 工具**：自动发现 OpenCode、Aider、Continue、Open WebUI 等并一键接入当前模型；缺少时可一键安装。修改前备份原文件。
- **图片与视频**：图生图、局部编辑（橡皮擦、撤销、羽化）、图生视频（首帧，LTX 还支持首尾帧）、更多尺寸、负面提示词、批量与 ZIP、取消排队任务。
- **安卓 App（预览）**：与桌面版相同的界面，包括聊天页。

- **Language models (preview):** a chat page and an OpenAI-compatible `/v1/chat/completions`. Enter a model name and the best GGUF build that fits dual T4 is chosen from Hugging Face (Qwen3-8B 27–37 tok/s, Qwen3-30B-A3B 55–66 tok/s). Tool calls, image input, and a CPU-only session that uses no GPU quota (about 5 tok/s for Qwen3-30B-A3B).
- **Several Kaggle accounts**, switched without ending the other sessions (each account must belong to a different person).
- **Local AI tools:** OpenCode, Aider, Continue, Open WebUI and others are found and connected in one click, or installed; changed files are backed up.
- **Images and video:** image-to-image, inpainting, image-to-video (LTX first and last frame), more sizes, negative prompts, batches with ZIP, cancelling queued tasks.
- **Android app (preview)** with the same pages, including chat.

详见源码中的 `CHANGELOG.md`、`docs/llm.md`、`docs/editing.md`。See `CHANGELOG.md`, `docs/llm.md` and `docs/editing.md` in the source.

## 下载 / Downloads

| 平台 / Platform | 文件 / File |
| --- | --- |
| Windows x64 | `*-setup.exe` |
| macOS Apple Silicon / Intel | `*-macos-arm64.dmg` / `*-macos-intel.dmg`（或 `.pkg`） |
| Linux x86_64 | `.deb` / `.rpm` |
| Android 10+（64 位） | `CC-Media-Studio-*-android.apk` |

本机只需填写 Kaggle 用户名和 API Key。Enter your Kaggle username and API key.

- **安卓（预览）**：允许“安装未知来源应用”后打开 APK。未配置发布签名时，APK 使用临时调试密钥签名：升级前需先卸载旧版本（会删除应用内的画廊与设置）。详见源码中的 `docs/android.md`。
- **Android (preview):** allow installing unknown apps, then open the APK. Without release signing configured, the APK is signed with a temporary debug key: uninstall the previous version before updating (this removes the in-app gallery and settings). See `docs/android.md` in the source.
- 从旧版本升级后，请重启本机后台，以启用语言模型和多账号。After updating, restart the local worker to enable language models and accounts.
- Windows 与 macOS 安装包未签名，Apple 公证未配置。Windows and macOS packages are unsigned and not notarized.
- 校验和见 `SHA256SUMS-*.txt`。Checksums are in `SHA256SUMS-*.txt`.
