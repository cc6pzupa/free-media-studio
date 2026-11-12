# 安装包 / Desktop installers

在 [Releases](https://github.com/cc6pzupa/free-media-studio/releases) 下载适合电脑的安装包。安装包自带 Python 与客户端依赖，无需单独安装 Python、Git、CUDA 或模型权重。

| 平台 / Platform | 文件 / Package |
| --- | --- |
| Windows 10/11 x64 | `windows-x64-setup.exe` |
| macOS 15+ Intel | `macos-intel.dmg` / `macos-intel.pkg` |
| macOS 15+ Apple Silicon | `macos-arm64.dmg` / `macos-arm64.pkg` |
| Linux x86_64 / amd64（glibc 2.35+） | `.deb` / `.rpm` |

Windows 运行安装程序，从开始菜单打开 CC Media Studio。macOS 将 DMG 内的应用拖入 Applications，或运行 PKG。Linux 使用软件安装器打开 deb/rpm，再从应用菜单启动；界面使用默认浏览器。

启动后填写 Kaggle 用户名和 API Key。账号、历史记录和原文件保存在用户目录 `~/.kaggle-qwen-client`；升级和卸载安装包不会删除这些作品。图片和视频模型继续在 Kaggle 运行。

当前 Windows 和 macOS 安装包未配置发行者签名或 Apple 公证；系统可能拦截或提示发行者未知。SHA256SUMS 用于核对下载完整性。

## 历史画廊

图片和视频工作台均有“历史画廊”入口，地址为本机 `/gallery`。可搜索提示词、筛选图片/视频、收藏、打开预览、播放 MP4、下载原文件或复制提示词。

旧版已保存的完成任务会自动显示。记录与文件都在这台电脑，浏览画廊不会启动 GPU；通过 GUI、MCP、ComfyUI 或本机 API 生成并保存到共享后台的作品都可显示。直接在 Kaggle Notebook 生成的文件需要自行下载，不会自动导入本机记录。

删除作品会删除本机原文件，不可撤销；不会删除云端文件。已被手动移走的原文件会显示缺失，但仍可删除其记录。

## English

Download a matching package from Releases. Python and client dependencies are included; no local GPU, CUDA, Git or Python installation is required. Install Windows setup, drag the macOS app into Applications (or use PKG), or install the Linux deb/rpm. Linux opens the interface in your default browser. macOS builds target macOS 15 or later; Linux targets x86_64 with glibc 2.35 or later.

Enter your Kaggle username and API key after launch. Settings, history and originals remain under `~/.kaggle-qwen-client` across upgrades and uninstall. Windows/macOS publisher signing and Apple notarization are not configured. Checksums are supplied with the downloads.

Open Gallery from either workspace to browse existing images and videos, search prompts, filter media, favorite works, play MP4, download originals or copy prompts. Browsing never allocates a GPU. The gallery includes completed jobs from the shared local GUI, MCP, ComfyUI and API worker. Standalone Kaggle Notebook files are not automatically imported. Deletion permanently removes the local original; cloud files are unaffected.

## 打包与验证 / Building

```bash
python -m pip install -r requirements-build.txt
python -m PyInstaller --noconfirm packaging/cc-media-studio.spec
python packaging/smoke.py dist/cc-media-studio/cc-media-studio
```

各平台在对应原生系统与 CPU 上构建；macOS smoke 路径为 `dist/CC Media Studio.app/Contents/MacOS/cc-media-studio`，Windows 为 `dist/cc-media-studio/cc-media-studio.exe`。

`Desktop installers` 工作流构建四个原生目标，执行源码和打包后测试，生成 Windows setup、两个 macOS 架构的 DMG/PKG、Linux DEB/RPM 和校验和。功能分支通过全部构建后发布预览 Release；`v*` 标签构建正式 Release。标签版本应与 `app_entry.py`、安装脚本和包版本保持一致。

安装版 MCP 可用同目录的 `cc-media-cli --mcp`，独立后台用 `cc-media-cli --worker`，停止后台用 `cc-media-cli --worker --stop`。macOS 命令位于应用包 `Contents/MacOS` 内；源码版的原配置保持有效。
