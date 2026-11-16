## 下载 / Downloads

| 平台 / Platform | 文件 / File |
| --- | --- |
| Windows x64 | `*-setup.exe` |
| macOS Apple Silicon / Intel | `*-macos-arm64.dmg` / `*-macos-intel.dmg`（或 `.pkg`） |
| Linux x86_64 | `.deb` / `.rpm` |
| Android 10+（64 位） | `CC-Media-Studio-*-android.apk` |

生成在你自己的 Kaggle 双 T4 上进行；本机只需填写 Kaggle 用户名和 API Key。Generation runs on your own Kaggle dual T4; enter your Kaggle username and API key.

- **安卓（预览）**：允许“安装未知来源应用”后打开 APK。未配置发布签名时，APK 使用临时调试密钥签名：升级前需先卸载旧版本（会删除应用内的画廊与设置）。详见源码中的 `docs/android.md`。
- **Android (preview):** allow installing unknown apps, then open the APK. Without release signing configured, the APK is signed with a temporary debug key: uninstall the previous version before updating (this removes the in-app gallery and settings). See `docs/android.md` in the source.
- Windows 与 macOS 安装包未签名，Apple 公证未配置。Windows and macOS packages are unsigned and not notarized.
- 校验和见 `SHA256SUMS-*.txt`。Checksums are in `SHA256SUMS-*.txt`.
