# 安卓应用（预览） / Android app (preview)

安卓应用与桌面版使用同一套本机程序和界面：生成仍在你的 Kaggle 双 T4 上进行，手机只负责界面、任务队列和画廊。

## 要求

- Android 10 或更新版本，64 位（arm64）手机。
- Kaggle 账号：用户名与 API Key（与桌面版相同）。

## 安装

1. 从 [Releases](https://github.com/cc6pzupa/free-media-studio/releases) 下载 `CC-Media-Studio-<版本>-android.apk`。预览版也可在 GitHub Actions 的 “Desktop installers” 运行结果中下载 `installers-android`。
2. 在手机上打开 APK，按提示允许“安装未知来源应用”。
3. 打开 **CC Media Studio**，在“账号与连接”中填写 Kaggle 用户名和 Key，保存后即可生成。

## 使用

- 图片、视频、画廊、批量、图生图与局部编辑与桌面版相同。涂抹蒙版支持手指操作。
- 上传参考图、首帧时使用系统文件选择器。
- 保存图片、视频与 ZIP 会写入 **下载/CC Media Studio**。
- 应用切到后台后，通知栏会显示“运行中”，任务与 15 分钟空闲释放继续进行。有任务排队或 Kaggle 会话占用时，应用会保持 CPU 唤醒；空闲时不会。
- 在通知中点 **退出**：先释放 Kaggle 会话，再关闭应用。直接划掉应用不会停止后台服务。

## 限制

- 只支持自动账号模式（默认模式）。需要 `kaggle` 命令行的批处理模式在手机上不可用。
- MCP 和 ComfyUI 入口仅限桌面。
- 不支持 32 位手机。
- 签名：配置了签名密钥的发布版可以直接覆盖升级。用临时调试密钥签名的预览版，每次构建的签名都不同，升级前需先卸载旧版本（卸载会删除应用内的画廊与设置）。

## 从源码构建

需要 JDK 17、Android SDK（API 35）和 Python 3.12（Chaquopy 用它安装应用内的 Python 依赖）。

```bash
cd android
./gradlew assembleRelease   # 输出 app/build/outputs/apk/release/app-release.apk
```

构建时会把仓库根目录的 Python 程序与页面复制进 APK。手机上的 Python 由 [Chaquopy](https://chaquo.com/chaquopy/) 提供，本机后台和界面服务以线程方式在应用内运行。

### 发布签名

在仓库 Settings → Secrets and variables → Actions 中设置以下四项后，CI 会用固定密钥签名，之后的版本可以覆盖安装：

| Secret | 内容 |
| --- | --- |
| `ANDROID_KEYSTORE_BASE64` | keystore 文件的 base64 |
| `ANDROID_KEYSTORE_PASSWORD` | keystore 密码 |
| `ANDROID_KEY_ALIAS` | 密钥别名 |
| `ANDROID_KEY_PASSWORD` | 密钥密码 |

```bash
keytool -genkeypair -v -keystore release.jks -alias ccmedia -keyalg RSA -keysize 4096 -validity 10000
base64 -w0 release.jks   # 填入 ANDROID_KEYSTORE_BASE64
```

请妥善备份 keystore：丢失后，新版本无法覆盖安装旧版本。

## English

The Android app runs the same local client and pages as the desktop app; generation still happens on your Kaggle dual T4.

- **Requirements:** Android 10 or later on a 64-bit (arm64) phone, plus a Kaggle username and API key.
- **Install:** download `CC-Media-Studio-<version>-android.apk` from Releases (preview builds: the `installers-android` artifact of the “Desktop installers” workflow), allow installing unknown apps, then enter your Kaggle username and key under Account & connection.
- **Use:** images, video, the gallery, batches, image-to-image and inpainting work as on the desktop; the mask brush works with touch. Uploads use the system file picker. Saved images, videos and ZIP files go to **Downloads/CC Media Studio**.
- **Background:** a notification keeps jobs and the 15-minute idle release running. The app keeps the CPU awake only while jobs are queued or a Kaggle session is held. Use **Exit** in the notification to release the Kaggle session and close the app; swiping the app away does not stop the service.
- **Limits:** automatic account mode only (batch mode needs the `kaggle` command line). MCP and ComfyUI are desktop-only. 32-bit phones are not supported. Builds signed with a temporary debug key change signature each time, so uninstall the old version before updating; that removes the in-app gallery and settings.
- **Build:** JDK 17, Android SDK (API 35) and Python 3.12, then `cd android && ./gradlew assembleRelease`. Python on the phone comes from Chaquopy; the worker and GUI server run as threads in the app.
- **Release signing:** set `ANDROID_KEYSTORE_BASE64`, `ANDROID_KEYSTORE_PASSWORD`, `ANDROID_KEY_ALIAS` and `ANDROID_KEY_PASSWORD` as Actions secrets. Keep the keystore safe; without it, new versions cannot update an installed app.
