# 视频预览 / Video preview

主分支已包含视频预览功能。使用前先按主 README 启动客户端并保存自动账号配置。

## 安装 / Install

```bash
git clone https://github.com/cc6pzupa/free-media-studio.git
cd free-media-studio
python launch.py
```

macOS / Linux 使用 `python3 launch.py`。启动器自动创建客户端环境并安装依赖。 / On macOS and Linux, use `python3 launch.py`. The launcher creates the client environment and installs dependencies.

## GUI

打开左侧 **视频 / Video Preview**，或访问 `http://127.0.0.1:8787/video`。选择模型、输入画面与动作描述、调整帧数，然后点击生成。完成后可播放和保存 MP4。图文操作与实际样片见 [HTML 教程](guide.html#video)。

提交响应丢失时，页面会保留原请求，点击“恢复上次提交”继续；刷新后仍可恢复。后台短暂离线时继续跟踪原任务，避免重复生成。

| 档位 | 模型 | 默认步数 |
| --- | --- | --- |
| 效率 | LTX-Video 2B Distilled 0.9.6 | 固定 8 |
| 平衡 | Wan2.1 T2V 1.3B | 30 |
| 质量 | Wan2.2 TI2V 5B | 30 |

帧数可选 9–81，按 8 递增。默认先生成短片；更多帧会占用更多显存和时间。

**图生视频（预览）**：上传首帧图片即可。LTX-Video 2B 与 Wan2.2 5B 支持，首帧居中裁剪为模型尺寸；Wan2.1 1.3B 仅文生视频。LTX 图生视频使用单卡路径，不走双卡序列并行。Wan 模型可填写负面提示词。详见 [图片编辑说明](editing.md)。

图片和视频共用本机后台、账号及串行队列。切换任务类型会释放当前模型；连续使用同一活跃视频模型会复用已加载实例。默认空闲保活 15 分钟，沿用 GUI 的设置。

## HTTP API

使用主 GUI 高级设置中的本机 API Key：

```http
POST /v1/videos/generations
Authorization: Bearer <local-api-key>
Content-Type: application/json

{"prompt":"An adult catgirl reading by a window, hair moving in the breeze","model_id":"wan21-t2v-1.3b","frames":33,"seed":42,"request_id":"video-001"}
```

接口返回任务 ID。用 `GET /v1/videos/jobs/<job_id>` 查询进度，完成后用 `GET /v1/videos/jobs/<job_id>/content` 下载 MP4，两者使用相同认证头。每次新生成使用不同的 `request_id`；断线后查询原任务。

## MCP 与 ComfyUI

请求体可加 `image`（首帧 base64）和 `negative_prompt`（仅 Wan）。

MCP 新增 `generate_video` 和 `get_video`，与图片共用 `get_job` 查询进度；`generate_video` 接受 `image_path` / `image_base64` 首帧。`get_video` 返回本机 MP4 文件路径、URI 和耗时信息。

ComfyUI 添加 **CC Media Video (Preview)** 节点，输出 `mp4_path` 字符串，可接支持视频路径输入的加载节点；可选 `image` 输入作为首帧。

## Kaggle Notebook

导入仓库中的 `kaggle_video.ipynb`，选择 **GPU T4 ×2**、开启 **Internet**，运行全部单元格。选择模型后生成，结果保存到 `/kaggle/working/cc-media-video.mp4`。

## English

The video preview is included on main. Save your automatic account configuration in the main GUI, then open `/video`. Choose the efficient LTX 2B distilled model, balanced Wan2.1 1.3B or quality Wan2.2 5B. First use downloads weights; repeated requests reuse the active model. Image and video tasks share the serial worker queue and idle-retention setting.

The HTTP API queues tasks at `POST /v1/videos/generations`, reports progress at `GET /v1/videos/jobs/<job_id>`, and serves completed MP4 files at `GET /v1/videos/jobs/<job_id>/content`. Authenticate with the local API key. Optional `image` (base64 first frame, LTX-Video 2B and Wan2.2 5B) enables image-to-video; `negative_prompt` applies to Wan. MCP provides `generate_video` and `get_video`; the ComfyUI preview node returns an MP4 path. A standalone Kaggle video notebook is included. If a submission response is lost, use Resume submission to retry the saved request with its original ID, even after refresh. Temporary backend outages preserve the active job. See the [illustrated guide and actual sample](guide.html#video-en).

视频 GPU 依赖在 Kaggle 的独立 Python 环境中运行。Notebook 重连复用存活的视频进程，切回图片模型时释放视频进程。

Video GPU dependencies run in an isolated Python environment on Kaggle. Reconnecting reuses the live video process; switching to an image model releases it.
