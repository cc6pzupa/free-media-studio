# 视频预览 / Video preview

视频功能位于 `feature/video-three-tiers` 开发分支。使用前先按主 README 启动客户端并保存自动账号配置。

## GUI

打开左侧 **视频 / Video Preview**，或访问 `http://127.0.0.1:8787/video`。选择模型、输入画面与动作描述、调整帧数，然后点击生成。完成后可播放和保存 MP4。

| 档位 | 模型 | 默认步数 |
| --- | --- | --- |
| 效率 | LTX-Video 2B Distilled 0.9.6 | 固定 8 |
| 平衡 | Wan2.1 T2V 1.3B | 30 |
| 质量 | Wan2.2 TI2V 5B | 30 |

帧数可选 9–81，按 8 递增。默认先生成短片；更多帧会占用更多显存和时间。当前界面提供文生视频。

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

MCP 新增 `generate_video` 和 `get_video`，与图片共用 `get_job` 查询进度。`get_video` 返回本机 MP4 文件路径、URI 和耗时信息。

ComfyUI 添加 **CC Media Video (Preview)** 节点，输出 `mp4_path` 字符串，可接支持视频路径输入的加载节点。

## Kaggle Notebook

导入仓库中的 `kaggle_video.ipynb`，选择 **GPU T4 ×2**、开启 **Internet**，运行全部单元格。选择模型后生成，结果保存到 `/kaggle/working/cc-media-video.mp4`。

## English

The video preview is on the `feature/video-three-tiers` development branch. Save your automatic account configuration in the main GUI, then open `/video`. Choose the efficient LTX 2B distilled model, balanced Wan2.1 1.3B or quality Wan2.2 5B. First use downloads weights; repeated requests reuse the active model. Image and video tasks share the serial worker queue and idle-retention setting.

The HTTP API queues tasks at `POST /v1/videos/generations`, reports progress at `GET /v1/videos/jobs/<job_id>`, and serves completed MP4 files at `GET /v1/videos/jobs/<job_id>/content`. Authenticate with the local API key. MCP provides `generate_video` and `get_video`; the ComfyUI preview node returns an MP4 path. A standalone Kaggle video notebook is included.
