# 图片编辑、尺寸与批量（预览） / Image editing, sizes and batches (preview)

> 预览功能：已在 Kaggle 双 T4 上实测通过（见下方实测表）。首次使用会下载对应模型。
>
> Preview: verified on Kaggle dual T4 (see the measurements below). First use downloads the model.

需要 **自动账号模式**（默认模式）。图生图、局部编辑和图生视频复用已加载的模型，不额外下载权重。

## 能力一览

| 功能 | 支持的模型 | 说明 |
| --- | --- | --- |
| 图生图（参考图） | Z-Image Turbo、SDXL-Lightning | 上传参考图，按描述重绘；重绘强度 0.05–1，默认 0.6 |
| 局部编辑（涂抹重绘） | Z-Image Turbo、SDXL-Lightning | 在图片上涂抹，白色区域重绘；默认强度 0.9 |
| 画面尺寸 | 全部图片模型 | 512×512（默认）、768×768、1024×1024、768×512、512×768、1024×576、576×1024 |
| 负面提示词 | Qwen-Image 2.1；Wan2.1 / Wan2.2 视频 | 其他模型不使用 CFG，填写会提示清空 |
| 图生视频（首帧） | LTX-Video 2B、Wan2.2 5B | Wan2.1 1.3B 仅文生视频 |
| GUI 批量与 ZIP | 全部图片模型 | 每个提示词 1–8 张，可每行一个提示词，种子递增；最多 32 个任务 |

Qwen-Image 2.1 暂不支持图生图和局部编辑：diffusers 没有它的编辑流程。指令式编辑需要另一个约 20B 的 Qwen-Image-Edit 模型，要先验证它在双 T4 上的显存，才能加入。

参考图、蒙版和首帧会在云端按目标尺寸**居中裁剪**。步数 × 重绘强度至少为 1，例如 SDXL-Lightning 的 4 步需要强度不低于 0.25。

## Kaggle 双 T4 实测（2026-11-13）

由 `edit_gpu_check.py` 经 GUI 相同的自动账号路径完成。耗时为模型就绪后的云端生成时间；首次下载与加载另计。

| 用例 | 生成耗时 | 峰值显存 GPU0 / GPU1 |
| --- | --- | --- |
| Z-Image 文生图 512 | 15.3 秒 | 12.6 / 4.4 GB |
| Z-Image 图生图 512（强度 0.6） | 6.6 秒 | 12.6 / 4.4 GB |
| Z-Image 局部编辑 512（强度 0.9） | 9.7 秒 | 12.6 / 4.4 GB |
| Z-Image 文生图 1024 | 38.3 秒 | 12.3 / 7.4 GB |
| SDXL-Lightning 文生图 1024 | 6.9 秒 | 8.2 / 1.5 GB |
| SDXL-Lightning 图生图 768×512 | 2.8 秒 | 6.3 / 1.5 GB |
| SDXL-Lightning 局部编辑 512 | 2.6 秒 | 6.0 / 1.5 GB |
| Qwen-Image 2.1 文生图 1024 + 负面提示词（CFG 双卡并行） | 121 秒 | 11.6 / 9.5 GB |
| LTX-Video 2B 图生视频 33 帧 | 12.5 秒 | 9.3 / 7.4 GB |
| Wan2.2 5B 图生视频 33 帧（CFG 双卡并行） | 210 秒 | 9.4 / 9.1 GB |

0.10 后续功能（2026-11-14）：

| 用例 | 生成耗时 | 峰值显存 GPU0 / GPU1 |
| --- | --- | --- |
| Z-Image 局部编辑 512 + 羽化 12 px（蒙版外与原图差异 0） | 12.4 秒 | 12.6 / 7.6 GB |
| LTX-Video 2B 图生视频 768×448（双卡序列并行） | 50.0 秒（含首次启动双卡） | 11.8 / 11.1 GB |
| LTX-Video 2B 文生视频竖向 448×768（双卡序列并行） | 57.9 秒 | 11.8 / 11.0 GB |
| LTX-Video 2B 首尾帧 512×512（单卡） | 8.3 秒 | 9.4 / 7.4 GB |
| Wan2.2 5B 图生视频竖向 448×768（CFG 双卡并行） | 202 秒 | 9.4 / 9.1 GB |
| Wan2.1 1.3B 文生视频方形 624×624（CFG 双卡并行） | 276 秒；首次使用约 12 分钟 | 4.5 / 9.1 GB |

已知限制：SDXL-Lightning 4 步局部编辑覆盖面部时细节较粗；LTX 首尾帧使用单卡路径。切换到图片模型后再用 LTX，会重新加载模型并启动双卡进程。Z-Image 大于 512×512 时，VAE 编解码在 GPU 1 执行以避免显存不足。

## GUI

- **创作方式** 选择“图生图”或“局部编辑”，上传 PNG / JPEG / WebP。局部编辑直接在图片上涂抹：可切换 **橡皮擦**、**撤销**（最多 20 步）或“清除涂抹”。选择 Qwen 时会自动切换到 Z-Image Turbo。
- **边缘羽化**（局部编辑，默认 8 px）：重绘区域与原图柔和过渡，涂抹区域以外的像素与原图完全一致。设为 0 则不羽化。
- 种子旁的 **随机** 按钮生成新种子；批量时从该种子开始递增。
- 生成进度下方的 **取消排队中的任务** 会取消尚未开始的任务（批量中剩余的任务也一并取消）。已在 Kaggle 上执行的任务不会被中断。
- 失败时会显示原因与建议，例如显存不足时提示改小尺寸；所有失败的任务都不会自动重发。
- 模型下方会显示 **首次使用预计耗时**（双 T4 实测，含申请 GPU、下载与加载），例如 Qwen 约 8–13 分钟、SDXL 约 2 分钟。
- 历史画廊的 **选择** 模式：点选作品（最多 100 件）后可 **打包所选** 或 **删除所选**。
- **画面尺寸** 与 **每个提示词张数** 在步数下方。勾选“每行一个提示词”后，每一行提交为独立任务。
- 批量完成后，点击 **打包下载 ZIP**。历史画廊也可把当前列表（最多 100 件）打包下载。
- 视频页可上传 **首帧图片**（LTX 还可上传 **尾帧**，生成从首帧过渡到尾帧的视频），选择 **画面尺寸**（横向、竖向、方形），并为 Wan 模型填写负面提示词。

## HTTP API

```http
POST http://127.0.0.1:8787/v1/images/generations
Authorization: Bearer <local API key>
Content-Type: application/json

{"prompt":"a straw hat","model_id":"z-image-turbo","size":"768x512",
 "task":"inpaint","strength":0.85,"image":"<base64 PNG/JPEG/WebP>","mask":"<base64 PNG>"}
```

- `task`：`text`（默认）、`img2img`、`inpaint`。`image` / `mask` 为 base64，也接受 `data:image/...;base64,` 前缀。单张不超过 8 MB。
- 蒙版白色为重绘区域，黑色保留。`feather`（0–64 像素，仅 inpaint）让重绘区域与原图柔和过渡。
- 返回尺寸与 `size` 一致；`usage` 包含 `task`、`strength`、`width`、`height`。
- `n`（1–8）一次生成多张，种子依次递增；`data` 按顺序返回每张图片，`usages` 列出每张的耗时信息（`usage` 为第一张）。图片依次生成，请求会等待全部完成。

视频：`POST /v1/videos/generations` 增加可选的 `image`（首帧 base64）、`negative_prompt`（仅 Wan）、`size`（如 `448x768`）和 `last_image`（尾帧 base64，仅 LTX，需同时提供首帧）。

## MCP

`cancel_job(job_id)` 取消尚未开始的图片或视频任务（已在执行的任务返回 `cancelled: false`，继续完成）。`generate_image` 新增 `size`、`negative_prompt`、`task`、`strength`、`feather`、`image_path` / `image_base64`、`mask_path` / `mask_base64`。`generate_video` 新增 `image_path` / `image_base64`、`last_image_path` / `last_image_base64`（仅 LTX）、`size` 与 `negative_prompt`。本机文件路径适合 AI 客户端直接引用图片。

```text
generate_image(prompt="watercolor style", request_id="edit-001", model_id="z-image-turbo",
               task="img2img", strength=0.6, image_path="C:/Pictures/photo.jpg")
```

## ComfyUI

- **CC Media Image** 新增可选的 `size` 与 `negative_prompt`。
- 新节点 **CC Media Image Edit (Preview)**：输入 `IMAGE`，局部编辑再接 `MASK`（1 / 白色为重绘区域），输出 `IMAGE`；可设 `feather` 边缘羽化。
- **CC Media Video (Preview)** 新增可选 `image` 首帧、`last_image` 尾帧（仅 LTX）与 `size`。

## 取消任务（HTTP）

```http
POST /v1/videos/jobs/<job_id>/cancel      # 本机 API Key
```

GUI 使用 `POST /jobs/<job_id>/cancel` 与 `POST /video/jobs/<job_id>/cancel`。只有排队中的任务会被取消，返回的任务中 `cancelled` 表示是否成功。

## Kaggle Notebook

- `kaggle_standalone.ipynb`：新增“方式”（文生图 / 图生图 / 局部编辑）、参考图上传、重绘强度、边缘羽化、尺寸和负面提示词（仅 Qwen）。局部编辑的蒙版可用 **形状**（椭圆或矩形，用滑块设置位置和大小，并在原图上预览重绘区域），也可上传蒙版图片（白色为重绘区域）。
- `kaggle_video.ipynb`：LTX-Video 2B 与 Wan2.2 5B 可上传首帧，LTX 可再上传尾帧；可选视频尺寸；Wan 模型可填负面提示词。

## English

## Measured on Kaggle dual T4 (2026-11-13)

Run with `edit_gpu_check.py` through the same automatic-account path as the GUI. Times are cloud generation after the model is loaded; first downloads and loading are extra.

| Case | Generation | Peak GPU0 / GPU1 |
| --- | --- | --- |
| Z-Image text 512 / img2img / inpaint | 15.3 / 6.6 / 9.7 s | 12.6 / 4.4 GB |
| Z-Image text 1024 | 38.3 s | 12.3 / 7.4 GB |
| SDXL-Lightning text 1024 / img2img 768×512 / inpaint | 6.9 / 2.8 / 2.6 s | ≤ 8.2 / 1.5 GB |
| Qwen-Image 2.1 1024 + negative prompt (dual-GPU CFG) | 121 s | 11.6 / 9.5 GB |
| LTX-Video 2B image-to-video, 33 frames | 12.5 s | 9.3 / 7.4 GB |
| Wan2.2 5B image-to-video, 33 frames (dual-GPU CFG) | 210 s | 9.4 / 9.1 GB |
| Z-Image inpaint 512 + feather 12 px (zero difference outside the mask) | 12.4 s | 12.6 / 7.6 GB |
| LTX image-to-video 768×448 (context parallel) / text-to-video portrait 448×768 | 50.0 (incl. dual-GPU start) / 57.9 s | 11.8 / 11.1 GB |
| LTX first and last frame 512×512 (single GPU) | 8.3 s | 9.4 / 7.4 GB |
| Wan2.2 image-to-video portrait 448×768 / Wan2.1 text-to-video 624×624 (dual-GPU CFG) | 202 / 276 s | ≤ 9.4 / 9.1 GB |

Known limits: 4-step SDXL-Lightning inpainting over faces is coarse; LTX first-and-last-frame runs on one GPU; after an image model runs, LTX reloads and restarts its dual-GPU workers. Above 512×512, Z-Image runs its VAE on GPU 1 to stay within memory.


All features need automatic account mode and reuse the loaded models; no extra weights are downloaded.

- **Image to image** and **inpainting** run on Z-Image Turbo and SDXL-Lightning. Strength is 0.05–1 (defaults 0.6 and 0.9); steps × strength must be at least 1. Inputs are center-cropped to the selected size. In masks, white is repainted; the GUI brush has an eraser and undo. `feather` (0–64 px, inpaint only; GUI default 8) blends the repainted area into the original, leaving pixels outside the mask unchanged.
- **Cancel** queued tasks from the GUI, `POST /v1/videos/jobs/<job_id>/cancel` or the MCP `cancel_job` tool. Running tasks are never interrupted.
- **Random seed** button in the GUI and video page. Failures show a reason and a suggestion (for example, use a smaller size when out of GPU memory); failed tasks are never resubmitted automatically.
- **Sizes**: 512×512 (default), 768×768, 1024×1024, 768×512, 512×768, 1024×576, 576×1024.
- **Negative prompts**: Qwen-Image 2.1 and the Wan video models (the others run without CFG).
- **Image to video** from a first frame: LTX-Video 2B and Wan2.2 5B. LTX also takes a last frame and generates a transition between the two. Each video model offers landscape (default), portrait and square sizes.
- **GUI batches**: 1–8 images per prompt, optional one prompt per line, incrementing seeds, up to 32 tasks; download results as a ZIP. The gallery can also export the current list (up to 100 works).
- Qwen-Image 2.1 has no editing pipeline in diffusers. Instruction editing would need the separate ~20B Qwen-Image-Edit model, which must be validated on dual T4 first.
- HTTP fields: `size`, `negative_prompt`, `task`, `strength`, `image`, `mask` (base64, at most 8 MB each). The video API adds `image`, `negative_prompt`, `size` and `last_image` (LTX). MCP adds the same fields and accepts local `image_path` / `mask_path` / `last_image_path`. ComfyUI adds **CC Media Image Edit (Preview)** plus optional `size` and `negative_prompt`; the video node takes `size`, a first-frame `image` and an LTX `last_image`. The standalone Kaggle notebooks add the same controls: upload a reference image or a video first frame; for inpainting, build an ellipse or rectangle mask with sliders and a live preview, or upload a mask image (white = repaint). The GUI shows each model's measured first-use time, and the gallery's Select mode downloads or deletes chosen works (up to 100).
