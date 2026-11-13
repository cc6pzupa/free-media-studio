# 图片编辑、尺寸与批量（预览） / Image editing, sizes and batches (preview)

> 预览功能：代码与本地测试已完成，尚未在 Kaggle 双 T4 上实测速度和显存。大尺寸或首次使用编辑流程时，请留意进度与 Notebook 日志。
>
> Preview: implemented and covered by local tests, but speed and memory have not yet been measured on Kaggle dual T4.

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

## GUI

- **创作方式** 选择“图生图”或“局部编辑”，上传 PNG / JPEG / WebP。局部编辑直接在图片上涂抹，“清除涂抹”可重来。选择 Qwen 时会自动切换到 Z-Image Turbo。
- **画面尺寸** 与 **每个提示词张数** 在步数下方。勾选“每行一个提示词”后，每一行提交为独立任务。
- 批量完成后，点击 **打包下载 ZIP**。历史画廊也可把当前列表（最多 100 件）打包下载。
- 视频页可上传 **首帧图片**，并为 Wan 模型填写负面提示词。

## HTTP API

```http
POST http://127.0.0.1:8787/v1/images/generations
Authorization: Bearer <local API key>
Content-Type: application/json

{"prompt":"a straw hat","model_id":"z-image-turbo","size":"768x512",
 "task":"inpaint","strength":0.85,"image":"<base64 PNG/JPEG/WebP>","mask":"<base64 PNG>"}
```

- `task`：`text`（默认）、`img2img`、`inpaint`。`image` / `mask` 为 base64，也接受 `data:image/...;base64,` 前缀。单张不超过 8 MB。
- 蒙版白色为重绘区域，黑色保留。
- 返回尺寸与 `size` 一致；`usage` 包含 `task`、`strength`、`width`、`height`。
- 每次请求仍生成 1 张（`n=1`）；批量请提交多个请求。

视频：`POST /v1/videos/generations` 增加可选的 `image`（首帧 base64）和 `negative_prompt`。

## MCP

`generate_image` 新增 `size`、`negative_prompt`、`task`、`strength`、`image_path` / `image_base64`、`mask_path` / `mask_base64`。`generate_video` 新增 `image_path` / `image_base64` 与 `negative_prompt`。本机文件路径适合 AI 客户端直接引用图片。

```text
generate_image(prompt="watercolor style", request_id="edit-001", model_id="z-image-turbo",
               task="img2img", strength=0.6, image_path="C:/Pictures/photo.jpg")
```

## ComfyUI

- **CC Media Image** 新增可选的 `size` 与 `negative_prompt`。
- 新节点 **CC Media Image Edit (Preview)**：输入 `IMAGE`，局部编辑再接 `MASK`（1 / 白色为重绘区域），输出 `IMAGE`。
- **CC Media Video (Preview)** 新增可选 `image` 首帧输入。

## Kaggle Notebook

- `kaggle_standalone.ipynb`：新增“方式”（文生图 / 图生图 / 局部编辑）、参考图与蒙版上传、重绘强度、尺寸和负面提示词（仅 Qwen）。Notebook 中蒙版需上传图片：白色为重绘区域。
- `kaggle_video.ipynb`：LTX-Video 2B 与 Wan2.2 5B 可上传首帧；Wan 模型可填负面提示词。

## English

All features need automatic account mode and reuse the loaded models; no extra weights are downloaded.

- **Image to image** and **inpainting** run on Z-Image Turbo and SDXL-Lightning. Strength is 0.05–1 (defaults 0.6 and 0.9); steps × strength must be at least 1. Inputs are center-cropped to the selected size. In masks, white is repainted.
- **Sizes**: 512×512 (default), 768×768, 1024×1024, 768×512, 512×768, 1024×576, 576×1024.
- **Negative prompts**: Qwen-Image 2.1 and the Wan video models (the others run without CFG).
- **Image to video** from a first frame: LTX-Video 2B and Wan2.2 5B.
- **GUI batches**: 1–8 images per prompt, optional one prompt per line, incrementing seeds, up to 32 tasks; download results as a ZIP. The gallery can also export the current list (up to 100 works).
- Qwen-Image 2.1 has no editing pipeline in diffusers. Instruction editing would need the separate ~20B Qwen-Image-Edit model, which must be validated on dual T4 first.
- HTTP fields: `size`, `negative_prompt`, `task`, `strength`, `image`, `mask` (base64, at most 8 MB each). MCP adds the same fields and accepts local `image_path` / `mask_path`. ComfyUI adds **CC Media Image Edit (Preview)** plus optional `size`, `negative_prompt` and a first-frame `image` on the video node. The standalone Kaggle notebooks add the same controls: upload a reference image, a mask image (white = repaint) or a video first frame.
