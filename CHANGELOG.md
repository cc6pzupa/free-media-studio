# 更新记录 / Changelog

## 0.10.0（预览 / preview）

> 新功能已在 Kaggle 双 T4 上实测通过，测量结果见 [图片编辑说明](docs/editing.md)。
> New features are verified on Kaggle dual T4; see the measurements in [docs/editing.md](docs/editing.md).

### 新增

- **图生图与局部编辑**：Z-Image Turbo 与 SDXL-Lightning 支持参考图重绘（强度 0.05–1）和涂抹局部编辑；复用已加载模型，不额外下载权重。
- **图生视频**：LTX-Video 2B 与 Wan2.2 5B 可上传首帧。
- **更多尺寸**：512×512、768×768、1024×1024、768×512、512×768、1024×576、576×1024。
- **负面提示词**：Qwen-Image 2.1 与 Wan 视频模型。
- **GUI 批量**：每个提示词 1–8 张、每行一个提示词、种子递增，最多 32 个任务；批量结果和画廊列表可打包 ZIP。
- **画廊**：显示作品使用的参考图与蒙版；“继续编辑”和“作为视频首帧”直接带入工作台。
- **全部入口**：HTTP API、MCP（支持本机图片路径）、ComfyUI（新增 CC Media Image Edit 节点，视频节点支持首帧）、两个 Kaggle Notebook。

### 修复（Kaggle 实测发现）

- 失败的生成不再在内核中保留显存；切换图片或视频模型时清理 cuBLAS 工作区（Qwen 双卡后曾占用 GPU 1 约 9.3 GB，导致视频模型显存不足）。
- 修正 Z-Image 编辑与 LTX 图生视频的 VAE 精度不一致；Z-Image 1024 的 VAE 编解码改在 GPU 1。

### 变化

- 上传图片在本机按文件签名校验（不超过 8 MB），按 SHA-256 保存，在云端按目标尺寸居中裁剪；删除作品时清理不再使用的输入。
- 本机后台需重启后才会启用新功能，旧后台会给出提示。
- Kaggle Notebook 根据当前运行时重新生成。

### Added

- Image-to-image and inpainting on Z-Image Turbo and SDXL-Lightning, reusing the loaded models.
- Image-to-video from a first frame on LTX-Video 2B and Wan2.2 5B.
- Seven output sizes up to 1024; negative prompts for Qwen-Image 2.1 and the Wan models.
- GUI batches (1–8 per prompt, one prompt per line, incrementing seeds, up to 32 tasks) and ZIP export from batches and the gallery.
- Gallery shows reference images and masks, with Edit further and Use as video first frame.
- Available in the HTTP API, MCP (local image paths), ComfyUI (CC Media Image Edit node, video first frame) and both Kaggle notebooks.

### Fixed (found on Kaggle)

- Failed generations no longer keep GPU memory in the kernel; model and video switches clear cuBLAS workspaces (after Qwen's dual-GPU CFG they pinned about 9.3 GB on GPU 1, so video models ran out of memory).
- VAE dtype mismatches in Z-Image editing and LTX image-to-video; Z-Image at 1024 runs its VAE on GPU 1.

### Changed

- Uploads are checked by signature (8 MB maximum), stored by SHA-256 and center-cropped on the GPU kernel; deleting a work releases unused inputs.
- Restart the local worker to enable the new features; an older worker reports this.
- The Kaggle notebooks are regenerated from the current runtime.
