# 模型与双 T4 性能研究

本版增加两个模型、可重复性能测试和运行时双卡诊断。当前单张延迟的优化先比较不同模型与步数；双卡采样加速仍处于实验研究阶段。

## 三个模型

| 模型 ID | 默认步数 | 用途 | 设备布局 |
| --- | --- | --- | --- |
| `qwen-image-2.1` | 20，可调 | 当前画质基线 | GPU 1 INT8 编码器，GPU 0 Q4 主模型与 VAE |
| `z-image-turbo` | 9，可调 | 效率优先 | GPU 1 INT8 Qwen3 编码器，GPU 0 FP16 Transformer 与 FP32 VAE |
| `sdxl-lightning` | 固定 4 | 快速预览，英文提示词 | GPU 0 FP16 SDXL 与 4 步专用 UNet |

模型来源、固定版本及各自私有缓存名称见 `model_registry.py`。切换时释放旧模型，磁盘文件保留；再次选择同一活跃模型时复用实例。新增模型使用自动账号模式，首次下载显示真实进度。Qwen 的缓存格式保留兼容，新增模型使用带模型 ID 与版本的独立缓存。

Z-Image 使用 9 个调度步，官方示例说明它对应 8 次 DiT 前向。SDXL-Lightning 使用官方 4 步 UNet、trailing Euler 调度与 `guidance_scale=0`。Qwen 保留 CFG 5 与已有 KV cache。

## 生成慢在哪一段

返回的 `usage` 及 GUI 完成状态新增：

- `encoding_seconds`：编码；
- `denoising_seconds`：GPU 采样段，包含管线准备，不包含前面的文本编码；
- `decode_seconds`：最后一个采样回调之后的解码与后处理；
- `denoising_steps`、`milliseconds_per_step`：实际回调次数与每步 GPU 时间；
- `parallel_strategy`：当前设备分工。

时间使用 CUDA event 在请求结束后统一同步，避免逐步同步拖慢采样。编码、GPU 时间与墙钟总耗时度量方式不同，不能要求严格相加相等。

## 可重复对比

先查看计划（不提交任务）：

```bash
python benchmark.py
```

使用 GUI 保存的账号，执行基准：

```bash
python benchmark.py --run --runs 3
```

同一英文提示词、512×512、seed 42；对比 Qwen 20/16/12 步、Z-Image 9 步、SDXL-Lightning 4 步。每组先生成一张预热，再测三张。预热记录单列；结果是中位数，保留每张 PNG 便于看速度与画质。默认保存到 `.benchmarks/`，该目录已排除在 Git 之外。跨模型换同一提示词和种子用于固定测试输入，不代表相同噪声语义或同等画质。

```bash
python benchmark.py --run --models sdxl-lightning --runs 2
python benchmark.py --run --models z-image-turbo --output .benchmarks/z-image
```

## 双卡到底能不能加速单张

当前默认分工把编码器放在第二张卡，采样仍主要在第一张卡，作用是容纳模型。不同并行方式需要分开看：

| 方式 | 对单张延迟的作用 | 本项目状态 |
| --- | --- | --- |
| 编码器 / 生图模型分卡 | 编码完成后第二张卡不参与主要采样 | 当前默认 |
| 两卡各生成不同图片 | 增加吞吐，单张延迟不因此减半 | 后续批量模式方向 |
| CFG 正负分支并行 | 可以研究同时计算 Qwen 的两个分支 | 需第二份模型、编码器显存迁移与独立 KV cache 验证 |
| Tensor parallel | 拆分每层矩阵计算 | 当前 Qwen 2.1 上游源码提供 `_tp_plan`；GGUF 分片与 KV cache 需另行适配验证 |
| Context parallel | 拆分序列，适合更长序列 | 需要该版本的模型专用计划与注意力后端支持 |

T4 使用 PCIe Gen3；逐层通信会抵消部分计算收益。Qwen 的 GGUF 矩阵是压缩存储，普通 FP16 线性层的张量分片计划不能直接视为已支持这条量化路径。Qwen 2.1 还涉及块因果注意力、前缀 KV cache 与 CFG 两个缓存域。

结论：双卡并行有研究路径；当前默认分卡配置不是双卡采样。先用诊断确认真实硬件互访与已安装模型的并行计划，再进行量化适配和同画质 A/B 测试，之后决定是否提供生产选项。

## 读取已连接会话的硬件诊断

先用 GUI 完成一次生图，再执行：

```bash
python benchmark.py --gpu-probe
```

它连接已记录、且只有一个内核的运行会话，返回 GPU 型号、显存、计算能力、peer-access、`nvidia-smi topo -m`、当前模型的 CP/TP 计划与量化状态；不申请新会话，也不切换模型。也可以把 `gpu_diagnostics.py` 放到 Kaggle 中运行。

诊断中的 `single_image_parallel_ready=false` 表示本项目尚未验收双卡采样路径。即便检测到上游 TP 计划，也还需要验证当前量化与设备布局。

## 验证状态

本地验证覆盖模型参数、缓存隔离、旧模型释放、重复请求去重、模型路由、生成代码和 Notebook 编译，以及基准工具的预热/采样分离。新增模型的 T4 画质、实际延迟、显存峰值与双卡性能尚未实测，本版不提供推测加速倍数。

## 一手资料

- [Z-Image Turbo 官方模型说明](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
- [SDXL-Lightning 官方模型与推理配置](https://huggingface.co/ByteDance/SDXL-Lightning)
- [Diffusers 分布式推理](https://huggingface.co/docs/diffusers/main/training/distributed_inference)
- [Qwen 2.1 Transformer 当前上游源码](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_qwenimage21.py)
- [Qwen 2.1 注意力与编译](https://huggingface.co/docs/diffusers/main/api/pipelines/qwenimage21)
- [NVIDIA T4 官方规格](https://www.nvidia.com/en-us/data-center/tesla-t4/)
