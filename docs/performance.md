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
| CFG 正负分支并行 | 两卡同时计算 Qwen 的两个分支 | 三轮采样中位数 37.621 秒，原路径 75.107 秒，逐像素一致；完整请求验收中 |
| Tensor parallel | 拆分每层矩阵计算 | 当前 Qwen 2.1 上游源码提供 `_tp_plan`；GGUF 分片与 KV cache 需另行适配验证 |
| Context parallel | 拆分序列，适合更长序列 | 需要该版本的模型专用计划与注意力后端支持 |

T4 使用 PCIe Gen3；逐层通信会抵消部分计算收益。Qwen 的 GGUF 矩阵是压缩存储，普通 FP16 线性层的张量分片计划不能直接视为已支持这条量化路径。Qwen 2.1 还涉及块因果注意力、前缀 KV cache 与 CFG 两个缓存域。

独立 Tesla T4 ×2 实测确认 CFG 双分支并行可以缩短单张采样时间。512×512、20 步的采样加解码中位耗时从 75.107 秒降至 37.621 秒，约 2.00 倍速度。三张输出与基线逐像素一致，退出实验后的原路径输出也一致。完整请求还包括编码和设备迁移，另行计时。默认生成流程保持原设备分工。

## 读取已连接会话的硬件诊断

先用 GUI 完成一次生图，再执行：

```bash
python benchmark.py --gpu-probe
```

它连接已记录、且只有一个内核的运行会话，返回 GPU 型号、显存、计算能力、peer-access、`nvidia-smi topo -m`、当前模型的 CP/TP 计划与量化状态；不申请新会话，也不切换模型。也可以把 `gpu_diagnostics.py` 放到 Kaggle 中运行。

诊断中的 `single_image_parallel_ready=false` 表示本项目尚未验收双卡采样路径。即便检测到上游 TP 计划，也还需要验证当前量化与设备布局。

## CFG 双卡实验

`dual_gpu.py` 将正分支保留在 GPU 0，把负分支交给 GPU 1 的模型副本。两分支使用各自的 KV cache；输入复制发生在并行启动之前，负分支结果通过 CUDA event 同步后返回 GPU 0。实验上下文退出时恢复原始 `forward`。

在独立 Kaggle Notebook 中安装运行依赖后，将项目源码放入工作目录，再执行：

```python
%pip install -q bitsandbytes==0.50.2 gguf 'diffusers>=0.36,<0.43' 'transformers>=4.51,<6' 'accelerate>=1.2'
!python dual_gpu_benchmark.py --assets /kaggle/temp/qwen-image-assets --runs 3
```

`--assets` 指向已准备好的 INT8 / Q4 缓存。实验先编码固定提示词，释放编码器，再加载主模型和副本，以分离采样并行的效果。GGUF 检查点先复制到本地临时目录；相同的 512×512、20 步、CFG 5、KV cache 与 seed 42 用于所有采样。

输出目录默认 `/kaggle/working/cc-dual-cfg`，包含原路径图片、三张双卡图片、回退图片和 `report.json`。报告分别记录预热、原路径中位数、副本加载、双卡中位数、像素误差与回退一致性。**该实验不包含编码器移出 / 恢复或重新加载的请求开销，因此结果只能衡量预编码后的采样阶段。**

### T4 ×2 实测（2026-11-10）

固定 512×512、20 步、CFG 5、KV cache、seed 42；预热一次后各测三次。该组仅计采样与 VAE 解码，编码器已释放。

| 路径 | 三次耗时（秒） | 中位数（秒） |
| --- | --- | --- |
| 原路径 | 75.525 / 75.107 / 75.065 | 75.107 |
| 双卡 CFG | 38.507 / 37.591 / 37.621 | 37.621 |

三次双卡输出的最大像素误差均为 0，每张完成 20 对 CFG 前向。副本首次加载 4.486 秒；恢复原始调用后的采样为 75.638 秒，图片与基线一致。详见 [JSON 报告](benchmarks/dual-cfg-sampling-2026-11-10.json)。

完整热请求测试使用 `--encoder-mode offload`，保留同一编码器和两个 Transformer 实例，逐次测量编码器与副本的 CPU / GPU 迁移、真实提示词编码、双卡采样和解码。模型文件只在初始化时加载；每次请求的设备迁移计入总耗时。

```bash
python dual_gpu_benchmark.py --assets /kaggle/temp/qwen-image-assets --runs 3 --encoder-mode offload
```

English: On two Tesla T4 GPUs, median sampling plus VAE decode fell from 75.107 s to 37.621 s across three warm runs. All outputs were pixel-identical; restoring the original forward path also preserved the output. These figures exclude prompt encoding and model transfers. Full warm requests are measured separately with `--encoder-mode offload`.

本地自动检查覆盖正负分支路由、各自的掩码与 KV cache、负分支异常后的原始调用恢复，以及副本和编码器的清理顺序。它们不替代 GPU 端的速度、画质和回退验收。

### 依赖与会话诊断

`python runtime_import_probe.py` 不安装或修改任何依赖。它在独立子进程中逐项导入运行组件，记录退出码、异常和耗时；即使组件进程收到终止信号，主诊断进程也能继续。完整结果保存在已排除提交的 `.benchmarks/import-diagnostics.json`。

独立环境的九项运行组件导入检查已通过。测试使用隔离虚拟环境，仅安装 bitsandbytes 0.50.2 和 gguf；原应用环境保持不变。

## 验证状态

本地验证覆盖模型参数、缓存隔离、旧模型释放、重复请求去重、模型路由、生成代码和 Notebook 编译，以及基准工具的预热/采样分离。新增模型的 T4 画质、实际延迟与显存峰值尚未实测。Qwen CFG 双卡采样、像素一致性与原路径回退已完成实测；完整请求结果另列。

## 一手资料

- [Z-Image Turbo 官方模型说明](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
- [SDXL-Lightning 官方模型与推理配置](https://huggingface.co/ByteDance/SDXL-Lightning)
- [Diffusers 分布式推理](https://huggingface.co/docs/diffusers/main/training/distributed_inference)
- [Qwen 2.1 Transformer 当前上游源码](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_qwenimage21.py)
- [Qwen 2.1 注意力与编译](https://huggingface.co/docs/diffusers/main/api/pipelines/qwenimage21)
- [NVIDIA T4 官方规格](https://www.nvidia.com/en-us/data-center/tesla-t4/)
