# 模型、双卡策略与耗时

默认 `auto`（双卡优先），所有入口使用同一套云端生成代码。可在 GUI 的“账号与连接”中选择 `split`（原分卡），或运行 `python launch.py --acceleration split`。Kaggle 独立 Notebook 提供相同选项。

| 模型 | 默认步数 | 自动策略 |
| --- | --- | --- |
| Qwen-Image 2.1 | 20，可调 | GPU 0 正分支、GPU 1 负分支同时采样；INT8 编码器与 Q4 副本按需迁移并复用 |
| Z-Image Turbo | 9，可调 | GPU 1 INT8 编码器，GPU 0 稳定 FP16 Transformer 与 FP32 VAE |
| SDXL-Lightning | 固定 4 | GPU 1 两个文本编码器，GPU 0 FP16 UNet 与 VAE |

Z-Image Turbo 与 SDXL-Lightning 使用各自的少步生成配置。界面和 `usage.parallel_strategy` 显示实际执行方式：`cfg_parallel` 为 CFG 双分支同时采样，`encoder_split` 为编码 / 生成分卡。双卡分工的速度收益取决于模型。

## 模型复用与回退

Qwen 首次生成准备一个 Q4 副本。后续提示词重新编码，主模型、编码器和副本继续复用，不重新读取权重。重新连接同一内核时更新共享生成函数，并保留已有实例。选择 `split` 会释放副本并恢复编码器到 GPU 1；加速异常时也恢复原始 `forward` 与编码器，以同一请求和种子完成原路径生成。

切换模型会释放当前模型与双卡副本，保留磁盘文件。再次选择同一活跃模型直接复用。各模型的来源、固定版本和独立缓存名称见 `model_registry.py`。

INT8 迁移同时处理参数与量化层单独保存的 CB / SCB 状态。连续三次 CPU / GPU 往返后，编码嵌入逐元素一致，移出后 GPU 1 的 PyTorch 分配量为 8,798,208 字节。见 [迁移报告](benchmarks/int8-encoder-swap-2026-11-10.json)。

## 同画质实测

Tesla T4 ×2；固定 512×512、20 步、CFG 5、KV cache、seed 42。预热一次后各测三次采样：

| 路径 | 三次采样与解码耗时（秒） | 中位数 |
| --- | --- | --- |
| 原路径 | 75.525 / 75.107 / 75.065 | 75.107 |
| CFG 双卡并行 | 38.507 / 37.591 / 37.621 | 37.621 |

三次输出最大像素误差均为 0，回退输出也与基线一致。见 [采样报告](benchmarks/dual-cfg-sampling-2026-11-10.json)。

完整热请求计入提示词编码、CPU / GPU 模型迁移、采样和解码：

| 路径 | 请求耗时（秒） | 统计 |
| --- | --- | --- |
| 原路径，编码器常驻 GPU 1 | 75.877 | 一次热基线 |
| CFG 双卡并行 | 56.378 / 52.030 / 51.946 | 三次中位数 52.030 |

该组完整热请求中位耗时减少 31.4%，速度约 1.46 倍；三张逐像素一致，每张完成 20 对 CFG 前向。编码器只加载一次、Transformer 加载两份；最终编码器与原始 `forward` 恢复成功，回退采样 75.321 秒且图片一致。首次加载、客户端和网络传输、保存 PNG 不计入上述热请求。见 [完整请求报告](benchmarks/dual-cfg-warm-2026-11-10.json)。

## 正式入口的三模型验收

共享 `bootstrap_code` 与 `generation_code` 在双 T4 上完成实际图片生成、换提示词和重连复用：

| 模型 | 步数 | 模型就绪后首次生成 | 后续热请求 | 实际策略 |
| --- | --- | --- | --- | --- |
| Qwen-Image 2.1 | 20 | 71.312 秒，含首次副本准备 | 49.744 秒 | CFG 双卡并行 |
| Z-Image Turbo | 9 | 17.740 秒 | 8.244 秒 | 编码 / 生成分卡 |
| SDXL-Lightning | 4 | 2.420 秒 | 1.547 秒 | 编码 / 生成分卡 |

![三模型实测图片](model-results.png)

本表计入编码、设备迁移、生成和 PNG 序列化 / 保存，不含首次下载、模型加载和客户端网络传输。各模型使用自己的默认步数。Qwen 重连检查 0.011 秒，主模型、编码器和副本实例保持一致；切回原模式的同提示词、同种子输出逐像素一致。新模型均完成两次生成，图像通过非空与有限数值检查。见 [正式入口报告](benchmarks/production-models-2026-11-10.json)。

Z-Image 的 FP16 兼容处理在后接 RMSNorm 的分支上提前缩放，并同步调整 epsilon，保持归一化数学关系；磁盘上的原权重保留。模型加载直接映射 GPU，避免完整主模型先在 CPU 展开。采样前检查编码嵌入，结束前检查 latent 有限值。相关上游问题见 [Z-Image FP16 数值溢出](https://github.com/Tongyi-MAI/Z-Image/issues/14)。

## 返回的耗时字段

| 字段 | 含义 |
| --- | --- |
| `request_seconds` | 客户端连接、准备、生成与取得结果的请求耗时 |
| `encoding_seconds` | 提示词编码 |
| `total_generation_seconds` | Qwen 本次云端生成总耗时，含编码、迁移、首次副本准备与生图 |
| `migration_seconds` | 模型 CPU / GPU 迁移与设备准备 |
| `replica_load_seconds` | 本次首次加载副本的耗时；复用时为 0 |
| `denoising_seconds` / `decode_seconds` | CUDA event 度量的采样与解码 / 后处理段 |
| `dual_model_reused` | Qwen 是否复用了现有加速副本 |
| `acceleration_mode` / `parallel_strategy` | 选择的策略与实际执行方式 |
| `acceleration_fallback` | 本次加速回退的异常类型；正常时为空 |

CUDA event 与墙钟时间度量方式不同，分段值不要求严格相加等于请求总耗时。

## 可重复测试

用已保存的账号配置比较三个模型：

```bash
python benchmark.py
python benchmark.py --run --runs 3
python benchmark.py --gpu-probe
```

第一条查看计划，第二条执行固定提示词、种子和分辨率的预热及三次热请求，第三条读取现有会话硬件与实际双卡副本就绪状态。

独立比较 Qwen 采样与完整请求：

```bash
python dual_gpu_benchmark.py --assets /kaggle/temp/qwen-image-assets --runs 3
python dual_gpu_benchmark.py --assets /kaggle/temp/qwen-image-assets --runs 3 --encoder-mode offload
```

`--assets` 指向准备好的 INT8 / Q4 缓存；GGUF 和 VAE 先复制到本地临时目录。`offload` 模式保留模型实例，计入完整请求迁移；可用 `--baseline-runs 1 --sampling-runs 0` 仅复测完整热请求。输出包含 PNG、像素误差、复用状态和回退报告。

English: Automatic dual-GPU preference is shared by every entry point. Qwen uses concurrent CFG branches; Z-Image Turbo and SDXL-Lightning split encoding from generation. The measured Qwen median fell from 75.107 s to 37.621 s for sampling/decode, and from one 75.877 s baseline to a 52.030 s median across three full warm requests. Outputs and rollback were pixel-identical. Reports distinguish cold preparation, model transfers, sampling, and actual reuse.

## 模型资料

- [Z-Image Turbo 官方说明](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
- [SDXL-Lightning 官方模型与配置](https://huggingface.co/ByteDance/SDXL-Lightning)
- [Diffusers Qwen 2.1](https://huggingface.co/docs/diffusers/main/api/pipelines/qwenimage21)
