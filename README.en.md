# CC Media Studio · Kaggle

[简体中文](README.md) | **English**

![CC Media Studio English interface](docs/gui-en.jpg)

The preview uses a previously verified generated image. No live account quota is connected in this screenshot.

A personal media studio powered by your own Kaggle GPU. Image generation is available now; the CC Media Studio name leaves room for future video features. Model downloads, loading, and inference run on Kaggle. Your computer does not need a GPU, CUDA, PyTorch, or model weights.

## Clone, install, run

Install Git and Python 3.11+ (3.11 or 3.12 recommended). Authenticate Git with a GitHub account that can access this private repository.

```bash
git clone https://github.com/cc6pzupa/free-media-studio.git qwen-image-studio
cd qwen-image-studio
```

Windows:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe desktop.py
```

Linux / macOS:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python desktop.py
```

The GUI opens in your browser at `http://127.0.0.1:8787`. Select **English** or **简体中文** in the top bar; the browser remembers your choice. The initial language follows your browser preference. Enter your Kaggle username and API key/token, then write a prompt and generate.

For an optional native window on Windows/macOS, install `requirements-desktop.txt` in the same environment. If a native window is unavailable, the app falls back to the browser. Use `--browser` to always use the browser or `--no-open` to serve without opening a window.

Alternatively, run `python launch.py`, or use `start_windows.bat`, `bash start_mac.command`, or `bash start_linux.sh`. The launcher creates `.easy-venv`, installs dependencies, and updates them when the dependency manifest changes.

## First use and model reuse

1. Use a verified Kaggle account with available GPU quota.
2. The client creates or recovers its private notebook and requests a dual T4 session.
3. A loaded model in a live kernel is reused. A new kernel first looks for your private model cache.
4. On a completely new account, public Hugging Face weights download automatically (about 23.51 GB), with actual byte progress and resumable downloads. INT8/Q4 preparation then saves an approximately 16 GB private dataset cache.
5. Images are single 512×512 PNG files. Later requests reuse the same loaded model while its session remains alive.

No Jupyter URL or existing dataset is required in automatic mode. Closing the notebook editor or GUI does not immediately stop the Kaggle session. A new client can recover a live session from the private notebook's operation record.

## Idle retention and GPU usage

The default idle retention is **15 minutes**, measured after the final local task finishes. Downloads, model loading, generation, and queued tasks are excluded from idle time. Status and quota checks do not reset the timer.

Set **Idle retention (minutes)** in the GUI, or pass a startup option:

```bash
python desktop.py --keep-alive-minutes 15
python launch.py --keep-alive-minutes 30
```

The range is 0–1440 minutes; **0 disables automatic release**. `CC_MEDIA_KEEP_ALIVE_MINUTES` sets the default when no saved setting exists. The GUI can override it. In automatic account mode, the local worker calls Kaggle's cancellation API for the exact connected session when idle time expires; the next generation starts or reconnects and reloads as needed. Manual Jupyter sessions and direct Kaggle notebooks remain externally managed.

The local worker must remain running and online for automatic release. Closing the GUI alone keeps it running; stopping the worker or powering off the computer prevents its cleanup call. If release fails, the GUI reports it and the worker retries after at least 60 seconds. Kaggle's own platform limits still apply.

The main GUI shows GPU used, reserved, available, total, quota reset time, and the idle countdown. Quota is fetched from Kaggle's official `GetAcceleratorQuotaStatistics` API and cached for 60 seconds across clients. Available quota is derived as `max(0, total − used − reserved)` and is not a billing estimate. Missing fields remain unavailable; no assumed weekly allowance is displayed. Usage checks never allocate a GPU.

## Four entry points

- **GUI:** `python desktop.py`.
- **ComfyUI:** clone into `ComfyUI/custom_nodes/qwen-image-studio`, install `requirements.txt` with ComfyUI's Python, restart, and add **CC Media Image** → **Save Image**. Save account configuration once in the GUI or use `KAGGLE_USERNAME` and `KAGGLE_API_TOKEN` (`KAGGLE_KEY` for older keys). The GUI does not need to stay open. ComfyUI supplies its own Torch, NumPy, and Pillow.
- **Kaggle notebook:** import `kaggle_standalone.ipynb`, enable Internet and dual T4 GPUs, and run all cells. No local installation is needed. Optional credentials enable cross-session private caching and are cleared from the input after use. Do not save notebook outputs containing credentials.
- **Local MCP:** install `requirements-mcp.txt`, then register `mcp_server.py` as a local stdio server. See [MCP setup](docs/mcp.en.md). GUI, ComfyUI, and MCP share one durable local queue, backend, and image store.

## Local API

```text
POST http://127.0.0.1:8787/v1/images/generations
Authorization: Bearer <local API key shown in advanced settings>
```

```json
{"prompt":"An orange fox beside a teacup, natural photography","size":"512x512","steps":20,"seed":42}
```

The result contains `data[0].b64_json` and usage statistics, including generation time, model reuse, load count, and cache status. This local API key differs from your Kaggle key.

```text
GET http://127.0.0.1:8787/v1/studio/usage
Authorization: Bearer <local API key>
```

Returns `quota`, `keep_alive_minutes`, `idle_remaining_seconds`, and `session_state`. MCP's `studio_status` exposes the same quota and retention information.

## Updating and stopping

Finish queued tasks before updating. Stop the local worker, pull changes, install dependencies, then restart GUI/MCP clients:

```bash
python studio_worker.py --stop
git pull
python -m pip install -r requirements.txt
python desktop.py
```

MCP users also install `requirements-mcp.txt`. These commands use the project's virtual-environment Python. Stopping the local worker does not stop a remote Kaggle session; explicitly stop unused sessions on Kaggle if needed.

Account settings live under `~/.kaggle-qwen-client/`; existing paths and internal identifiers remain compatible with earlier releases. The repository excludes credentials, connection records, local databases, outputs, virtual environments, and weights. Never share local connection or settings files.

## Validation

```bash
python -m unittest discover -s tests -v
node --test tests/test_ui.cjs
```

MCP tests use the real SDK with a simulated GPU backend. Prior real-GPU checks covered first-time Hugging Face downloads, private cache reloads, session recovery, and repeated generation with model reuse. New quota and idle-release behavior is tested with mocked API responses; it has not been validated against a live Kaggle account in this update. Windows/macOS native windows and a full ComfyUI canvas have not been tested on physical hosts.
