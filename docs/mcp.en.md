# Local MCP for CC Media Studio

[简体中文](mcp.md) | **English**

Install with the project's virtual-environment Python:

```bash
python -m pip install -r requirements-mcp.txt
```

Save your Kaggle username and key once in the GUI, then close the GUI if desired. Alternatively set `KAGGLE_USERNAME` and `KAGGLE_API_TOKEN` (`KAGGLE_KEY` for older keys). Saved configuration takes precedence. Credentials are never MCP tool arguments.

For MCP clients using `mcpServers`, use absolute paths:

```json
{
  "mcpServers": {
    "cc-media-studio": {
      "command": "/absolute/path/free-media-studio/.easy-venv/bin/python",
      "args": ["/absolute/path/free-media-studio/mcp_server.py"]
    }
  }
}
```

On Windows use `C:/absolute/path/free-media-studio/.easy-venv/Scripts/python.exe`. Before connecting MCP on Windows, save your account and submit a task in the GUI to start the independent worker. Alternatively, run `.\\.easy-venv\\Scripts\\python.exe studio_worker.py` in a separate terminal and keep that terminal open. macOS / Linux can start the worker automatically. Complete examples are in `examples/mcp/`.

| Tool | Inputs | Output |
| --- | --- | --- |
| `studio_status` | None | Configuration, queue, previous model usage, GPU quota, idle retention, countdown |
| `generate_image` | `prompt`, unique `request_id`, optional `seed=42`, `model_id`, `steps` (model default when omitted), `size`, `negative_prompt`, `task`, `strength`, `feather` (inpaint edge feathering, 0–64), `image_path` / `image_base64`, `mask_path` / `mask_base64` | Immediate durable `job_id` |
| `get_job` | `job_id` | State, real progress, downloaded bytes, model reuse |
| `get_image` | Completed `job_id` | PNG image content and usage metadata |
| `cancel_job` | Queued `job_id` | Cancels an image or video task that has not started; a running task returns `cancelled: false` |
| `generate_video` | `prompt`, `request_id`; optional `model_id`, `frames`, `seed`, `steps`, `size`, `negative_prompt`, `image_path` / `image_base64` (first frame), `last_image_path` / `last_image_base64` (last frame, LTX only) | Video job ID |
| `get_video` | Completed video job ID | Local MP4 path, URI, file size and usage |

Call status, submit once, poll the job at least two seconds apart, then retrieve the image. Reuse a request ID only when retrying the same intended submission. A different payload with the same ID is rejected. Do not automatically resubmit errored or interrupted tasks; the remote task may still be running.

The local worker survives GUI closure and MCP disconnection. Completed images persist in `~/.kaggle-qwen-client/studio/images/`; SQLite holds task metadata without Kaggle keys. Private local connection records authenticate the localhost-only worker. Different accounts should use different `EASY_CONFIG_PATH` values, and account changes are rejected while tasks are pending.

Default idle retention is 15 minutes after the queue finishes. Change it in the GUI or saved configuration; 0 disables automatic release. Only automatic account mode releases its exact Kaggle session, and the local worker must remain running and online. Quota is cached for 60 seconds and querying status never allocates a GPU or extends idle retention. Last model usage is historical and does not guarantee a currently live GPU.

Before upgrading, finish tasks and run `python studio_worker.py --stop`. This stops the local worker only; explicitly stop unused remote sessions on Kaggle. An interrupted running job is never replayed automatically on recovery, while completed images remain available and unstarted queued jobs can resume with the same account configuration.


## Model selection

`generate_image` accepts `model_id`: `qwen-image-2.1`, `z-image-turbo`, or `sdxl-lightning`. Omit `steps` to use 20, 9, or 4 respectively. SDXL-Lightning requires exactly 4 steps. For sizes, image-to-image, inpainting and image-to-video, see the [editing guide](editing.md).

```text
generate_image(prompt="An adult catgirl by a window", request_id="catgirl-turbo-001", model_id="z-image-turbo")
```

Switching models releases the previous model; repeated requests for the same active model reuse it. Model selection uses the automatic account configuration saved in the GUI. Returned usage includes the model ID and encoding, sampling and decoding timings.

Acceleration follows the saved GUI preference and defaults to `auto` (dual GPU preferred). `usage.parallel_strategy` reports `cfg_parallel` or `encoder_split`. For configurations without a saved preference, set `CC_MEDIA_ACCELERATION=auto` or `split`.

## Video generation

The video preview is included on main. Use the same saved account, MCP configuration and worker as images.

| Tier | `model_id` | Default steps |
| --- | --- | --- |
| Efficient | `ltx-video-2b-distilled` | Exactly 8 |
| Balanced | `wan21-t2v-1.3b` | 30 |
| Quality | `wan22-ti2v-5b` | 30 |

Frames default to 33; supported values are 9, 17, 25, …, 81. Omit steps to use the model default.

```text
studio_status()
generate_video(prompt="An adult anime catgirl reading by a sunlit window. Her hair moves gently in the breeze.", request_id="cc-video-001", model_id="ltx-video-2b-distilled", frames=33, seed=42)
get_job(job_id="<returned job_id>")
get_video(job_id="<same job_id>")
```

Submission returns immediately. Keep the job ID, poll `get_job` at least two seconds apart, and call `get_video` after state becomes `complete`. Progress includes real download bytes, sampling steps, decoding and export.

`get_video` returns `path`, `uri`, `mime_type=video/mp4`, `size_bytes` and `usage`. Open or download the local MP4 using the returned path. Videos persist as `studio/images/<job_id>.mp4` beside completed PNG files.

Image and video requests share one serial queue. Repeated requests reuse the active video process; switching models releases the previous one. Video tasks use the acceleration and idle-retention settings saved in the GUI.
