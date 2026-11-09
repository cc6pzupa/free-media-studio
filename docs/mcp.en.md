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
      "command": "/absolute/path/qwen-image-studio/.venv/bin/python",
      "args": ["/absolute/path/qwen-image-studio/mcp_server.py"]
    }
  }
}
```

On Windows use `C:/absolute/path/qwen-image-studio/.venv/Scripts/python.exe`. If using the automatic launcher environment, replace `.venv` with `.easy-venv` and install the MCP requirements there. Complete examples are in `examples/mcp/`.

| Tool | Inputs | Output |
| --- | --- | --- |
| `studio_status` | None | Configuration, queue, previous model usage, GPU quota, idle retention, countdown |
| `generate_image` | `prompt`, unique `request_id`, optional `seed=42`, `steps=20` | Immediate durable `job_id` |
| `get_job` | `job_id` | State, real progress, downloaded bytes, model reuse |
| `get_image` | Completed `job_id` | PNG image content and usage metadata |

Call status, submit once, poll the job at least two seconds apart, then retrieve the image. Reuse a request ID only when retrying the same intended submission. A different payload with the same ID is rejected. Do not automatically resubmit errored or interrupted tasks; the remote task may still be running.

The local worker survives GUI closure and MCP disconnection. Completed images persist in `~/.kaggle-qwen-client/studio/images/`; SQLite holds task metadata without Kaggle keys. Private local connection records authenticate the localhost-only worker. Different accounts should use different `EASY_CONFIG_PATH` values, and account changes are rejected while tasks are pending.

Default idle retention is 15 minutes after the queue finishes. Change it in the GUI or saved configuration; 0 disables automatic release. Only automatic account mode releases its exact Kaggle session, and the local worker must remain running and online. Quota is cached for 60 seconds and querying status never allocates a GPU or extends idle retention. Last model usage is historical and does not guarantee a currently live GPU.

Before upgrading, finish tasks and run `python studio_worker.py --stop`. This stops the local worker only; explicitly stop unused remote sessions on Kaggle. An interrupted running job is never replayed automatically on recovery, while completed images remain available and unstarted queued jobs can resume with the same account configuration.
