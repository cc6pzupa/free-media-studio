"""Local stdio MCP adapter. Protocol stdout is owned exclusively by the SDK."""
import base64
import json
import sys
from typing import Any

try:
    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError
    from mcp.types import CallToolResult, ImageContent, TextContent
except ImportError:
    raise SystemExit('请先安装 MCP 依赖：python -m pip install -r requirements-mcp.txt')

from studio_client import StudioClient
from app_entry import VERSION


def build_server(client=None):
    service=client or StudioClient()
    server=MCPServer('CC Media Studio',version=VERSION,log_level='WARNING',instructions=(
        'Generate images on the user\'s Kaggle GPU. Credentials are configured locally; never ask for a key in tool arguments. '
        'Call studio_status, then generate_image once with a unique request_id. Poll get_job with at least two seconds '
        'between calls. When complete, call get_image to display the PNG. Reuse the same request_id only for transport retries. '
        'Never automatically resubmit jobs marked error or interrupted. A last ready model status is historical, not a live GPU guarantee.'))

    def call(function,*args):
        try:return function(*args)
        except (ValueError,RuntimeError,TimeoutError) as error:
            raise ToolError(str(error)) from None

    @server.tool(structured_output=True)
    def studio_status() -> dict[str,Any]:
        """Check local configuration, queued jobs and last model usage. Does not allocate a GPU."""
        return dict(call(service.status),quota=call(service.quota))

    @server.tool(structured_output=True)
    def generate_video(prompt:str,request_id:str,seed:int=42,steps:int|None=None,
                       model_id:str='wan21-t2v-1.3b',frames:int|None=None) -> dict[str,Any]:
        """Queue one experimental MP4 on Kaggle. Models: ltx-video-2b-distilled (8 steps),
        wan21-t2v-1.3b (balanced), wan22-ti2v-5b (quality). Use a unique request_id.
        Poll get_job every two seconds, then get_video for its local file URI.
        Initial weights and video generation can take many minutes. Never resubmit automatically."""
        return call(service.submit_video,prompt,seed,steps,request_id,model_id,frames)

    @server.tool(structured_output=True)
    def get_video(job_id:str) -> dict[str,Any]:
        """Get the completed local MP4 path/URI and usage. Credentials are not included."""
        job=call(service.get_job,job_id)
        if not job.get('video_ready'):raise ToolError('Video is not ready')
        raw=call(service.get_video,job_id)
        path=(service.directory/'images'/(job_id+'.mp4')).resolve()
        return {'job_id':job_id,'path':str(path),'uri':path.as_uri(),
            'mime_type':'video/mp4','size_bytes':len(raw),'usage':job['usage']}

    @server.tool(structured_output=True)
    def generate_image(prompt:str,request_id:str,seed:int=42,steps:int|None=None,model_id:str="qwen-image-2.1") -> dict[str,Any]:
        """Queue one 512x512 PNG and return its job_id immediately. Models: qwen-image-2.1 (20 steps default),
        z-image-turbo (9 steps default), sdxl-lightning (exactly 4 steps). Omit steps for the model default. Use a unique request_id per intended image;
        reuse it when retrying the same submission. The saved acceleration preference defaults to auto (dual GPU preferred); get_job reports the actual parallel_strategy. First-time preparation may take many minutes.
        Retrieve progress with get_job and the final image with get_image. No credentials are accepted here.
        """
        return call(service.submit,prompt,seed,steps,request_id,model_id)

    @server.tool(structured_output=True)
    def get_job(job_id:str) -> dict[str,Any]:
        """Get durable task state, real preparation progress and model reuse statistics. Poll at most once every two seconds.
        queued/running means keep waiting; complete means get_image is ready. Do not resubmit error/interrupted jobs automatically.
        """
        return call(service.get_job,job_id)

    @server.tool()
    def get_image(job_id:str) -> CallToolResult:
        """Return a completed job as MCP image/png content plus usage metadata. The PNG is retained locally by the worker."""
        job=call(service.get_job,job_id)
        raw=call(service.get_image,job_id)
        info={'job_id':job_id,'width':512,'height':512,'usage':job['usage']}
        return CallToolResult(content=[ImageContent(type='image',data=base64.b64encode(raw).decode('ascii'),mime_type='image/png'),
            TextContent(type='text',text=json.dumps(info,ensure_ascii=False))],structured_content=info)

    return server


if __name__=='__main__':
    build_server().run(transport='stdio')
