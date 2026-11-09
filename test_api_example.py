import base64
import json
from pathlib import Path
from urllib.request import Request, urlopen

# 只需要改这一行提示词。
prompt = "橙色长发的猫耳少女坐在窗边，清爽的卡通插画，浅色背景"

config = json.loads((Path.home() / ".kaggle-qwen-client" / "settings.json").read_text())
request = Request(
    "http://127.0.0.1:8787/v1/images/generations",
    data=json.dumps({"prompt": prompt, "size": "512x512", "steps": 20, "seed": 42}).encode(),
    headers={"Content-Type": "application/json", "Authorization": "Bearer " + config["api_key"]},
    method="POST",
)
with urlopen(request, timeout=3700) as response:
    result = json.load(response)
Path("result.png").write_bytes(base64.b64decode(result["data"][0]["b64_json"]))
print("完成，图片已保存到当前文件夹的 result.png")
