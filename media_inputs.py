"""Client-side checks for uploaded reference images and masks.

Local clients have no Pillow dependency. Images are validated by signature and
size only; decoding, resizing and cropping happen on the remote GPU kernel.
"""
import base64
import binascii
import hashlib
import re
import struct

MAX_INPUT_BYTES=8*1024*1024
# Base64 for two inputs plus JSON overhead.
MAX_REQUEST_BYTES=24*1024*1024
SIGNATURES=((b'\x89PNG\r\n\x1a\n','png'),(b'\xff\xd8\xff','jpeg'))


def image_kind(raw):
    for signature,kind in SIGNATURES:
        if raw.startswith(signature):return kind
    if len(raw)>=12 and raw[:4]==b'RIFF' and raw[8:12]==b'WEBP':return 'webp'
    return None


def decode_image(value,label='参考图'):
    """Accept raw bytes, base64 or a data: URL; return validated bytes."""
    if isinstance(value,(bytes,bytearray)):raw=bytes(value)
    elif isinstance(value,str):
        text=value.strip()
        match=re.fullmatch(r'data:image/(?:png|jpeg|jpg|webp);base64,(.*)',text,re.S)
        if match:text=match.group(1)
        if len(text)>MAX_INPUT_BYTES*4//3+8:raise ValueError(label+'不能超过 8 MB。')
        try:raw=base64.b64decode(re.sub(r'\s+','',text),validate=True)
        except (binascii.Error,ValueError):raise ValueError(label+'需要为 base64 编码的 PNG、JPEG 或 WebP。') from None
    else:raise ValueError(label+'需要为 base64 编码的 PNG、JPEG 或 WebP。')
    if not raw or len(raw)>MAX_INPUT_BYTES:raise ValueError(label+'不能为空或超过 8 MB。')
    if image_kind(raw) is None:raise ValueError(label+'只支持 PNG、JPEG 或 WebP。')
    return raw


def digest(raw):return hashlib.sha256(raw).hexdigest()


def is_digest(value):return isinstance(value,str) and re.fullmatch(r'[a-f0-9]{64}',value) is not None


def png_size(raw):
    if not raw.startswith(b'\x89PNG\r\n\x1a\n') or len(raw)<24 or raw[12:16]!=b'IHDR':return None
    return struct.unpack('>II',raw[16:24])


def encode(raw):return base64.b64encode(raw).decode('ascii')
