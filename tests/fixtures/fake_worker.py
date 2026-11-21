"""Subprocess fixture: real worker/SQLite/HTTP, deterministic image backend."""
import struct
import sys
import time
import zlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from studio_worker import serve


def png(width=512,height=512):
    def chunk(kind,data):return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data)&0xffffffff)
    pixels=b''.join(b'\0'+bytes((240,120,60))*width for _ in range(height))
    return b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',width,height,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(pixels))+chunk(b'IEND',b'')


class FakeBackend:
    def __init__(self,config):
        self.config=config;self.count=0
    def generate_with_progress(self,prompt,seed=42,steps=20,progress=None,model_id="qwen-image-2.1",**options):
        self.count+=1
        if progress:progress({'phase':'download','message':'should not leak '+self.config['token'],
            'downloaded_bytes':1024,'total_bytes':4096,'percent':25,'token':self.config['token']})
        if prompt=='fail':raise RuntimeError('private connection and '+self.config['token'])
        time.sleep(.6)
        if progress:progress({'phase':'generating'})
        time.sleep(.3)
        width,height=options.get('width',512),options.get('height',512)
        return png(width,height),{'model_id':model_id,'runtime_id':'fixture-runtime','model_load_count':1,'model_reused':self.count>1,
            'generation_number':self.count,'request_seconds':.9,'seed':seed,'steps':steps,'width':width,'height':height,
            'task':options.get('task','text'),'strength':options.get('strength'),
            'token':self.config['token']}
    def close(self):pass
    def generate_video_with_progress(self,prompt,seed=42,steps=30,model_id='wan21-t2v-1.3b',frames=33,progress=None):
        if progress:progress({'phase':'exporting'})
        return b'\0\0\0\x18ftypisom'+b'fixture-protocol-only',{
            'model_id':model_id,'media_type':'video','frames':frames,'fps':16,'steps':steps,
            'seed':seed,'parallel_strategy':'cfg_parallel','cfg_pairs':steps,'token':self.config['token']}


if __name__=='__main__':serve(Path(sys.argv[1]),backend_factory=FakeBackend)
