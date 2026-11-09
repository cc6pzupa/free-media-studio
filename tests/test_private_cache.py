import contextlib
import io
import json
import os
import struct
import threading
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from session_backend import bootstrap_code
from session_runtime import decode_message,RemoteError
from asset_manager import BASE_REVISION,GGUF_REVISION
import asset_manager
import cache_manager

# These helpers execute together in the remote kernel.
for name in ('os','json','Path','requests','BASE_REVISION','GGUF_REVISION','safe_relative','time','quote','download_file','emit_progress','hashlib'):
    setattr(cache_manager,name,getattr(asset_manager,name))

class PrivateCacheTest(unittest.TestCase):
    def test_legacy_cache_key_temporarily_replaces_stale_modern_token(self):
        with patch.dict(os.environ,{'KAGGLE_API_TOKEN':'KGAT_old','KAGGLE_KEY':'old-key'}):
            with cache_manager.cache_credentials('owner/cache','legacy-new-key'):
                self.assertNotIn('KAGGLE_API_TOKEN',os.environ)
                self.assertEqual(os.environ['KAGGLE_KEY'],'legacy-new-key')
            self.assertEqual(os.environ['KAGGLE_API_TOKEN'],'KGAT_old')
            self.assertEqual(os.environ['KAGGLE_KEY'],'old-key')

    def test_private_cache_transfer_downloads_manifest_files_with_real_progress(self):
        files={'transformer-q4.gguf':b'GGUF'+b'x'*1000,'text_encoder/config.json':b'{"load_in_8bit":true}'}
        manifest={'studio_cache_version':1,'base_revision':BASE_REVISION,'gguf_revision':GGUF_REVISION,
                  'cache_files':{k:len(v) for k,v in files.items()},
                  'cache_sha256':{k:cache_manager.hashlib.sha256(v).hexdigest() for k,v in files.items()}}
        files['asset-sources.json']=json.dumps(manifest).encode();seen=[]
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_GET(self):
                name=self.path.removeprefix('/');seen.append(name)
                content=files[name];self.send_response(200);self.send_header('Content-Length',str(len(content)))
                self.end_headers();self.wfile.write(content)
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler);worker=threading.Thread(target=server.serve_forever);worker.start()
        try:
            with tempfile.TemporaryDirectory() as d,patch.object(cache_manager,'cache_file_url',side_effect=lambda h,v,n:f'http://127.0.0.1:{server.server_port}/'+n),patch.object(cache_manager,'emit_progress') as progress:
                cache_manager.download_cache_assets(d,'owner/cache','KGAT_test',{'currentVersionNumber':1})
                self.assertEqual((Path(d)/'transformer-q4.gguf').read_bytes(),files['transformer-q4.gguf'])
                self.assertEqual(set(seen),set(files))
                last=progress.call_args.args
                self.assertEqual(last[0],'cache-download');self.assertEqual(last[2],sum(manifest['cache_files'].values()))
                self.assertEqual(last[2],last[3])
        finally:server.shutdown();worker.join();server.server_close()

    def test_credentials_are_restored_after_failure(self):
        with patch.dict(os.environ,{'KAGGLE_API_TOKEN':'existing','KAGGLE_USERNAME':'other'}):
            with self.assertRaises(RuntimeError):
                with cache_manager.cache_credentials('owner/cache','KGAT_new'):
                    self.assertEqual(os.environ['KAGGLE_API_TOKEN'],'KGAT_new')
                    raise RuntimeError('failure')
            self.assertEqual(os.environ['KAGGLE_API_TOKEN'],'existing')
            self.assertEqual(os.environ['KAGGLE_USERNAME'],'other')

    def test_hidden_missing_slug_requires_authenticated_absence_check(self):
        http=Mock();http.headers={};http.__enter__=Mock(return_value=http);http.__exit__=Mock(return_value=False)
        missing=Mock(status_code=403);listing=Mock();listing.json.return_value=[]
        http.get.side_effect=[missing,listing]
        with patch.object(cache_manager.requests,'Session',return_value=http):
            self.assertIsNone(cache_manager.cache_metadata('owner/cache','KGAT_test'))
        self.assertIn('/datasets/list',http.get.call_args.args[0])
        self.assertEqual(http.get.call_args.kwargs['params']['user'],'owner')

    def test_existing_but_inaccessible_cache_never_becomes_create_permission(self):
        http=Mock();http.headers={};http.__enter__=Mock(return_value=http);http.__exit__=Mock(return_value=False)
        missing=Mock(status_code=403);listing=Mock();listing.json.return_value=[{'ref':'owner/cache'}]
        http.get.side_effect=[missing,listing]
        with patch.object(cache_manager.requests,'Session',return_value=http):
            with self.assertRaisesRegex(RuntimeError,'inaccessible'):
                cache_manager.cache_metadata('owner/cache','KGAT_test')

    def test_rejects_public_cache(self):
        http=Mock();http.headers={};http.__enter__=Mock(return_value=http);http.__exit__=Mock(return_value=False)
        response=Mock(status_code=200);response.json.return_value={'isPrivate':False};http.get.return_value=response
        with patch.object(cache_manager.requests,'Session',return_value=http):
            with self.assertRaisesRegex(RuntimeError,'私有'):cache_manager.cache_metadata('owner/cache','KGAT_test')

    def test_verifies_cache_marker_revision_paths_and_file_sizes(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'transformer-q4.gguf').write_bytes(b'gguf')
            manifest={'studio_cache_version':1,'base_revision':BASE_REVISION,'gguf_revision':GGUF_REVISION,'cache_files':{'transformer-q4.gguf':4}}
            file=root/'asset-sources.json';file.write_text(json.dumps(manifest))
            self.assertEqual(cache_manager.validate_cache(root),manifest)
            manifest['cache_files']['transformer-q4.gguf']=5;file.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(RuntimeError,'不完整'):cache_manager.validate_cache(root)
            manifest['cache_files']={'transformer-q4.gguf':4,'../secret':1};file.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):cache_manager.validate_cache(root)

    def test_hf_and_auto_share_same_live_model_key(self):
        source="pipe=object()\nencoder=object()\ndef generate_image(*a):pass\n"
        scope={}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(bootstrap_code('owner/assets',source,install=False,asset_mode='huggingface'),scope)
            identity=scope['_COMFY_QWEN_STATE']['runtime_id']
            exec(bootstrap_code('other/account',"raise AssertionError('must reuse')",install=False,asset_mode='auto'),scope)
        self.assertEqual(scope['_COMFY_QWEN_STATE']['runtime_id'],identity)
        self.assertEqual(scope['_COMFY_QWEN_STATE']['load_count'],1)

    def test_decodes_widget_binary_buffers_and_rejects_corrupt_offsets(self):
        message={'header':{'msg_type':'comm_msg'},'parent_header':{},'content':{'data':{'state':{}}}}
        payload=json.dumps(message).encode();image=b'\x89PNG\0\xff'
        envelope=struct.pack('!III',2,12,12+len(payload))+payload+image
        self.assertEqual(decode_message(envelope),message)
        for raw in [b'\0',struct.pack('!II',1,900),struct.pack('!III',2,12,10)]:
            with self.subTest(raw=raw),self.assertRaises(RemoteError):decode_message(raw)

if __name__=='__main__':unittest.main()
