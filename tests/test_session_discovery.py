import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import Mock,patch
import requests
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from account_backend import AccountBackend
URL='https://kkb-production.jupyter-proxy.kaggle.net/k/42/secret/proxy'
OP='operations/create-kernel-session/42'

class DiscoveryTest(unittest.TestCase):
    def test_private_notebook_keeps_only_task_identifier_and_preserves_cells(self):
        with tempfile.TemporaryDirectory() as d:
            b=AccountBackend('owner','KGAT_test',cache_dir=d)
            notebook={'cells':[{'cell_type':'markdown','source':['QWEN_STUDIO_MANAGED']},{'cell_type':'code','source':['valuable custom cell']}], 'metadata':{'keep':True}}
            saved=[]
            def api(service,method,body):
                if method=='GetKernel':return {'blob':{'source':json.dumps(notebook)}}
                saved.append(body);return {}
            with patch.object(b,'_call',side_effect=api):b._remember_operation(OP)
            body=saved[0];result=json.loads(body['text'])
            self.assertEqual(result['cells'],notebook['cells']);self.assertTrue(result['metadata']['keep'])
            self.assertEqual(result['metadata']['qwen_studio']['operation'],OP)
            self.assertTrue(body['isPrivate']);self.assertEqual(body['kernelExecutionType'],'QUICK_SAVE')
            self.assertNotIn(b.token,body['text']);self.assertNotIn('jupyter-proxy',body['text'])
            with patch.object(b,'_call',return_value={'blob':{'source':body['text']}}):self.assertEqual(b._ensure_notebook(),OP)
            b.close()

    def test_empty_local_cache_recovers_existing_operation_without_starting_gpu(self):
        with tempfile.TemporaryDirectory() as d:
            b=AccountBackend('owner','KGAT_test',cache_dir=d)
            with patch.object(b,'_ensure_notebook',return_value=OP),patch.object(b,'_recover_operation',return_value=OP),patch.object(b,'_call',return_value={'done':True,'response':{'jupyterUrl':URL}}) as api:
                self.assertEqual(b._ensure_session(lambda e:None),URL)
            self.assertEqual([c.args[1] for c in api.call_args_list],['GetOperation'])
            self.assertEqual(json.loads(b.cache_file.read_text())['operation'],OP);b.close()

    def test_expired_recorded_operation_starts_one_new_session(self):
        with tempfile.TemporaryDirectory() as d:
            b=AccountBackend('owner','KGAT_test',cache_dir=d);calls=[]
            def api(service,method,body):
                calls.append((method,body))
                if method=='CreateKernelSession':return {'name':OP+'new'}
                return {'done':True,'response':{'jupyterUrl':URL}}
            with patch.object(b,'_ensure_notebook',return_value=OP),patch.object(b,'_remember_operation'),patch.object(b,'_call',side_effect=api),patch('account_backend.requests.get',return_value=Mock(ok=False,status_code=404)):
                self.assertEqual(b._ensure_session(lambda e:None),URL)
            self.assertEqual([c[0] for c in calls],['GetOperation','CreateKernelSession','GetOperation']);b.close()

    def test_network_failure_during_discovery_does_not_start_another_gpu(self):
        with tempfile.TemporaryDirectory() as d:
            b=AccountBackend('owner','KGAT_test',cache_dir=d)
            with patch.object(b,'_ensure_notebook',return_value=OP),patch.object(b,'_call',return_value={'done':True,'response':{'jupyterUrl':URL}}) as api,patch('account_backend.requests.get',side_effect=requests.ConnectionError()):
                with self.assertRaisesRegex(RuntimeError,'没有重复申请'):b._ensure_session(lambda e:None)
            self.assertEqual([c.args[1] for c in api.call_args_list],['GetOperation']);b.close()

if __name__=='__main__':unittest.main()
