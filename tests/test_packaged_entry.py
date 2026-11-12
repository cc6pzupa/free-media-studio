import io
import sys
import unittest
from unittest.mock import patch
import app_entry


class PackagedEntryTest(unittest.TestCase):
    def test_frozen_english_windows_stream_can_print_chinese_startup(self):
        raw=io.BytesIO()
        stream=io.TextIOWrapper(raw,encoding='cp1252')
        try:
            with patch.object(sys,'frozen',True,create=True),patch.object(sys,'stdout',stream),patch.object(sys,'stderr',stream),patch.object(sys,'argv',['cc-media-studio','--version']):
                self.assertEqual(app_entry.main(),0)
                print('保持此进程运行；退出仅关闭本机服务。',flush=True)
            self.assertIn('保持此进程运行',raw.getvalue().decode('utf-8'))
        finally:stream.close()

class WindowsWorkerTest(unittest.TestCase):
    def exercise(self,allow):
        import contextlib
        import subprocess
        from studio_client import spawn_worker
        denied=OSError('Job denies breakaway');denied.winerror=5
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch('studio_client.os.name','nt'))
            for name,value in [('CREATE_NEW_PROCESS_GROUP',4),('DETACHED_PROCESS',8),('CREATE_BREAKAWAY_FROM_JOB',16)]:
                stack.enter_context(patch.object(subprocess,name,value,create=True))
            launch=stack.enter_context(patch('studio_client.subprocess.Popen',side_effect=[denied,object()]))
            if allow:
                spawn_worker(['app','--worker'],None,'folder',{},True)
                self.assertEqual(launch.call_count,2)
                self.assertEqual(launch.call_args.kwargs['creationflags'],12)
            else:
                with self.assertRaises(OSError):spawn_worker(['app','--worker'],None,'folder',{})
                self.assertEqual(launch.call_count,1)
    def test_gui_can_inherit_an_enclosing_job(self):self.exercise(True)
    def test_mcp_never_falls_back_to_a_host_bound_worker(self):self.exercise(False)


if __name__=='__main__':unittest.main()
