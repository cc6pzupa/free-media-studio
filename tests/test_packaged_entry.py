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


if __name__=='__main__':unittest.main()
