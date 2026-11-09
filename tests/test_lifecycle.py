import sys, threading, time, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from session_runtime import ModelSession

class FakeTransport:
    def __init__(self): self.closed=False; self.loads=0; self.active=0; self.peak=0
    def start(self): pass
    def close(self): self.closed=True
    def execute(self, code, timeout):
        if code=='bootstrap': self.loads+=1; return ''
        self.active+=1; self.peak=max(self.peak,self.active)
        try:
            if code=='slow': time.sleep(.12)
            if code=='error': raise ValueError('remote error')
            return code
        finally: self.active-=1

class LifecycleTest(unittest.TestCase):
    def setUp(self):
        self.created=[]
        def factory():
            t=FakeTransport(); self.created.append(t); return t
        self.factory=factory
    def test_reuse_and_serialization(self):
        s=ModelSession(self.factory,'bootstrap',10)
        try:
            self.assertEqual(s.execute('first'),'first')
            threads=[threading.Thread(target=s.execute,args=('slow',)) for _ in range(3)]
            for t in threads:t.start()
            for t in threads:t.join()
            self.assertEqual(len(self.created),1)
            self.assertEqual(self.created[0].loads,1)
            self.assertEqual(self.created[0].peak,1)
        finally:s.close()
    def test_deadline_does_not_extend_on_request(self):
        now=[0.0]; s=ModelSession(self.factory,'bootstrap',100,clock=lambda:now[0])
        try:
            s.execute('first'); now[0]=99; s.execute('second')
            self.assertEqual(len(self.created),1)
            now[0]=100; s.execute('third')
            self.assertEqual(len(self.created),2); self.assertTrue(self.created[0].closed)
        finally:s.close()
    def test_timer_waits_for_current_generation(self):
        s=ModelSession(self.factory,'bootstrap',.04)
        self.assertEqual(s.execute('slow'),'slow')
        deadline=time.monotonic()+2
        while not self.created[0].closed and time.monotonic()<deadline:time.sleep(.01)
        self.assertTrue(self.created[0].closed)
        self.assertEqual(s.execute('next'),'next'); self.assertEqual(len(self.created),2)
        s.close()
    def test_error_not_retried(self):
        s=ModelSession(self.factory,'bootstrap',100)
        try:
            with self.assertRaises(ValueError):s.execute('error')
            self.assertEqual(len(self.created),1)
        finally:s.close()

if __name__=='__main__':unittest.main()
