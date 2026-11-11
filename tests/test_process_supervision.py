import os
import subprocess
import sys
import unittest
import tempfile
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


@unittest.skipUnless(os.name=='nt','Windows training supervisor')
class SupervisionTests(unittest.TestCase):
    def test_child_cannot_execute_before_job_assignment(self):
        from unittest.mock import patch
        from training.processes import ProcessJob
        scratch=ROOT/'.test_tmp';scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as folder, ProcessJob() as job:
            marker=Path(folder)/'started.txt'
            original=job.api.AssignProcessToJobObject
            observed=[]
            def delayed_assignment(*args):
                time.sleep(1)
                observed.append(marker.exists())
                return original(*args)
            with patch.object(job.api,'AssignProcessToJobObject',side_effect=delayed_assignment):
                child=job.popen([sys.executable,'-c',
                    "from pathlib import Path; import sys,time; Path(sys.argv[1]).write_text('started'); time.sleep(1.5)",str(marker)])
                self.assertEqual(child.wait(timeout=5),0)
            self.assertEqual(observed,[False],'child ran before supervisor owned its lifetime')
            self.assertTrue(marker.exists())

    def test_abrupt_supervisor_exit_terminates_owned_child(self):
        import ctypes
        from ctypes import wintypes
        script=("from training.processes import ProcessJob; import subprocess,sys,time; "
            "job=ProcessJob(); child=job.popen([sys.executable,'-c','import time; time.sleep(120)'],stdout=subprocess.DEVNULL); "
            "print(child.pid,flush=True); time.sleep(120)")
        parent=subprocess.Popen([sys.executable,'-u','-c',script],cwd=ROOT,stdout=subprocess.PIPE,text=True)
        api=ctypes.WinDLL('kernel32',use_last_error=True)
        api.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
        api.OpenProcess.restype=wintypes.HANDLE
        api.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD]
        api.WaitForSingleObject.restype=wintypes.DWORD
        api.CloseHandle.argtypes=[wintypes.HANDLE]
        handle=None
        try:
            child_id=int(parent.stdout.readline().strip())
            handle=api.OpenProcess(0x100000,False,child_id)
            self.assertTrue(handle)
            parent.kill();parent.wait(timeout=5)
            self.assertEqual(api.WaitForSingleObject(handle,5000),0,'orphan survived supervisor exit')
        finally:
            if parent.poll() is None:parent.kill();parent.wait()
            parent.stdout.close()
            if handle:api.CloseHandle(handle)

    def test_job_close_stops_child_before_restart_can_write(self):
        from training.processes import ProcessJob
        child=None
        try:
            with ProcessJob() as job:
                child=job.popen([sys.executable,'-c','import time; time.sleep(120)'],stdout=subprocess.DEVNULL)
                self.assertIsNone(child.poll())
            self.assertIsNotNone(child.wait(timeout=5))
        finally:
            if child is not None and child.poll() is None:
                child.kill();child.wait()


if __name__=='__main__':unittest.main()
