"""Windows job lifetime owns GPU subprocesses and their loader descendants."""
import ctypes
from ctypes import wintypes
import subprocess


class _Limits(ctypes.Structure):
    _fields_=[('process_time',ctypes.c_longlong),('job_time',ctypes.c_longlong),
        ('flags',wintypes.DWORD),('minimum_ws',ctypes.c_size_t),('maximum_ws',ctypes.c_size_t),
        ('active_processes',wintypes.DWORD),('affinity',ctypes.c_size_t),
        ('priority',wintypes.DWORD),('scheduling',wintypes.DWORD)]


class _IO(ctypes.Structure):
    _fields_=[(name,ctypes.c_ulonglong) for name in ('read_ops','write_ops','other_ops','read_bytes','write_bytes','other_bytes')]


class _ExtendedLimits(ctypes.Structure):
    _fields_=[('basic',_Limits),('io',_IO),('process_memory',ctypes.c_size_t),
        ('job_memory',ctypes.c_size_t),('peak_process_memory',ctypes.c_size_t),('peak_job_memory',ctypes.c_size_t)]


class _ThreadEntry(ctypes.Structure):
    _fields_=[('size',wintypes.DWORD),('usage',wintypes.DWORD),('thread_id',wintypes.DWORD),
        ('process_id',wintypes.DWORD),('base_priority',wintypes.LONG),('delta_priority',wintypes.LONG),('flags',wintypes.DWORD)]


class ProcessJob:
    def __init__(self):
        self.api=ctypes.WinDLL('kernel32',use_last_error=True)
        self.api.CreateJobObjectW.argtypes=[ctypes.c_void_p,wintypes.LPCWSTR]
        self.api.CreateJobObjectW.restype=wintypes.HANDLE
        self.api.SetInformationJobObject.argtypes=[wintypes.HANDLE,ctypes.c_int,ctypes.c_void_p,wintypes.DWORD]
        self.api.SetInformationJobObject.restype=wintypes.BOOL
        self.api.AssignProcessToJobObject.argtypes=[wintypes.HANDLE,wintypes.HANDLE]
        self.api.AssignProcessToJobObject.restype=wintypes.BOOL
        self.api.CloseHandle.argtypes=[wintypes.HANDLE]
        self.api.CloseHandle.restype=wintypes.BOOL
        self.api.CreateToolhelp32Snapshot.argtypes=[wintypes.DWORD,wintypes.DWORD]
        self.api.CreateToolhelp32Snapshot.restype=wintypes.HANDLE
        for name in ('Thread32First','Thread32Next'):
            function=getattr(self.api,name)
            function.argtypes=[wintypes.HANDLE,ctypes.POINTER(_ThreadEntry)]
            function.restype=wintypes.BOOL
        self.api.OpenThread.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
        self.api.OpenThread.restype=wintypes.HANDLE
        self.api.ResumeThread.argtypes=[wintypes.HANDLE]
        self.api.ResumeThread.restype=wintypes.DWORD
        self.handle=self.api.CreateJobObjectW(None,None)
        if not self.handle:raise ctypes.WinError(ctypes.get_last_error())
        limits=_ExtendedLimits();limits.basic.flags=0x2000  # KILL_ON_JOB_CLOSE
        if not self.api.SetInformationJobObject(self.handle,9,ctypes.byref(limits),ctypes.sizeof(limits)):
            error=ctypes.WinError(ctypes.get_last_error());self.close();raise error

    def popen(self,arguments,**kwargs):
        # CREATE_SUSPENDED prevents any user code/descendants before ownership.
        # https://learn.microsoft.com/en-us/windows/win32/procthread/process-creation-flags
        kwargs['creationflags']=kwargs.get('creationflags',0)|0x4
        child=subprocess.Popen(arguments,**kwargs)
        try:
            if not self.api.AssignProcessToJobObject(self.handle,wintypes.HANDLE(child._handle)):
                raise ctypes.WinError(ctypes.get_last_error())
            self._resume_initial_thread(child.pid)
        except BaseException:
            child.kill();child.wait();raise
        return child

    def _resume_initial_thread(self,pid):
        snapshot=self.api.CreateToolhelp32Snapshot(0x4,0)  # TH32CS_SNAPTHREAD
        if snapshot==ctypes.c_void_p(-1).value:raise ctypes.WinError(ctypes.get_last_error())
        try:
            entry=_ThreadEntry();entry.size=ctypes.sizeof(entry)
            found=self.api.Thread32First(snapshot,ctypes.byref(entry))
            while found:
                if entry.process_id==pid:
                    thread=self.api.OpenThread(0x2,False,entry.thread_id)  # THREAD_SUSPEND_RESUME
                    if not thread:raise ctypes.WinError(ctypes.get_last_error())
                    try:
                        if self.api.ResumeThread(thread)!=1:raise RuntimeError('unexpected initial thread suspend count')
                    finally:self.api.CloseHandle(thread)
                    return
                found=self.api.Thread32Next(snapshot,ctypes.byref(entry))
            raise RuntimeError('suspended child initial thread not found')
        finally:self.api.CloseHandle(snapshot)

    def close(self):
        if self.handle:
            self.api.CloseHandle(self.handle);self.handle=None

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
