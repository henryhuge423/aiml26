"""Transactional authority for teacher rows; CSV is a reproducible derived export."""
import json
import math
import sqlite3
from pathlib import Path


class TeacherCacheStore:
    def __init__(self,path: Path,fingerprint: dict):
        path.parent.mkdir(parents=True,exist_ok=True)
        self.connection=sqlite3.connect(path)
        self.connection.execute('PRAGMA journal_mode=WAL')
        self.connection.execute('PRAGMA synchronous=FULL')
        self.connection.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY,value TEXT NOT NULL)')
        self.connection.execute('CREATE TABLE IF NOT EXISTS records (sample_id TEXT PRIMARY KEY,row_json TEXT NOT NULL)')
        encoded=json.dumps(fingerprint,sort_keys=True,allow_nan=False)
        previous=self.connection.execute("SELECT value FROM metadata WHERE key='fingerprint'").fetchone()
        if previous and previous[0]!=encoded:
            self.connection.close();raise ValueError('teacher cache journal fingerprint mismatch')
        with self.connection:
            self.connection.execute("INSERT OR IGNORE INTO metadata VALUES ('fingerprint',?)",(encoded,))

    def records(self):
        return {key:json.loads(row) for key,row in self.connection.execute('SELECT sample_id,row_json FROM records')}

    def put(self,row):
        encoded=json.dumps(row,sort_keys=True,allow_nan=False)
        previous=self.connection.execute('SELECT row_json FROM records WHERE sample_id=?',(row['sample_id'],)).fetchone()
        if previous and previous[0]!=encoded:raise ValueError('conflicting teacher cache identity/target')
        with self.connection:
            self.connection.execute('INSERT OR IGNORE INTO records VALUES (?,?)',(row['sample_id'],encoded))

    def close(self):self.connection.close()


def validated_reuse_rows(allowed_rows,previous,allow_missing=False):
    result=[]
    for r in allowed_rows:
        old=previous.get(r['sample_id'])
        if r['split'] not in {'train','val'}:
            raise ValueError('cache reuse requires an allowed quality identity')
        if old is None:
            if allow_missing:continue
            raise ValueError('cache reuse requires an allowed quality identity')
        if any(old[field]!=r[field] for field in ('sample_id','sha256','path','split')):
            raise ValueError('cache reuse image identity/hash/split mismatch')
        if not math.isfinite(float(old['teacher_q'])) or not math.isfinite(float(old['crop_std'])):
            raise ValueError('cache reuse target nonfinite')
        result.append(dict(old))
    return result
