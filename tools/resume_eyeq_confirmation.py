"""Guarded entry point for resuming the immutable EyeQ continuation."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CONTINUATION=ROOT/'tools/continue_eyeq_confirmation.py'


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_implementation(screen,implementation_path):
    evidence_path=Path(screen)/'cached_confirmation.json'
    implementation_hash=sha256(implementation_path)
    if evidence_path.exists():
        evidence=json.loads(evidence_path.read_text(encoding='utf-8'))
        if evidence.get('implementation_sha256')!=implementation_hash:
            raise ValueError('cached confirmation implementation hash mismatch')
    binding_path=Path(screen)/'confirmation_cache'/'binding.json'
    if binding_path.exists():
        binding=json.loads(binding_path.read_text(encoding='utf-8'))
        bound_implementation=binding.get('implementation_sha256')
        if bound_implementation is not None and bound_implementation!=implementation_hash:
            raise ValueError('confirmation cache binding implementation hash mismatch')


def guarded_resume(screen,implementation_path,continuation_main):
    verify_implementation(screen,implementation_path)
    return continuation_main()


def load_continuation(path=CONTINUATION):
    spec=importlib.util.spec_from_file_location('eyeq_confirmation_implementation',path)
    if spec is None or spec.loader is None:
        raise ImportError(f'cannot load EyeQ continuation implementation: {path}')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--screen',type=Path,default=ROOT/'data/audit/session_20261110/eyeq_overlap')
    args=parser.parse_args()
    implementation=load_continuation()
    guarded_resume(args.screen,Path(implementation.__file__).resolve(),implementation.main)


if __name__=='__main__':main()
