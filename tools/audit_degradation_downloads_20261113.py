"""Download pinned author releases and inventory code, licenses, and checkpoints.

Downloaded research material stays in ignored models/; compact provenance is
written beside the new progress report. This tool does not execute third-party
training or inference scripts.
"""
from __future__ import annotations
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/progress/2026-11-13/training-results"
DOWNLOADS = ROOT / "models/degradation_sources_20261113"
SELECTED = {
    "joanshen0508/Fundus-correction-cofe-Net", "liamheng/ArcNet-Medical-Image-Enhancement",
    "heeheee888/FD3", "HudenJear/Catintell", "Retinal-Research/CUNSB-RFIE",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_release(record):
    repo, commit = record["repository"], record["commit"]
    destination = DOWNLOADS / (repo.split("/")[-1] + "-" + commit[:12])
    archive = destination.with_suffix(".zip")
    url = f"https://codeload.github.com/{repo}/zip/{commit}"
    try:
        if not archive.exists():
            with urllib.request.urlopen(url, timeout=180) as response, archive.open("wb") as stream:
                while chunk := response.read(4 * 1024 * 1024):
                    stream.write(chunk)
        with zipfile.ZipFile(archive) as package:
            if package.testzip() is not None:
                raise ValueError("archive CRC failure")
            if sum(item.file_size for item in package.infolist()) > 1024 ** 3:
                raise ValueError("release exceeds expected extraction size")
            for item in package.infolist():
                relative = Path(*Path(item.filename).parts[1:])
                if not relative.parts:
                    continue
                path = (destination / relative).resolve()
                if not path.is_relative_to(destination.resolve()):
                    raise ValueError("unsafe archive entry")
                if item.is_dir():
                    path.mkdir(parents=True, exist_ok=True)
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with package.open(item) as source, path.open("wb") as target:
                        while chunk := source.read(4 * 1024 * 1024):
                            target.write(chunk)
        files = sorted(p for p in destination.rglob("*") if p.is_file())
        relevant = [p for p in files if p.suffix.lower() in {".py", ".pth", ".pt"}
                    or p.name.lower().startswith(("readme", "license", "copying"))]
        return dict(repository=repo, commit=commit, url=url, archive_sha256=sha256(archive),
                    archive_bytes=archive.stat().st_size, local_directory=destination.relative_to(ROOT).as_posix(),
                    status="downloaded_crc_verified", files=[dict(path=p.relative_to(destination).as_posix(),
                    bytes=p.stat().st_size, sha256=sha256(p)) for p in relevant])
    except Exception as exc:
        return dict(repository=repo, commit=commit, url=url, status="download_failed", error=str(exc))


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    records = json.loads((ROOT / "data/audit/data_expansion_20261113/degradation_sources.json").read_text(encoding="utf-8"))["repositories"]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(fetch_release, [r for r in records if r["repository"] in SELECTED]))
    output = dict(date="2026-11-13", pinned_to_prior_audit=True, releases=results)
    (OUTPUT / "degradation-downloads.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    for result in results:
        print(result["repository"], result["status"], result.get("archive_bytes"), flush=True)
    if any(r["status"] != "downloaded_crc_verified" for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
