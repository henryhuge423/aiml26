"""Wait for the user's official ZIP, selectively extract EyeQ and audit originals.

CPU only. Does not change formal_v3, grade images, or release a training cohort.
The downloaded archive is never deleted or modified. Output must be a new folder.
"""
from __future__ import annotations
import argparse,collections,csv,datetime,hashlib,json,os,re,shutil,subprocess,sys,tempfile,time,zipfile
from pathlib import Path,PurePosixPath
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'data/audit/eyeq_acquisition_20261110'
SEVENZIP=Path('C:/Program Files/7-Zip/7z.exe')

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def select_members(entries,wanted):
    result=[];seen=set()
    for entry in entries:
        path=entry['path'].replace('\\','/');name=PurePosixPath(path).name
        if name not in wanted:continue
        if (path.startswith('/') or ':' in path or any(p in ('..','.') for p in path.split('/'))
                or any(c in path for c in '\r\n\x00') or not re.fullmatch(r'\d+_(left|right)\.jpeg',name)):
            raise ValueError(f'unsafe image member: {path}')
        if name in seen:raise ValueError(f'duplicate image identity: {name}')
        seen.add(name);result.append(dict(entry,path=path))
    return result

def parse_index(text):
    records=[];record={}
    for line in text.replace('\r','').splitlines()+['Path = END']:
        if line.startswith('Path = '):
            if ('Path' in record and 'Size' in record and record.get('Folder')!='+'
                    and not record['Path'].endswith('/') and not record.get('Attributes','').startswith('D')):
                records.append(dict(path=record['Path'],size=int(record['Size'])))
            record={}
        if ' = ' in line:
            key,value=line.split(' = ',1);record[key]=value
    return records

def write_status(**fields):
    path=STATE/'import_status.json'
    # A new watch is a new run, not a merge with a previous timeout/failure.
    fresh=fields.get('state')=='waiting_for_completed_official_zip'
    state=json.loads(path.read_text()) if path.exists() and not fresh else {}
    state.update(fields,heartbeat_at=now(),pid=os.getpid())
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2),encoding='utf-8');temporary.replace(path)

def labels():
    result={}
    for split in ('train','test'):
        for row in csv.DictReader((ROOT/f'data/external/eyeq/data/Label_EyeQ_{split}.csv').open(encoding='utf-8-sig')):
            if row['image'] in result:raise ValueError('EyeQ image occurs in both official splits')
            result[row['image']]=dict(image=row['image'],eyeq_split=split,quality=row['quality'],DR_grade=row['DR_grade'])
    if len(result)!=28792:raise ValueError('unexpected official label count')
    return result

def capacity(path,size):
    anchor=Path(path)
    while not anchor.exists():anchor=anchor.parent
    if shutil.disk_usage(anchor).free<size+10_000_000_000:
        raise RuntimeError(f'insufficient free space for {size} bytes plus 10GB reserve at {anchor}')

def find_archive(downloads,wanted,previous):
    matches=[]
    for path in downloads.glob('*.zip'):
        info=path.stat();signature=(info.st_size,info.st_mtime_ns)
        stable=previous.get(str(path))==signature;previous[str(path)]=signature
        if not stable:continue
        try:
            with zipfile.ZipFile(path) as z:
                names={PurePosixPath(i.filename.replace('\\','/')).name for i in z.infolist()}
                multipart=all(f'train.zip.{i:03d}' in names for i in range(1,6))
                if multipart or len(names & set(wanted))>=1000:matches.append(path)
        except (OSError,zipfile.BadZipFile):continue
    if len(matches)>1:raise ValueError('multiple matching EyePACS ZIPs; refusing ambiguous import')
    return matches[0] if matches else None

def sevenzip(args,log):
    sys.path.insert(0,str(ROOT))
    from training.processes import ProcessJob
    with log.open('wb') as stream:
        job=ProcessJob()
        try:
            child=job.popen([str(SEVENZIP),*map(str,args)],stdout=stream,stderr=subprocess.STDOUT)
            while child.poll() is None:time.sleep(5);write_status()
            if child.returncode:raise RuntimeError(f'7-Zip failed ({child.returncode}); see {log}')
        finally:job.close()
    return log.read_text(encoding='utf-8',errors='replace')

def import_archive(archive,output,wanted):
    if output.exists():raise ValueError(f'output already exists; refusing overwrite: {output}')
    signature=(archive.stat().st_size,archive.stat().st_mtime_ns)
    write_status(state='hashing_archive',archive=str(archive),output=str(output))
    archive_hash=digest(archive)
    if signature!=(archive.stat().st_size,archive.stat().st_mtime_ns):raise ValueError('archive changed while hashing')
    extracted={};output.mkdir(parents=True)
    write_status(state='extracting',archive_sha256=archive_hash)
    with zipfile.ZipFile(archive) as z:
        infos=z.infolist()
        direct=select_members([dict(path=i.filename,size=i.file_size) for i in infos if not i.is_dir()],wanted)
        if direct:
            capacity(output,sum(x['size'] for x in direct))
            for entry in direct:
                name=PurePosixPath(entry['path']).name;target=output/name
                with z.open(entry['path']) as source,target.open('xb') as dest:shutil.copyfileobj(source,dest,1024*1024)
                extracted[name]=target
        else:
            by_name=collections.defaultdict(list)
            for info in infos:by_name[PurePosixPath(info.filename).name].append(info)
            for partition,count in (('train',5),('test',7)):
                if set(extracted)==set(wanted):break
                part_names=[f'{partition}.zip.{i:03d}' for i in range(1,count+1)]
                if any(len(by_name[name])!=1 for name in part_names):raise ValueError(f'missing/ambiguous {partition} volume')
                parts=[by_name[name][0] for name in part_names]
                capacity(tempfile.gettempdir(),sum(i.file_size for i in parts))
                with tempfile.TemporaryDirectory(prefix='eyeq_official_parts_') as scratch:
                    scratch=Path(scratch)
                    for info in parts:
                        with z.open(info) as source,(scratch/PurePosixPath(info.filename).name).open('xb') as dest:
                            shutil.copyfileobj(source,dest,1024*1024)
                    first=scratch/part_names[0]
                    index=parse_index(sevenzip(['l','-slt','-sccUTF-8',first],STATE/f'{partition}_archive_index.log'))
                    selected=select_members(index,wanted)
                    if set(extracted)&{PurePosixPath(e['path']).name for e in selected}:raise ValueError('identity duplicated across upstream volumes')
                    capacity(output,sum(e['size'] for e in selected))
                    if selected:
                        include=scratch/'include.txt';include.write_text(''.join(e['path']+'\n' for e in selected),encoding='utf-8')
                        sevenzip(['x','-y','-scsUTF-8','-sccUTF-8','-mmt=2',f'-i@{include}',f'-o{output}',first],STATE/f'{partition}_extract.log')
                        for entry in selected:
                            name=PurePosixPath(entry['path']).name;target=output/entry['path']
                            if not target.is_file() or target.stat().st_size!=entry['size']:raise ValueError(f'extraction mismatch: {name}')
                            extracted[name]=target
                write_status(extracted=len(extracted),upstream_partition_finished=partition)
    if set(extracted)!=set(wanted):raise ValueError(f'missing {len(set(wanted)-set(extracted))} official EyeQ images')
    if signature!=(archive.stat().st_size,archive.stat().st_mtime_ns):raise ValueError('source archive changed')
    write_status(state='auditing_full_decode_and_sha256',extracted=len(extracted))
    active=collections.defaultdict(set)
    with (ROOT/'data/prepared/formal_v3/manifest.csv').open(encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):active[row['sha256']].add(row['split'])
    rows=[];subjects=collections.defaultdict(set);content=collections.defaultdict(set)
    for i,(name,path) in enumerate(sorted(extracted.items()),1):
        before=path.stat()
        with Image.open(path) as image:image.load();width,height=image.size
        hashed=digest(path)
        if (before.st_size,before.st_mtime_ns)!=(path.stat().st_size,path.stat().st_mtime_ns):raise ValueError('extracted image changed during audit')
        row=dict(wanted[name],patient_id=name.rsplit('_',1)[0],path=str(path),sha256=hashed,width=width,height=height,
            active_exact_roles='|'.join(sorted(active[hashed])),training_released=False)
        rows.append(row);subjects[row['patient_id']].add(row['eyeq_split']);content[hashed].add(row['eyeq_split'])
        if i%100==0:write_status(audited=i)
    with (STATE/'imported_originals.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    cross_subject=sum(len(v)>1 for v in subjects.values());cross_content=sum(len(v)>1 for v in content.values())
    summary=dict(checked_at=now(),archive=str(archive),archive_sha256=archive_hash,images=len(rows),
        all_decoded=True,official_counts=dict(collections.Counter(r['eyeq_split'] for r in rows)),
        subject_cross_split_groups=cross_subject,exact_content_cross_split_groups=cross_content,
        active_exact_overlap_images=sum(bool(r['active_exact_roles']) for r in rows),
        near_duplicate_and_patient_family_audit='pending; not a training release',training_released=False)
    (STATE/'import_verification.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    write_status(state='originals_imported_pending_full_overlap_audit',audited=len(rows),completed_at=now(),summary=summary)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--downloads',type=Path,default=Path('D:/downloads'))
    parser.add_argument('--output',type=Path,default=Path('D:/datasets/EyeQ_20261110/originals'))
    parser.add_argument('--watch-hours',type=float,default=4)
    args=parser.parse_args();STATE.mkdir(exist_ok=True)
    import msvcrt
    with (STATE/'import.lock').open('a+b') as lock:
        lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:
            wanted=labels();deadline=time.monotonic()+args.watch_hours*3600;previous={}
            write_status(state='waiting_for_completed_official_zip',downloads=str(args.downloads),started_at=now(),training_released=False)
            while time.monotonic()<deadline:
                archive=find_archive(args.downloads,wanted,previous)
                if archive:import_archive(archive,args.output,wanted);return
                time.sleep(20);write_status()
            write_status(state='waiting_timed_out',detail='No complete matching official ZIP appeared within watch period; original download untouched')
        except BaseException as exc:
            write_status(state='failed',error=str(exc));raise

if __name__=='__main__':main()
