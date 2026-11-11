"""Freeze a quality-only release from the audited EyeQ proposal and formal_v3."""
import collections
import csv
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.finalize_eyeq_overlap import known_edges
from audit_eyeq_overlap import digest, read, save_csv

BASE = ROOT / 'data/prepared/formal_v3'
SCREEN = ROOT / 'data/audit/session_20261110/eyeq_overlap'
OUTPUT = ROOT / 'data/prepared/eyeq_quality_v1'


class Components:
    def __init__(self):
        self.parent = {}

    def find(self, path):
        self.parent.setdefault(path, path)
        root = path
        while self.parent[root] != root:
            root = self.parent[root]
        while path != root:
            path, self.parent[path] = self.parent[path], root
        return root

    def join(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def protected_components(find, protected, candidates):
    """Block possible aliases of protected families without equating uncertain edges."""
    blocked = set(protected)
    for a, b in candidates:
        a, b = find(a), find(b)
        if a in protected:
            blocked.add(b)
        if b in protected:
            blocked.add(a)
    return blocked


def freeze():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    summary = SCREEN / 'final_summary.json'
    proposal = SCREEN / 'eyeq_quality_split_proposal.csv'
    base_release = BASE / 'release.json'
    metadata = json.loads(summary.read_text(encoding='utf-8'))
    bindings = dict(metadata['finalizer_input_fingerprints'])
    for path in (summary, proposal, base_release, Path(__file__)):
        bindings[path.relative_to(ROOT).as_posix()] = digest(path)

    def verify_inputs():
        for path, expected in bindings.items():
            if digest(ROOT / path) != expected:
                raise ValueError(f'audit input changed: {path}')

    verify_inputs()
    originals = read(ROOT / 'data/audit/eyeq_acquisition_20261110/imported_originals.csv')
    historical = read(ROOT / 'data/audit/2026-11-09/image_manifest.csv')
    active = read(BASE / 'manifest.csv')
    proposed = {r['path']: r for r in read(proposal)}
    if len(proposed) != len(originals) or len(originals) != metadata['eyeq_images']:
        raise ValueError('EyeQ proposal does not cover the imported cohort')

    def absolute(path):
        return str((ROOT / 'data' / path).resolve())

    components = Components()
    identities = {}
    for row in originals + historical + active:
        path = absolute(row['path'])
        components.find(path)
        for field in ('sha256', 'patient_id', 'related_group'):
            value = row.get(field)
            if value:
                components.join(path, identities.setdefault((field, value), path))
    work = read(ROOT / 'data/audit/2026-11-09/glaucoma_candidate_worklist.csv')
    fiqs = read(ROOT / 'data/prepared/1109/fiqs_quality.csv')
    for a, b in known_edges(work, fiqs):
        components.join(absolute(a), absolute(b))
    for file in (ROOT / 'data/audit/deep/release_inputs/confirmed_edges_v1.csv',
                 SCREEN / 'confirmed_edges.csv'):
        for row in read(file):
            components.join(absolute(row['path_a']), absolute(row['path_b']))

    def group(path):
        return components.find(absolute(path))

    protected = {group(r['path']) for r in originals if r['eyeq_split'] == 'test'}
    protected.update(group(r['path']) for r in active if r['split'] not in {'train', 'val'})
    uncertain = set()
    with (SCREEN / 'unresolved_candidates.csv').open(encoding='utf-8-sig', newline='') as stream:
        def candidates():
            for row in csv.DictReader(stream):
                a, b = components.find(row['path_a']), components.find(row['path_b'])
                uncertain.update((a, b))
                yield a, b

        blocked = protected_components(lambda root: root, protected, candidates())

    def component_id(root):
        return hashlib.sha256(root.encode()).hexdigest()[:24]

    active_roles = collections.defaultdict(set)
    for row in active:
        active_roles[group(row['path'])].add(row['split'])
    rows, excluded, selected = [], [], []
    for row in active:
        if row['task'] != 'quality':
            continue
        root = group(row['path'])
        if root in blocked or len(active_roles[root]) > 1:
            excluded.append(dict(sample_id=row['sample_id'], split=row['split'],
                                 related_group=component_id(root),
                                 reason='protected_family_or_unconfirmed_neighbor' if root in blocked else 'multiple_active_roles'))
        else:
            rows.append(dict(row, related_group=component_id(root), eyeq_grade=''))
    for row in originals:
        draft = proposed[row['path']]
        root = group(row['path'])
        if draft['component'] != component_id(root) or any(
                draft[k] != row[k] for k in ('sha256', 'eyeq_split', 'quality', 'patient_id')):
            raise ValueError(f'EyeQ proposal identity/association mismatch: {row["image"]}')
        role = draft['proposed_role']
        if role not in {'train', 'val'}:
            continue
        if row['eyeq_split'] != 'train' or root in blocked or root in uncertain or active_roles[root] - {role}:
            raise ValueError(f'proposed EyeQ row crosses a protected/uncertain group: {row["image"]}')
        if row['quality'] not in {'0', '1', '2'} or Path(row['image']).name != row['image']:
            raise ValueError('invalid EyeQ grade/image identifier')
        path = (OUTPUT / 'images' / row['image']).relative_to(ROOT / 'data').as_posix()
        rows.append(dict(sample_id='eyeq:' + row['image'], path=path, sha256=row['sha256'],
                         source='EyeQ', label='', original_label=row['quality'],
                         target_definition='FTHNet_continuous_and_EyeQ_human_ordinal',
                         related_group=component_id(root), patient_id_status='known', split=role,
                         task='quality', eligible='true', review_status='audited_official_train',
                         eyeq_grade=row['quality']))
        selected.append(row)

    counts = collections.Counter((r['split'], r['source']) for r in rows)
    eyeq_counts = {split: counts[split, 'EyeQ'] for split in ('train', 'val')}
    if any(eyeq_counts[split] != metadata['proposed_roles'][split] for split in eyeq_counts):
        raise ValueError('released EyeQ counts disagree with audited proposal')
    split_by_group = {}
    for row in rows:
        if split_by_group.setdefault(row['related_group'], row['split']) != row['split']:
            raise ValueError('quality family crosses train/val')
    if any({r['eyeq_grade'] for r in rows if r['source'] == 'EyeQ' and r['split'] == split}
           != {'0', '1', '2'} for split in ('train', 'val')):
        raise ValueError('EyeQ train/val must each cover all three grades')

    with tempfile.TemporaryDirectory(prefix='eyeq_quality_', dir=OUTPUT.parent) as tmp:
        stage = Path(tmp) / 'release'
        (stage / 'images').mkdir(parents=True)
        for row in selected:
            copy = stage / 'images' / row['image']
            shutil.copyfile(row['path'], copy)
            if digest(copy) != row['sha256']:
                raise ValueError(f'imported image changed: {row["image"]}')
        save_csv(stage / 'manifest.csv', rows, list(rows[0]))
        save_csv(stage / 'excluded_formal_v3_quality.csv', excluded,
                 ['sample_id', 'split', 'related_group', 'reason'])
        verify_inputs()
        release = dict(version='formal-research-v1', cohort_version=OUTPUT.name,
                       project_root_from_release='../../..', research_training_released=True,
                       manifest_sha256=digest(stage / 'manifest.csv'), input_fingerprints=bindings,
                       eyeq_counts=eyeq_counts, source_counts={f'{a}/{b}': n for (a, b), n in sorted(counts.items())},
                       excluded_formal_v3_rows=len(excluded),
                       excluded_formal_v3_sha256=digest(stage / 'excluded_formal_v3_quality.csv'),
                       protected_policy='Known families of official EyeQ test and formal_v3 protected roles, plus their direct unconfirmed neighbors, excluded from quality development.')
        (stage / 'release.json').write_text(json.dumps(release, indent=2), encoding='utf-8')
        stage.rename(OUTPUT)
    print(json.dumps(release, indent=2), flush=True)


if __name__ == '__main__':
    freeze()
