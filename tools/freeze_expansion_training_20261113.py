"""Freeze joint disease-training roles from the reviewed November 13 pool.

Original releases stay immutable. All decisions precede model-score inspection.
Run with tools/training_python.cmd to use the audit's RGB decoder.
"""
from __future__ import annotations

import argparse
import collections
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

from PIL import Image, __version__ as PILLOW_VERSION
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
AUDIT = DATA / 'audit/data_expansion_20261113'
DEST = DATA / 'prepared/expanded_20261113'
SEED = 20261113
sys.path.insert(0, str(ROOT))


def read(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def norm(path):
    return os.path.normcase(os.path.abspath(str(path)))


class Groups:
    def __init__(self):
        self.parent = {}

    def find(self, item):
        self.parent.setdefault(item, item)
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return root

    def union(self, first, second):
        first, second = self.find(first), self.find(second)
        if first != second:
            self.parent[max(first, second)] = min(first, second)


def bind_group(graph, records, group_field, path_field='path', data_relative=False):
    first = {}
    for row in records:
        key = row[group_field]
        if not key:
            continue
        path = norm(DATA / row[path_field] if data_relative else row[path_field])
        if key in first:
            graph.union(path, first[key])
        else:
            first[key] = path


def poor(row):
    return row['label'] == '1' and (
        (row['source'] == 'fives' and row.get('quality_fives_any_poor_dimension') == 'True')
        or (row['source'] == 'mshf' and row.get('quality_overall_official') == '0'))


def group_id(graph, row):
    return 'joint13:' + hashlib.sha256(graph.find(norm(DATA / row['path'])).encode()).hexdigest()[:24]


def allocate_groups(group_rows):
    """Stratify whole components; source-family links are allocated only once."""
    strata = collections.defaultdict(list)
    for key, rows in group_rows.items():
        family = '|'.join(sorted({row['source_family'] for row in rows}))
        labels = '|'.join(sorted({row['label'] for row in rows}))
        quality = '|'.join(sorted({row['source'] for row in rows if poor(row)}))
        sources = '|'.join(sorted({row['source'] for row in rows}))
        strata[(family, sources, labels, quality)].append(key)
    roles = {}
    for stratum, keys in sorted(strata.items()):
        keys.sort(key=lambda key: hashlib.sha256(f'{SEED}:{stratum}:{key}'.encode()).hexdigest())
        # Tiny strata keep some fitting rows and validation takes precedence
        # over an artificial exact percentage split.
        count = len(keys)
        n_val = max(1, round(count * .10)) if count >= 5 else 0
        n_boundary = max(1, round(count * .10)) if count >= 10 else 0
        n_test = max(1, round(count * .20)) if count >= 10 else 0
        for index, key in enumerate(keys):
            role = 'val' if index < n_val else 'boundary' if index < n_val + n_boundary else (
                'test' if index < n_val + n_boundary + n_test else 'train')
            roles[key] = role
    return roles


def construct():
    old_path = DATA / 'prepared/formal_v3/manifest.csv'
    pool_path = DATA / 'prepared/expansion_20261113/pool_manifest.csv'
    old_release = json.loads((old_path.parent / 'release.json').read_text())
    pool_release = json.loads((pool_path.parent / 'release.json').read_text())
    assert digest(old_path) == old_release['manifest_sha256']
    assert digest(pool_path) == pool_release['manifest_sha256']
    assert pool_release['audit_complete'] and pool_release['images'] == 9070
    summary = json.loads((AUDIT / 'local_summary.json').read_text())
    assert summary['completed'] and summary['merge_complete']
    for source, expected in summary['inputs'].items():
        assert digest(ROOT / source) == expected, source
    for source, expected in pool_release['inputs'].items():
        assert digest(ROOT / source) == expected, source
    for source, expected in old_release['input_fingerprints'].items():
        assert digest(ROOT / source) == expected, source

    old_all, pool = read(old_path), read(pool_path)
    old = [dict(row, origin_cohort='formal_v3', source_family=row['source'],
                imaging_domain='fundus', original_related_group=row['related_group'])
           for row in old_all if row['task'] == 'glaucoma']
    pool = [dict(row, origin_cohort='expansion_20261113', original_related_group=row['related_group'])
            for row in pool]
    assert len(pool) == 9070 and all(row['label'] in {'0', '1'} for row in pool)
    eyeq_path = DATA / 'audit/session_20261110/eyeq_overlap/full_components.csv'
    eyeq = read(eyeq_path)
    decisions_path = AUDIT / 'local_decisions.csv'
    decisions = read(decisions_path)
    graph = Groups()
    bind_group(graph, old_all, 'related_group', data_relative=True)
    bind_group(graph, decisions, 'related_group')
    bind_group(graph, eyeq, 'component')
    first_sha = {}
    for records, data_relative in [(old_all, True), (eyeq, False), (decisions, False)]:
        for row in records:
            path = norm(DATA / row['path']) if data_relative else norm(row['path'])
            key = row['sha256']
            if key in first_sha:
                graph.union(path, first_sha[key])
            else:
                first_sha[key] = path
    edge_paths = [AUDIT / 'final_exact_edges.csv', AUDIT / 'final_confirmed_edges.csv',
                  AUDIT / 'local_legacy_confirmed_edges.csv',
                  DATA / 'audit/deep/release_inputs/confirmed_edges_v1.csv',
                  DATA / 'audit/session_20261110/eyeq_overlap/confirmed_edges.csv']
    edge_count = 0
    for path in edge_paths:
        for edge in read(path):
            graph.union(norm(DATA / edge['path_a']), norm(DATA / edge['path_b']))
            edge_count += 1
    # MSHF equal-RGB PNG training copies must stay linked to original TIFFs.
    for row in pool:
        graph.union(norm(DATA / row['path']), norm(DATA / row['original_path']))
    for row in old + pool:
        row['related_group'] = group_id(graph, row)

    eyeq_roles = collections.defaultdict(set)
    for row in eyeq:
        root = graph.find(norm(row['path']))
        if row['eyeq_split'] == 'test':
            eyeq_roles[root].add('test')
        elif row.get('proposal_eligible') == 'True' and row.get('proposed_role') in {'train', 'val'}:
            eyeq_roles[root].add(row['proposed_role'])
    old_groups = collections.defaultdict(list)
    for row in old:
        old_groups[row['related_group']].append(row)
    excluded, retained = [], []
    repaired = {row['sample_id'] for row in read(DATA / 'prepared/formal_v3_boundary_repaired_20261113/manifest.csv')}
    original_conflicts = collections.Counter()
    protected = {'test', 'external_acrima', 'external_hygd'}
    for key, rows in old_groups.items():
        roles = {row['split'] for row in rows}
        qroles = eyeq_roles[graph.find(norm(DATA / rows[0]['path']))]
        if len(roles) > 1:
            original_conflicts['|'.join(sorted(roles))] += 1
        locked = roles & protected
        assert len(locked) <= 1, ('protected role crossing requires adjudication', key, roles)
        for row in rows:
            reason = ''
            if row['split'] == 'boundary' and row['sample_id'] not in repaired:
                reason = 'November_13_identity_only_boundary_repair'
            elif locked and row['split'] not in locked:
                reason = 'confirmed_group_contains_protected_disease_test'
            elif row['split'] == 'train' and 'val' in roles:
                reason = 'confirmed_group_contains_original_checkpoint_validation'
            elif row['split'] == 'boundary' and roles & {'train', 'val'}:
                reason = 'confirmed_group_contains_original_model_fitting_or_validation'
            elif row['split'] not in protected and 'test' in qroles:
                reason = 'confirmed_group_contains_official_EyeQ_test'
            elif row['split'] == 'train' and 'val' in qroles:
                reason = 'confirmed_group_contains_EyeQ_model_validation'
            elif row['split'] in {'val', 'boundary'} and 'train' in qroles:
                reason = 'confirmed_group_contains_EyeQ_quality_training'
            if reason:
                excluded.append(dict(row, exclusion_reason=reason))
            else:
                retained.append(row)
    original_roles = collections.defaultdict(set)
    for row in retained:
        original_roles[row['related_group']].add(row['split'])
    assert all(len(roles) == 1 for roles in original_roles.values())
    pool_groups = collections.defaultdict(list)
    for row in pool:
        root = graph.find(norm(DATA / row['path']))
        assert not eyeq_roles[root], ('pool overlaps an EyeQ fitting/evaluation role', row['sample_id'])
        assert row['related_group'] not in original_roles, ('pool overlaps existing disease release', row['sample_id'])
        pool_groups[row['related_group']].append(row)
    forced_test, free = {}, {}
    for key, rows in pool_groups.items():
        if any(row['source'] == 'papila' or row['official_split'] == 'test' for row in rows):
            forced_test[key] = 'test'
        else:
            free[key] = rows
    assignments = dict(allocate_groups(free), **forced_test)
    for row in pool:
        row['split'] = assignments[row['related_group']]
        row['allocation_reason'] = ('PAPILA_whole_source_holdout' if row['source'] == 'papila' else
            'official_test_or_linked_patient_image_group' if row['related_group'] in forced_test else
            'seeded_joint_source_label_real_quality_group_stratification')
    rows = sorted(retained + pool, key=lambda row: (row['split'], row['origin_cohort'], row['sample_id']))
    assert len({row['sample_id'] for row in rows}) == len(rows)
    assert len({row['sha256'] for row in rows}) == len(rows)
    role_groups = collections.defaultdict(set)
    for row in rows:
        role_groups[row['related_group']].add(row['split'])
    assert all(len(roles) == 1 for roles in role_groups.values())
    old_locked = {row['sample_id']: row for row in old if row['split'] in protected}
    new_locked = {row['sample_id']: row for row in retained if row['split'] in protected}
    assert old_locked == new_locked
    inputs = [old_path, old_path.parent / 'release.json', pool_path, pool_path.parent / 'release.json',
              eyeq_path, decisions_path, AUDIT / 'local_summary.json',
              DATA / 'prepared/formal_v3_boundary_repaired_20261113/manifest.csv',
              DATA / 'prepared/formal_v3_boundary_repaired_20261113/release.json',
              Path(__file__)] + edge_paths
    report = dict(seed=SEED, input_pool_images=len(pool), retained_original_images=len(retained),
                  excluded_original_images=len(excluded),
                  excluded_original_by_role=dict(collections.Counter(row['split'] for row in excluded)),
                  excluded_original_reasons=dict(collections.Counter(row['exclusion_reason'] for row in excluded)),
                  original_cross_role_components=dict(original_conflicts),
                  counts=dict(collections.Counter(row['split'] for row in rows)),
                  origin_role_counts=dict(collections.Counter(f"{row['origin_cohort']}/{row['split']}" for row in rows)),
                  origin_role_positives=dict(collections.Counter(f"{row['origin_cohort']}/{row['split']}" for row in rows if row['label'] == '1')),
                  source_role_counts=dict(collections.Counter(f"{row['source']}/{row['split']}" for row in rows)),
                  real_poor_positive_counts=dict(collections.Counter(f"{row['source']}/{row['split']}" for row in pool if poor(row))),
                  independent_source_holdout='papila', whole_source_holdout_images=420,
                  protected_old_disease_rows_preserved=len(old_locked),
                  known_patient_components=sum(any(row.get('patient_id') for row in rs) for rs in pool_groups.values()),
                  new_groups=len(pool_groups), role_crossings=0, duplicate_file_hashes=0,
                  graph_edge_records=edge_count, official_test_images=sum(row['official_split'] == 'test' for row in pool),
                  source_family_grouping='LAG_GRAPE_FIVES_FundusAVSeg_MSHF_joint_graph',
                  auxiliary_suspects_excluded=204)
    decision_by_path = {norm(row['path']): row for row in decisions}
    return rows, excluded, inputs, report, decision_by_path


def verify_images(rows, decisions):
    assert PILLOW_VERSION == '13.0.0', 'Use tools/training_python.cmd (Pillow 13.0.0)'
    def check(row):
        path = (DATA / row['path']).resolve()
        assert path.is_relative_to(DATA.resolve())
        assert digest(path) == row['sha256'], path
        with Image.open(path) as image:
            image.load()
            rgb = image.convert('RGB')
            assert rgb.width > 0 and rgb.height > 0
            pixel_hash = hashlib.sha256(rgb.tobytes()).hexdigest()
        if row['origin_cohort'] == 'expansion_20261113':
            original = DATA / row['original_path']
            original_record = decisions[norm(original)]
            assert digest(original) == row['original_sha256'], original
            assert pixel_hash == original_record['pixel_sha256'], path
            if original.resolve() != path:
                array = cv2.imdecode(np.frombuffer(original.read_bytes(), np.uint8),
                                     cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                assert array is not None, original
                assert hashlib.sha256(cv2.cvtColor(array, cv2.COLOR_BGR2RGB).tobytes()).hexdigest() == pixel_hash
        return row['sample_id'], pixel_hash
    outputs = []
    with ThreadPoolExecutor(max_workers=4) as workers:
        for index, value in enumerate(workers.map(check, rows), 1):
            outputs.append(value)
            if index % 2000 == 0:
                print(f'file SHA256/full RGB verification {index}/{len(rows)}', flush=True)
    return dict(outputs)


def write(path, rows, fields):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='raise')
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    rows, excluded, inputs, report, decisions = construct()
    if args.plan_only:
        print(json.dumps(report, indent=2))
        return
    if DEST.exists() and not args.verify:
        raise ValueError('Immutable destination exists; use --verify')
    if args.verify:
        release = json.loads((DEST / 'release.json').read_text())
        assert digest(DEST / 'manifest.csv') == release['manifest_sha256']
        for path, expected in release['input_fingerprints'].items():
            assert digest(ROOT / path) == expected, path
        actual = read(DEST / 'manifest.csv')
        assert {row['sample_id']: row for row in actual} == {
            row['sample_id']: {field: row.get(field, '') for field in actual[0]} for row in rows}
    pixel_hashes = verify_images(rows, decisions)
    # Exact decoded pixel identity may merge different encodings; never let it
    # cross a role even if a previously reviewed geometric edge was missed.
    pixel_roles = collections.defaultdict(set)
    for row in rows:
        pixel_roles[pixel_hashes[row['sample_id']]].add(row['split'])
    assert all(len(roles) == 1 for roles in pixel_roles.values()), 'cross-role equal decoded RGB'
    report.update(passed=True, unique_paths_file_hash_and_full_RGB_verified=len(rows),
                  expansion_original_pixel_hashes_verified=9070,
                  cross_role_equal_RGB_hashes=0, pillow_version=PILLOW_VERSION,
                  original_MSHF_decoder=cv2.__version__)
    if not args.verify:
        DEST.mkdir(parents=True)
        fields = sorted({field for row in rows for field in row})
        write(DEST / 'manifest.csv', rows, fields)
        write(DEST / 'exclusions.csv', excluded, sorted({field for row in excluded for field in row}))
        release = dict(version='formal-research-v1', cohort_version='expanded_20261113', seed=SEED,
                       project_root_from_release='../../..', research_training_released=True,
                       manifest_sha256=digest(DEST / 'manifest.csv'), images=len(rows),
                       disease_only=True, quality_supervision='Existing quality model frozen; no degraded-image quality target reused',
                       input_fingerprints={path.relative_to(ROOT).as_posix(): digest(path) for path in inputs},
                       allocation=report, final_tests_locked=True,
                       author_test_partitions_preserved=True, all_patient_independence_proven=False,
                       source_endpoint_policy='Keep source-defined diagnosis, referable glaucoma, suspect exclusion and imaging domain explicit',
                       test_policy='PAPILA whole-source, original ACRIMA/HYGD/internal test and new official/group test rows remain unavailable to training selection')
        (DEST / 'release.json').write_text(json.dumps(release, indent=2), encoding='utf-8')
        report['manifest_sha256'] = release['manifest_sha256']
        (DEST / 'cohort_verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    from training.data import load_rows
    for split, count in report['counts'].items():
        assert len(load_rows(DEST / 'manifest.csv', 'glaucoma', split, DEST / 'release.json')) == count
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
