import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]


class EyeqImportTests(unittest.TestCase):
    def module(self):
        spec=importlib.util.spec_from_file_location('eyeq_import',ROOT/'tools/import_eyeq_download.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module

    def test_select_by_official_filename_not_archive_partition(self):
        m=self.module()
        wanted={'100_left.jpeg':dict(eyeq_split='test')}
        entries=[dict(path='train/100_left.jpeg',size=12),dict(path='train/200_left.jpeg',size=90)]
        self.assertEqual(m.select_members(entries,wanted),entries[:1])

    def test_reject_unsafe_paths_and_ambiguous_identity(self):
        m=self.module();wanted={'100_left.jpeg':{}}
        for path in ('../100_left.jpeg','C:/100_left.jpeg','/100_left.jpeg','train/../100_left.jpeg'):
            with self.assertRaises(ValueError):m.select_members([dict(path=path,size=12)],wanted)
        with self.assertRaisesRegex(ValueError,'duplicate'):
            m.select_members([dict(path='a/100_left.jpeg',size=12),dict(path='b/100_left.jpeg',size=12)],wanted)

    def test_parse_sevenzip_member_index(self):
        m=self.module()
        text='Path = archive.zip.001\nType = zip\n\nPath = train/100_left.jpeg\nFolder = -\nSize = 100\n\nPath = train\nFolder = +\nSize = 0\n'
        self.assertEqual(m.parse_index(text),[dict(path='train/100_left.jpeg',size=100)])

    def test_new_watch_does_not_inherit_previous_failure_or_completion(self):
        m=self.module()
        with tempfile.TemporaryDirectory(dir=ROOT/'.test_tmp') as scratch:
            m.STATE=Path(scratch)
            m.write_status(state='failed',error='old error',detail='old timeout',
                           completed_at='old completion',extracted=123,archive='old.zip')
            m.write_status(state='waiting_for_completed_official_zip',started_at='new run',training_released=False)
            state=json.loads((m.STATE/'import_status.json').read_text())
            for key in ('error','detail','completed_at','extracted','archive'):
                self.assertNotIn(key,state)
            m.write_status()
            self.assertEqual(json.loads((m.STATE/'import_status.json').read_text())['started_at'],'new run')

    def test_successful_small_import_reports_final_audited_count(self):
        m=self.module()
        with tempfile.TemporaryDirectory(dir=ROOT/'.test_tmp') as scratch:
            directory=Path(scratch);m.STATE=directory
            image=directory/'100_left.jpeg';Image.new('RGB',(32,24),'red').save(image)
            archive=directory/'images.zip'
            with zipfile.ZipFile(archive,'w') as z:z.write(image,'train/100_left.jpeg')
            wanted={'100_left.jpeg':dict(image='100_left.jpeg',eyeq_split='test',quality='0',DR_grade='0')}
            m.import_archive(archive,directory/'originals',wanted)
            state=json.loads((directory/'import_status.json').read_text())
            self.assertEqual(state.get('audited'),1)
            self.assertEqual(state['summary']['images'],1)
            self.assertFalse(state['summary']['training_released'])


if __name__=='__main__':unittest.main()
