import contextlib
import csv
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pipeline

class SafetyTests(unittest.TestCase):
    def test_all_buffer_entrypoints_disabled(self):
        with patch.object(pipeline.urllib.request, 'urlopen', side_effect=AssertionError('network called')):
            with contextlib.redirect_stdout(io.StringIO()):
                pipeline.cmd_push(SimpleNamespace(file='unused', account='tech'))
                pipeline.cmd_publish(None)
                self.assertFalse(pipeline.buffer_push('text', 'tech', {'BUFFER_TOKEN':'dummy'})[0])

    def test_feedback_goes_into_pending_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); kb=root/'knowledge_base'
            for d in [kb/'topics',kb/'articles',kb/'archive',root/'feedback']: d.mkdir(parents=True)
            pool=kb/'topics'/'topic_pool.md'
            pool.write_text('# topics\n## 待做队列\n\n## 衍生选题区\n',encoding='utf-8')
            with (root/'feedback'/'metrics.csv').open('w',encoding='utf-8',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=['platform','date','title','reads','likes','comments','url'])
                writer.writeheader(); writer.writerow({'platform':'zhihu','title':'测试','reads':'2000'})
            with patch.multiple(pipeline, ROOT=root, KB=kb), patch.object(pipeline,'load_env',return_value={}):
                with contextlib.redirect_stdout(io.StringIO()):
                    pipeline.cmd_feedback(None); pipeline.cmd_feedback(None)
            text=pool.read_text(encoding='utf-8')
            self.assertEqual(text.count('测试 的续作/反方角度'),1)
            self.assertIn('测试',pipeline.next_pending(text))

if __name__=='__main__': unittest.main()
