import tempfile
import unittest
from pathlib import Path
import pipeline

def result(status='pass'):
    return {'status':status,'summary':status,'blocking_issues':[] if status=='pass' else ['问题'], 'evidence':['测试证据']}

class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name); (self.root/'docs').mkdir()
        self.state={'request':'test','status':'pending','stages':{},'issues':[],'repairs':[]}
    def tearDown(self): self.tmp.cleanup()
    def test_failure_repair_recheck_and_resume(self):
        calls=[]
        def runner(root,role,prompt,timeout,executable):
            calls.append(role)
            if role=='pm' and calls.count('pm')==1: return result('fail')
            if role!='lead':
                filename=next(x[1] for x in pipeline.ROLES if x[0]==role)
                (root/'docs'/filename).write_text('证据',encoding='utf-8')
            return result()
        out=pipeline.run(self.root,self.state,runner,2,10,'fake')
        self.assertEqual(out['status'],'passed'); self.assertEqual(calls[:4],['gate','pm','lead','pm'])
        self.assertEqual(len(out['repairs']),1)
        calls.clear(); pipeline.run(self.root,out,runner,2,10,'fake'); self.assertEqual(calls,[])
    def test_missing_artifact_never_passes(self):
        out=pipeline.run(self.root,self.state,lambda *args:result(),0,10,'fake')
        self.assertEqual(out['status'],'needs_attention'); self.assertEqual(out['stages']['gate']['status'],'fail')
    def test_blocked_does_not_repair_or_continue(self):
        calls=[]
        def runner(root,role,*args): calls.append(role); return result('blocked')
        out=pipeline.run(self.root,self.state,runner,2,10,'fake')
        self.assertEqual(out['status'],'blocked'); self.assertEqual(calls,['gate'])
    def test_conditional_is_not_pass(self):
        out=pipeline.run(self.root,self.state,lambda *args:result('conditional'),0,10,'fake')
        self.assertEqual(out['status'],'needs_attention')
    def test_invalid_response(self):
        self.assertFalse(pipeline.valid({'status':'pass'}))
        self.assertTrue(pipeline.valid(result()))
    def test_resume_rechecks_missing_artifact(self):
        calls=[]
        for role, filename, _ in pipeline.ROLES:
            self.state['stages'][role]=result()
            (self.root/'docs'/filename).write_text('ok',encoding='utf-8')
        (self.root/'docs'/'01-PRD.md').unlink()
        def runner(root,role,prompt,timeout,executable):
            calls.append(role)
            filename=next(x[1] for x in pipeline.ROLES if x[0]==role)
            (root/'docs'/filename).write_text('restored',encoding='utf-8')
            return result()
        pipeline.run(self.root,self.state,runner,0,10,'fake')
        self.assertEqual(calls,[x[0] for x in pipeline.ROLES[1:]])
    def test_empty_evidence_is_not_pass(self):
        def runner(root,role,*args):
            (root/'docs'/'00-任务书.md').write_text('ok',encoding='utf-8')
            return {**result(), 'evidence':[]}
        out=pipeline.run(self.root,self.state,runner,0,10,'fake')
        self.assertEqual(out['status'],'needs_attention')
    def test_qa_repair_requires_fresh_review(self):
        calls=[]
        def runner(root,role,prompt,timeout,executable):
            calls.append(role)
            if role=='qa' and calls.count('qa')==1: return result('fail')
            if role!='lead':
                filename=next(x[1] for x in pipeline.ROLES if x[0]==role)
                (root/'docs'/filename).write_text('ok',encoding='utf-8')
            return result()
        out=pipeline.run(self.root,self.state,runner,2,10,'fake')
        self.assertEqual(out['status'],'passed')
        self.assertEqual(calls[calls.index('reviewer'):calls.index('media')],['reviewer','qa','lead','reviewer','qa'])

if __name__=='__main__': unittest.main()
