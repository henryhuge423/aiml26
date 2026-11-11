import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        scratch=ROOT/'.test_tmp';scratch.mkdir(exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=scratch)
        assert Path(self.tmp.name).resolve().is_relative_to(scratch.resolve())
        self.addCleanup(self.tmp.cleanup);self.path=Path(self.tmp.name)/'last.pt'

    def engine(self):
        self.assertTrue((ROOT/'training/engine.py').exists(),'checkpoint engine not implemented')
        from training import engine
        return engine

    def test_resume_reproduces_next_dropout_and_optimizer_step(self):
        engine=self.engine();torch.manual_seed(17)
        model=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.Dropout(.3),torch.nn.Linear(4,1))
        optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
        x=torch.randn(6,3);y=torch.ones(6,1)
        optimizer.zero_grad();torch.nn.functional.mse_loss(model(x),y).backward();optimizer.step()
        engine.save_checkpoint(self.path,model,optimizer,None,{'epoch':2,'next_batch':3,'global_step':8},'bound-inputs')
        optimizer.zero_grad();loss=torch.nn.functional.mse_loss(model(x),y);loss.backward();optimizer.step()
        expected={k:v.clone() for k,v in model.state_dict().items()}
        restored=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.Dropout(.3),torch.nn.Linear(4,1))
        opt=torch.optim.AdamW(restored.parameters(),lr=.01)
        state=engine.restore_checkpoint(self.path,restored,opt,None,'bound-inputs','cpu')
        self.assertEqual(state['next_batch'],3)
        opt.zero_grad();resumed=torch.nn.functional.mse_loss(restored(x),y);resumed.backward();opt.step()
        self.assertEqual(loss.item(),resumed.item())
        for k,v in restored.state_dict().items():self.assertTrue(torch.equal(v,expected[k]),k)

    def test_mismatched_manifest_or_config_fingerprint_rejected(self):
        engine=self.engine();model=torch.nn.Linear(2,1);optimizer=torch.optim.AdamW(model.parameters())
        engine.save_checkpoint(self.path,model,optimizer,None,{'epoch':0,'next_batch':0,'global_step':0},'one')
        with self.assertRaisesRegex(ValueError,'fingerprint'):
            engine.restore_checkpoint(self.path,model,optimizer,None,'different','cpu')


if __name__=='__main__':unittest.main()
