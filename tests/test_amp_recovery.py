import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class AmpRecoveryTests(unittest.TestCase):
    def test_cuda_overflow_skips_update_and_reduces_scale(self):
        import torch
        from training.engine import optimizer_step
        if not torch.cuda.is_available():
            self.skipTest('CUDA required for real GradScaler overflow')
        parameter = torch.nn.Parameter(torch.tensor([1.0], device='cuda'))
        optimizer = torch.optim.AdamW([parameter], lr=.01)
        scaler = torch.amp.GradScaler('cuda', init_scale=1024)
        before = parameter.detach().clone()
        loss = (parameter * 2).sum()
        hook = parameter.register_hook(lambda gradient: gradient * float('inf'))
        event = optimizer_step(loss, optimizer, scaler, [parameter], 5.)
        hook.remove()
        self.assertFalse(event['optimizer_updated'])
        self.assertEqual(event['scale_after'], 512.)
        self.assertIsNone(event['gradient_norm'])
        self.assertTrue(torch.equal(parameter, before))
        self.assertEqual(len(optimizer.state), 0)
        optimizer.zero_grad(set_to_none=True)
        event = optimizer_step((parameter * 2).sum(), optimizer, scaler, [parameter], 5.)
        self.assertTrue(event['optimizer_updated'])
        self.assertTrue(torch.isfinite(parameter).all())
        self.assertFalse(torch.equal(parameter, before))

    def test_unscaled_nonfinite_gradients_still_fail(self):
        import torch
        from training.engine import optimizer_step
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.AdamW([parameter], lr=.01)
        scaler = torch.amp.GradScaler('cuda', enabled=False)
        parameter.register_hook(lambda gradient: gradient * float('inf'))
        with self.assertRaisesRegex(RuntimeError, 'nonfinite gradients'):
            optimizer_step(parameter.sum(), optimizer, scaler, [parameter], 5.)
        self.assertEqual(parameter.item(), 1.)


if __name__ == '__main__':
    unittest.main()
