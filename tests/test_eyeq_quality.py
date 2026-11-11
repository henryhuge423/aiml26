import unittest

import torch

from tools.freeze_eyeq_quality import protected_components
from training.eyeq_quality import ordinal_loss, ordinal_logits


class EyeQQualityTests(unittest.TestCase):
    def test_unconfirmed_neighbors_of_protected_families_are_blocked(self):
        groups = {'test': 't', 'test-alias': 't', 'train': 'a',
                  'train-alias': 'a', 'other': 'b'}
        edges = [('train', 'test-alias'), ('train-alias', 'other')]
        blocked = protected_components(groups.__getitem__, {'t'}, edges)
        self.assertEqual(blocked, {'t', 'a'})

    def test_higher_q_favors_better_grade_and_missing_labels_do_not_contribute(self):
        scores = torch.tensor([.9, .1, .5], requires_grad=True)
        logits = ordinal_logits(scores, torch.tensor(.5), torch.tensor(-2.0))
        self.assertTrue(bool((logits[:, 0] >= logits[:, 1]).all()))
        loss = ordinal_loss(logits, torch.tensor([0, 2, -1]))
        reversed_loss = ordinal_loss(logits, torch.tensor([2, 0, -1]))
        self.assertLess(loss.item(), reversed_loss.item())
        loss.backward()
        self.assertEqual(scores.grad[-1].item(), 0.)

    def test_batch_without_human_labels_has_finite_zero_ordinal_loss(self):
        logits = torch.randn(4, 2, requires_grad=True)
        loss = ordinal_loss(logits, torch.full((4,), -1))
        self.assertEqual(loss.item(), 0.)
        loss.backward()
        self.assertTrue(bool((logits.grad == 0).all()))
