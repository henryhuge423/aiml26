"""Replayable defocus and exposure calibrated on same-eye photo pairs."""
from __future__ import annotations

import hashlib

import numpy as np
from PIL import Image, ImageFilter

from .fundus_degradation import FundusDegradation, retinal_context

VERSION = 'fundus-calibrated-defocus-v1'


class CalibratedDefocus(FundusDegradation):
    def __init__(self, probability, profile, families=None):
        super().__init__(probability, families)
        bank = np.asarray(profile, dtype=float)
        if (bank.ndim != 2 or bank.shape[1] != 2 or len(bank) == 0
                or not np.isfinite(bank).all() or (bank[:, 0] < 0).any()
                or (bank[:, 1] <= 0).any()):
            raise ValueError('expected nonempty pairs of nonnegative blur radius and positive exposure gain')
        self.profile = tuple(tuple(row) for row in bank.tolist())

    def augment(self, image, *, seed):
        baseline, record = super().augment(image, seed=seed)
        if not record['applied'] or record['family'] != 'defocus':
            return baseline, record
        bank_seed = int.from_bytes(hashlib.sha256(f'{VERSION}:{seed}'.encode()).digest()[:8], 'big')
        radius, gain = self.profile[bank_seed % len(self.profile)]
        record.update(version=VERSION, parameters=dict(radius_at_512=radius, exposure_gain=gain))
        return self.replay(image, record), record

    @staticmethod
    def replay(image, record):
        if record.get('version') != VERSION:
            return FundusDegradation.replay(image, record)
        original = image.convert('RGB')
        context = retinal_context(original)
        if (context != record['context'] or
                hashlib.sha256(original.tobytes()).hexdigest() != record['original_rgb_sha256']):
            raise ValueError('calibrated replay source image differs')
        params = record['parameters']
        radius = params['radius_at_512'] * context['field_diameter_pixels'] / 512
        blurred = np.asarray(original.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32)
        rendered = np.rint(np.clip(blurred * params['exposure_gain'], 0, 255)).astype(np.uint8)
        source = np.asarray(original)
        background = source.max(axis=2) <= 2
        rendered[background] = source[background]
        return Image.fromarray(rendered)
