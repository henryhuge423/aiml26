"""Deterministic, mild fundus acquisition augmentation for disease training.

This is an independent implementation of exposure/illumination, defocus, and
scattering mechanisms discussed by CoFe-Net and ArcNet. It does not import their
source code or restore/replace retinal anatomy. Apply after square padding and
resizing, before normalization. Original quality grades must not be inherited.
"""
from __future__ import annotations
import io
import hashlib
import math
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

VERSION = "fundus-mild-v1"
FAMILIES = ("illumination", "defocus", "scatter")


def retinal_context(image: Image.Image) -> dict[str, Any]:
    """Estimate the visible field's bounding diameter, excluding black padding.

    Rows/columns need at least 10% nonblack occupancy, limiting influence from
    small border annotations. This is a field estimate, not an anatomical mask.
    Disc crops use their visible image extent; they are not full-fundus fields.
    """
    rgb = np.asarray(image.convert("RGB"))
    active = rgb.max(axis=2) > 8
    rows = np.flatnonzero(active.mean(axis=1) >= 0.1)
    columns = np.flatnonzero(active.mean(axis=0) >= 0.1)
    width, height = image.size
    if not len(rows) or not len(columns):
        box = [0, 0, width, height]
        empty = True
    else:
        box = [int(columns[0]), int(rows[0]), int(columns[-1]) + 1, int(rows[-1]) + 1]
        empty = False
    diameter = float(max(box[2] - box[0], box[3] - box[1]))
    return dict(image_size=[width, height], field_box=box,
                field_diameter_pixels=diameter, empty_field=empty)


class FundusDegradation:
    def __init__(self, probability: float = 0.3):
        probability = float(probability)
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("degradation probability must be between zero and one")
        self.probability = probability

    def __call__(self, image: Image.Image, *, seed: int) -> Image.Image:
        return self.augment(image, seed=seed)[0]

    def augment(self, image: Image.Image, *, seed: int) -> tuple[Image.Image, dict[str, Any]]:
        """Return RGB image and complete deterministic replay parameters.

        The API receives no disease label, source label, or quality target.
        Probability and family draws therefore do not depend on those labels.
        """
        original = image.convert("RGB")
        context = retinal_context(original)
        rng = np.random.default_rng(int(seed) % (2 ** 64))
        applied = bool(rng.random() < self.probability and not context["empty_field"])
        record = dict(version=VERSION, seed=int(seed), probability=self.probability,
                      original_rgb_sha256=hashlib.sha256(original.tobytes()).hexdigest(),
                      applied=applied, family="original", context=context, parameters={})
        if not applied:
            return original.copy(), record
        family = str(rng.choice(FAMILIES))
        record["family"] = family
        if family == "illumination":
            parameters = dict(exposure=float(rng.uniform(0.72, 1.12)),
                              gamma=float(rng.uniform(0.9, 1.15)),
                              edge_attenuation=float(rng.uniform(0.05, 0.20)),
                              center_x_fraction=float(rng.uniform(0.4, 0.6)),
                              center_y_fraction=float(rng.uniform(0.4, 0.6)))
        elif family == "defocus":
            parameters = dict(gaussian_sigma_at_512=float(rng.uniform(0.5, 2.0)),
                              secondary=str(rng.choice(("jpeg", "noise"))))
            if parameters["secondary"] == "jpeg":
                parameters["jpeg_quality"] = int(rng.integers(75, 96))
            else:
                parameters["noise_std"] = float(rng.uniform(0.004, 0.012))
                parameters["noise_seed"] = int(rng.integers(0, 2 ** 63))
        else:
            parameters = dict(gaussian_sigma_at_512=float(rng.uniform(0.5, 1.0)),
                              veiling_opacity=float(rng.uniform(0.03, 0.10)),
                              field_sigma_fraction=float(rng.uniform(0.6, 1.0)),
                              veil_rgb=[0.90, 0.88, 0.82])
        record["parameters"] = parameters
        if "gaussian_sigma_at_512" in parameters:
            parameters["gaussian_sigma_pixels"] = parameters["gaussian_sigma_at_512"] * context["field_diameter_pixels"] / 512.0
        return self.replay(original, record), record

    @staticmethod
    def replay(image: Image.Image, record: dict[str, Any]) -> Image.Image:
        if record.get("version") != VERSION:
            raise ValueError("unknown fundus degradation version")
        original = image.convert("RGB")
        context = record["context"]
        if list(original.size) != context["image_size"]:
            raise ValueError("replay image dimensions differ")
        if hashlib.sha256(original.tobytes()).hexdigest() != record["original_rgb_sha256"]:
            raise ValueError("replay source image differs")
        if retinal_context(original) != context:
            raise ValueError("replay retinal field differs")
        if not record["applied"]:
            return original.copy()
        rgb = np.asarray(original).astype(np.float32) / 255.0
        x0, y0, x1, y1 = context["field_box"]
        diameter = context["field_diameter_pixels"]
        parameters = record["parameters"]
        family = record["family"]
        if family == "illumination":
            y, x = np.ogrid[:original.height, :original.width]
            cx = x0 + parameters["center_x_fraction"] * (x1 - x0)
            cy = y0 + parameters["center_y_fraction"] * (y1 - y0)
            radius_squared = ((x - cx) ** 2 + (y - cy) ** 2) / max((diameter / 2) ** 2, 1)
            field = 1 - parameters["edge_attenuation"] * np.clip(radius_squared, 0, 1)
            changed = np.power(rgb, parameters["gamma"]) * parameters["exposure"] * field[..., None]
        elif family in {"defocus", "scatter"}:
            sigma = parameters["gaussian_sigma_at_512"] * diameter / 512.0
            blurred = original.filter(ImageFilter.GaussianBlur(radius=sigma))
            changed = np.asarray(blurred).astype(np.float32) / 255.0
            if family == "defocus":
                if parameters["secondary"] == "jpeg":
                    stream = io.BytesIO()
                    blurred.save(stream, format="JPEG", quality=parameters["jpeg_quality"], subsampling=0)
                    stream.seek(0)
                    with Image.open(stream) as decoded:
                        changed = np.asarray(decoded.convert("RGB")).astype(np.float32) / 255.0
                elif parameters["secondary"] == "noise":
                    rng = np.random.default_rng(parameters["noise_seed"])
                    changed += rng.normal(0, parameters["noise_std"], changed.shape).astype(np.float32)
                else:
                    raise ValueError("unknown defocus secondary operation")
            else:
                y, x = np.ogrid[:original.height, :original.width]
                sigma_field = max(parameters["field_sigma_fraction"] * diameter, 1)
                field = np.exp(-((x - (x0 + x1) / 2) ** 2 + (y - (y0 + y1) / 2) ** 2) / (2 * sigma_field ** 2))
                alpha = parameters["veiling_opacity"] * field[..., None]
                changed = (1 - alpha) * changed + alpha * np.asarray(parameters["veil_rgb"], dtype=np.float32)
        else:
            raise ValueError("unknown fundus degradation family")
        output = np.rint(np.clip(changed, 0, 1) * 255).astype(np.uint8)
        # Keep source black padding/border pixels exactly black rather than
        # teaching a noisy/bright border as a surrogate for acquisition quality.
        background = np.asarray(original).max(axis=2) <= 2
        output[background] = np.asarray(original)[background]
        return Image.fromarray(output)
