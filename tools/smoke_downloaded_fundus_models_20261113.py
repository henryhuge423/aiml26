"""Run isolated CPU smoke checks against pinned, locally downloaded releases.

No third-party sources are copied into the project's training package. CUNSB
is licensed for academic/non-commercial research. The supplied generator is
tested as a restoration research component, not inserted into disease training.
"""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/progress/2026-11-13/training-results"
RELEASES = ROOT / "models/degradation_sources_20261113"


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def main():
    torch.set_num_threads(2)
    image_path = RELEASES / "Fundus-correction-cofe-Net-5366d53808bc/degradation_code/data/sample/13_left.jpeg"
    with Image.open(image_path) as image:
        image.load()
        image = image.convert("RGB").resize((320, 320), Image.Resampling.BILINEAR)
    checks = {}
    source = RELEASES / "Fundus-correction-cofe-Net-5366d53808bc/degradation_code"
    sys.path.insert(0, str(source))
    try:
        degradation = module(source / "degrad_de.py", "audit_cofe_degradation")
        original = np.asarray(image).astype(np.float32).transpose(2, 0, 1) / 255
        np.random.seed(20261113)
        import random
        random.seed(20261113)
        changed, parameters = degradation.DE_BLUR(original, 320, 320)
        checks["CoFe_author_blur"] = dict(status="executed_cpu", shape=list(changed.shape), finite=bool(np.isfinite(changed).all()), parameters=parameters)
        try:
            degradation.DE_HALO(original, 320, 320, .8)
            checks["CoFe_author_illumination"] = dict(status="executed_cpu")
        except Exception as exc:
            checks["CoFe_author_illumination"] = dict(status="execution_failed", error=type(exc).__name__ + ": " + str(exc))
    except Exception as exc:
        checks["CoFe_import"] = dict(status="import_failed", error=type(exc).__name__ + ": " + str(exc))
    finally:
        sys.path.remove(str(source))
    source = RELEASES / "CUNSB-RFIE-fc0a00f1ade7"
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(RELEASES / "dependencies"))
    started = time.time()
    try:
        from models.DSCNet import DSCNet_pro
        model = DSCNet_pro(3, 3, 9, 1, True, torch.device("cpu"), 32, 1, 9, "reflect", False).eval()
        weights = torch.load(source / "pretrained/pretrain_net_G.pth", map_location="cpu", weights_only=True)
        model.load_state_dict(weights, strict=True)
        generator = torch.Generator().manual_seed(20261113)
        tensor = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float().unsqueeze(0) / 127.5 - 1
        with torch.inference_mode():
            output = model(tensor, torch.zeros(1, dtype=torch.long), torch.randn((1, 256), generator=generator))
        finite = bool(torch.isfinite(output).all())
        if output.shape != tensor.shape or not finite:
            raise ValueError("generator returned wrong shape or nonfinite output")
        values = ((output[0].permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
        Image.fromarray(values).save(OUT / "cunsb-generator-smoke.png")
        image.save(OUT / "cunsb-generator-input.png")
        checks["CUNSB_author_generator"] = dict(status="strict_checkpoint_loaded_forward_executed_cpu", tensors=len(weights), output_shape=list(output.shape),
                finite=finite, output_min=float(output.min()), output_max=float(output.max()), seconds=time.time() - started,
                mean_absolute_rgb_change=float(np.abs(values.astype(float) - np.asarray(image).astype(float)).mean()),
                scope="One generator forward at t=0; not the complete five-step bridge, a clinical reproduction, or a disease result")
    except Exception as exc:
        checks["CUNSB_author_generator"] = dict(status="execution_failed", error=type(exc).__name__ + ": " + str(exc), seconds=time.time() - started)
    (OUT / "degradation-author-smoke.json").write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(checks, indent=2), flush=True)


if __name__ == "__main__":
    main()
