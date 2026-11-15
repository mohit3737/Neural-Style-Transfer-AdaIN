"""
AdaIN Neural Style Transfer — Automated Test Suite
Clean, readable, and comprehensive verification of all 11 project stages.
Author: Senior AI/ML Engineer
"""

from pathlib import Path
import time
import psutil
import torch
from PIL import Image
from torchvision.transforms import functional as TF

import adain_model as am

# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent
PROJECT_DIR = PROJECT_ROOT / "project"
CONTENT_DIR = PROJECT_DIR / "dataset" / "test2017"
STYLE_DIR = PROJECT_DIR / "dataset" / "style"
CHECKPOINT_PATH = PROJECT_DIR / "checkpoints" / "adain_decoder_100_batches.pth"
OUTPUT_DIR = PROJECT_ROOT / "generated_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_all_tests():
    print("=" * 65)
    print("AdaIN NEURAL STYLE TRANSFER — AUTOMATED TEST SUITE")
    print("=" * 65)

    results = {}

    def test(test_num: int, name: str, fn):
        try:
            detail = fn()
            print(f"[TEST {test_num:2d}] {name:30s}: PASS {detail or ''}")
            results[f"Test {test_num:2d}: {name}"] = "PASS"
        except Exception as e:
            print(f"[TEST {test_num:2d}] {name:30s}: FAIL ({e})")
            results[f"Test {test_num:2d}: {name}"] = "FAIL"

    # Shared state across sequential tests
    state = {}

    # 1. Dataset Paths
    def t1():
        assert CONTENT_DIR.exists(), f"Content dir missing: {CONTENT_DIR}"
        assert STYLE_DIR.exists(), f"Style dir missing: {STYLE_DIR}"
        assert CHECKPOINT_PATH.exists(), f"Checkpoint missing: {CHECKPOINT_PATH}"
        exts = {".jpg", ".jpeg", ".png", ".webp"}
        state["c_paths"] = [p for p in CONTENT_DIR.rglob("*") if p.suffix.lower() in exts]
        state["s_paths"] = [p for p in STYLE_DIR.rglob("*") if p.suffix.lower() in exts]
        assert len(state["c_paths"]) > 0 and len(state["s_paths"]) > 0
        return f"(Found {len(state['c_paths']):,} content, {len(state['s_paths']):,} style images)"

    # 2. Image Loading
    def t2():
        c_img = Image.open(state["c_paths"][0]).convert("RGB")
        s_img = Image.open(state["s_paths"][0]).convert("RGB")
        assert c_img.mode == "RGB" and s_img.mode == "RGB"
        return f"(Content: {state['c_paths'][0].name}, Style: {state['s_paths'][0].name})"

    # 3. Image Dimensions
    def t3():
        state["c_tensor"] = am.load_and_preprocess_image(state["c_paths"][0], size=256, device="cpu")
        state["s_tensor"] = am.load_and_preprocess_image(state["s_paths"][0], size=256, device="cpu")
        assert state["c_tensor"].shape == (1, 3, 256, 256)
        assert state["s_tensor"].shape == (1, 3, 256, 256)
        return f"(Shape: (1, 3, 256, 256))"

    # 4. VGG19 Features
    def t4():
        state["encoder"] = am.VGG19Encoder().eval()
        with torch.inference_mode():
            f1, f2, f3, f4 = state["encoder"](state["c_tensor"])
        state["c_f4"] = f4
        assert f1.shape == (1, 64, 256, 256)
        assert f2.shape == (1, 128, 128, 128)
        assert f3.shape == (1, 256, 64, 64)
        assert f4.shape == (1, 512, 32, 32)
        return "(Layers: 64, 128, 256, 512 channels)"

    # 5. AdaIN Dimensions
    def t5():
        with torch.inference_mode():
            state["s_f4"] = state["encoder"](state["s_tensor"])[-1]
            state["target_adain"] = am.adaptive_instance_normalization(state["c_f4"], state["s_f4"])
        assert state["target_adain"].shape == (1, 512, 32, 32)
        return f"(Shape: {tuple(state['target_adain'].shape)})"

    # 6. Decoder Dimensions
    def t6():
        state["decoder"] = am.Decoder().eval()
        dummy_in = torch.randn(1, 512, 32, 32)
        with torch.inference_mode():
            dummy_out = state["decoder"](dummy_in)
        assert dummy_out.shape == (1, 3, 256, 256)
        return "(512x32x32 -> 3x256x256)"

    # 7. Checkpoint Loading
    def t7():
        ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
        assert "decoder_state_dict" in ckpt
        state["decoder"].load_state_dict(ckpt["decoder_state_dict"])
        return f"(Loaded {CHECKPOINT_PATH.name}, avg loss: {ckpt.get('average_loss', 0.0):.4f})"

    # 8. CPU Inference
    def t8():
        t0 = time.time()
        with torch.inference_mode():
            state["cpu_out"] = am.style_transfer(
                state["encoder"].to("cpu"), state["decoder"].to("cpu"),
                state["c_tensor"], state["s_tensor"], alpha=1.0
            )
        elapsed = (time.time() - t0) * 1000
        assert state["cpu_out"].shape == (1, 3, 256, 256)
        return f"(Time: {elapsed:.2f} ms)"

    # 9. CUDA Inference
    def t9():
        if not torch.cuda.is_available():
            return "(CUDA not available - skipped)"
        dev = torch.device("cuda")
        t0 = time.time()
        with torch.inference_mode():
            cuda_out = am.style_transfer(
                state["encoder"].to(dev), state["decoder"].to(dev),
                state["c_tensor"].to(dev), state["s_tensor"].to(dev), alpha=1.0
            )
            torch.cuda.synchronize()
        elapsed = (time.time() - t0) * 1000
        assert cuda_out.shape == (1, 3, 256, 256)
        vram_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        return f"(Time: {elapsed:.2f} ms, VRAM Alloc: {vram_mb:.1f} MB)"

    # 10. Output Validity
    def t10():
        out = state["cpu_out"]
        assert not torch.isnan(out).any(), "Output has NaN"
        assert not torch.isinf(out).any(), "Output has Inf"
        return f"(No NaN/Inf, min={out.min():.4f}, max={out.max():.4f})"

    # 11. Image Saving & Style Strength
    def t11():
        alphas = [0.0, 0.5, 1.0]
        outputs = []
        for a in alphas:
            with torch.inference_mode():
                out = am.style_transfer(
                    state["encoder"].to("cpu"), state["decoder"].to("cpu"),
                    state["c_tensor"], state["s_tensor"], alpha=a
                )
                outputs.append(out)
                p = OUTPUT_DIR / f"test_suite_alpha_{a:.1f}.png"
                TF.to_pil_image(out[0].clamp(0, 1).cpu()).save(p)
                assert p.exists() and p.stat().st_size > 0
        diff = (outputs[0] - outputs[2]).abs().mean().item()
        assert diff > 0.001, "Alpha interpolation produced identical output"
        return f"(Saved 3 images, alpha diff={diff:.4f})"

    # Execute all 11 tests
    test(1, "Dataset Paths", t1)
    test(2, "Image Loading", t2)
    test(3, "Image Dimensions", t3)
    test(4, "VGG19 Features", t4)
    test(5, "AdaIN Dimensions", t5)
    test(6, "Decoder Dimensions", t6)
    test(7, "Checkpoint Loading", t7)
    test(8, "CPU Inference", t8)
    test(9, "CUDA Inference", t9)
    test(10, "Output Validity", t10)
    test(11, "Save & Style Strength", t11)

    # Summary
    print("\n" + "=" * 65)
    print("TEST SUITE SUMMARY")
    print("=" * 65)
    passed = sum(1 for v in results.values() if v == "PASS")
    total = len(results)
    for k, v in results.items():
        print(f"  {k:35s}: {v}")
    print(f"\nTotal: {passed}/{total} Tests Passed.")

    # Hardware stats
    ram = psutil.virtual_memory()
    print(f"System RAM: Used {ram.used / 1024**3:.2f} GB / Total {ram.total / 1024**3:.2f} GB ({ram.percent}%)")
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU VRAM:   Allocated {torch.cuda.memory_allocated() / 1024**2:.1f} MB / Total {vram:.2f} GB")
    print("=" * 65)


if __name__ == "__main__":
    run_all_tests()
