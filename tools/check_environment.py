"""Fail loudly rather than counting skipped GPU tests as GPU validation."""
import warp as wp
wp.init()
if not wp.is_cuda_available():
    raise SystemExit("CUDA unavailable: on CUDA 12 drivers install the official Warp +cu12 wheel (README).")
print("GPU validation enabled:", wp.get_device("cuda:0"))
