"""Phase 1 smoke test: confirm ROCm PyTorch sees the MI300X.

Writes evidence/env_torch_check.txt with a verdict line at the top so the
verify-gate has a real artifact to consult.
"""
import os
import sys
import textwrap

EVIDENCE = "evidence/env_torch_check.txt"


def main() -> int:
    lines = []
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        device_count = torch.cuda.device_count() if cuda_available else 0
        lines.append(f"verdict: {'PASS' if cuda_available and device_count >= 1 else 'FAIL'}")
        lines.append(f"torch.__version__: {torch.__version__}")
        lines.append(f"torch.version.hip: {getattr(torch.version, 'hip', None)}")
        lines.append(f"torch.cuda.is_available(): {cuda_available}")
        lines.append(f"torch.cuda.device_count(): {device_count}")
        if cuda_available and device_count >= 1:
            lines.append(f"torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}")
            lines.append(f"torch.cuda.get_device_properties(0).total_memory: {torch.cuda.get_device_properties(0).total_memory}")
            x = torch.randn(1024, 1024, device="cuda")
            y = x @ x.T
            lines.append(f"sample matmul on GPU OK: shape={tuple(y.shape)}, dtype={y.dtype}, device={y.device}")
    except Exception as e:
        lines.insert(0, "verdict: FAIL")
        lines.append(f"exception: {type(e).__name__}: {e}")
    os.makedirs(os.path.dirname(EVIDENCE), exist_ok=True)
    with open(EVIDENCE, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if lines[0].endswith("PASS") else 1


if __name__ == "__main__":
    sys.exit(main())
