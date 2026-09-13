import sys
import os
import unittest
import torch

# Ensure repository and forge neo are on path
ext_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
forge_dir = os.path.dirname(os.path.dirname(ext_dir))
if ext_dir not in sys.path:
    sys.path.insert(0, ext_dir)
if forge_dir not in sys.path:
    sys.path.insert(0, forge_dir)

import backend.operations
import backend.operations_mixed_precision as omp
from backend.quant_ops import QuantizedTensor, TensorWiseINT8Layout
from triton_gemm import (
    apply_triton_gemm_patch,
    remove_triton_gemm_patch,
    get_triton_gemm_diagnostics,
    reset_triton_gemm_diagnostics,
)


class TestTritonPatcher(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        apply_triton_gemm_patch()

    @classmethod
    def tearDownClass(cls):
        remove_triton_gemm_patch()

    def setUp(self):
        reset_triton_gemm_diagnostics()

    def test_patch_idempotency(self):
        """Verify that calling apply_triton_gemm_patch multiple times does not corrupt class methods."""
        apply_triton_gemm_patch()
        apply_triton_gemm_patch()
        diag = get_triton_gemm_diagnostics()
        self.assertTrue(diag["patched"])

    def test_cuda_fused_int8_linear_execution(self):
        """Verify true Triton INT8 GEMM dispatch under both parameters_manual_cast states."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available for Triton tests")

        ops = omp.mixed_precision_ops()
        layer = ops.Linear(128, 256, bias=True).to("cuda", dtype=torch.bfloat16)
        layer.layout_type = TensorWiseINT8Layout
        layer.quant_format = "int8_tensorwise"
        layer._full_precision_mm = False
        layer.weight_function = []
        layer.bias_function = []

        w = torch.randn(256, 128, dtype=torch.bfloat16, device="cuda")
        layer.weight = QuantizedTensor.from_float(w, "TensorWiseINT8Layout")
        layer.bias = torch.nn.Parameter(torch.randn(256, dtype=torch.bfloat16, device="cuda"))

        x = torch.randn(2, 64, 128, dtype=torch.bfloat16, device="cuda")

        # Test Case A: parameters_manual_cast = False
        layer.parameters_manual_cast = False
        reset_triton_gemm_diagnostics()
        out1 = layer(x)
        diag1 = get_triton_gemm_diagnostics()
        self.assertEqual(out1.shape, (2, 64, 256))
        self.assertEqual(diag1["triton_calls"], 1)
        self.assertEqual(diag1["fallback_calls"], 0)

        # Test Case B: parameters_manual_cast = True (Forge GPU memory manager default)
        layer.parameters_manual_cast = True
        reset_triton_gemm_diagnostics()
        out2 = layer(x)
        diag2 = get_triton_gemm_diagnostics()
        self.assertEqual(out2.shape, (2, 64, 256))
        self.assertEqual(diag2["triton_calls"], 1)
        self.assertEqual(diag2["fallback_calls"], 0)

    def test_disabled_toggle(self):
        """Verify that creating .disabled cleanly forces fallback to baseline without crashing."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available for Triton tests")

        ops = omp.mixed_precision_ops()
        layer = ops.Linear(128, 256, bias=True).to("cuda", dtype=torch.bfloat16)
        layer.layout_type = TensorWiseINT8Layout
        layer.quant_format = "int8_tensorwise"
        w = torch.randn(256, 128, dtype=torch.bfloat16, device="cuda")
        layer.weight = QuantizedTensor.from_float(w, "TensorWiseINT8Layout")
        layer.bias = torch.nn.Parameter(torch.randn(256, dtype=torch.bfloat16, device="cuda"))
        x = torch.randn(2, 64, 128, dtype=torch.bfloat16, device="cuda")

        flag_path = os.path.join(ext_dir, ".disabled")
        try:
            with open(flag_path, "w") as f:
                f.write("1")
            reset_triton_gemm_diagnostics()
            out = layer(x)
            diag = get_triton_gemm_diagnostics()
            self.assertEqual(out.shape, (2, 64, 256))
            self.assertEqual(diag["triton_calls"], 0)
            self.assertEqual(diag["fallback_calls"], 1)
            self.assertEqual(diag["fallback_reasons"].get("disabled_by_user"), 1)
        finally:
            if os.path.exists(flag_path):
                os.remove(flag_path)

    def test_clean_unpatch(self):
        """Verify that remove_triton_gemm_patch cleanly restores original operations."""
        remove_triton_gemm_patch()
        diag = get_triton_gemm_diagnostics()
        self.assertFalse(diag["patched"])
        # Re-apply for tearDownClass
        apply_triton_gemm_patch()


if __name__ == "__main__":
    unittest.main()
