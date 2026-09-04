import logging
import torch

from .operations_triton import triton_int8_linear, triton_int8_linear_per_row
from .quant_rotation import build_hadamard, rotate_activation

logger = logging.getLogger("triton_int8_gemm")

_STATE = {
    "enabled": True,
    "patched": False,
    "orig_mixed_precision_ops": None,
    "per_row_quant": False,
}


def is_triton_gemm_enabled() -> bool:
    return _STATE["enabled"]


def set_triton_gemm_enabled(enabled: bool):
    _STATE["enabled"] = enabled


def _register_custom_ops_for_torch_compile():
    """Register opaque custom ops with fake tensor handlers so TorchDynamo/torch.compile
    can trace custom C++/CUDA kernels (e.g. comfy_kitchen ConvRot W4A4) without graph breaks.
    """
    try:
        import comfy_kitchen.tensor.convrot_w4a4 as ckw4
        from comfy_kitchen.tensor.convrot_w4a4 import (
            TensorCoreConvRotW4A4Layout,
            QuantizedTensor,
            convrot_w4a4_linear,
            dequantize_convrot_w4a4_weight,
        )

        if not hasattr(torch.ops, "ck") or not hasattr(torch.ops.ck, "convrot_w4a4_linear"):
            @torch.library.custom_op("ck::convrot_w4a4_linear", mutates_args=())
            def ck_convrot_w4a4_linear(
                x: torch.Tensor,
                qweight: torch.Tensor,
                wscales: torch.Tensor,
                bias: torch.Tensor | None,
                convrot_groupsize: int,
                quant_group_size: int,
                linear_dtype: str,
            ) -> torch.Tensor:
                return convrot_w4a4_linear(
                    x,
                    qweight,
                    wscales,
                    bias=bias,
                    convrot_groupsize=convrot_groupsize,
                    quant_group_size=quant_group_size,
                    linear_dtype=linear_dtype,
                )

            @ck_convrot_w4a4_linear.register_fake
            def _(x, qweight, wscales, bias, convrot_groupsize, quant_group_size, linear_dtype):
                out_features = qweight.shape[0]
                return torch.empty((*x.shape[:-1], out_features), dtype=x.dtype, device=x.device)

        if not hasattr(torch.ops, "ck") or not hasattr(torch.ops.ck, "dequantize_convrot_w4a4_weight"):
            @torch.library.custom_op("ck::dequantize_convrot_w4a4_weight", mutates_args=())
            def ck_dequantize_convrot_w4a4_weight(
                qdata: torch.Tensor,
                scales: torch.Tensor,
                convrot_groupsize: int,
                quant_group_size: int,
                output_dtype: torch.dtype,
            ) -> torch.Tensor:
                return dequantize_convrot_w4a4_weight(
                    qdata,
                    scales,
                    convrot_groupsize=convrot_groupsize,
                    quant_group_size=quant_group_size,
                    output_dtype=output_dtype,
                )

            @ck_dequantize_convrot_w4a4_weight.register_fake
            def _(qdata, scales, convrot_groupsize, quant_group_size, output_dtype):
                k = qdata.shape[-1] * 2
                return torch.empty((qdata.shape[0], k), dtype=output_dtype, device=qdata.device)

        def patched_convrot_w4a4_forward(input_tensor: torch.Tensor, weight: QuantizedTensor, bias: torch.Tensor | None):
            qweight, wscales = TensorCoreConvRotW4A4Layout.get_plain_tensors(weight)
            params = weight._params
            return torch.ops.ck.convrot_w4a4_linear(
                input_tensor,
                qweight,
                wscales,
                bias,
                params.convrot_groupsize,
                params.quant_group_size,
                params.linear_dtype,
            )

        ckw4._convrot_w4a4_forward = patched_convrot_w4a4_forward

        def patched_dequantize(cls, qdata: torch.Tensor, params: TensorCoreConvRotW4A4Layout.Params) -> torch.Tensor:
            return torch.ops.ck.dequantize_convrot_w4a4_weight(
                qdata,
                params.scale,
                params.convrot_groupsize,
                params.quant_group_size,
                params.orig_dtype,
            )

        TensorCoreConvRotW4A4Layout.dequantize = classmethod(patched_dequantize)
        logger.info("sd-forge-triton-int8-gemm: Successfully registered ConvRot W4A4 custom ops for torch.compile.")
    except Exception as e:
        logger.warning(f"sd-forge-triton-int8-gemm: Failed to register ConvRot W4A4 custom ops ({e}).")


def apply_triton_gemm_patch():
    if _STATE["patched"]:
        return

    _register_custom_ops_for_torch_compile()

    try:
        import backend.operations
        import backend.operations_mixed_precision as omp
        from backend.operations import main_stream_worker, weights_manual_cast
        from backend.quant_ops import QuantizedTensor, TensorWiseINT8Layout

        orig_fn = omp.mixed_precision_ops
        _STATE["orig_mixed_precision_ops"] = orig_fn

        def hooked_mixed_precision_ops(*args, **kwargs):
            cls = orig_fn(*args, **kwargs)
            original_forward = cls.Linear.forward

            def triton_fused_linear_forward(self, input, *f_args, **f_kwargs):
                if not _STATE["enabled"]:
                    return original_forward(self, input, *f_args, **f_kwargs)

                _use_quantized = (
                    getattr(self, "layout_type", None) is not None
                    and not isinstance(input, QuantizedTensor)
                    and not getattr(self, "_full_precision_mm", False)
                    and not getattr(self, "forge_force_cast_weights", False)
                    and len(self.weight_function) == 0
                    and len(self.bias_function) == 0
                )

                is_int8_tensorwise = getattr(self, "quant_format", None) == "int8_tensorwise"

                if _use_quantized and is_int8_tensorwise and isinstance(self.weight, QuantizedTensor):
                    try:
                        if self.parameters_manual_cast:
                            weight, bias, signal = weights_manual_cast(self, x=None, dtype=torch.int8, device=input.device, bias_dtype=input.dtype)
                            scale = self.weight.params.scale.to(device=input.device, non_blocking=True)
                        else:
                            weight, bias, signal = self.weight._qdata, self.bias, None
                            scale = self.weight.params.scale.to(device=input.device, non_blocking=True)

                        if getattr(self.weight.params, "convrot", False):
                            group_size = getattr(self.weight.params, "convrot_groupsize", 256)
                            H = build_hadamard(group_size, device=input.device, dtype=input.dtype)
                            input = rotate_activation(input, H, group_size=group_size)

                        compute_dtype = input.dtype if input.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16

                        with main_stream_worker(weight, bias, signal):
                            if getattr(self, "_per_row", False) or _STATE["per_row_quant"]:
                                output = triton_int8_linear_per_row(input, weight, scale, bias, compute_dtype)
                            else:
                                output = triton_int8_linear(input, weight, scale, bias, compute_dtype)

                        return output
                    except Exception as e:
                        logger.debug(f"Triton INT8 GEMM fallback to standard linear: {e}")
                        return original_forward(self, input, *f_args, **f_kwargs)

                return original_forward(self, input, *f_args, **f_kwargs)

            cls.Linear.forward = triton_fused_linear_forward
            return cls

        omp.mixed_precision_ops = hooked_mixed_precision_ops
        backend.operations.mixed_precision_ops = hooked_mixed_precision_ops
        _STATE["patched"] = True
        logger.info("sd-forge-triton-int8-gemm: Successfully hooked backend.operations_mixed_precision.mixed_precision_ops with Fused Triton INT8 GEMM.")

    except Exception as e:
        logger.warning(f"sd-forge-triton-int8-gemm: Patching failed ({e}), using default linear.")
