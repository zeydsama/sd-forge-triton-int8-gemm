import os
import logging
import traceback
import torch

from .operations_triton import triton_int8_linear, triton_int8_linear_per_row
from .quant_rotation import build_hadamard, rotate_activation

logger = logging.getLogger("triton_int8_gemm")
_EXT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_STATE = {
    "enabled": True,
    "patched": False,
    "orig_mixed_precision_ops": None,
    "per_row_quant": False,
    "diagnostics": {
        "total_calls": 0,
        "triton_calls": 0,
        "fallback_calls": 0,
        "fallback_reasons": {},
        "last_error": None,
        "last_traceback": None,
        "first_fallback_logged": False,
    },
}


def is_triton_gemm_enabled() -> bool:
    """Returns True if Triton GEMM is enabled and not suppressed by .disabled flag."""
    if _is_compiling():
        return _STATE.get("enabled", True)
    if os.path.exists(os.path.join(_EXT_DIR, ".disabled")):
        return False
    return _STATE.get("enabled", True)


def set_triton_gemm_enabled(enabled: bool):
    _STATE["enabled"] = enabled


def get_triton_gemm_diagnostics() -> dict:
    """Returns runtime telemetry regarding Triton execution vs fallbacks."""
    diag = dict(_STATE["diagnostics"])
    diag["enabled"] = _STATE["enabled"]
    diag["patched"] = _STATE["patched"]
    diag["per_row_quant"] = _STATE["per_row_quant"]
    return diag


def reset_triton_gemm_diagnostics():
    """Resets the diagnostic counters."""
    _STATE["diagnostics"] = {
        "total_calls": 0,
        "triton_calls": 0,
        "fallback_calls": 0,
        "fallback_reasons": {},
        "last_error": None,
        "last_traceback": None,
        "first_fallback_logged": False,
    }


def _is_compiling() -> bool:
    """Returns True if TorchDynamo / torch.compile is currently tracing or compiling."""
    try:
        return hasattr(torch.compiler, "is_compiling") and torch.compiler.is_compiling()
    except Exception:
        return False


def _record_fallback(reason: str, error: Exception | str | None = None):
    """Records a fallback event in runtime diagnostics and logs the first occurrence prominently."""
    if _is_compiling():
        return
    diag = _STATE["diagnostics"]
    diag["fallback_calls"] += 1
    diag["fallback_reasons"][reason] = diag["fallback_reasons"].get(reason, 0) + 1

    if error:
        diag["last_error"] = str(error)
        if isinstance(error, Exception):
            diag["last_traceback"] = traceback.format_exc()
        else:
            diag["last_traceback"] = str(error)

        if not diag["first_fallback_logged"]:
            diag["first_fallback_logged"] = True
            logger.warning(
                f"[sd-forge-triton-int8-gemm] Runtime fallback triggered: {reason} ({error}). "
                "Subsequent identical fallbacks will be counted in diagnostics."
            )
    else:
        logger.debug(f"[sd-forge-triton-int8-gemm] Expected fallback: {reason}")


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
            params = getattr(weight, "_params", getattr(weight, "params", None))
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
        logger.info("[sd-forge-triton-int8-gemm] Successfully registered ConvRot W4A4 custom ops for torch.compile.")
    except Exception as e:
        logger.warning(f"[sd-forge-triton-int8-gemm] Failed to register ConvRot W4A4 custom ops ({e}).")

    try:
        if not hasattr(torch.ops, "triton_gemm") or not hasattr(torch.ops.triton_gemm, "int8_linear"):
            @torch.library.custom_op("triton_gemm::int8_linear", mutates_args=())
            def op_triton_int8_linear(
                x: torch.Tensor,
                weight: torch.Tensor,
                weight_scale: torch.Tensor,
                bias: torch.Tensor | None,
                compute_dtype: torch.dtype,
            ) -> torch.Tensor:
                return triton_int8_linear(x, weight, weight_scale, bias=bias, compute_dtype=compute_dtype)

            @op_triton_int8_linear.register_fake
            def _(x, weight, weight_scale, bias, compute_dtype):
                out_features = weight.shape[0]
                return torch.empty((*x.shape[:-1], out_features), dtype=compute_dtype, device=x.device)

            def _int8_linear_setup_context(ctx, inputs, output):
                x, weight, weight_scale, bias, compute_dtype = inputs
                ctx.save_for_backward(weight, weight_scale)
                ctx.has_bias = bias is not None

            def _int8_linear_backward(ctx, grad_output):
                weight, weight_scale = ctx.saved_tensors
                w_dequant = weight.to(grad_output.dtype) * weight_scale
                grad_x = torch.matmul(grad_output, w_dequant)
                grad_bias = grad_output.sum(dim=tuple(range(grad_output.ndim - 1))) if ctx.has_bias else None
                return grad_x, None, None, grad_bias, None

            op_triton_int8_linear.register_autograd(_int8_linear_backward, setup_context=_int8_linear_setup_context)

        if not hasattr(torch.ops, "triton_gemm") or not hasattr(torch.ops.triton_gemm, "int8_linear_per_row"):
            @torch.library.custom_op("triton_gemm::int8_linear_per_row", mutates_args=())
            def op_triton_int8_linear_per_row(
                x: torch.Tensor,
                weight: torch.Tensor,
                weight_scale: torch.Tensor,
                bias: torch.Tensor | None,
                compute_dtype: torch.dtype,
            ) -> torch.Tensor:
                return triton_int8_linear_per_row(x, weight, weight_scale, bias=bias, compute_dtype=compute_dtype)

            @op_triton_int8_linear_per_row.register_fake
            def _(x, weight, weight_scale, bias, compute_dtype):
                out_features = weight.shape[0]
                return torch.empty((*x.shape[:-1], out_features), dtype=compute_dtype, device=x.device)

            def _int8_linear_per_row_setup_context(ctx, inputs, output):
                x, weight, weight_scale, bias, compute_dtype = inputs
                ctx.save_for_backward(weight, weight_scale)
                ctx.has_bias = bias is not None

            def _int8_linear_per_row_backward(ctx, grad_output):
                weight, weight_scale = ctx.saved_tensors
                scale_b = weight_scale.view(-1, 1) if weight_scale.ndim == 1 else weight_scale
                w_dequant = weight.to(grad_output.dtype) * scale_b
                grad_x = torch.matmul(grad_output, w_dequant)
                grad_bias = grad_output.sum(dim=tuple(range(grad_output.ndim - 1))) if ctx.has_bias else None
                return grad_x, None, None, grad_bias, None

            op_triton_int8_linear_per_row.register_autograd(_int8_linear_per_row_backward, setup_context=_int8_linear_per_row_setup_context)

        logger.info("[sd-forge-triton-int8-gemm] Successfully registered Triton INT8 custom ops for torch.compile.")
    except Exception as e:
        logger.warning(f"[sd-forge-triton-int8-gemm] Failed to register Triton INT8 custom ops ({e}).")


def apply_triton_gemm_patch():
    """Hooks into Forge's mixed_precision_ops to route int8_tensorwise layers through Fused Triton GEMM."""
    if _STATE["patched"]:
        return

    _register_custom_ops_for_torch_compile()

    try:
        import backend.operations
        import backend.operations_mixed_precision as omp
        from backend.operations import main_stream_worker, weights_manual_cast
        from backend.quant_ops import QuantizedTensor

        orig_fn = omp.mixed_precision_ops
        _STATE["orig_mixed_precision_ops"] = orig_fn

        def hooked_mixed_precision_ops(*args, **kwargs):
            cls = orig_fn(*args, **kwargs)

            # Idempotency guard: never double-wrap an already patched class
            if getattr(cls.Linear.forward, "_is_triton_fused", False):
                return cls

            original_forward = cls.Linear.forward

            def triton_fused_linear_forward(self, input, *f_args, **f_kwargs):
                is_tracing = _is_compiling()
                if not is_tracing:
                    _STATE["diagnostics"]["total_calls"] += 1

                if not is_triton_gemm_enabled():
                    _record_fallback("disabled_by_user")
                    return original_forward(self, input, *f_args, **f_kwargs)

                # Qualification check: verify that this layer is eligible for fused INT8 execution
                if getattr(self, "layout_type", None) is None:
                    _record_fallback("not_quantized_layout")
                    return original_forward(self, input, *f_args, **f_kwargs)

                if isinstance(input, QuantizedTensor):
                    _record_fallback("input_already_quantized")
                    return original_forward(self, input, *f_args, **f_kwargs)

                if getattr(self, "_full_precision_mm", False):
                    _record_fallback("full_precision_mm_forced")
                    return original_forward(self, input, *f_args, **f_kwargs)

                if getattr(self, "forge_force_cast_weights", False):
                    _record_fallback("forge_force_cast_weights")
                    return original_forward(self, input, *f_args, **f_kwargs)

                if len(getattr(self, "weight_function", [])) > 0:
                    _record_fallback("has_weight_function_lora")
                    return original_forward(self, input, *f_args, **f_kwargs)

                if len(getattr(self, "bias_function", [])) > 0:
                    _record_fallback("has_bias_function_lora")
                    return original_forward(self, input, *f_args, **f_kwargs)

                quant_format = getattr(self, "quant_format", None)
                if quant_format != "int8_tensorwise":
                    _record_fallback(f"unsupported_quant_format_{quant_format}")
                    return original_forward(self, input, *f_args, **f_kwargs)

                if not isinstance(self.weight, QuantizedTensor):
                    _record_fallback("weight_not_quantized_tensor")
                    return original_forward(self, input, *f_args, **f_kwargs)

                # Execute Fused Triton INT8 GEMM
                try:
                    # 1. Cast or fetch weights and bias
                    if getattr(self, "parameters_manual_cast", False):
                        weight, bias, signal = weights_manual_cast(
                            self,
                            x=None,
                            dtype=torch.int8,
                            device=input.device,
                            bias_dtype=input.dtype,
                        )
                    else:
                        weight, bias, signal = self.weight, self.bias, None

                    # 2. Extract underlying raw int8 tensor (CRITICAL: prevents dtype mismatch in Triton)
                    raw_weight = getattr(weight, "_qdata", weight)
                    if hasattr(raw_weight, "_qdata"):
                        raw_weight = raw_weight._qdata

                    if not isinstance(raw_weight, torch.Tensor) or raw_weight.dtype != torch.int8:
                        _record_fallback(
                            "raw_weight_not_int8",
                            f"Weight extraction resulted in dtype {getattr(raw_weight, 'dtype', type(raw_weight))}"
                        )
                        return original_forward(self, input, *f_args, **f_kwargs)

                    # 3. Defensive scale extraction from either params or _params
                    params = getattr(self.weight, "params", None) or getattr(self.weight, "_params", None)
                    scale = getattr(params, "scale", None) if params is not None else None
                    if scale is None:
                        scale = getattr(self.weight, "scale", None)

                    if scale is None:
                        _record_fallback("missing_scale_parameter", "No scale found on weight or weight params")
                        return original_forward(self, input, *f_args, **f_kwargs)

                    if isinstance(scale, torch.Tensor):
                        scale = scale.to(device=input.device, non_blocking=True)

                    # 4. Optional Hadamard rotation for ConvRot models
                    if params is not None and getattr(params, "convrot", False):
                        group_size = getattr(params, "convrot_groupsize", 256)
                        H = build_hadamard(group_size, device=input.device, dtype=input.dtype)
                        input = rotate_activation(input, H, group_size=group_size)

                    compute_dtype = input.dtype if input.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16

                    # 5. Kernel execution inside stream context
                    with main_stream_worker(weight, bias, signal):
                        if getattr(self, "_per_row", False) or _STATE["per_row_quant"]:
                            if hasattr(torch.ops, "triton_gemm") and hasattr(torch.ops.triton_gemm, "int8_linear_per_row"):
                                output = torch.ops.triton_gemm.int8_linear_per_row(input, raw_weight, scale, bias, compute_dtype)
                            else:
                                output = triton_int8_linear_per_row(input, raw_weight, scale, bias, compute_dtype)
                        else:
                            if hasattr(torch.ops, "triton_gemm") and hasattr(torch.ops.triton_gemm, "int8_linear"):
                                output = torch.ops.triton_gemm.int8_linear(input, raw_weight, scale, bias, compute_dtype)
                            else:
                                output = triton_int8_linear(input, raw_weight, scale, bias, compute_dtype)

                    if not is_tracing:
                        _STATE["diagnostics"]["triton_calls"] += 1
                    return output

                except Exception as e:
                    _record_fallback("triton_execution_error", e)
                    return original_forward(self, input, *f_args, **f_kwargs)

            triton_fused_linear_forward._is_triton_fused = True
            triton_fused_linear_forward._original_forward = original_forward
            cls.Linear.forward = triton_fused_linear_forward
            return cls

        omp.mixed_precision_ops = hooked_mixed_precision_ops
        backend.operations.mixed_precision_ops = hooked_mixed_precision_ops
        _STATE["patched"] = True
        logger.info("[sd-forge-triton-int8-gemm] Successfully hooked backend.operations_mixed_precision.mixed_precision_ops.")

    except Exception as e:
        logger.warning(f"[sd-forge-triton-int8-gemm] Patching failed ({e}), using default linear.")


def remove_triton_gemm_patch():
    """Cleanly restores original Forge mixed_precision_ops without leaving orphaned hooks."""
    if not _STATE["patched"] or _STATE["orig_mixed_precision_ops"] is None:
        return

    try:
        import backend.operations
        import backend.operations_mixed_precision as omp

        omp.mixed_precision_ops = _STATE["orig_mixed_precision_ops"]
        backend.operations.mixed_precision_ops = _STATE["orig_mixed_precision_ops"]
        _STATE["patched"] = False
        _STATE["orig_mixed_precision_ops"] = None
        logger.info("[sd-forge-triton-int8-gemm] Successfully restored original mixed_precision_ops.")
    except Exception as e:
        logger.warning(f"[sd-forge-triton-int8-gemm] Failed to unpatch ({e}).")
