import sys
import os

ext_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ext_dir not in sys.path:
    sys.path.insert(0, ext_dir)

import gradio as gr
import modules.scripts as scripts
from triton_gemm.patcher import (
    apply_triton_gemm_patch,
    set_triton_gemm_enabled,
    get_triton_gemm_diagnostics,
    _STATE,
)

# Apply the runtime hook on extension discovery
apply_triton_gemm_patch()


class TritonInt8GemmScript(scripts.Script):
    sorting_priority = 2024

    def title(self):
        return "Triton INT8 Fused GEMM"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion(self.title(), open=False, elem_id="triton-int8-gemm-panel"):
            enable = gr.Checkbox(
                label="Enable Fused Triton INT8 Linear (ConvRot)",
                value=True,
                elem_id="triton-int8-gemm-enable",
                info="Executes true INT8 TensorCore GEMM without FP16 dequantization for INT8/ConvRot models",
            )
            per_row = gr.Checkbox(
                label="Per-Row Scale Mode",
                value=False,
                elem_id="triton-int8-gemm-per-row",
                info="Use per-row weight scaling (for specialized per-row INT8 models)",
            )
            gr.Markdown(
                "**Fused Triton INT8 GEMM**: Eliminates the intermediate FP16 dequantization bottleneck for `int8_tensorwise` Comfy-Kitchen/Forge models with autotuned Tensor Core execution."
            )

            # Telemetry status display
            patch_status = "Active (Hooked into mixed_precision_ops)" if _STATE["patched"] else "Not Active"
            gr.Markdown(
                f"*Runtime Hook Status*: `{patch_status}` | *Per-Row Scaling*: `{'Enabled' if _STATE['per_row_quant'] else 'Disabled'}`",
                elem_id="triton-int8-gemm-telemetry",
            )

        return [enable, per_row]

    def process(self, p, enable: bool = True, per_row: bool = False, *args, **kwargs):
        set_triton_gemm_enabled(enable)
        _STATE["per_row_quant"] = per_row
