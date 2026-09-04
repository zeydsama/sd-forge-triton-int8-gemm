import sys
import os

ext_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ext_dir not in sys.path:
    sys.path.insert(0, ext_dir)

import gradio as gr
import modules.scripts as scripts
from triton_gemm.patcher import apply_triton_gemm_patch, set_triton_gemm_enabled, _STATE

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
                info="Executes true INT8 TensorCore GEMM without FP16 dequantization for INT8/ConvRot models"
            )
            per_row = gr.Checkbox(
                label="Per-Row Scale Mode",
                value=False,
                elem_id="triton-int8-gemm-per-row",
                info="Use per-row weight scaling (for specialized per-row INT8 models)"
            )
            gr.Markdown(
                "**Fused Triton INT8 GEMM**: Restores high-speed INT8 matrix multiplication from commit `dac2375f` with zero on-the-fly dequantization overhead."
            )
        return [enable, per_row]

    def process(self, p, enable: bool = True, per_row: bool = False, *args, **kwargs):
        set_triton_gemm_enabled(enable)
        _STATE["per_row_quant"] = per_row
