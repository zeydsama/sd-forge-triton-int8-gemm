# SD Forge Triton INT8 GEMM

High-speed fused INT8 matrix multiplication (GEMM) extension for **SD WebUI Forge Neo** using OpenAI Triton.

## Features

- **True INT8 Tensor Core Execution**: Eliminates the intermediate FP16 dequantization bottleneck for `int8_tensorwise` Comfy-Kitchen/Forge models.
- **Autotuned Triton Kernels**: Fused row-wise activation quantization + GEMM execution achieving ~2.8x speedup over standard 16-bit linear layers on Ampere/Ada/Blackwell GPUs.
- **Hadamard / ConvRot Support**: Outlier channel suppression via group-wise normalized Hadamard rotation.
- **`torch.compile` Tracing Compatibility**: Registers custom fake tensor schemas (`ck::convrot_w4a4_linear`, `ck::dequantize_convrot_w4a4_weight`) to prevent PyTorch Inductor / Dynamo graph breaks.
- **Resilient Runtime Interception & Telemetry**: Unpacks raw INT8 storage across Forge dynamic memory management (`parameters_manual_cast`), guards against double-wrapping with idempotent hooks, and provides runtime diagnostic telemetry.
- **WebUI Accordion Toggle**: Quick toggle in txt2img/img2img with optional per-row weight scaling mode and live runtime status.

## Verification & Tests

Run the test suite inside your Forge virtual environment:

```bash
python -m unittest discover tests
```

## Requirements

- SD WebUI Forge / Forge Neo
- PyTorch >= 2.1
- Triton (installed in the Forge Python environment)
- NVIDIA GPU (RTX 30xx, 40xx, 50xx series recommended)

## Installation

Clone this repository into the `extensions` directory of your SD WebUI Forge installation:

```bash
cd extensions
git clone https://github.com/zeydsama/sd-forge-triton-int8-gemm.git
```

## License

MIT License
