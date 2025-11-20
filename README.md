# GPU Model Bench

This repository provides a production-ready Python script for running and benchmarking various AI models locally on NVIDIA GPUs. It supports generative models like text-to-image and text-to-video using Hugging Face libraries.

## Features
- **Model Support**: Stable Diffusion for images, Stable Video Diffusion for videos, and LLMs for text generation.
- **Optimizations**: Mixed precision, memory-efficient attention, attention slicing, and model offloading for efficient GPU usage.
- **Benchmarking**: Measures execution time, CPU/RAM delta, VRAM usage, and FPS (for videos).
- **Configuration**: YAML-based config for easy customization.
- **Error Handling**: Logging and exception decorators for robustness.
- ~**Tested On**: NVIDIA GeForce RTX 5090 with 32GB VRAM (compatible with other CUDA-capable GPUs).~

## Requirements
Install dependencies using the provided `requirements.txt`:

```
torch
torchvision
torchaudio
diffusers
transformers
accelerate
pyyaml
psutil
GPUtil
xformers
```

### Installation Instructions
1. Ensure CUDA 12.x is installed and compatible with your GPU.
2. Install PyTorch with CUDA support:
   ```
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```
3. Install remaining packages:
   ```
   pip install -r requirements.txt
   ```

If using a different CUDA version, adjust the PyTorch index URL accordingly.

## Usage
Run the script with a YAML configuration file:

```
python gpu_model_bench.py --config config.yaml
```

### Example `config.yaml`
```yaml
model_type: stable_video_diffusion  # Options: stable_diffusion, stable_video_diffusion, llm
model_id: stabilityai/stable-video-diffusion-img2vid-xt
prompt: "A provocative music video scene with dynamic lighting"
num_frames: 16  # For video models
output_path: "generated_video"  # Output file path (e.g., .png for images, adapt for video/text)
# Additional params: resolution: [720, 1280], max_length: 100 (for LLMs), input_image: "path/to/image.jpg" (for img2vid)
```

- **Output**: Generated content saved to the specified path. Benchmarks logged to console.

## Script Overview (`gpu_model_bench.py`)
The script includes:
- Decorators for logging exceptions and measuring resources.
- Model loading with GPU optimizations.
- Inference execution with mixed precision.
- Support for custom parameters via YAML.

For full script details, see `gpu_model_bench.py`.

## Benchmarking
Each run logs metrics like:
- Duration (seconds)
- CPU usage delta (%)
- RAM usage delta (GB)
- VRAM usage delta (MB)
- FPS (for video models)

Example log:
```
2023-10-01 12:00:00 - INFO - Benchmark for run_inference: {'duration_sec': 5.2, 'cpu_usage_delta': 10.5, 'ram_usage_delta_gb': 0.8, 'vram_usage_delta_mb': 2048, 'fps': 3.07}
```

## Extending the Script
- Add new model types by extending `load_model` and `run_inference`.
- For video outputs, integrate libraries like OpenCV for saving (e.g., `cv2.VideoWriter`).
- Ensure models are downloaded from Hugging Face (requires login for some gated models).

## Limitations
- Requires NVIDIA GPU with sufficient VRAM (e.g., 32GB for high-res videos).
- No internet access during inference; models must be pre-downloaded.
- Adapt for non-diffusion models as needed.

## License
MIT License. See `LICENSE` for details.
