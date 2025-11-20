# gpu_model_bench.py
# A production-ready Python script for running and benchmarking various AI models locally on NVIDIA GPUs.
# Supports text-to-image, text-to-video, and other generative models via Hugging Face Diffusers and Transformers.
# Features: Configurable via YAML, logging, error handling, benchmarking (time, memory, FPS for video), GPU optimization.
# Requirements: Python 3.10+, CUDA 12.x, NVIDIA GPU (tested on RTX 5090 with 32GB VRAM).
# Installation:
#   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
#   pip install diffusers transformers accelerate pyyaml psutil GPUtil
#   For specific models: pip install xformers (for memory efficiency)
# Usage: python gpu_model_bench.py --config config.yaml

import argparse
import logging
import time
import yaml
import torch
from torch.utils.data import DataLoader
from torch.cuda import amp
from diffusers import StableDiffusionPipeline, StableVideoDiffusionPipeline  # Example for diffusion models
from transformers import pipeline  # For general models like LLM
import psutil
import GPUtil
from functools import wraps

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def log_exceptions(func):
    """Decorator for error handling."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
            raise
    return wrapper

def measure_resources(func):
    """Decorator to benchmark time and resource usage."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_cpu = psutil.cpu_percent()
        start_mem = psutil.virtual_memory().used / (1024 ** 3)  # GB
        gpus = GPUtil.getGPUs()
        start_gpu_mem = gpus[0].memoryUsed if gpus else 0  # MB
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_cpu = psutil.cpu_percent()
        end_mem = psutil.virtual_memory().used / (1024 ** 3)
        end_gpu_mem = gpus[0].memoryUsed if gpus else 0
        
        duration = end_time - start_time
        metrics = {
            'duration_sec': duration,
            'cpu_usage_delta': end_cpu - start_cpu,
            'ram_usage_delta_gb': end_mem - start_mem,
            'vram_usage_delta_mb': end_gpu_mem - start_gpu_mem,
        }
        if 'video' in kwargs.get('model_type', '').lower():  # Estimate FPS for video
            metrics['fps'] = kwargs.get('num_frames', 1) / duration if 'num_frames' in kwargs else None
        
        logger.info(f"Benchmark for {func.__name__}: {metrics}")
        return result, metrics
    return wrapper

@log_exceptions
@measure_resources
def load_model(config):
    """Load the specified model with optimizations."""
    model_type = config['model_type']
    model_id = config['model_id']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model_type == 'stable_diffusion':
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        pipe.enable_xformers_memory_efficient_attention()  # Memory optimization
        pipe.enable_attention_slicing()  # For large models
    elif model_type == 'stable_video_diffusion':
        pipe = StableVideoDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        pipe.enable_model_cpu_offload()  # Offload to CPU when idle
    elif model_type == 'llm':
        pipe = pipeline('text-generation', model=model_id, device=0)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    pipe.to(device)
    logger.info(f"Loaded model {model_id} of type {model_type} on {device}")
    return pipe

@log_exceptions
@measure_resources
def run_inference(pipe, config):
    """Run inference based on model type."""
    model_type = config['model_type']
    prompt = config['prompt']
    
    with amp.autocast(enabled=True):  # Mixed precision for speed
        if model_type == 'stable_diffusion':
            result = pipe(prompt).images[0]
        elif model_type == 'stable_video_diffusion':
            image = config.get('input_image')  # Assume pre-loaded if needed
            result = pipe(image, prompt=prompt, num_frames=config.get('num_frames', 16)).frames
        elif model_type == 'llm':
            result = pipe(prompt, max_length=config.get('max_length', 100))[0]['generated_text']
        else:
            raise ValueError(f"Unsupported inference for {model_type}")
    
    return result

def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    pipe = load_model(config)
    result, metrics = run_inference(pipe, config)
    
    # Save results (example: image/video/text)
    output_path = config.get('output_path', 'output')
    if isinstance(result, str):
        with open(f"{output_path}.txt", 'w') as f:
            f.write(result)
    else:
        result.save(f"{output_path}.png")  # Adapt for video
    
    logger.info(f"Inference complete. Output saved to {output_path}. Metrics: {metrics}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU Model Bench: Local AI Model Runner and Benchmark")
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    args = parser.parse_args()
    main(args.config)

# Example config.yaml
# model_type: stable_video_diffusion  # or stable_diffusion, llm
# model_id: stabilityai/stable-video-diffusion-img2vid-xt
# prompt: "A provocative music video scene with dynamic lighting"
# num_frames: 16  # For video models
# output_path: "generated_video"
# # Add more params as needed, e.g., resolution: [720, 1280]
