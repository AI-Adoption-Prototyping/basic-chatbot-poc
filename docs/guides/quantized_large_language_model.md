# Quantized Large Language Model for CPU Inference

## Overview

This project uses **GGUF** (Generalized GGML Unified Format) quantized models for efficient CPU-based inference. The primary model is **Mistral-7B-Instruct-v0.2-GGUF**, which provides high-quality responses while running efficiently on modest hardware without requiring GPUs.

## What is GGUF?

**GGUF** (Generalized GGML Unified Format) is a binary format designed for efficient inference of large language models on CPU-based systems. It's the successor to GGML and provides:

- **Multiple Quantization Levels**: From Q2 (very compressed) to Q8 (high quality)
- **CPU Optimization**: Designed specifically for CPU inference
- **Cross-Platform Support**: Works on x86, ARM, and Apple Silicon
- **Efficient Memory Usage**: Significantly reduces RAM requirements

## Why Quantization?

Quantization reduces model size and memory footprint by storing weights in lower precision formats. This is essential for CPU-based deployments.

### Benefits

- **Lower RAM Usage**: Enables deployment on commodity hardware

  - Q4 quantization: ~4-5GB RAM (vs. ~14GB for FP16)
  - Q8 quantization: ~8-10GB RAM (vs. ~14GB for FP16)

- **Faster Inference**: Especially important for CPU-only environments

  - Reduced memory bandwidth requirements
  - Faster computation with lower precision

- **Minimal Accuracy Loss**: Q4_K_M or Q5_K_M often retain near-original performance

  - Most tasks show <5% quality degradation
  - Often imperceptible in practice

- **Smaller Download Size**: Faster model downloads and updates
  - Q4: ~4.5GB (vs. ~14GB for FP16)
  - Q8: ~8GB (vs. ~14GB for FP16)

### Trade-offs

- **Slight Quality Degradation**: Very low quantization (Q2/Q3) may reduce reasoning quality
- **Memory vs. Quality**: Higher quantization (Q8) improves accuracy but increases memory needs
- **Inference Speed**: Higher quantization may be slightly slower due to larger model size

## Quantization Levels

### Q2 (2-bit)

- **Size**: ~2GB
- **RAM**: ~3GB
- **Quality**: Lower, may struggle with complex reasoning
- **Use Case**: Very resource-constrained environments

### Q4 (4-bit)

- **Size**: ~4.5GB
- **RAM**: ~4-5GB
- **Quality**: Good balance, recommended for most use cases
- **Use Case**: General purpose, this project's default

### Q5 (5-bit)

- **Size**: ~5.5GB
- **RAM**: ~6GB
- **Quality**: Very good, slight improvement over Q4
- **Use Case**: When you need better quality and have more RAM

### Q8 (8-bit)

- **Size**: ~8GB
- **RAM**: ~8-10GB
- **Quality**: Near-original, minimal degradation
- **Use Case**: When quality is critical and resources allow

## Mistral-7B-Instruct Model

### Model Details

- **Base Model**: Mistral 7B Instruct v0.2
- **Parameters**: 7 billion
- **Context Window**: ~32K tokens
- **Format**: GGUF quantized
- **Recommended Quantization**: Q4_K_M

### Why Mistral-7B?

- **High Quality**: Excellent instruction-following and reasoning
- **Efficient**: 7B parameters provide good quality-to-size ratio
- **Instruct-Tuned**: Optimized for following instructions and answering questions
- **Well-Supported**: Extensive community support and resources

### Performance Characteristics

- **Response Quality**: High-quality, coherent responses
- **Speed**: Moderate inference speed on CPU
- **Memory**: ~4-5GB RAM with Q4 quantization
- **Context Handling**: Good at maintaining context in conversations

## Configuration

### Environment Variables

```env
MODEL_TYPE=gguf
MODEL_REPO_ID=TheBloke/Mistral-7B-Instruct-v0.2-GGUF
MODEL_FILENAME=mistral-7b-instruct-v0.2.Q4_K_M.gguf
MODEL_N_CTX=4096
MODEL_N_THREADS=None
MODEL_N_GPU_LAYERS=1
```

### Configuration Options

- **MODEL_N_CTX**: Context window size (default: 4096)

  - Larger values = more context but more memory
  - Adjust based on your use case

- **MODEL_N_THREADS**: Number of CPU threads (default: None = auto-detect)

  - Set to specific number for fine-tuning
  - None uses all available cores

- **MODEL_N_GPU_LAYERS**: GPU layers for acceleration (default: 1 for M1/M2 Mac)
  - 0 = CPU only
  - 1+ = Use GPU/Metal acceleration (Apple Silicon)
  - Higher values = more GPU usage, faster inference

## Using Different Quantization Levels

### Switching to Q8 for Higher Quality

```bash
export MODEL_FILENAME="mistral-7b-instruct-v0.2.Q8_0.gguf"
uvicorn main:app --reload
```

**Pros**: Better quality, near-original performance  
**Cons**: Larger download, more RAM required

### Switching to Q2 for Lower Resources

```bash
export MODEL_FILENAME="mistral-7b-instruct-v0.2.Q2_K.gguf"
uvicorn main:app --reload
```

**Pros**: Very small, low RAM  
**Cons**: Lower quality, may struggle with complex queries

## Inference Engine: llama.cpp

The project uses **llama.cpp** through the `llama-cpp-python` library for inference.

### Why llama.cpp?

- **CPU Optimized**: Designed specifically for CPU inference
- **Efficient**: Highly optimized C++ implementation
- **Cross-Platform**: Works on x86, ARM, Apple Silicon
- **Well-Maintained**: Active development and community support

### Integration

```python
from llama_cpp import Llama

model = Llama(
    model_path=model_path,
    n_ctx=4096,
    n_threads=None,  # Auto-detect
    n_gpu_layers=1,  # Metal acceleration on Apple Silicon
    verbose=False,
)
```

## Apple Silicon Optimization

### Metal Acceleration

On Apple Silicon (M1/M2/M3 Macs), you can use Metal acceleration:

```python
n_gpu_layers=1  # Use Metal acceleration
```

This offloads some computation to the GPU, improving inference speed.

### Performance Tips

- **n_gpu_layers**: Start with 1, increase if you have more GPU memory
- **n_threads**: Leave as None for auto-detection
- **n_ctx**: Adjust based on your needs (larger = more memory)

## Model Loading

### First Run

On first run, the model is downloaded from Hugging Face:

```python
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
    filename="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
)
```

The model is cached locally for subsequent runs.

### Loading Time

- **First Load**: ~10-30 seconds (depends on hardware)
- **Subsequent Loads**: Faster due to caching
- **Memory Allocation**: Happens during load

## Best Practices

### Memory Management

- Monitor RAM usage, especially with larger context windows
- Use appropriate quantization level for your hardware
- Consider reducing `n_ctx` if memory is constrained

### Performance Optimization

- Use Metal acceleration on Apple Silicon
- Adjust `n_threads` based on your CPU
- Consider batch processing for multiple queries

### Quality vs. Speed

- Q4_K_M provides good balance for most use cases
- Use Q8 for critical applications requiring highest quality
- Use Q2 only if resources are severely constrained

## Troubleshooting

### Out of Memory Errors

If you encounter memory issues:

1. Reduce `MODEL_N_CTX` (context window size)
2. Use lower quantization (Q4 instead of Q8)
3. Close other applications
4. Consider using a smaller model

### Slow Inference

If inference is too slow:

1. Enable Metal acceleration (Apple Silicon)
2. Increase `n_gpu_layers` if using GPU
3. Check CPU usage and available cores
4. Consider using a smaller model or lower quantization

### Model Download Issues

If model download fails:

1. Check internet connection
2. Verify Hugging Face access
3. Check available disk space
4. Try downloading manually from Hugging Face

## Alternatives

### Other GGUF Models

- **Llama 2/3**: Alternative 7B models
- **Phi-2**: Smaller, faster model
- **Mistral-NeMo-Minitron-2B**: Smaller model with larger context

### Other Formats

- **Ollama**: Alternative inference engine with model management
- **Transformers**: Full precision models (requires more resources)
- **ONNX**: Optimized format for various runtimes

See the [LLM Model Selection](llm_model_selection.md) guide for detailed comparisons.

## Resources

- [GGUF Format Specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [llama.cpp GitHub](https://github.com/ggerganov/llama.cpp)
- [TheBloke Models on Hugging Face](https://huggingface.co/TheBloke)
- [Mistral-7B-Instruct-GGUF](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF)
