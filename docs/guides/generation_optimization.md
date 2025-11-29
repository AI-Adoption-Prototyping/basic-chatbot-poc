# Generation Optimization Guide

## Overview

This guide provides optimization strategies for improving LLM generation speed in the RAG chatbot. Generation is typically the slowest part of the pipeline, so these optimizations can significantly improve response times.

## Quick Wins (Immediate Impact)

### 1. Increase GPU Layers (Apple Silicon)

**Impact**: High - Can improve speed by 2-4x on M1/M2/M3 Macs

**Current**: `MODEL_N_GPU_LAYERS=1` (only 1 layer on GPU)

**Optimization**: Increase to use more GPU layers

```env
MODEL_N_GPU_LAYERS=20  # Start with 20, increase if you have more GPU memory
```

**How to find optimal value**:
- Start with 20 layers
- Monitor GPU memory usage
- Increase gradually until you hit memory limits
- Typical range: 20-35 layers for M1/M2, 30-40+ for M3

**Trade-off**: Uses more GPU memory, but much faster inference

### 2. Reduce Max Tokens

**Impact**: High - Linear improvement (fewer tokens = faster)

**Current**: `RAG_MAX_TOKENS=200`

**Optimization**: Reduce if responses are longer than needed

```env
RAG_MAX_TOKENS=150  # Or even 100 for shorter responses
```

**Trade-off**: Shorter responses, but significantly faster

### 3. Optimize Temperature

**Impact**: Medium - Lower temperature = faster sampling

**Current**: `RAG_TEMPERATURE=0.7`

**Optimization**: Lower temperature for faster, more deterministic generation

```env
RAG_TEMPERATURE=0.5  # Faster, more focused responses
# Or even 0.3 for very fast, deterministic answers
```

**Trade-off**: Less creative/diverse responses, but faster generation

### 4. Reduce Context Window

**Impact**: Medium - Smaller context = faster processing

**Current**: `MODEL_N_CTX=4096`

**Optimization**: Reduce if you don't need the full context window

```env
MODEL_N_CTX=2048  # Or 1024 if context is typically small
```

**Trade-off**: Less context capacity, but faster processing

## Advanced Optimizations

### 5. Optimize Sampling Parameters

Add these to the generation call for faster sampling:

```python
# In models/gguf_model.py generate() method
output = self._model(
    formatted_prompt,
    max_tokens=max_tokens,
    temperature=temperature,
    top_p=0.5,  # Reduce from 0.9 for faster sampling
    top_k=40,   # Limit candidate tokens (faster)
    repeat_penalty=1.1,
    stop=stop,
    echo=False,
    # Add these for speed:
    tfs_z=0.5,      # Tail-free sampling (faster)
    typical_p=0.95, # Typical sampling (faster)
)
```

### 6. Reduce Top-K Retrieval

**Impact**: Medium - Less context to process

**Current**: `RAG_TOP_K=5`

**Optimization**: Reduce if 3 documents provide sufficient context

```env
RAG_TOP_K=3  # Faster, but may miss relevant context
```

**Trade-off**: Less context, but faster generation

### 7. Optimize Prompt Length

**Impact**: Medium - Shorter prompts = faster processing

**Current**: Full context included in prompt

**Optimization**: 
- Limit context to most relevant parts
- Summarize context if very long
- Use more concise formatting

### 8. Use Smaller Model Variant

**Impact**: Very High - Smaller model = much faster

**Options**:
- **Q2_K**: Very fast, lower quality (~2GB)
- **Q3_K_M**: Fast, acceptable quality (~3GB)
- **Q4_K_S**: Faster than Q4_K_M, similar quality

```env
MODEL_FILENAME=mistral-7b-instruct-v0.2.Q3_K_M.gguf
```

**Trade-off**: Lower quality, but 2-3x faster

## Model Configuration Optimizations

### 9. Explicit Thread Configuration

**Impact**: Low-Medium - Better CPU utilization

**Current**: `MODEL_N_THREADS=None` (auto-detect)

**Optimization**: Set explicitly to match CPU cores

```env
MODEL_N_THREADS=8  # Match your CPU core count
```

**How to find**: `sysctl -n hw.ncpu` on macOS

### 10. Batch Size Optimization

If processing multiple requests, consider batching (requires code changes).

## Performance Monitoring

Use the timing logs to identify bottlenecks:

```
REQUEST TIMING - Query: '...'
  ┌─ RAG Pipeline:         2.123s
  │  ├─ Embedding:        0.045s
  │  ├─ Retrieval:        0.234s
  │  └─ Generation:       1.844s  ← Focus here
```

If generation is >1s, apply the optimizations above.

## Recommended Configuration for Speed

For maximum speed (balanced with quality):

```env
# Model
MODEL_N_GPU_LAYERS=25        # Increase GPU usage
MODEL_N_CTX=2048            # Reduce context window
MODEL_N_THREADS=8           # Explicit thread count

# RAG
RAG_MAX_TOKENS=150          # Shorter responses
RAG_TEMPERATURE=0.5         # Faster sampling
RAG_TOP_K=3                 # Less context
```

## Expected Performance Improvements

| Optimization | Speed Improvement | Quality Impact |
|-------------|------------------|----------------|
| GPU Layers (1→25) | 2-4x faster | None |
| Max Tokens (200→150) | ~25% faster | Slightly shorter |
| Temperature (0.7→0.5) | ~10-15% faster | Less creative |
| Context (4096→2048) | ~10-20% faster | Less capacity |
| Top-K (5→3) | ~5-10% faster | Less context |
| Model (Q4→Q3) | 2-3x faster | Lower quality |

## Testing Optimizations

1. **Baseline**: Run a few queries and note generation time
2. **Apply one optimization**: Change one setting at a time
3. **Measure**: Compare generation time in logs
4. **Iterate**: Keep what works, revert what doesn't

## When to Optimize

- Generation time > 2 seconds → Apply optimizations
- Generation time > 1 second → Consider optimizations
- Generation time < 0.5 seconds → Already well optimized

## Trade-offs Summary

**Speed vs Quality**:
- Lower quantization (Q2/Q3) = faster but lower quality
- Lower temperature = faster but less creative
- Fewer tokens = faster but shorter responses

**Speed vs Context**:
- Smaller context window = faster but less capacity
- Fewer retrieved documents = faster but less context

**Speed vs Resources**:
- More GPU layers = faster but more memory
- More threads = faster but more CPU usage

Choose optimizations based on your priorities: speed, quality, or resource usage.

