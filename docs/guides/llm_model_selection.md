# LLM Model Selection Guide

## Overview

This document compares two strong candidates for CPU-based AI deployments in environments where cost, control, and data privacy matter. Both models are quantized GGUF formats optimized for efficient inference without requiring GPUs.

## Model Comparison

| Feature              | **Mistral-7B-Instruct (GGUF)**                       | **Mistral-NeMo-Minitron-2B-128K-Instruct** |
| -------------------- | ---------------------------------------------------- | ------------------------------------------ |
| **Parameters**       | 7B                                                   | 2B                                         |
| **Context Window**   | ~32K tokens                                          | **128K tokens**                            |
| **Performance**      | High-quality reasoning, better instruction-following | Lightweight, faster, less nuanced          |
| **Hardware Needs**   | ~8–10 GB RAM (Q4 quantization)                       | ~3–4 GB RAM                                |
| **Latency on CPU**   | Moderate                                             | Very fast                                  |
| **Best Use Case**    | Complex chatbots, multi-turn conversations           | Large-context RAG, document ingestion      |
| **Integration**      | Ollama, llama.cpp, FastAPI                           | Same                                       |
| **Function Calling** | Supported                                            | Supported                                  |

### Summary Recommendation

- **Phase One (Chatbots, IT Help, HR FAQs)** → **Mistral-7B-Instruct-GGUF**

  - Better instruction-following and conversational depth
  - More robust reasoning capabilities
  - Ideal for interactive, multi-turn conversations

- **Future Phase (Student Success Insights, Large Document RAG)** → **Minitron-2B-128K**
  - Handles **128K context window**, perfect for ingesting large documents
  - Suitable for Workday + SIS + Canvas data ingestion
  - Lower memory footprint enables deployment on more constrained hardware

## Impact of Quantization

Quantization reduces model size and memory footprint by storing weights in lower precision formats (e.g., Q4, Q5, Q8). This is essential for CPU-based deployments.

### Benefits

- **Lower RAM usage** → Enables deployment on commodity hardware without GPUs
- **Faster inference** → Especially important for CPU-only environments
- **Minimal accuracy loss** → Q4_K_M or Q5_K_M often retain near-original performance
- **Smaller download size** → Faster model downloads and updates

### Trade-offs

- **Slight degradation in reasoning** → Very low quantization (Q2/Q3) may reduce quality
- **Memory vs. Quality** → Larger quantization (Q8) improves accuracy but increases memory needs
- **Inference speed** → Higher quantization may be slightly slower

### Quantization Recommendations

- **For Mistral-7B-Instruct**: Q4_K_M is the sweet spot for CPU deployment, balancing quality and performance
- **For Minitron-2B**: Even Q4 quantization makes it extremely lightweight while maintaining good performance

## Ability to Augment

Both models can be augmented with various techniques to enhance their capabilities:

### RAG (Retrieval-Augmented Generation)

- Combine with vector databases (e.g., Chroma, Weaviate) for contextual answers
- Enables the model to access up-to-date information not in its training data
- Critical for domain-specific applications

### Function Calling

- Structured outputs for APIs (JSON responses)
- Enables integration with external systems
- Useful for building agentic applications

### Fine-Tuning / LoRA

- Add domain-specific knowledge (e.g., Workday HR policies, SIS data)
- Improves performance on specialized tasks
- Can be done efficiently with LoRA (Low-Rank Adaptation)

### Hybrid Pipeline

- Use Mistral-7B for conversational depth and complex reasoning
- Use Minitron-2B for large-context ingestion and document processing
- Route queries to the appropriate model based on requirements

## Architecture Suggestion

A recommended hybrid pipeline architecture:

- **FastAPI** as the serving layer → Provides REST API and web interface
- **Ollama / llama.cpp** for model hosting → Efficient CPU inference
- **Role-based RAG** for contextual responses → Weaviate with filtering
- **Function calling** for structured outputs → JSON responses for APIs
- **Quantized GGUF models** for CPU efficiency → No GPU required

This architecture enables:

- Scalable deployment on commodity hardware
- Flexible model selection based on use case
- Integration with existing systems
- Cost-effective operation

## Decision Matrix

| Phase         | Model                    | Why                                   |
| ------------- | ------------------------ | ------------------------------------- |
| **Phase One** | Mistral-7B-Instruct-GGUF | Robust chatbot, better reasoning      |
| **Phase Two** | Minitron-2B-128K         | Large-context RAG for historical data |

## Key Takeaways

1. **Start with Mistral-7B-Instruct-GGUF** for quality and robustness in conversational applications
2. **Add Minitron-2B-128K later** for massive context ingestion when needed
3. **Quantization is critical** for CPU deployment without sacrificing too much accuracy
4. **Both models can be augmented** with RAG, function calling, and fine-tuning for domain-specific tasks
5. **Choose based on use case**: Quality and reasoning → Mistral-7B; Large context and speed → Minitron-2B

The current project uses **Mistral-7B-Instruct-GGUF** with Q4_K_M quantization, which provides an excellent balance of quality, performance, and resource usage for most chatbot applications.
