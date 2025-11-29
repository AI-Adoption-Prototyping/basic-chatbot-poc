from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import os
import dotenv

dotenv.load_dotenv()  # Load environment variables from .env file

# GGUF models work excellently on M1 Mac with llama-cpp-python
# They're optimized for Apple Silicon and use ~4-5GB RAM with 4-bit quantization

# Download and use a pre-quantized 4-bit GGUF model from TheBloke
# This is the Q4_K_M quantized version (4-bit, medium quality)
print("Loading pre-quantized 4-bit GGUF model...")
print("This will download the model on first run (~4.5GB)")

# Download the model file from Hugging Face
model_filename = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
model_path = hf_hub_download(
    repo_id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
    filename=model_filename,
    local_dir=None,  # Use Hugging Face cache
)

print(f"Model downloaded to: {model_path}")

# Load the GGUF model with llama.cpp
# n_ctx: context window size
# n_threads: number of threads (automatically detects CPU cores)
# n_gpu_layers: layers to offload to GPU/Metal (1+ uses Metal acceleration on M1)
model = Llama(
    model_path=model_path,
    n_ctx=4096,  # Context window size
    n_threads=None,  # Auto-detect CPU cores
    n_gpu_layers=1,  # Use Metal acceleration on M1 (can increase for faster inference)
    verbose=False,
)

print("Model loaded successfully!")
print("Model uses approximately 4-5GB of RAM (4-bit quantization)")


def generate_response(prompt):
    print("Generating response...")

    # Format prompt for Mistral Instruct
    # llama.cpp handles the <s> token automatically, so we don't need to add it
    formatted_prompt = f"[INST] {prompt} [/INST]"

    # Generate response using llama.cpp
    output = model(
        formatted_prompt,
        max_tokens=100,
        temperature=0.7,
        top_p=0.9,
        repeat_penalty=1.1,
        stop=["</s>", "[INST]", "[/INST]"],
        echo=False,  # Don't echo the prompt
    )

    # Extract the generated text
    response = output["choices"][0]["text"].strip()
    print(f"Generated response: {response}")
    return response


print("Loading model...")
response = generate_response("Hello, how are you?")
print("Response: ", response)
