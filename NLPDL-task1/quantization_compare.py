import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

# Define the model and tokenizer
MODEL_NAME = "gpt2"  # Using GPT-2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)

# Helper function to measure throughput
def measure_throughput(model, tokenizer, dataset, max_new_tokens=50, use_kv_cache=False, quantization=None):
    """
    Measures inference throughput (tokens/second) and GPU memory usage for a dataset.
    """
    # Apply quantization if specified

    # Prepare for throughput measurement
    total_tokens = 0
    total_time = 0

    # Iterate through each line in the dataset
    for input_text in dataset:
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(DEVICE)

        # Measure the time for generating tokens
        
        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()
        torch.cuda.synchronize()
        with torch.no_grad():
            if use_kv_cache:
                # Enable KV caching
                if quantization:
                    outputs = model.generate(input_ids, max_new_tokens=max_new_tokens, use_cache=True,
                    cache_implementation="quantized", cache_config={"nbits":quantization, "backend":"quanto"})
                    memory_usage = torch.cuda.max_memory_allocated(device='cuda') / 1024**2
                else:
                    outputs = model.generate(input_ids, max_new_tokens=max_new_tokens, use_cache=True)
                    memory_usage = torch.cuda.max_memory_allocated(device='cuda') / 1024**2
            else:
                # Naive implementation
                outputs = model.generate(input_ids, max_new_tokens=max_new_tokens, use_cache=False)
                memory_usage = torch.cuda.max_memory_allocated(device='cuda') / 1024**2
        torch.cuda.synchronize()
        elapsed_time = time.time() - start_time

        # Update total tokens and time
        total_tokens += max_new_tokens
        total_time += elapsed_time

    # Compute throughput
    throughput = total_tokens / total_time
    return elapsed_time, memory_usage

# Load dataset
with open("/home/mohan/NLP_assignment3/data.txt", "r") as f:
    dataset = f.readlines()

# Experiment parameters
max_new_tokens = 50

# Measure throughput and memory for each implementation
results = {}

# Naive implementation
results["Naive"] = measure_throughput(model, tokenizer, dataset, max_new_tokens, use_kv_cache=False)

# KV-cache-enabled implementation
results["KV-Cache"] = measure_throughput(model, tokenizer, dataset, max_new_tokens, use_kv_cache=True)

# KV-cache with INT4 quantization
results["KV-Cache INT4"] = measure_throughput(model, tokenizer, dataset, max_new_tokens, use_kv_cache=True, quantization=4)

# KV-cache with INT2 quantization
results["KV-Cache INT2"] = measure_throughput(model, tokenizer, dataset, max_new_tokens, use_kv_cache=True, quantization=2)

# Display results
print(f"\nResults on dataset (data.txt):\n")
for key, (throughput, memory) in results.items():
    throughput = float(throughput)
    memory = float(memory)
    print(f"{key} - Throughput: {throughput:.2f} tokens/sec, Memory Usage: {memory:.2f} MB")