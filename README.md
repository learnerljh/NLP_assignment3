# NLP-DL Assignment 3 Repository

This repository contains the code and resources for **NLP-DL Assignment 3**, which focuses on two main tasks: **Task 1 - Quantization and Throughput Comparison** and **Task 2 - Prompt Evaluation on GSM8k Dataset**. Below is an overview of the repository structure, the tasks, and how to run the code.

---

## Repository Structure

```
NLPDL-task1/
├── customized_gpt2.py
├── data.txt
├── main.py
├── quantization_compare.py
├── report1.md
└── requirements.txt

NLPDL-task2/
├── prompt_evaluation.py
├── rag.py
├── reflextion.py
├── report2.md

.gitignore
LICENSE
README.md
NLP_DL_Assignment_3_Report.pdf
```

---

## Task 1: Quantization and Throughput Comparison

### Overview
This task focuses on comparing the performance of GPT-2 with and without **Key-Value (KV) cache** and **quantization**. The goal is to measure the **throughput** (tokens/second) and **memory usage** for different configurations.

### Key Files
- **`quantization_compare.py`**:
  - Implements the main logic for measuring throughput and memory usage.
  - Supports different configurations:
    - BF16 precision.
    - INT4 quantization.
    - No cache (disabling KV-cache).
  - Uses the `transformers` library to load the `Qwen/Qwen-7B-Chat` model and the `torch` library for GPU operations.

- **`data.txt`**:
  - A dataset file containing input text for inference.

### How to Run
1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the script with the desired quantization type:
   ```bash
   python quantization_compare.py bf16
   ```
   or
   ```bash
   python quantization_compare.py int4
   ```
   or
   ```bash
   python quantization_compare.py nocache
   ```

3. The script will output the following:
   - **Average generation speed (tokens/second)**.
   - **GPU memory usage (in GB)**.
   - **Experiment settings**, including seed, number of experiments, context length, generation length, and quantization type.

---

## Task 2: Prompt Evaluation on GSM8k Dataset

### Overview
This task evaluates the performance of different **prompting strategies** on the **GSM8k dataset**. The goal is to measure the accuracy of generated answers using various prompts, such as **CoT (Chain of Thought)** and **few-shot examples**.

### Key Files
- **`prompt_evaluation.py`**:
  - Implements the main logic for generating prompts and evaluating the accuracy of responses.
  - Uses the `datasets` library to load the GSM8k dataset.
  - Integrates with the **DeepSeek API** for generating responses.

- **`data/`**:
  - Contains the GSM8k dataset used for evaluation.

### How to Run
1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the script with the desired prompt strategy:
   ```bash
   python prompt_evaluation.py cot
   ```
   or
   ```bash
   python prompt_evaluation.py few-shot
   ```
   or
   ```bash
   python prompt_evaluation.py naive
   ```
   or
   ```bash
   python reflexion.py
   ```

4. The script will output the accuracy of the generated answers for the specified prompt strategy.

---

## Dependencies

Install all dependencies using:
```bash
pip install -r requirements.txt
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contributors

- [Jiahao Li]

---

If you have any questions or issues, feel free to open an issue or contact the repository owner.
