# 🌱 Green LLM Optimization Lab

A reproducible experimental project exploring the performance, memory, energy-efficiency, quality, and serving trade-offs involved in optimizing Large Language Model inference.

## Research Question

> **Can I make an LLM faster and more energy-efficient without significantly reducing its output quality?**

Rather than building a new AI application, this project benchmarks different LLM inference configurations and studies how optimization techniques affect:

- Generation throughput
- Latency
- GPU memory
- GPU power consumption
- Energy per generated token
- Output quality
- Multi-user serving throughput

---

## Model

The experiments use:

**Qwen/Qwen2.5-0.5B-Instruct**

A small instruction-tuned model was intentionally selected so the experiments could run on consumer GPU hardware.

---

## Project Structure

```text
green-llm-optimization-lab/
│
├── configs/
│   ├── baseline.yaml
│   ├── int8.yaml
│   ├── int4.yaml
│   └── serving.yaml
│
├── prompts/
│   └── benchmark_prompts.json
│
├── scripts/
│   ├── benchmark_fp16.py
│   ├── benchmark_int8.py
│   ├── benchmark_int4.py
│   ├── benchmark_energy_fp16.py
│   ├── benchmark_energy_int8.py
│   ├── benchmark_energy_int4.py
│   ├── benchmark_vllm.py
│   ├── benchmark_concurrency.py
│   ├── compare_results.py
│   ├── compare_energy_results.py
│   └── evaluate_quality.py
│
├── results/
│   ├── fp16.json
│   ├── int8.json
│   ├── int4.json
│   ├── energy_fp16.json
│   ├── energy_int8.json
│   ├── energy_int4.json
│   ├── quality_evaluation.json
│   ├── manual_quality_review.json
│   ├── vllm_fp16_c1.json
│   └── vllm_concurrency.json
│
├── report/
│   └── findings.md
│
├── requirements.txt
└── README.md
```

---

# Experiments

The project contains four main groups of experiments.

## 1. Quantization

Three inference configurations were evaluated:

- FP16 baseline
- INT8 using bitsandbytes
- INT4 NF4 using bitsandbytes

The same model, prompts, generation settings, and GPU were used for these experiments.

The goal was to determine whether reducing model precision improves inference performance and resource efficiency.

---

## 2. Energy Efficiency

GPU power was sampled during inference using NVIDIA GPU telemetry.

Approximate GPU energy consumption was calculated by integrating sampled power over generation time.

The primary efficiency metric is:

```text
Energy per token = GPU energy consumed / generated tokens
```

This is more informative than power draw alone because a low-power configuration can still consume more total energy if inference takes significantly longer.

---

## 3. LLM Serving

The model was served using **vLLM** to investigate production-style inference.

Concurrency levels tested:

```text
1 request
4 concurrent requests
8 concurrent requests
```

The experiment measured:

- Aggregate tokens/second
- Average request latency
- Requests/second

This demonstrates the effect of batching and concurrent workloads on GPU utilization.

---

## 4. Output Quality

A small prompt suite was used to test:

- Explanation
- Arithmetic reasoning
- Coding
- Instruction following
- LLM inference concepts

Automatic checks were combined with manual review.

The quality experiment is exploratory and is not intended to replace a large-scale model evaluation benchmark.

---

# Hardware

Two environments were used.

## Local Optimization Environment

```text
GPU: NVIDIA GeForce RTX 4050 Laptop GPU
VRAM: 6 GB
```

Used for:

- FP16 baseline
- INT8
- INT4
- GPU memory experiments
- Power measurements
- Energy measurements
- Output quality evaluation

## Serving Environment

```text
GPU: NVIDIA Tesla T4
VRAM: 15 GB
Environment: Google Colab
Inference Engine: vLLM
```

Used for:

- vLLM inference
- Concurrent request experiments
- Continuous batching experiments

> **Important:** Absolute Transformers and vLLM performance numbers should not be directly interpreted as an engine-only comparison because the experiments ran on different GPUs.

---

# Results

## Energy & Performance — RTX 4050

| Configuration | Throughput | Avg. Power | Energy | Energy / Token | Latency |
|---|---:|---:|---:|---:|---:|
| **FP16** | **36.21 tok/s** | 18.24 W | **49.31 J** | **0.4931 J/token** | **2.765 s** |
| INT8 | 3.22 tok/s | **7.44 W** | 233.42 J | 2.3342 J/token | 32.170 s |
| INT4 NF4 | 17.31 tok/s | 9.97 W | 63.48 J | 0.6348 J/token | 6.522 s |

### Changes Relative to FP16

| Configuration | Throughput | Avg. Power | Energy / Token | Latency |
|---|---:|---:|---:|---:|
| INT8 | -91.12% | -59.21% | +373.40% | +1063.25% |
| INT4 | -52.21% | -45.36% | +28.74% | +135.82% |

### Observation

Quantization substantially reduced instantaneous GPU power consumption.

However, both quantized configurations generated tokens more slowly.

INT8 reduced average GPU power by approximately **59%**, but its large increase in execution time caused energy per token to increase by approximately **373%**.

INT4 provided a better trade-off than INT8, but still consumed approximately **29% more energy per generated token** than FP16.

For this particular model, GPU, and software configuration:

> **FP16 provided the highest throughput, lowest latency, and lowest measured GPU energy per generated token.**

---

# vLLM Serving Results

Serving experiments were performed on a Tesla T4.

| Concurrency | Throughput | Avg. Latency | Requests/sec |
|---:|---:|---:|---:|
| 1 | 127.42 tok/s | 0.785 s | 1.27 |
| 4 | 470.18 tok/s | 0.849 s | 4.70 |
| 8 | **1187.08 tok/s** | **0.672 s** | **11.87** |

Increasing concurrency from 1 to 8 increased aggregate output throughput by approximately:

```text
9.32×
```

The workload had not yet reached an obvious saturation point at concurrency 8.

These results demonstrate how an inference engine can use batching to increase GPU utilization when multiple sequences are active.

---

# Quality Evaluation

The same benchmark prompts were used across FP16, INT8, and INT4.

The small evaluation did **not show consistent evidence that quantization caused systematic output-quality degradation**.

However, several outputs from all configurations contained weaknesses, particularly when explaining the difference between LLM prefill and decoding.

Because only a small number of prompts were evaluated, quality results should be considered exploratory.

A larger benchmark would be required to make strong claims about quality retention.

---

# Key Findings

### 1. Lower precision does not automatically mean faster inference

Although INT8 and INT4 reduce model weight precision, the quantized configurations were slower than FP16 in this experiment.

Quantization/dequantization overhead, GPU architecture, kernels, and model size all affect actual performance.

### 2. Lower GPU power does not necessarily mean lower energy

Energy depends on both power and execution time:

```text
Energy = Power × Time
```

INT8 consumed substantially less instantaneous power but ran much longer, resulting in significantly higher energy consumption per generated token.

### 3. FP16 was the most energy-efficient local configuration

For Qwen2.5-0.5B-Instruct on the RTX 4050 Laptop GPU:

```text
FP16: 0.4931 J/token
INT4: 0.6348 J/token
INT8: 2.3342 J/token
```

### 4. INT4 was a better compromise than INT8

INT4 was significantly faster and more energy-efficient than INT8, although it still did not outperform FP16.

### 5. Serving optimization can produce large throughput gains

vLLM throughput increased substantially when concurrency increased.

This demonstrates that optimization is not limited to model precision.

**Request scheduling, batching, KV-cache management, and GPU utilization are also important optimization dimensions.**

### 6. A model fitting in VRAM does not guarantee a serving engine will fit

The Qwen model could run locally using Hugging Face Transformers on the 6 GB RTX 4050.

However, attempts to run vLLM locally encountered GPU-memory limitations during engine initialization.

Serving engines require additional memory for components such as:

- KV cache
- runtime buffers
- scheduling infrastructure
- execution structures

The serving experiment was therefore moved to a Tesla T4 with 15 GB VRAM.

---

# Reproducing the Experiments

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/green-llm-optimization-lab.git
cd green-llm-optimization-lab
```

## 2. Create a Python Environment

Windows example:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install Dependencies

Install a CUDA-compatible PyTorch build appropriate for your system first.

Then install the remaining dependencies:

```bash
pip install transformers accelerate bitsandbytes pyyaml
```

---

# Run FP16 Baseline

```bash
python scripts/benchmark_fp16.py
```

Output:

```text
results/fp16.json
```

---

# Run INT8

```bash
python scripts/benchmark_int8.py
```

Output:

```text
results/int8.json
```

---

# Run INT4

```bash
python scripts/benchmark_int4.py
```

Output:

```text
results/int4.json
```

---

# Compare Quantization Results

```bash
python scripts/compare_results.py
```

---

# Energy Experiments

Run:

```bash
python scripts/benchmark_energy_fp16.py
python scripts/benchmark_energy_int8.py
python scripts/benchmark_energy_int4.py
```

Then compare:

```bash
python scripts/compare_energy_results.py
```

Results are written to:

```text
results/energy_fp16.json
results/energy_int8.json
results/energy_int4.json
```

---

# Quality Evaluation

Run:

```bash
python scripts/evaluate_quality.py
```

This creates:

```text
results/quality_evaluation.json
results/manual_quality_review.json
```

The second file is intended for human review of open-ended model outputs.

---

# vLLM Serving

Start the vLLM server:

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --dtype float16 \
  --generation-config vllm \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.80 \
  --port 8000
```

Then run the single-request benchmark:

```bash
python scripts/benchmark_vllm.py
```

And the concurrency benchmark:

```bash
python scripts/benchmark_concurrency.py
```

---

# Methodology Notes

For reproducibility:

- The same prompt set is used across precision configurations.
- Generation is deterministic where possible.
- Each benchmark includes GPU warm-up.
- CUDA synchronization is performed before recording elapsed time.
- Quantization experiments use the same local GPU.
- Energy measurements sample NVIDIA GPU power telemetry during generation.
- Serving experiments keep the same model while varying request concurrency.

---

# Limitations

This project has several important limitations:

1. Quantization and vLLM serving experiments were performed on different GPUs.
2. Energy measurements represent estimated **GPU-side energy**, not full-system wall power.
3. The model contains only approximately 0.5B parameters.
4. The quality benchmark contains only a small number of prompts.
5. Only bitsandbytes INT8 and NF4 INT4 quantization were evaluated.
6. GPTQ and AWQ were not benchmarked.
7. Serving concurrency was tested only up to eight requests.
8. The serving saturation point was not identified.
9. Energy measurements should be repeated across multiple trials to estimate variance.
10. Results should not be generalized to larger models or different GPU architectures without additional experiments.

---

# Conclusion

The experiments show that **LLM optimization is a system-level problem**.

Reducing model precision alone did not make inference faster or more energy-efficient in the tested environment.

Although INT8 and INT4 reduced average GPU power consumption, their slower generation speeds increased total energy consumption per token.

For the tested Qwen2.5-0.5B-Instruct model on an RTX 4050 Laptop GPU, **FP16 achieved the best measured combination of throughput, latency, and GPU energy efficiency**.

Separately, the vLLM experiments demonstrated that serving optimizations such as batching and concurrency can dramatically increase aggregate throughput.

The central finding of this project is therefore:

> **Green LLM inference cannot be optimized using a single metric. Model precision, execution speed, GPU power, memory usage, output quality, batching, and serving configuration must be evaluated together.**

---

## Future Work

Potential extensions include:

- Test larger 1B–7B parameter models
- Benchmark GPTQ and AWQ
- Compare BF16 with FP16
- Measure TTFT and TPOT
- Find the vLLM saturation point with higher concurrency
- Measure full-system wall power
- Run multiple energy trials and calculate confidence intervals
- Test optimized quantized inference engines
- Compare additional GPUs
- Use established quality benchmarks instead of a small custom prompt set
- Measure KV-cache memory under different context lengths

---

## Technologies

- Python
- PyTorch
- Hugging Face Transformers
- bitsandbytes
- vLLM
- CUDA
- NVIDIA GPU telemetry
- Google Colab
- WSL2

---

## Purpose

This repository was created as a hands-on study of **LLM inference optimization, GPU performance, quantization, energy efficiency, and production model serving**.

It is intended as an experimental learning project rather than a production benchmarking suite.