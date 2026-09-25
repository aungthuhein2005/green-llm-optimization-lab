<div align="center">

# 🌱 Green LLM Optimization Lab

### Measure what “efficient” really means for LLM inference.

An end-to-end benchmarking lab for comparing **speed, latency, GPU power, energy per token, output quality, and serving throughput** across FP16, INT8, INT4, and vLLM workloads.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Transformers-FFD21E)](https://huggingface.co/docs/transformers)
[![Model](https://img.shields.io/badge/Model-Qwen2.5--0.5B--Instruct-7C3AED)](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
[![vLLM](https://img.shields.io/badge/Serving-vLLM-2F80ED)](https://docs.vllm.ai/)

[Key findings](#-key-findings) · [Results](#-benchmark-results) · [Quick start](#-quick-start) · [Methodology](#-methodology) · [Limitations](#-limitations)

</div>

---

## Why this project?

Quantization is often presented as a simple route to faster, greener inference. In practice, lower precision can reduce power draw while increasing latency enough to use **more total energy**.

This repository tests one central question:

> **Can an LLM become faster and more energy-efficient without meaningfully reducing output quality?**

| Track | What is measured |
|---|---|
| **Quantization** | FP16 vs. bitsandbytes INT8 vs. NF4 INT4 |
| **Performance** | Throughput, latency, and GPU memory |
| **Energy** | GPU power, estimated energy, and joules per token |
| **Quality** | Automatic checks and manual review across a shared prompt suite |
| **Serving** | vLLM throughput and latency at 1, 4, and 8 concurrent requests |

> [!IMPORTANT]
> Optimization is a systems problem. Precision, kernels, hardware, execution time, batching, memory, and serving strategy all affect the outcome.

## ✨ Key findings

1. **FP16 won on the tested local hardware.** It delivered the highest throughput, lowest latency, and lowest measured GPU energy per token.
2. **Lower power did not mean lower energy.** INT8 drew about 59% less average GPU power but required much longer to generate, increasing energy per token by about 373%.
3. **INT4 was the stronger quantized compromise.** It substantially outperformed INT8, but still did not beat FP16 on this small model and GPU.
4. **Serving optimization produced the largest throughput gain.** vLLM aggregate throughput scaled from 127.42 to 1,187.08 tokens/s as concurrency increased from 1 to 8.
5. **No consistent quality collapse was observed.** The small evaluation did not show systematic degradation from quantization, though the sample is too limited for broad claims.

## 📊 Benchmark results

### Local inference — RTX 4050 Laptop GPU

| Precision | Throughput ↑ | Avg. power ↓ | Energy/token ↓ | Latency ↓ |
|---|---:|---:|---:|---:|
| **FP16** | **36.21 tok/s** | 18.24 W | **0.4931 J/token** | **2.765 s** |
| INT4 NF4 | 17.31 tok/s | 9.97 W | 0.6348 J/token | 6.522 s |
| INT8 | 3.22 tok/s | **7.44 W** | 2.3342 J/token | 32.170 s |

```text
Energy per generated token

FP16  █████                     0.4931 J/token
INT4  ██████                    0.6348 J/token
INT8  ███████████████████████   2.3342 J/token
```

The INT8 configuration used less instantaneous power, but its slower execution dominated the final energy cost: `Energy = Power × Time`.

### vLLM serving — Tesla T4

| Concurrent requests | Aggregate throughput ↑ | Avg. latency ↓ | Requests/s ↑ |
|---:|---:|---:|---:|
| 1 | 127.42 tok/s | 0.785 s | 1.27 |
| 4 | 470.18 tok/s | 0.849 s | 4.70 |
| **8** | **1,187.08 tok/s** | **0.672 s** | **11.87** |

At concurrency 8, aggregate throughput was **9.32×** the single-request result. The workload had not reached an obvious saturation point.

> [!NOTE]
> Local Transformers and vLLM measurements were collected on different GPUs. Treat them as separate experiments, not as a direct engine-to-engine comparison.

## 🧪 Experiment design

All experiments use [`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct). The small instruction-tuned model keeps the lab accessible on consumer hardware.

| Environment | Hardware | Used for |
|---|---|---|
| **Local** | NVIDIA RTX 4050 Laptop GPU, 6 GB VRAM | FP16, INT8, INT4, memory, power, energy, quality |
| **Serving** | NVIDIA Tesla T4, 15 GB VRAM, Google Colab | vLLM and concurrency benchmarks |

The lab evaluates:

- **Performance:** output tokens per second and end-to-end generation latency
- **Energy:** sampled NVIDIA GPU power integrated over generation time
- **Efficiency:** GPU joules divided by generated tokens
- **Quality:** shared deterministic prompt suite with automatic and manual review
- **Serving:** aggregate throughput, average request latency, and requests per second

## 🚀 Quick start

### 1. Clone and create an environment

```bash
git clone https://github.com/aungthuhein2005/green-llm-optimization-lab.git
cd green-llm-optimization-lab
python -m venv .venv
```

```powershell
# Windows
.venv\Scripts\Activate.ps1
```

```bash
# Linux/macOS
source .venv/bin/activate
```

### 2. Install dependencies

Install the [PyTorch build](https://pytorch.org/get-started/locally/) matching your CUDA environment, then run:

```bash
pip install -r requirements.txt
pip install bitsandbytes pynvml
```

### 3. Run the benchmarks

```bash
# Performance and quantization
python scripts/benchmark_fp16.py
python scripts/benchmark_int8.py
python scripts/benchmark_int4.py
python scripts/compare_results.py

# Energy
python scripts/benchmark_energy_fp16.py
python scripts/benchmark_energy_int8.py
python scripts/benchmark_energy_int4.py
python scripts/compare_energy_results.py

# Quality
python scripts/evaluate_quality.py
```

Results are written to [`results/`](results/).

## ⚡ vLLM serving

Start an OpenAI-compatible server:

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --dtype float16 \
  --generation-config vllm \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.80 \
  --port 8000
```

Then run:

```bash
python scripts/benchmark_vllm.py
python scripts/benchmark_concurrency.py
```

> [!TIP]
> vLLM needs memory beyond model weights for its KV cache, runtime buffers, scheduler, and execution structures. A model that fits with Transformers may still fail during serving-engine initialization.

## 🗂️ Repository structure

```text
green-llm-optimization-lab/
├── configs/       # Precision and serving configurations
├── notebooks/     # Interactive experiments
├── prompts/       # Shared benchmark prompt suite
├── report/        # Detailed findings
├── results/       # Raw benchmark and evaluation outputs
├── scripts/       # Benchmark, comparison, and evaluation tools
├── requirements.txt
└── README.md
```

<details>
<summary><strong>Benchmark scripts</strong></summary>

| Script | Purpose |
|---|---|
| `benchmark_fp16.py` | FP16 baseline |
| `benchmark_int8.py` | bitsandbytes INT8 inference |
| `benchmark_int4.py` | bitsandbytes NF4 INT4 inference |
| `benchmark_energy_*.py` | Power sampling and energy measurement |
| `compare_results.py` | Performance comparison |
| `compare_energy_results.py` | Energy comparison |
| `evaluate_quality.py` | Automatic checks and manual-review output |
| `benchmark_vllm.py` | Single-request vLLM benchmark |
| `benchmark_concurrency.py` | Concurrent vLLM workload |

</details>

## 🔬 Methodology

- Every precision configuration uses the same model and prompt set.
- Generation is deterministic where possible.
- Benchmarks include a GPU warm-up pass.
- CUDA is synchronized before elapsed time is recorded.
- Local quantization experiments run on the same GPU.
- GPU power is sampled during generation and integrated over time.
- Serving experiments hold the model constant while changing concurrency.

The core efficiency metric is:

```text
Energy per token = estimated GPU energy consumed / generated tokens
```

This is more useful than power alone because a low-power configuration may run long enough to consume more total energy.

## ⚠️ Limitations

- Measurements estimate **GPU-side energy**, not full-system wall power.
- Quantization and serving experiments use different GPUs.
- The tested model has only about 0.5B parameters.
- Quality evaluation uses a small custom prompt set.
- Only bitsandbytes INT8 and NF4 INT4 are evaluated; GPTQ and AWQ are not included.
- Serving concurrency is tested only through eight requests, before a clear saturation point.
- Results are hardware- and software-specific and should not be generalized without more experiments.

## 🧭 Roadmap

- [ ] Benchmark 1B–7B parameter models
- [ ] Add BF16, GPTQ, and AWQ
- [ ] Measure time to first token (TTFT) and time per output token (TPOT)
- [ ] Find the vLLM saturation point at higher concurrency
- [ ] Measure full-system wall power
- [ ] Repeat energy trials and report variance or confidence intervals
- [ ] Evaluate established quality benchmarks
- [ ] Compare GPU architectures and optimized quantized runtimes

## 💡 Takeaway

For `Qwen2.5-0.5B-Instruct` on the tested RTX 4050 Laptop GPU, quantization reduced average power but did not improve speed or energy per token. FP16 delivered the best local balance, while vLLM batching and concurrency delivered the strongest throughput gains in the separate serving experiment.

> **Green inference is not the precision with the smallest number. It is the configuration that performs the required work with the best measured system-level trade-off.**

---

<div align="center">

Built as a hands-on study of efficient LLM inference, GPU performance, quantization, and production serving.

If this project helps your research, consider giving it a ⭐.

</div>
