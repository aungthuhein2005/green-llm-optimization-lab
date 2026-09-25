# Green LLM Optimization Lab

## Research Question

Can an LLM be made faster and more energy-efficient without
significantly reducing output quality?

## Model

Qwen/Qwen2.5-0.5B-Instruct

## Experimental Environments

### Local Optimization Experiments

GPU: NVIDIA GeForce RTX 4050 Laptop GPU
VRAM: 6 GB

Used for:

- FP16 baseline
- INT8 quantization
- INT4 NF4 quantization
- latency measurements
- throughput measurements
- GPU power sampling
- energy-per-token estimation
- output quality comparison

### Serving Experiments

GPU: NVIDIA Tesla T4
VRAM: 15 GB
Environment: Google Colab

Used for:

- vLLM serving
- continuous batching
- concurrency experiments

The two environments are analyzed separately because hardware
differences prevent a controlled Transformers-vs-vLLM engine
comparison.

---

# Experiment 1 — Quantization and Performance

| Configuration | Throughput | Latency |
|---|---:|---:|
| FP16 | 36.21 tok/s | 2.765 s |
| INT8 | 3.22 tok/s | 32.170 s |
| INT4 NF4 | 17.31 tok/s | 6.522 s |

FP16 achieved the highest generation throughput and lowest
latency.

INT8 reduced numerical precision but was approximately 91%
slower in throughput than FP16.

INT4 performed substantially better than INT8 but remained
slower than FP16.

These results demonstrate that reducing weight precision does
not guarantee faster inference. Performance depends on GPU
architecture, model size, quantization kernels, and
quantization/dequantization overhead.

---

# Experiment 2 — Energy Efficiency

GPU power was sampled during generation using NVIDIA GPU
telemetry.

Estimated energy was calculated by integrating sampled power
over generation time.

Energy per token was calculated as:

Energy per token = total estimated GPU energy / generated tokens

| Configuration | Average Power | Energy | Energy/Token |
|---|---:|---:|---:|
| FP16 | 18.24 W | 49.31 J | 0.4931 J/token |
| INT8 | 7.44 W | 233.42 J | 2.3342 J/token |
| INT4 NF4 | 9.97 W | 63.48 J | 0.6348 J/token |

Although INT8 reduced average GPU power by approximately 59%,
its much longer execution time caused energy per token to
increase by approximately 373%.

INT4 reduced average GPU power by approximately 45%, but its
energy per token was still approximately 29% higher than FP16.

Therefore, lower instantaneous GPU power did not translate into
lower total energy consumption.

For this model and GPU, FP16 provided the best measured energy
efficiency.

---

# Experiment 3 — LLM Serving

vLLM was evaluated on a Tesla T4 using multiple simultaneous
requests.

| Concurrency | Throughput | Average Latency | Requests/sec |
|---:|---:|---:|---:|
| 1 | 127.42 tok/s | 0.785 s | 1.27 |
| 4 | 470.18 tok/s | 0.849 s | 4.70 |
| 8 | 1187.08 tok/s | 0.672 s | 11.87 |

Increasing concurrency from 1 to 8 increased aggregate
throughput by approximately 9.3x.

This demonstrates how batching multiple active sequences can
increase GPU utilization and serving throughput.

The tested concurrency levels did not reach an obvious
saturation point.

---

# Experiment 4 — Output Quality

Five prompts covering explanation, arithmetic reasoning,
coding, instruction following, and LLM concepts were tested
using identical generation settings.

No consistent quality degradation caused by INT8 or INT4 was
observed in this small prompt set.

However, all configurations produced weaknesses on some tasks,
particularly the prefill-versus-decoding explanation.

Because only five prompts were evaluated, these results should
be treated as exploratory rather than as a statistically
reliable model-quality benchmark.

---

# Key Findings

1. Quantization successfully changes the memory and power
   characteristics of inference, but lower precision does not
   automatically produce higher throughput.

2. Lower average GPU power does not necessarily mean lower
   energy consumption. Execution time must also be considered.

3. On the RTX 4050 and Qwen2.5-0.5B tested here, FP16 achieved
   the lowest latency and lowest measured energy per token.

4. INT8 had the lowest average power draw but the highest
   energy per generated token because inference was much slower.

5. INT4 represented a better compromise than INT8 but did not
   outperform FP16 in latency or energy efficiency.

6. vLLM demonstrated substantial throughput scaling as
   concurrency increased on the Tesla T4.

7. A model fitting in memory under Hugging Face Transformers
   does not guarantee that a production inference server such
   as vLLM will fit on the same GPU because serving introduces
   additional KV-cache and runtime memory requirements.

---

# Limitations

- Quantization and serving experiments used different GPUs.
- GPU energy measurements represent sampled GPU power rather
  than full-system wall power.
- The tested model contains only approximately 0.5B parameters.
- Only five quality-evaluation prompts were used.
- Only bitsandbytes INT8 and NF4 INT4 quantization were tested.
- Concurrency testing stopped at eight requests and therefore
  did not identify the server's saturation point.
- Energy measurements should be repeated multiple times to
  estimate variance.

---

# Conclusion

The experiments do not support the assumption that aggressive
quantization automatically makes LLM inference greener.

For Qwen2.5-0.5B-Instruct on the tested RTX 4050 Laptop GPU,
FP16 achieved the best combination of throughput, latency, and
measured GPU energy per generated token.

Quantization substantially reduced average GPU power, but the
resulting inference slowdown increased total energy consumption.

Separately, the serving experiment demonstrated that increasing
concurrency can dramatically improve aggregate throughput by
allowing an inference engine to use GPU parallelism more
effectively.

The broader conclusion is that green LLM inference requires
optimizing the complete system rather than minimizing a single
metric such as numerical precision, power draw, or latency.