import json
from pathlib import Path


FP16_PATH = Path("results/fp16.json")
INT8_PATH = Path("results/int8.json")
INT4_PATH = Path("results/int4.json")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def average_metric(results, key):
    values = [item[key] for item in results]
    return sum(values) / len(values)


fp16 = load_json(FP16_PATH)
int8 = load_json(INT8_PATH)
int4 = load_json(INT4_PATH)


fp16_latency = average_metric(
    fp16["results"],
    "generation_time_sec_avg"
)

int8_latency = average_metric(
    int8["results"],
    "generation_time_sec_avg"
)

int4_latency = average_metric(
    int4["results"],
    "generation_time_sec_avg"
)


fp16_speed = average_metric(
    fp16["results"],
    "tokens_per_sec_avg"
)

int8_speed = average_metric(
    int8["results"],
    "tokens_per_sec_avg"
)

int4_speed = average_metric(
    int4["results"],
    "tokens_per_sec_avg"
)


fp16_memory = average_metric(
    fp16["results"],
    "peak_gpu_memory_gb"
)

int8_memory = average_metric(
    int8["results"],
    "peak_gpu_memory_gb"
)

int4_memory = average_metric(
    int4["results"],
    "peak_gpu_memory_gb"
)


memory_reduction = (
    (fp16_memory - int8_memory)
    / fp16_memory
    * 100
)

latency_change = (
    (int8_latency - fp16_latency)
    / fp16_latency
    * 100
)

throughput_change = (
    (int8_speed - fp16_speed)
    / fp16_speed
    * 100
)

int4_memory_reduction = (
    (fp16_memory - int4_memory)
    / fp16_memory
    * 100
)

int4_latency_change = (
    (int4_latency - fp16_latency)
    / fp16_latency
    * 100
)

int4_throughput_change = (
    (int4_speed - fp16_speed)
    / fp16_speed
    * 100
)


print("\n=== FP16 vs INT8 ===")

print(f"FP16 model memory: {fp16_memory:.3f} GB")
print(f"INT8 model memory: {int8_memory:.3f} GB")
print(f"Memory reduction: {memory_reduction:.2f}%")

print()

print(f"FP16 avg latency: {fp16_latency:.3f} s")
print(f"INT8 avg latency: {int8_latency:.3f} s")
print(f"Latency change: {latency_change:+.2f}%")

print()

print(f"FP16 throughput: {fp16_speed:.2f} tok/s")
print(f"INT8 throughput: {int8_speed:.2f} tok/s")
print(f"Throughput change: {throughput_change:+.2f}%")

print("\n=== FP16 vs INT4 ===")

print(f"FP16 model memory: {fp16_memory:.3f} GB")
print(f"INT4 model memory: {int4_memory:.3f} GB")
print(f"Memory reduction: {int4_memory_reduction:.2f}%")

print()

print(f"FP16 avg latency: {fp16_latency:.3f} s")
print(f"INT4 avg latency: {int4_latency:.3f} s")
print(f"Latency change: {int4_latency_change:+.2f}%")

print()

print(f"FP16 throughput: {fp16_speed:.2f} tok/s")
print(f"INT4 throughput: {int4_speed:.2f} tok/s")
print(f"Throughput change: {int4_throughput_change:+.2f}%")