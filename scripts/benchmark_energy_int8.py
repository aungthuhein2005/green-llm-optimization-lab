import json
import subprocess
import threading
import time
from pathlib import Path

import torch
import yaml
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)


CONFIG_PATH = "configs/baseline.yaml"
PROMPTS_PATH = "prompts/benchmark_prompts.json"
OUTPUT_PATH = "results/energy_int8.json"

SAMPLE_INTERVAL = 0.1  # seconds


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_gpu_power_watts():
    """
    Read instantaneous GPU power from nvidia-smi.
    Returns None if unavailable.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        value = result.stdout.strip().splitlines()[0]
        return float(value)

    except Exception:
        return None


def power_sampler(stop_event, samples):
    while not stop_event.is_set():

        power = read_gpu_power_watts()

        if power is not None:
            samples.append(
                {
                    "time": time.perf_counter(),
                    "power_watts": power,
                }
            )

        time.sleep(SAMPLE_INTERVAL)


def calculate_energy_joules(samples):
    """
    Integrate power over time using trapezoidal approximation.
    """

    if len(samples) < 2:
        return 0.0

    energy = 0.0

    for i in range(1, len(samples)):

        t1 = samples[i - 1]["time"]
        t2 = samples[i]["time"]

        p1 = samples[i - 1]["power_watts"]
        p2 = samples[i]["power_watts"]

        dt = t2 - t1

        avg_power = (p1 + p2) / 2

        energy += avg_power * dt

    return energy


def main():

    config = load_yaml(CONFIG_PATH)
    prompts = load_json(PROMPTS_PATH)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required.")

    model_name = config["model_name"]
    generation_cfg = config["generation"]

    print("GPU:", torch.cuda.get_device_name(0))
    print("Model:", model_name)
    print("Precision: int8")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    quantization_config = BitsAndBytesConfig(
    load_in_8bit=True
)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="cuda",
    )

    model.eval()

    all_results = []

    for item in prompts:

        prompt_id = item["id"]
        prompt = item["prompt"]

        print(f"\n=== {prompt_id} ===")

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
        ).to("cuda")

        prompt_tokens = inputs["input_ids"].shape[1]

        # Warm-up
        with torch.no_grad():
            _ = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
            )

        torch.cuda.synchronize()

        del _

        samples = []
        stop_event = threading.Event()

        sampler_thread = threading.Thread(
            target=power_sampler,
            args=(stop_event, samples),
        )

        sampler_thread.start()

        start = time.perf_counter()

        with torch.no_grad():

            output = model.generate(
                **inputs,
                max_new_tokens=generation_cfg["max_new_tokens"],
                do_sample=generation_cfg["do_sample"],
            )

        torch.cuda.synchronize()

        end = time.perf_counter()

        stop_event.set()
        sampler_thread.join()

        generation_time = end - start

        output_tokens = (
            output.shape[1] - prompt_tokens
        )

        tokens_per_second = (
            output_tokens / generation_time
        )

        energy_joules = calculate_energy_joules(samples)

        avg_power = (
            sum(s["power_watts"] for s in samples)
            / len(samples)
            if samples
            else 0
        )

        joules_per_token = (
            energy_joules / output_tokens
            if output_tokens > 0
            else 0
        )

        print(
            f"Time: {generation_time:.3f}s"
        )

        print(
            f"Output tokens: {output_tokens}"
        )

        print(
            f"Throughput: {tokens_per_second:.2f} tok/s"
        )

        print(
            f"Average power: {avg_power:.2f} W"
        )

        print(
            f"Energy: {energy_joules:.2f} J"
        )

        print(
            f"Energy/token: {joules_per_token:.4f} J/token"
        )

        all_results.append(
            {
                "prompt_id": prompt_id,
                "generation_time_sec": generation_time,
                "output_tokens": output_tokens,
                "tokens_per_sec": tokens_per_second,
                "average_power_watts": avg_power,
                "energy_joules": energy_joules,
                "joules_per_token": joules_per_token,
                "power_samples": samples,
            }
        )

    final_result = {
        "model": model_name,
        "precision": "int8",
        "gpu": torch.cuda.get_device_name(0),
        "sample_interval_sec": SAMPLE_INTERVAL,
        "results": all_results,
    }

    Path(OUTPUT_PATH).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            final_result,
            f,
            indent=2,
        )

    print(
        f"\nSaved results to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()