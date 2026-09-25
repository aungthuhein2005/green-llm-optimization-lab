import json
import time
from pathlib import Path

import torch
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM


CONFIG_PATH = "configs/baseline.yaml"
PROMPTS_PATH = "prompts/benchmark_prompts.json"


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def load_prompts():
    with open(PROMPTS_PATH, "r") as f:
        return json.load(f)


def main():
    config = load_config()
    prompts = load_prompts()

    model_name = config["model_name"]
    generation_cfg = config["generation"]
    benchmark_cfg = config["benchmark"]
    output_file = config["output_file"]

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this benchmark.")

    device = "cuda"

    print("GPU:", torch.cuda.get_device_name(0))
    print("Model:", model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda"
    )

    model.eval()

    results = []

    for item in prompts:
        prompt_id = item["id"]
        prompt = item["prompt"]

        print(f"\nRunning prompt: {prompt_id}")

        inputs = tokenizer(
            prompt,
            return_tensors="pt"
        ).to(device)

        prompt_tokens = inputs["input_ids"].shape[1]

        # Warm-up
        for _ in range(benchmark_cfg["warmup_runs"]):
            with torch.no_grad():
                _ = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False
                )

            torch.cuda.synchronize()

        times = []
        speeds = []
        outputs = []

        for run in range(benchmark_cfg["measured_runs"]):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            start = time.perf_counter()

            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=generation_cfg["max_new_tokens"],
                    do_sample=generation_cfg["do_sample"]
                )

            torch.cuda.synchronize()

            end = time.perf_counter()

            generation_time = end - start

            output_tokens = (
                output.shape[1] - prompt_tokens
            )

            tokens_per_second = (
                output_tokens / generation_time
            )

            peak_memory_gb = (
                torch.cuda.max_memory_allocated()
                / 1024**3
            )

            generated_text = tokenizer.decode(
                output[0][prompt_tokens:],
                skip_special_tokens=True
            )

            times.append(generation_time)
            speeds.append(tokens_per_second)
            outputs.append(generated_text)

            print(
                f"Run {run + 1}: "
                f"{generation_time:.3f}s, "
                f"{tokens_per_second:.2f} tok/s"
            )

        result = {
            "prompt_id": prompt_id,
            "prompt": prompt,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "generation_time_sec_avg": sum(times) / len(times),
            "tokens_per_sec_avg": sum(speeds) / len(speeds),
            "peak_gpu_memory_gb": peak_memory_gb,
            "output": outputs[-1]
        }

        results.append(result)

    final_result = {
        "model": model_name,
        "precision": "fp16",
        "gpu": torch.cuda.get_device_name(0),
        "results": results
    }

    Path(output_file).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(output_file, "w") as f:
        json.dump(
            final_result,
            f,
            indent=2
        )

    print(f"\nSaved results to: {output_file}")


if __name__ == "__main__":
    main()