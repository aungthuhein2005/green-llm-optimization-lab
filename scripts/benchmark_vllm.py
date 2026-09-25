import json
import time
from pathlib import Path

import requests
import yaml


CONFIG_PATH = "configs/serving.yaml"
PROMPTS_PATH = "prompts/benchmark_prompts.json"


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def send_request(url, model_name, prompt, max_tokens, temperature):
    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    start = time.perf_counter()

    response = requests.post(
        url,
        json=payload,
        timeout=300,
    )

    end = time.perf_counter()

    response.raise_for_status()

    data = response.json()

    return data, end - start


def main():
    config = load_yaml(CONFIG_PATH)
    prompts = load_json(PROMPTS_PATH)

    model_name = config["model_name"]
    url = config["server"]["url"]

    max_tokens = config["generation"]["max_new_tokens"]
    temperature = config["generation"]["temperature"]

    warmup_runs = config["benchmark"]["warmup_runs"]
    measured_runs = config["benchmark"]["measured_runs"]

    output_file = config["output_file"]

    print("Engine: vLLM")
    print("Model:", model_name)
    print("Concurrency: 1")

    all_results = []

    for item in prompts:
        prompt_id = item["id"]
        prompt = item["prompt"]

        print(f"\n=== {prompt_id} ===")

        # -------------------------
        # Warm-up
        # -------------------------

        for _ in range(warmup_runs):
            send_request(
                url,
                model_name,
                prompt,
                10,
                temperature,
            )

        run_results = []
        generated_text = ""

        # -------------------------
        # Measured runs
        # -------------------------

        for run_id in range(measured_runs):

            response, latency = send_request(
                url,
                model_name,
                prompt,
                max_tokens,
                temperature,
            )

            generated_text = response["choices"][0]["text"]

            usage = response.get("usage", {})

            prompt_tokens = usage.get(
                "prompt_tokens",
                0
            )

            output_tokens = usage.get(
                "completion_tokens",
                0
            )

            tokens_per_second = (
                output_tokens / latency
                if latency > 0
                else 0
            )

            result = {
                "run": run_id + 1,
                "latency_sec": latency,
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "tokens_per_sec": tokens_per_second,
            }

            run_results.append(result)

            print(
                f"Run {run_id + 1}: "
                f"{latency:.3f}s | "
                f"{output_tokens} tokens | "
                f"{tokens_per_second:.2f} tok/s"
            )

        avg_latency = sum(
            r["latency_sec"]
            for r in run_results
        ) / len(run_results)

        avg_speed = sum(
            r["tokens_per_sec"]
            for r in run_results
        ) / len(run_results)

        all_results.append({
            "prompt_id": prompt_id,
            "prompt": prompt,
            "average_latency_sec": avg_latency,
            "average_tokens_per_sec": avg_speed,
            "runs": run_results,
            "output": generated_text,
        })

    final_result = {
        "engine": "vllm",
        "model": model_name,
        "precision": "fp16",
        "concurrency": 1,
        "results": all_results,
    }

    Path(output_file).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            final_result,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nSaved results to: {output_file}")


if __name__ == "__main__":
    main()