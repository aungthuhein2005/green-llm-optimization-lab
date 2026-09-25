import asyncio
import json
import statistics
import time
from pathlib import Path

import aiohttp
import yaml


CONFIG_PATH = "configs/serving.yaml"
PROMPTS_PATH = "prompts/benchmark_prompts.json"
OUTPUT_PATH = "results/vllm_concurrency.json"


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


async def send_request(
    session,
    url,
    model_name,
    prompt,
    max_tokens,
    temperature,
):
    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    start = time.perf_counter()

    async with session.post(
        url,
        json=payload,
        timeout=aiohttp.ClientTimeout(total=300),
    ) as response:

        data = await response.json()

    end = time.perf_counter()

    latency = end - start

    usage = data.get("usage", {})

    output_tokens = usage.get(
        "completion_tokens",
        0
    )

    return {
        "latency_sec": latency,
        "output_tokens": output_tokens,
    }


async def benchmark_concurrency(
    concurrency,
    prompt,
    config,
):
    url = config["server"]["url"]
    model_name = config["model_name"]

    max_tokens = config["generation"]["max_new_tokens"]
    temperature = config["generation"]["temperature"]

    async with aiohttp.ClientSession() as session:

        start = time.perf_counter()

        tasks = [
            send_request(
                session,
                url,
                model_name,
                prompt,
                max_tokens,
                temperature,
            )
            for _ in range(concurrency)
        ]

        results = await asyncio.gather(*tasks)

        total_time = time.perf_counter() - start

    latencies = [
        r["latency_sec"]
        for r in results
    ]

    total_output_tokens = sum(
        r["output_tokens"]
        for r in results
    )

    avg_latency = statistics.mean(latencies)

    throughput = (
        total_output_tokens / total_time
        if total_time > 0
        else 0
    )

    requests_per_sec = (
        concurrency / total_time
        if total_time > 0
        else 0
    )

    return {
        "concurrency": concurrency,
        "total_time_sec": total_time,
        "average_latency_sec": avg_latency,
        "throughput_tokens_sec": throughput,
        "requests_per_sec": requests_per_sec,
        "total_output_tokens": total_output_tokens,
        "individual_latencies": latencies,
    }


async def main():
    config = load_yaml(CONFIG_PATH)
    prompts = load_json(PROMPTS_PATH)

    # Use one fixed prompt to isolate concurrency effect
    prompt = prompts[0]["prompt"]

    concurrency_levels = [1, 4, 8]

    all_results = []

    # Warm-up
    print("Warm-up...")
    await benchmark_concurrency(
        1,
        prompt,
        config,
    )

    for concurrency in concurrency_levels:

        print(
            f"\n=== Concurrency {concurrency} ==="
        )

        result = await benchmark_concurrency(
            concurrency,
            prompt,
            config,
        )

        all_results.append(result)

        print(
            f"Total time: "
            f"{result['total_time_sec']:.3f}s"
        )

        print(
            f"Average latency: "
            f"{result['average_latency_sec']:.3f}s"
        )

        print(
            f"Throughput: "
            f"{result['throughput_tokens_sec']:.2f} tok/s"
        )

        print(
            f"Requests/sec: "
            f"{result['requests_per_sec']:.2f}"
        )

    final = {
        "engine": "vllm",
        "model": config["model_name"],
        "precision": "fp16",
        "prompt": prompt,
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
            final,
            f,
            indent=2,
        )

    print(
        f"\nSaved results to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    asyncio.run(main())