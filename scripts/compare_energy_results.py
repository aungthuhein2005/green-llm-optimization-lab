import json
from pathlib import Path
from statistics import mean


FILES = {
    "FP16": Path("results/energy_fp16.json"),
    "INT8": Path("results/energy_int8.json"),
    "INT4": Path("results/energy_int4.json"),
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def average_metric(data, key):
    values = [
        item[key]
        for item in data["results"]
    ]

    return mean(values)


def calculate_summary(data):
    return {
        "tokens_per_sec": average_metric(
            data,
            "tokens_per_sec"
        ),
        "average_power_watts": average_metric(
            data,
            "average_power_watts"
        ),
        "energy_joules": average_metric(
            data,
            "energy_joules"
        ),
        "joules_per_token": average_metric(
            data,
            "joules_per_token"
        ),
        "generation_time_sec": average_metric(
            data,
            "generation_time_sec"
        ),
    }


def percent_change(new, baseline):
    return (
        (new - baseline)
        / baseline
        * 100
    )


def main():
    raw_results = {
        name: load_json(path)
        for name, path in FILES.items()
    }

    summaries = {
        name: calculate_summary(data)
        for name, data in raw_results.items()
    }

    print("\n=== GREEN LLM ENERGY COMPARISON ===\n")

    header = (
        f"{'Config':<8}"
        f"{'Tok/s':>12}"
        f"{'Power(W)':>12}"
        f"{'Energy(J)':>14}"
        f"{'J/token':>12}"
        f"{'Latency(s)':>14}"
    )

    print(header)
    print("-" * len(header))

    for name, summary in summaries.items():
        print(
            f"{name:<8}"
            f"{summary['tokens_per_sec']:>12.2f}"
            f"{summary['average_power_watts']:>12.2f}"
            f"{summary['energy_joules']:>14.2f}"
            f"{summary['joules_per_token']:>12.4f}"
            f"{summary['generation_time_sec']:>14.3f}"
        )

    fp16 = summaries["FP16"]

    print("\n=== CHANGE VS FP16 BASELINE ===\n")

    for name in ["INT8", "INT4"]:
        current = summaries[name]

        throughput_change = percent_change(
            current["tokens_per_sec"],
            fp16["tokens_per_sec"]
        )

        power_change = percent_change(
            current["average_power_watts"],
            fp16["average_power_watts"]
        )

        energy_change = percent_change(
            current["energy_joules"],
            fp16["energy_joules"]
        )

        energy_per_token_change = percent_change(
            current["joules_per_token"],
            fp16["joules_per_token"]
        )

        latency_change = percent_change(
            current["generation_time_sec"],
            fp16["generation_time_sec"]
        )

        print(name)
        print(
            f"  Throughput change: "
            f"{throughput_change:+.2f}%"
        )

        print(
            f"  Average power change: "
            f"{power_change:+.2f}%"
        )

        print(
            f"  Energy change: "
            f"{energy_change:+.2f}%"
        )

        print(
            f"  Joules/token change: "
            f"{energy_per_token_change:+.2f}%"
        )

        print(
            f"  Latency change: "
            f"{latency_change:+.2f}%"
        )

        print()

    print("=== BEST BY METRIC ===\n")

    best_throughput = max(
        summaries,
        key=lambda x: summaries[x]["tokens_per_sec"]
    )

    lowest_power = min(
        summaries,
        key=lambda x: summaries[x]["average_power_watts"]
    )

    lowest_energy = min(
        summaries,
        key=lambda x: summaries[x]["energy_joules"]
    )

    lowest_joules_token = min(
        summaries,
        key=lambda x: summaries[x]["joules_per_token"]
    )

    lowest_latency = min(
        summaries,
        key=lambda x: summaries[x]["generation_time_sec"]
    )

    print(
        f"Highest throughput: {best_throughput} "
        f"({summaries[best_throughput]['tokens_per_sec']:.2f} tok/s)"
    )

    print(
        f"Lowest average power: {lowest_power} "
        f"({summaries[lowest_power]['average_power_watts']:.2f} W)"
    )

    print(
        f"Lowest energy/request: {lowest_energy} "
        f"({summaries[lowest_energy]['energy_joules']:.2f} J)"
    )

    print(
        f"Lowest energy/token: {lowest_joules_token} "
        f"({summaries[lowest_joules_token]['joules_per_token']:.4f} J/token)"
    )

    print(
        f"Lowest latency: {lowest_latency} "
        f"({summaries[lowest_latency]['generation_time_sec']:.3f} s)"
    )


if __name__ == "__main__":
    main()