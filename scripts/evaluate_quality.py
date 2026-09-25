import ast
import json
import re
import textwrap
from pathlib import Path


FILES = {
    "FP16": Path("results/fp16.json"),
    "INT8": Path("results/int8.json"),
    "INT4": Path("results/int4.json"),
}

OUTPUT_PATH = Path("results/quality_evaluation.json")
MANUAL_REVIEW_PATH = Path("results/manual_quality_review.json")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_outputs(data):
    return {
        item["prompt_id"]: {
            "prompt": item["prompt"],
            "output": item["output"],
        }
        for item in data["results"]
    }


# --------------------------------------------------
# Reasoning evaluation
# --------------------------------------------------

def evaluate_reasoning(text):
    """
    Expected:
    240 / 8 = 30 tokens/sec

    Score:
    0 = incorrect
    1 = final answer 30 present
    2 = answer + valid calculation evidence
    """

    normalized = text.lower()

    score = 0

    has_answer = bool(
        re.search(r"\b30(?:\.0+)?\b", normalized)
    )

    if has_answer:
        score += 1

    calculation_patterns = [
        r"240\s*/\s*8",
        r"240\s*÷\s*8",
        r"240\s+tokens?.*8\s+seconds?",
        r"240.*8.*30",
    ]

    has_calculation = any(
        re.search(pattern, normalized, re.DOTALL)
        for pattern in calculation_patterns
    )

    if has_answer and has_calculation:
        score += 1

    return min(score, 2)


# --------------------------------------------------
# Instruction-following evaluation
# --------------------------------------------------

def extract_candidate_items(text):
    """
    Try to identify list-like semantic items.
    """

    text = text.strip()

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    items = []

    # Numbered / bullet lists
    for line in lines:
        match = re.match(
            r"^(?:\d+[.)]|[-*•])\s*(.+)",
            line
        )

        if match:
            item = match.group(1).strip()

            if item:
                items.append(item)

    if items:
        return items

    # Fallback: split prose on semicolons
    if ";" in text:
        parts = [
            p.strip()
            for p in text.split(";")
            if p.strip()
        ]

        if len(parts) >= 2:
            return parts

    # Fallback: split short sentence-like answers
    sentences = [
        s.strip()
        for s in re.split(r"[.!?]+", text)
        if s.strip()
    ]

    return sentences


def evaluate_instruction(text):
    """
    Prompt:
    Give exactly three advantages of using GPUs for LLM inference.
    Do not provide explanations.

    Score:
    2 = exactly 3 concise advantages
    1 = roughly follows instruction
    0 = does not follow instruction
    """

    items = extract_candidate_items(text)

    count = len(items)

    if count == 3:
        # Penalize very long explanatory items
        avg_words = sum(
            len(item.split())
            for item in items
        ) / 3

        if avg_words <= 15:
            return 2

        return 1

    if 2 <= count <= 4:
        return 1

    return 0


# --------------------------------------------------
# Coding evaluation
# --------------------------------------------------

def extract_python_code(text):
    """
    Prefer fenced Python blocks.
    Otherwise try to extract from first `def`.
    """

    fenced = re.findall(
        r"```(?:python)?\s*(.*?)```",
        text,
        re.DOTALL | re.IGNORECASE,
    )

    if fenced:
        return fenced[0].strip()

    index = text.find("def ")

    if index != -1:
        return text[index:].strip()

    return None


def find_function_name(code):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            return node.name

    return None


def evaluate_coding(text):
    """
    Actually executes the generated factorial function.

    Score:
    0 = no valid runnable function
    1 = partially correct
    2 = passes all test cases
    """

    code = extract_python_code(text)

    if not code:
        return 0

    function_name = find_function_name(code)

    if not function_name:
        return 0

    namespace = {}

    try:
        exec(
            compile(code, "<generated>", "exec"),
            {"__builtins__": __builtins__},
            namespace,
        )
    except Exception:
        return 0

    func = namespace.get(function_name)

    if not callable(func):
        return 0

    tests = {
        0: 1,
        1: 1,
        2: 2,
        3: 6,
        5: 120,
        7: 5040,
    }

    passed = 0

    for value, expected in tests.items():
        try:
            result = func(value)

            if result == expected:
                passed += 1

        except Exception:
            pass

    if passed == len(tests):
        return 2

    if passed >= len(tests) // 2:
        return 1

    return 0


# --------------------------------------------------
# Manual review preparation
# --------------------------------------------------

def build_manual_review(outputs):
    """
    Creates a human-review structure for open-ended tasks.
    """

    manual = {}

    for config_name, config_outputs in outputs.items():

        manual[config_name] = {
            "explanation": {
                "prompt": config_outputs["explanation"]["prompt"],
                "output": config_outputs["explanation"]["output"],
                "rubric": {
                    "technical_correctness_0_to_2": None,
                    "relevance_0_to_1": None,
                },
                "max_score": 3,
            },

            "llm_concept": {
                "prompt": config_outputs["llm_concept"]["prompt"],
                "output": config_outputs["llm_concept"]["output"],
                "rubric": {
                    "technical_correctness_0_to_2": None,
                    "completeness_0_to_1": None,
                },
                "max_score": 3,
            },
        }

    return manual


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    raw = {
        name: load_json(path)
        for name, path in FILES.items()
    }

    outputs = {
        name: get_outputs(data)
        for name, data in raw.items()
    }

    evaluation = {}

    print("\n=== AUTOMATIC QUALITY EVALUATION ===\n")

    for config_name, config_outputs in outputs.items():

        reasoning = evaluate_reasoning(
            config_outputs["reasoning"]["output"]
        )

        instruction = evaluate_instruction(
            config_outputs["instruction"]["output"]
        )

        coding = evaluate_coding(
            config_outputs["coding"]["output"]
        )

        automatic_total = (
            reasoning
            + instruction
            + coding
        )

        evaluation[config_name] = {
            "automatic": {
                "reasoning": {
                    "score": reasoning,
                    "max_score": 2,
                },
                "instruction_following": {
                    "score": instruction,
                    "max_score": 2,
                },
                "coding": {
                    "score": coding,
                    "max_score": 2,
                },
                "total_score": automatic_total,
                "max_score": 6,
            },
            "manual_score": None,
            "final_score": None,
            "final_max_score": 12,
        }

        print(config_name)
        print(f"  Reasoning:   {reasoning}/2")
        print(f"  Instruction: {instruction}/2")
        print(f"  Coding:      {coding}/2")
        print(f"  Auto total:  {automatic_total}/6")
        print()

    manual_review = build_manual_review(outputs)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            evaluation,
            f,
            indent=2,
            ensure_ascii=False,
        )

    with open(
        MANUAL_REVIEW_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manual_review,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"Saved automatic evaluation to: "
        f"{OUTPUT_PATH}"
    )

    print(
        f"Saved manual review file to: "
        f"{MANUAL_REVIEW_PATH}"
    )


if __name__ == "__main__":
    main()