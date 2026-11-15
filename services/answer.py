"""Extract the final answer from the model's chain of thought."""

import re

# \boxed{...} allowing one level of nested braces, e.g. \boxed{\frac{1}{2}}.
_BOXED = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def extract_answer(text: str) -> str:
    """Return the last \\boxed{} value, falling back to the last number.

    This is the same priority used in evaluation. An empty string means no
    answer could be found.
    """
    boxed = _BOXED.findall(text)
    if boxed:
        return boxed[-1].strip()

    numbers = _NUMBER.findall(text)
    return numbers[-1].rstrip(",") if numbers else ""
