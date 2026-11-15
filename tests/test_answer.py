"""Answer extraction follows the evaluation rule: last \\boxed{}, else last number."""

import pytest

from services.answer import extract_answer


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("8 + 7 = 15\n\nJAWABAN: \\boxed{15}", "15"),
        ("First \\boxed{3}, corrected: \\boxed{4}", "4"),
        ("Half of it: \\boxed{\\frac{1}{2}}", "\\frac{1}{2}"),
        ("She pays $1,200 in total.", "1,200"),
        ("The temperature drops to -3 degrees.", "-3"),
        ("Each costs 2.50 dollars.", "2.50"),
        ("No number here.", ""),
    ],
)
def test_extract_answer(text: str, expected: str) -> None:
    assert extract_answer(text) == expected
