"""Live JUW RAG evaluation.

Run:
    python tests/evaluate.py

This is intentionally a small release-gate harness, not a fabricated accuracy
benchmark. Expand the cases with verified answers whenever the source corpus
changes. The script requires GROQ_API_KEY and the project dependencies.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app import chat, ChatRequest, NO_ANSWER, OUT_OF_SCOPE


CASES = [
    {
        "question": "What university do you provide information about?",
        "must_contain": ["Jinnah University for Women"],
    },
    {
        "question": "What are the minimum marks for a BS program at JUW?",
        "must_contain": ["45%"],
    },
    {
        "question": "What minimum marks are required for Computer Science, Food Science, Biotechnology, Biochemistry and Business Administration?",
        "must_contain": ["50%"],
    },
    {
        "question": "What minimum marks are required for Pharmacy and DPT?",
        "must_contain": ["60%"],
    },
    {
        "question": "What percentage of the admission merit is based on the university admission test?",
        "must_contain": ["60%"],
    },
    {
        "question": "What is the JUW campus address?",
        "must_contain": ["Nazimabad"],
    },
    {
        "question": "What is the general JUW email address?",
        "must_contain": ["info@juw.edu.pk"],
    },
    {
        "question": "What email should I use for admission queries at JUW?",
        "must_contain": ["admission@juw.edu.pk"],
    },
    {
        "question": "Does JUW offer Pharm-D?",
        "must_contain": ["Pharm-D"],
    },
    {
        "question": "How many faculties and departments are listed in the JUW knowledge base?",
        "must_contain": ["5", "28"],
    },
    {
        "question": "How many times a year does the Student Handbook say admissions are conducted?",
        "must_contain": ["twice"],
    },
    {
        "question": "What does a credit hour mean at JUW?",
        "must_contain": ["credit hour"],
    },
    {
        "question": "What is the date range listed for the 2025 final examinations?",
        "must_contain": ["18th June", "3rd July"],
    },
    {
        "question": "Tell me about admissions at Karachi University.",
        "equals": OUT_OF_SCOPE,
    },
    {
        "question": "Does JUW have a campus on Mars?",
        "equals": NO_ANSWER,
    },
    {
        "question": "What is the capital of France?",
        "equals": NO_ANSWER,
    },
]


def matches(answer, case):
    if "equals" in case:
        return answer.strip() == case["equals"]
    answer_lower = answer.lower()
    return all(term.lower() in answer_lower for term in case["must_contain"])


def main():
    passed = 0
    for case in CASES:
        result = chat(ChatRequest(message=case["question"]))
        ok = matches(result.answer, case)
        passed += ok
        print(
            f"{'PASS' if ok else 'FAIL'} | {case['question']}\n"
            f"  {result.answer}\n"
            f"  Sources: {', '.join(result.sources) or 'none'}\n"
        )

    print(f"Passed: {passed}/{len(CASES)}")
    raise SystemExit(0 if passed == len(CASES) else 1)


if __name__ == "__main__":
    main()
