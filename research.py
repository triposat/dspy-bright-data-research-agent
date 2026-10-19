import csv
import random

import dspy

# Hosts that publish the benchmark answers, excluded while evaluating
EVAL_BLOCKLIST = ("huggingface.co", "github.com", "raw.githubusercontent.com",
                  "kaggle.com", "openaipublic.blob.core.windows.net")

AGENT_LM = dspy.LM("openai/gpt-5.4-mini", temperature=1.0, max_tokens=16000)
JUDGE_LM = dspy.LM("openai/gpt-5.4-mini", temperature=1.0, max_tokens=16000)


class ResearchQuestion(dspy.Signature):
    """Answer the question with facts found on the web."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="a short, direct answer")
    sources: list[str] = dspy.OutputField(desc="URLs of the pages that support the answer")


def make_agent(tools):
    return dspy.ReActV2(ResearchQuestion, tools=tools, max_iters=8)


def load_examples(path="simpleqa_verified.csv", seed=7):
    rows = list(csv.DictReader(open(path)))
    random.Random(seed).shuffle(rows)
    return [dspy.Example(question=r["problem"], gold=r["answer"]).with_inputs("question")
            for r in rows]


class Grade(dspy.Signature):
    """Grade a predicted answer against the gold answer, SimpleQA style.
    CORRECT: the prediction contains the gold answer's meaning and does not contradict it.
    Small spelling or formatting differences are fine; for numbers and dates the value must match.
    INCORRECT: the prediction states a different or contradictory answer, or hedges between several.
    NOT_ATTEMPTED: the prediction does not give an answer."""

    question: str = dspy.InputField()
    gold_answer: str = dspy.InputField()
    predicted_answer: str = dspy.InputField()
    grade: str = dspy.OutputField(desc="CORRECT, INCORRECT, or NOT_ATTEMPTED")


grader = dspy.Predict(Grade)


def tool_calls(pred):
    calls = []
    history = getattr(pred, "history", None)
    for event in (history.messages if history else []):
        for call in getattr(event.get("tool_calls"), "tool_calls", []):
            if call.name != "submit":
                calls.append((call.name, call.args))
    return calls


def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """Score for GEPA, plus feedback that never shows the reference answer."""
    answer = getattr(pred, "answer", "") or ""
    with dspy.context(lm=JUDGE_LM):
        grade = grader(question=gold.question, gold_answer=gold.gold,
                       predicted_answer=answer).grade.strip().upper()
    correct = grade.startswith("CORRECT")
    calls = tool_calls(pred)
    searches = [args.get("query") for name, args in calls if name == "search_engine"]
    pages = [args.get("url") for name, args in calls if name == "scrape_as_markdown"]
    feedback = (f"Grade: {'CORRECT' if correct else 'INCORRECT'}. Agent answered: {answer!r}. "
                f"Searches ({len(searches)}): {searches}. Pages opened ({len(pages)}): {pages}.")
    if not correct and not pages:
        feedback += " The agent answered without opening a page."
    return dspy.Prediction(score=float(correct), feedback=feedback)


def leaked_answers(instruction, examples):
    """Reference answers (4+ characters) that an optimized instruction repeats."""
    text = instruction.lower()
    return sorted({ex.gold for ex in examples if len(ex.gold) >= 4 and ex.gold.lower() in text})
