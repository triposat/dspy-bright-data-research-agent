import argparse
import json
from concurrent.futures import ThreadPoolExecutor

import dspy
from brightdata_tools import BrightDataTools
from research import AGENT_LM, EVAL_BLOCKLIST, ResearchQuestion, load_examples, make_agent, metric

parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, default=48)
parser.add_argument("--n", type=int, default=60)
parser.add_argument("--program", help="saved program from optimize.py")
parser.add_argument("--out", default="results.json")
parser.add_argument("--csv", default="simpleqa_verified.csv")
parser.add_argument("--local", action="store_true", help="local npx server instead of hosted")
parser.add_argument("--no-tools", action="store_true", help="model alone, for a baseline")
args = parser.parse_args()

dspy.configure(lm=AGENT_LM, adapter=dspy.ChatAdapter(use_native_function_calling=True))
bd = None
if args.no_tools:
    agent = dspy.ChainOfThought(ResearchQuestion)
else:
    bd = BrightDataTools(blocked_domains=EVAL_BLOCKLIST, remote=not args.local)
    agent = make_agent(bd.tools())
    if args.program:
        agent.load(args.program)

examples = load_examples(args.csv)[args.start:args.start + args.n]


def run(ex):
    try:
        pred = agent(question=ex.question)
        return {"question": ex.question, "gold": ex.gold, "answer": pred.answer,
                "correct": metric(ex, pred).score == 1.0}
    except Exception as e:  # count a crash as a wrong answer
        return {"question": ex.question, "gold": ex.gold, "answer": None,
                "correct": False, "error": str(e)[:200]}


with ThreadPoolExecutor(8) as pool:
    results = list(pool.map(run, examples))
json.dump(results, open(args.out, "w"), indent=1)
print(f"{sum(r['correct'] for r in results)} of {len(results)} correct")
if bd:
    print("Tool errors:", bd.stats["errors"])
    bd.close()
