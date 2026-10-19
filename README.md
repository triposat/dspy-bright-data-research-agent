# DSPy web research agent with the Bright Data MCP server

A DSPy agent that searches Google and reads web pages through the [Bright Data MCP server](https://brightdata.com/ai/mcp-server), plus the scripts to evaluate it and optimize it with GEPA.

## Files

| File | What it does |
|---|---|
| `agent.py` | Connects to the hosted MCP server and answers one question with `dspy.ReActV2` |
| `brightdata_tools.py` | Keeps one MCP session (hosted or local) on a background event loop and exposes its tools as sync `dspy.Tool`s, so `dspy.Evaluate` and GEPA can call them from threads. Pages are limited to 20,000 characters, and the scrape tool takes `start` and `find` to read the rest of a long page |
| `research.py` | Signature, agent, SimpleQA-style grader, GEPA metric, and a check for training answers copied into an optimized instruction |
| `evaluate.py` | Evaluates the agent on a slice of a question CSV |
| `optimize.py` | Runs GEPA on a training set built from the agent's wrong answers |
| `post_cutoff_questions.csv` | 14 questions about 2025 and 2026 events, answers checked against Wikipedia |

## Setup

You need Python 3.10 or later, [uv](https://docs.astral.sh/uv/), a [Bright Data API key](https://brightdata.com/cp/setting/users), and an OpenAI API key. Node.js is needed only for the local server (`--local`).

```bash
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
cp .env.example .env   # then fill in both keys
set -a; source .env; set +a
```

## Run

Answer one question:

```bash
python agent.py
```

Evaluate the agent on the post-cutoff questions, then the model alone for comparison:

```bash
python evaluate.py --csv post_cutoff_questions.csv --start 0 --n 14
python evaluate.py --csv post_cutoff_questions.csv --start 0 --n 14 --no-tools
```

Download the [SimpleQA Verified](https://huggingface.co/datasets/google/simpleqa-verified) questions into this folder, then evaluate the 60 test questions:

```bash
curl -L -o simpleqa_verified.csv https://huggingface.co/datasets/google/simpleqa-verified/resolve/main/simpleqa_verified.csv
python evaluate.py
```

Build a pool of questions outside the test set, then optimize with GEPA:

```bash
python evaluate.py --start 108 --n 400 --out pool.json
python optimize.py
python evaluate.py --program optimized_agent.json
```

`optimize.py` prints the optimized instruction, any training answers it repeats, how many requests reached the MCP server, and how many tool calls failed. Read the instruction before you use it. GEPA can copy training answers into it.

Add `--local` to any `evaluate.py` run to use the local server (`npx -y @brightdata/mcp@latest`) instead of the hosted one. For reproducible evaluations, replace `@latest` with the version you tested.

## Requests and pricing

You can start with free monthly requests, and the [MCP server pricing page](https://brightdata.com/pricing/mcp-server) lists the current allowance. Each tool call that the cache in `BrightDataTools` can't answer sends one request to the Bright Data MCP server.

## Versions

Tested with DSPy 3.4.1, MCP Python SDK 2.3.0, and MLflow 3.17.1 on the hosted Bright Data MCP server. Starting with DSPy 3.5, `dspy.ReAct` uses the ReActV2 implementation, and DSPy 3.6 removes the `ReActV2` name. When you upgrade, change `dspy.ReActV2` to `dspy.ReAct` and rerun `evaluate.py` to compare with your saved results.
