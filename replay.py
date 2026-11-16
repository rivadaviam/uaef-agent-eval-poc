"""Replay the 10 Gate 3 sessions through UAEF and compare with AgentCore's built-in scores.

Inputs (all in data/, from a 2026-07-30 AgentCore lab run):
  - data/transcripts/q{N}.txt        agent output as the AgentCore CLI printed it
  - data/ground-truth.json           answers verified by hand against Cost Explorer
  - data/agentcore-evals/eval_*.json AgentCore Evaluations scores for the same sessions

Usage:
  uv run python replay.py --offline            # deterministic metrics only, no AWS
  AWS_PROFILE=<profile> UAEF_POC_ACCOUNT_ID=<12-digit id> uv run python replay.py --runs 3
"""
import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "data"
# The run aborts unless the caller is exactly this account (guards against a wrong SSO profile).
EXPECTED_ACCOUNT = os.environ.get("UAEF_POC_ACCOUNT_ID", "")

LLM_METRICS = ["accuracy", "hallucination_score", "answer_relevance", "completeness"]
DETERMINISTIC_METRICS = ["tool_selection_accuracy"]

TOOL_NAME = "get_cost_breakdown"
# The tool output quoted by AgentCore's own evaluator explanations (same for every
# 7-day session; q9 asked for 30 days and quotes its own window).
FALLBACK_TOOL_OUTPUT = (
    "Gross cost by service (2026-07-23 to 2026-07-30):\n"
    "AWS Skill Builder Individual: $29.0000\n"
    "Credits applied: $-0.0000 (net may be ~$0)"
)


def assert_expected_account() -> None:
    import boto3

    if not EXPECTED_ACCOUNT:
        sys.exit("ABORT: set UAEF_POC_ACCOUNT_ID to the 12-digit account you intend to run in")

    ident = boto3.Session().client("sts").get_caller_identity()
    if ident["Account"] != EXPECTED_ACCOUNT:
        sys.exit(f"ABORT: running in account {ident['Account']}, expected {EXPECTED_ACCOUNT}")
    print(f"AWS account verified: {ident['Account']}")


def parse_transcript(path: Path) -> dict:
    raw = path.read_text()
    body = raw.split("\x1b[K")[-1]  # drop the spinner frames
    answer_part = body.split("Session:")[0]
    final = re.split(r"</thinking>", answer_part)[-1].strip()
    session_id = (path.with_suffix(".session")).read_text().strip()
    return {"final_response": final, "session_id": session_id}


def load_agentcore_scores() -> dict:
    """{evaluator: {qN: score}} from the two saved AgentCore eval runs."""
    out: dict = {}
    for f in sorted((DATA / "agentcore-evals").glob("eval_*.json")):
        for res in json.load(open(f))["results"]:
            name = res["evaluator"].split("/")[-1]
            for s in res["sessionScores"]:
                m = re.match(r"gate3q(\d+)-", s["sessionId"])
                if not m:
                    continue  # the smoke-test session
                out.setdefault(name, {}).setdefault(f"q{m.group(1)}", []).append(
                    {"value": s["value"], "label": s.get("label"), "explanation": s.get("explanation")}
                )
    return out


def build_case(q: dict, parsed: dict):
    from uaef.adapters import AgentCoreAdapter
    from uaef.models import GroundTruth, ToolCall

    now = datetime.now(timezone.utc)
    trace = AgentCoreAdapter().transform_to_canonical(
        {
            "user_input": q["question"],
            "final_response": parsed["final_response"],
            "session_id": parsed["session_id"],
            "agent_runtime_id": "InvoiceExplainer_InvoiceExplainer-6OivUsDx4c",
        }
    )
    # The adapter has no span data here, so the tool call is attached from what the
    # transcript and AgentCore's evaluator output show. Arguments are unknown: left empty.
    trace.tool_calls = [ToolCall(name=TOOL_NAME, arguments={}, result=FALLBACK_TOOL_OUTPUT, timestamp=now)]
    gt = GroundTruth(
        expected_output=q["expected_response"],
        expected_tool_calls=[ToolCall(name=q["expected_trajectory"], arguments={}, timestamp=now)],
    )
    return trace, gt


def flatten(result) -> dict:
    scores = {}
    for dim in result.dimension_results:
        for ms in dim.metric_scores:
            scores[ms.metric_name] = {"score": ms.score, "reasoning": ms.reasoning}
    return scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--offline", action="store_true", help="deterministic metrics only, no AWS calls")
    ap.add_argument("--only", type=int, nargs="*", help="question ids to run (default: all 10)")
    ap.add_argument("--tag", default="", help="suffix for the output file, e.g. sonnet or opus")
    args = ap.parse_args()

    from uaef.api import evaluate

    if not args.offline:
        assert_expected_account()

    gt_file = json.load(open(DATA / "ground-truth.json"))
    agentcore = load_agentcore_scores()
    metrics = DETERMINISTIC_METRICS if args.offline else DETERMINISTIC_METRICS + LLM_METRICS
    runs = 1 if args.offline else args.runs

    rows = []
    world_ctx = [f"{k}: {v}" for k, v in gt_file["_meta"]["verified_facts"].items()]
    judge = os.environ.get("UAEF_LLM_MODEL_ID", "default (us.anthropic.claude-sonnet-4-6)")
    for q in gt_file["questions"]:
        if args.only and q["id"] not in args.only:
            continue
        key = f"q{q['id']}"
        parsed = parse_transcript(DATA / "transcripts" / f"{key}.txt")
        trace, gt = build_case(q, parsed)
        per_run = []
        for _ in range(runs):
            # context = what the agent's tool returned: judges the answer against the trace
            res = evaluate(trace=trace, ground_truth=gt, metrics=metrics, context=[trace.tool_calls[0].result])
            scores = flatten(res)
            if not args.offline:
                # context = facts verified by hand against Cost Explorer: judges against the world
                res_w = evaluate(trace=trace, ground_truth=gt, metrics=["hallucination_score"], context=world_ctx)
                scores["hallucination_score@world"] = flatten(res_w)["hallucination_score"]
            per_run.append(scores)
        rows.append(
            {
                "id": q["id"],
                "question": q["question"],
                "expected": q["expected_response"],
                "agent_answer": parsed["final_response"],
                "uaef_runs": per_run,
                "agentcore": {ev: agentcore.get(ev, {}).get(key) for ev in agentcore},
            }
        )
        print(f"{key} done")

    name = "offline" if args.offline else "uaef-vs-agentcore"
    out = HERE / "results" / f"{name}{'-' + args.tag if args.tag else ''}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"judge": judge, "caching": os.environ.get("UAEF_ENABLE_CACHING"), "metrics": metrics, "runs": runs, "rows": rows}, indent=2, default=str))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
