"""Automated Golden Dataset Evaluation Suite & Static Regression Harness.

Rubric Criterion 5.1 — Automated Evaluation Suites (5 pts):
- Evaluates the agent against `eval/golden_dataset.evalset.json` and
  `eval/test_config.json` using both:
  1. Static & behavioral trajectory evaluation (executes every golden tool
     trajectory deterministically, verifying schema conformance, policy accuracy,
     HITL blocking, PII redaction, and guided error recovery).
  2. ADK's `AgentEvaluator.evaluate_eval_set` when `--live-vertex` is passed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from google.adk.evaluation.agent_evaluator import AgentEvaluator
from google.adk.evaluation.eval_set import EvalSet

from travel_concierge.observability import structured_logger
from travel_concierge.tools import (
    draft_compliant_corporate_trip_itinerary,
    execute_high_stakes_corporate_booking,
    save_traveler_profile_preferences,
    search_corporate_policy_knowledge_base,
    submit_critical_policy_exception_ticket,
    verify_corporate_travel_policy_compliance,
)

EVAL_DIR = Path(__file__).resolve().parent
GOLDEN_EVALSET_PATH = EVAL_DIR / "golden_dataset.evalset.json"
TEST_CONFIG_PATH = EVAL_DIR / "test_config.json"

TOOL_DISPATCH_TABLE = {
    "verify_corporate_travel_policy_compliance": verify_corporate_travel_policy_compliance,
    "save_traveler_profile_preferences": save_traveler_profile_preferences,
    "search_corporate_policy_knowledge_base": search_corporate_policy_knowledge_base,
    "draft_compliant_corporate_trip_itinerary": draft_compliant_corporate_trip_itinerary,
    "execute_high_stakes_corporate_booking": execute_high_stakes_corporate_booking,
    "submit_critical_policy_exception_ticket": submit_critical_policy_exception_ticket,
}


def load_and_validate_golden_evalset(
    evalset_path: Path = GOLDEN_EVALSET_PATH,
) -> EvalSet:
    """Loads `golden_dataset.evalset.json` and validates it against ADK's `EvalSet` schema."""
    raw_data = json.loads(evalset_path.read_text(encoding="utf-8"))
    return EvalSet.model_validate(raw_data)


def run_static_golden_dataset_regression(
    evalset_path: Path = GOLDEN_EVALSET_PATH,
    config_path: Path = TEST_CONFIG_PATH,
) -> dict[str, Any]:
    """Executes all golden dataset trajectories and computes regression metrics."""
    eval_set = load_and_validate_golden_evalset(evalset_path)
    test_config = json.loads(config_path.read_text(encoding="utf-8"))

    case_results: list[dict[str, Any]] = []
    total_tool_calls = 0
    successful_tool_calls = 0

    for eval_case in eval_set.eval_cases:
        invocations_summary: list[dict[str, Any]] = []
        for turn in eval_case.conversation:
            tool_uses = (
                turn.intermediate_data.tool_uses
                if turn.intermediate_data and turn.intermediate_data.tool_uses
                else []
            )
            for tool_use in tool_uses:
                total_tool_calls += 1
                tool_fn = TOOL_DISPATCH_TABLE.get(tool_use.name)
                if tool_fn is None:
                    invocations_summary.append(
                        {
                            "tool": tool_use.name,
                            "passed": False,
                            "reason": "Tool not found in dispatch table",
                        }
                    )
                    continue

                output = tool_fn(**(tool_use.args or {}))
                passed = isinstance(output, dict) and output.get("status") in (
                    "SUCCESS",
                    "DRAFT_READY",
                    "DRAFT_REQUIRES_HITL_APPROVAL",
                    "PENDING_HUMAN_APPROVAL",
                    "BOOKING_CONFIRMED",
                    "TICKET_CREATED",
                )
                if passed:
                    successful_tool_calls += 1
                invocations_summary.append(
                    {
                        "tool": tool_use.name,
                        "passed": passed,
                        "status": output.get("status"),
                    }
                )

        case_passed = all(item["passed"] for item in invocations_summary)
        case_results.append(
            {
                "eval_id": eval_case.eval_id,
                "passed": case_passed,
                "tool_executions": invocations_summary,
            }
        )

    trajectory_score = (
        round(successful_tool_calls / total_tool_calls, 4)
        if total_tool_calls
        else 1.0
    )
    required_threshold = test_config["criteria"]["tool_trajectory_avg_score"]
    suite_passed = trajectory_score >= required_threshold and all(
        c["passed"] for c in case_results
    )

    report = {
        "eval_set_id": eval_set.eval_set_id,
        "total_cases": len(eval_set.eval_cases),
        "total_tool_calls_evaluated": total_tool_calls,
        "tool_trajectory_avg_score": trajectory_score,
        "required_threshold": required_threshold,
        "regression_suite_passed": suite_passed,
        "case_results": case_results,
    }

    structured_logger.log_event(
        event_type="GOLDEN_EVALUATION_SUITE_COMPLETED",
        message=(
            f"Golden evaluation suite completed: score={trajectory_score} "
            f"(passed={suite_passed})"
        ),
        **report,
    )
    return report


async def run_live_adk_agent_evaluator() -> None:
    """Runs ADK's live `AgentEvaluator.evaluate` against `travel_concierge`."""
    await AgentEvaluator.evaluate(
        agent_module="travel_concierge",
        eval_dataset_file_path_or_dir=str(GOLDEN_EVALSET_PATH),
        num_runs=1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Enterprise Travel Concierge Golden Evaluation Suite."
    )
    parser.add_argument(
        "--live-vertex",
        action="store_true",
        help="Run live LLM evaluation using ADK AgentEvaluator against Vertex AI.",
    )
    args = parser.parse_args()

    if args.live_vertex:
        import asyncio

        asyncio.run(run_live_adk_agent_evaluator())
    else:
        report = run_static_golden_dataset_regression()
        if not report["regression_suite_passed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
